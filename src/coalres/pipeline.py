"""Orkestrasi pipeline estimasi sumberdaya, dari Excel + DXF sampai laporan."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .classify import CLASS_NAMES, classification_frame, classify_seam, spacing_statistics
from .config import Config
from .grid import Grid, interpolate_to_grid
from .io.dxf import load_topo_points
from .io.excel import load_drillholes
from .model import build_stratmodel, collapse_plies, structural_residuals
from .quality import add_derived_quality
from .report import (
    map_classification, map_continuous, reporting_notes, write_excel_report,
    write_grid_ascii,
)
from .resource import apply_cutoffs, summarise_model, totals_by_class
from .validate import ValidationReport, check_collar_vs_topo, run_all


@dataclass
class Results:
    config: Config
    grid: Grid
    topo: np.ndarray
    model: object
    summary: pd.DataFrame
    totals: pd.DataFrame
    findings: pd.DataFrame
    class_grids: dict[str, np.ndarray]
    spacing: pd.DataFrame
    residuals: pd.DataFrame
    outputs: list[Path] = field(default_factory=list)


def _sample_grid_at(grid: Grid, surface: np.ndarray, x, y) -> np.ndarray:
    col = np.clip(((np.asarray(x, float) - grid.xmin) / grid.cell_size).round().astype(int),
                  0, grid.ncols - 1)
    row = np.clip(((np.asarray(y, float) - grid.ymin) / grid.cell_size).round().astype(int),
                  0, grid.nrows - 1)
    return surface[row, col]


def run(
    drillholes_xlsx: str | Path,
    topo_dxf: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> Results:
    cfg = Config.load(config_path)
    out_dir = Path(output_dir or cfg["output.dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # --- 1. Import ---------------------------------------------------------- #
    log("[1/7] Membaca data lubang bor ...")
    data = load_drillholes(drillholes_xlsx, cfg)
    log(f"      {data.n_holes} lubang, {len(data.seam)} interval, "
        f"{len(data.seams)} seam: {', '.join(data.seams)}")

    log("[2/7] Membaca topografi ...")
    topo_points, topo_info = load_topo_points(topo_dxf)
    log(f"      {topo_info['n_points']:,} titik, RL {topo_info['z_min']:.1f}-{topo_info['z_max']:.1f} m")
    if topo_info["dropped_zero_z"]:
        log(f"      PERINGATAN: {topo_info['dropped_zero_z']} titik ber-Z=0 dibuang")

    # --- 2. Grid ------------------------------------------------------------ #
    collar = data.collar
    grid = Grid.from_extent(
        min(collar["east"].min(), topo_info["extent"][0]),
        max(collar["east"].max(), topo_info["extent"][1]),
        min(collar["north"].min(), topo_info["extent"][2]),
        max(collar["north"].max(), topo_info["extent"][3]),
        cell_size=cfg["grid.cell_size"], margin=cfg["grid.margin"],
    )
    log(f"[3/7] Grid {grid.ncols} x {grid.nrows} sel @ {grid.cell_size} m")

    spacing_stats = spacing_statistics(collar)
    if "nearest_median" in spacing_stats:
        log(f"      Spasi bor median {spacing_stats['nearest_median']:.0f} m "
            f"(saran ukuran sel ~{spacing_stats['suggested_cell_size']:.0f} m)")

    topo = interpolate_to_grid(
        grid, topo_points[:, 0], topo_points[:, 1], topo_points[:, 2],
        method="linear", max_extrapolation=None,
    )

    # --- 3. Validasi -------------------------------------------------------- #
    log("[4/7] Validasi ...")
    report = ValidationReport()
    topo_at_collar = _sample_grid_at(grid, topo, collar["east"], collar["north"])
    run_all(data, report, cfg, topo_at_collar=topo_at_collar)
    log(f"      {report.summary()}")
    for finding in report.findings:
        if finding.severity == "ERROR":
            where = f" [{finding.hole_id or ''}{'/' + finding.seam if finding.seam else ''}]"
            log(f"      ERROR {finding.code}{where}: {finding.message}")

    # --- 4. Turunan kualitas & compositing ---------------------------------- #
    log("[5/7] Membangun model stratigrafi ...")
    seam = add_derived_quality(data.seam, cfg)
    collapsed = collapse_plies(seam)

    # Parting tebal memisahkan seam; dicatat sebagai temuan, bukan didiamkan.
    max_parting = cfg.get("seam_rules.max_parting_thickness_m")
    if max_parting is not None and "parting_thickness" in collapsed:
        thick = collapsed[collapsed["parting_thickness"] > float(max_parting)]
        for _, row in thick.iterrows():
            report.add(
                "WARNING", "PARTING_EXCEEDS_CUTOFF",
                f"parting total {row['parting_thickness']:.2f} m melebihi cutoff "
                f"{max_parting} m - seam ini semestinya dipecah, bukan dikompositkan",
                hole_id=row["hole_id"], seam=row["seam"],
                value=float(row["parting_thickness"]),
            )

    order = cfg.get("stratigraphy") or []
    model = build_stratmodel(collapsed, collar, topo, grid, order, cfg)

    # --- 5. Klasifikasi ----------------------------------------------------- #
    log("[6/7] Klasifikasi & perhitungan sumberdaya ...")
    quality_cols = [c for c in ("ash_adb", "cv_adb", "rd_adb") if c in collapsed.columns]
    class_grids: dict[str, np.ndarray] = {}
    for name in model.order:
        pts = collapsed[collapsed["seam"] == name].merge(
            collar[["hole_id", "east", "north"]], on="hole_id", how="inner"
        )
        pts["has_quality"] = (
            pts[quality_cols].notna().any(axis=1) if quality_cols else False
        )
        # Cutoff diterapkan LANGSUNG pada grid kelas, sehingga luas di peta,
        # di tabel, dan di grid keluaran menyatakan hal yang sama. Tanpa ini
        # legenda peta melaporkan luas yang lebih besar daripada yang benar-benar
        # ikut terhitung sebagai tonase.
        cutoff_mask, rejected = apply_cutoffs(model.seam(name), cfg)
        class_grids[name] = classify_seam(grid, pts, cfg, present_mask=cutoff_mask)
        for reason, n in rejected.items():
            if n:
                report.add(
                    "INFO", "CUTOFF_APPLIED",
                    f"{n} sel ditolak oleh cutoff '{reason}'", seam=name, value=float(n),
                )

    summary = summarise_model(model, class_grids, cfg)
    totals = totals_by_class(summary)

    residual_frames = [structural_residuals(model, name) for name in model.order]
    residuals = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    if not residuals.empty:
        # Residual struktur besar yang mengelompok adalah petunjuk sesar atau
        # salah korelasi - keduanya tak terlihat dari kehalusan grid.
        sd = residuals["residual"].std()
        if sd and sd > 0:
            for _, row in residuals[residuals["residual"].abs() > 3 * sd].iterrows():
                report.add(
                    "WARNING", "STRUCTURE_RESIDUAL",
                    f"RL floor menyimpang {row['residual']:+.2f} m dari permukaan model "
                    f"({row['residual'] / sd:+.1f} sigma). Periksa korelasi atau kemungkinan sesar.",
                    hole_id=row["hole_id"], seam=row["seam"], value=float(row["residual"]),
                )

    # --- 6. Keluaran -------------------------------------------------------- #
    log("[7/7] Menulis keluaran ...")
    outputs: list[Path] = []
    findings = report.to_frame()

    spacing_df = pd.DataFrame([spacing_stats])
    excel = write_excel_report(
        out_dir / "resource_report.xlsx", summary, totals, findings,
        spacing=spacing_df, residuals=residuals, notes=reporting_notes(),
    )
    outputs.append(excel)

    if cfg["output.write_maps"]:
        holes_xy = collar[["east", "north"]]
        map_continuous(grid, topo, "Topografi", "RL (m)",
                       out_dir / "maps" / "topo.png", holes=holes_xy)
        for name in model.order:
            s = model.seam(name)
            sub = f"n = {s.n_holes} lubang menembus seam ini"
            map_continuous(grid, np.where(s.present, s.thickness_vertical, np.nan),
                           f"Isopach {name}", "Ketebalan vertikal (m)",
                           out_dir / "maps" / f"isopach_{name}.png",
                           holes=holes_xy, subtitle=sub)
            map_continuous(grid, s.floor_rl, f"Struktur floor {name}", "RL (m)",
                           out_dir / "maps" / f"structure_{name}.png",
                           holes=holes_xy, subtitle=sub)
            map_classification(grid, class_grids[name], f"Klasifikasi {name}",
                               out_dir / "maps" / f"class_{name}.png",
                               holes=holes_xy,
                               subtitle="Berdasarkan spasi titik data - bukan klasifikasi final")
        outputs.extend(sorted((out_dir / "maps").glob("*.png")))

    if cfg["output.write_grids"]:
        for name in model.order:
            s = model.seam(name)
            write_grid_ascii(grid, s.thickness_vertical, out_dir / "grids" / f"{name}_thickness.asc")
            write_grid_ascii(grid, s.floor_rl, out_dir / "grids" / f"{name}_floor_rl.asc")
            write_grid_ascii(grid, s.roof_rl, out_dir / "grids" / f"{name}_roof_rl.asc")
        write_grid_ascii(grid, topo, out_dir / "grids" / "topo.asc")
        outputs.extend(sorted((out_dir / "grids").glob("*.asc")))

    return Results(
        config=cfg, grid=grid, topo=topo, model=model, summary=summary,
        totals=totals, findings=findings, class_grids=class_grids,
        spacing=spacing_df, residuals=residuals, outputs=outputs,
    )
