"""Orkestrasi estimasi penuh, dari berkas masukan sampai deliverable."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .audit.checks import AuditReport, run_audit
from .config import Config, file_digest
from .density import resolve_in_situ_rd
from .errors import MissingDataError
from .estimate import HolePoint, ResourcePolygon, build_polygons, rd_sensitivity, to_frame
from .io.dxf import load_topography
from .io.excel import Workbook, load_workbook, normalise_hole_id
from .io.las import load_las
from .io.quality_table import load_quality_table
from .logging_setup import get_logger
from .quality import basis_report, resolve_quality, summarise
from .report import (
    assumptions_frame, map_contours, map_polygons, map_surface, write_ascii_grid,
    write_excel, write_geotiff, write_grid_sidecar, write_qaqc_markdown,
    write_run_log, write_vectors,
)
from .rpeee import apply_constraints, label_warning, reconciliation_table
from .seams import SeamIntersection, assumptions, build_intersections, to_frame as seams_frame
from .topo import Surface, build_surface, depth_limit_extent, difference, subcrop_extent

log = get_logger("pipeline")

GRID_SPACING_M = 25.0
GRID_MARGIN_M = 250.0


@dataclass
class Inputs:
    workbooks: list[Workbook]
    las_files: dict[str, object]
    quality: object | None
    topo_points: object | None
    digests: dict[str, str] = field(default_factory=dict)

    @property
    def by_hole(self) -> dict[str, Workbook]:
        return {normalise_hole_id(w.hole_id): w for w in self.workbooks}


@dataclass
class Results:
    config: Config
    audit: AuditReport
    polygons: list[ResourcePolygon]
    frames: dict[str, pd.DataFrame]
    surfaces: dict[str, dict[str, Surface]]
    outputs: list[Path] = field(default_factory=list)


def gather(cfg: Config) -> Inputs:
    workbook_dir = cfg.paths.workbook_dir
    if not workbook_dir.exists():
        raise MissingDataError(f"paths.workbook_dir tidak ada: {workbook_dir}")
    paths = sorted(p for p in workbook_dir.glob("*.xls*") if not p.name.startswith("~$"))
    if not paths:
        raise MissingDataError(f"tidak ada workbook Excel di {workbook_dir}")

    digests: dict[str, str] = {}
    workbooks = []
    for path in paths:
        workbooks.append(load_workbook(path))
        digests[path.name] = file_digest(path)

    las_files = {}
    if cfg.paths.las_dir and cfg.paths.las_dir.exists():
        for path in sorted(cfg.paths.las_dir.glob("*.[Ll][Aa][Ss]")):
            las = load_las(path)
            las_files[normalise_hole_id(las.well_name or path.stem)] = las
            digests[path.name] = file_digest(path)

    quality = None
    if cfg.paths.quality_table:
        quality = load_quality_table(cfg.paths.quality_table)
        digests[cfg.paths.quality_table.name] = file_digest(cfg.paths.quality_table)

    topo_points = None
    if cfg.paths.topography_dxf and cfg.paths.topography_dxf.exists():
        topo_points = load_topography(cfg.paths.topography_dxf)
        digests[cfg.paths.topography_dxf.name] = file_digest(cfg.paths.topography_dxf)

    return Inputs(workbooks, las_files, quality, topo_points, digests)


def _collar(wb: Workbook, cfg: Config) -> tuple[float, float, float]:
    """Koordinat dari sumber otoritatif yang dipilih konfigurasi."""
    if cfg.authoritative_coordinate_source == "collar_ts":
        row = wb.sheets["Collar"].frame.iloc[0]
        return (float(row["east"]), float(row["north"]), float(row["rl"]))
    bhc = wb.sheets.get("BHC")
    if bhc is None or bhc.frame.empty:
        raise MissingDataError(f"{wb.hole_id}: BHC tidak tersedia untuk sumber koordinat bhc_gps")
    row = bhc.frame.iloc[0]
    return (float(row["gps_east"]), float(row["gps_north"]), float(row["gps_rl"]))


def build_hole_points(
    inputs: Inputs, cfg: Config
) -> tuple[dict[str, list[HolePoint]], pd.DataFrame, pd.DataFrame, list[str]]:
    """Bangun titik per seam, lengkap dengan tebal, RD, dan kualitasnya."""
    intersections: list[SeamIntersection] = []
    for wb in inputs.workbooks:
        intersections.extend(build_intersections(wb, cfg))

    quality_by_key: dict[tuple[str, str], object] = {}
    quality_issues = pd.DataFrame()
    if inputs.quality is not None:
        resolved, quality_issues = resolve_quality(
            intersections, inputs.quality, inputs.by_hole, cfg
        )
        quality_by_key = {(q.hole_id, q.seam): q for q in resolved}

    points: dict[str, list[HolePoint]] = {}
    rows, rd_notes = [], []
    excluded = []

    for item in intersections:
        wb = inputs.by_hole[item.hole_id]
        east, north, rl = _collar(wb, cfg)
        q = quality_by_key.get(item.key)

        try:
            rd, assumed, note = resolve_in_situ_rd(
                rd_value=q.rd_t_per_m3 if q else None,
                rd_basis=q.rd_basis if q else "unknown",
                total_moisture_ar_pct=(q.values.get("TM_ar") if q else None),
                inherent_moisture_adb_pct=(q.values.get("M_adb") if q else None),
                assumed_rd_t_per_m3=cfg.assumed_rd_t_per_m3,
            )
        except MissingDataError as exc:
            excluded.append({"hole_id": item.hole_id, "seam": item.seam,
                             "alasan": str(exc)})
            log.warning(f"{item.hole_id}/{item.seam} dikeluarkan: {exc}")
            continue

        rd_notes.append(f"{item.hole_id}/{item.seam}: {note}")
        values = dict(q.values) if q else {}
        point = HolePoint(
            hole_id=item.hole_id, east=east, north=north,
            coal_thickness_m=item.coal_thickness_m, rd_t_per_m3=rd,
            rd_basis=q.rd_basis if q else "unknown", rd_is_assumed=assumed,
            quality_coverage_frac=q.coverage_frac if q else float("nan"),
            quality=values,
        )
        points.setdefault(item.seam, []).append(point)

        rows.append({
            "hole_id": item.hole_id, "seam": item.seam, "east": east, "north": north,
            "collar_rl_m": rl, "roof_m": item.roof_m, "floor_m": item.floor_m,
            "roof_rl_m": rl - item.roof_m, "floor_rl_m": rl - item.floor_m,
            "gross_thickness_m": item.gross_thickness_m,
            "coal_thickness_m": item.coal_thickness_m,
            "parting_thickness_m": item.parting_thickness_m,
            "core_loss_counted_as_coal_m": item.core_loss_in_coal_m,
            "rd_t_per_m3": rd, "rd_basis": point.rd_basis, "rd_is_assumed": assumed,
            "quality_coverage_frac": point.quality_coverage_frac,
            "n_quality_samples": q.n_samples if q else 0,
            "quality_covered_interval": q.covered_interval if q else "tidak ada hasil",
            **values,
        })

    intercepts = pd.DataFrame(rows)
    if not quality_issues.empty and excluded:
        quality_issues = pd.concat([quality_issues, pd.DataFrame(excluded)], ignore_index=True)
    elif excluded:
        quality_issues = pd.DataFrame(excluded)
    return points, intercepts, quality_issues, rd_notes


def build_surfaces(
    inputs: Inputs, intercepts: pd.DataFrame, cfg: Config
) -> tuple[Surface, dict[str, dict[str, Surface]]]:
    topo_points = inputs.topo_points
    if topo_points is None:
        raise MissingDataError("topografi DXF tidak dipasok; subcrop dan batas kedalaman "
                               "tidak dapat dihitung.")
    topo = build_surface(
        topo_points.points[:, 0], topo_points.points[:, 1], topo_points.points[:, 2],
        spacing=GRID_SPACING_M, margin=0.0, name="topografi",
    )

    surfaces: dict[str, dict[str, Surface]] = {}
    for seam, group in intercepts.groupby("seam", sort=False):
        if len(group) < 3:
            log.warning(f"seam {seam}: {len(group)} lubang, permukaan tidak dibangun")
            continue
        roof = build_surface(group["east"], group["north"], group["roof_rl_m"],
                             spacing=topo.spacing, name=f"{seam}_roof_rl")
        floor = build_surface(group["east"], group["north"], group["floor_rl_m"],
                              spacing=topo.spacing, name=f"{seam}_floor_rl")
        thickness = build_surface(group["east"], group["north"], group["coal_thickness_m"],
                                  spacing=topo.spacing, name=f"{seam}_coal_thickness")

        # Permukaan seam diselaraskan ke grid topografi agar dapat dikurangkan.
        roof = _align(roof, topo)
        floor = _align(floor, topo)
        thickness = _align(thickness, topo)
        depth = difference(topo, roof, f"{seam}_depth_below_surface")

        entry = {"roof": roof, "floor": floor, "thickness": thickness, "depth": depth,
                 "subcrop": subcrop_extent(topo, roof)}
        if cfg.rpeee_constraints.max_depth_m is not None:
            entry["depth_limit"] = depth_limit_extent(depth, cfg.rpeee_constraints.max_depth_m)
        surfaces[seam] = entry
    return topo, surfaces


def _align(surface: Surface, reference: Surface) -> Surface:
    """Contoh ulang permukaan ke grid referensi supaya bisa dikurangkan."""
    gx, gy = np.meshgrid(reference.x, reference.y)
    z = surface.sample(gx.ravel(), gy.ravel()).reshape(gx.shape)
    return Surface(name=surface.name, x=reference.x, y=reference.y, z=z,
                   spacing=reference.spacing, method=surface.method,
                   n_points=surface.n_points, hull=surface.hull)


def run(config_path: str | Path, verbose: bool = True) -> Results:
    cfg = Config.load(config_path)
    warning = label_warning(cfg)
    if warning:
        log.warning(warning)

    inputs = gather(cfg)
    audit = run_audit(inputs.workbooks, inputs.las_files, inputs.quality,
                      inputs.topo_points, cfg)
    if not audit.passed:
        raise MissingDataError(
            f"Phase 0 tidak lulus: {len(audit.stops)} gerbang terbuka. "
            "Jalankan `coalres audit` untuk rinciannya."
        )

    points, intercepts, quality_issues, rd_notes = build_hole_points(inputs, cfg)
    topo, surfaces = build_surfaces(inputs, intercepts, cfg)

    polygons: list[ResourcePolygon] = []
    for seam, seam_points in points.items():
        try:
            polygons.extend(build_polygons(seam, seam_points, cfg))
        except MissingDataError as exc:
            log.warning(f"seam {seam} dilewati: {exc}")

    polygons, steps = apply_constraints(polygons, surfaces, cfg)

    frame = to_frame(polygons, cfg)
    frames = {
        "polygons": frame,
        "by_seam_class": summarise(frame, ["seam", "class"]) if not frame.empty else pd.DataFrame(),
        "by_seam": summarise(frame, ["seam"]) if not frame.empty else pd.DataFrame(),
        "grand_total": summarise(frame.assign(_all="Total"), ["_all"]) if not frame.empty else pd.DataFrame(),
        "intercepts": intercepts,
        "rpeee": reconciliation_table(steps, cfg),
        "rd_sensitivity": rd_sensitivity(polygons),
        "quality_issues": quality_issues,
        "basis": basis_report(frame) if not frame.empty else pd.DataFrame(),
        "audit": audit.to_frame(),
        "rd_notes": pd.DataFrame({"catatan konversi RD": rd_notes}),
    }
    return Results(config=cfg, audit=audit, polygons=polygons, frames=frames,
                   surfaces={"topo": {"topo": topo}, **surfaces})


def write_outputs(results: Results, inputs: Inputs, config_path: Path) -> list[Path]:
    cfg = results.config
    out_dir = cfg.paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    frames = results.frames

    seam_assumptions = assumptions(cfg) + frames["rd_notes"]["catatan konversi RD"].tolist()
    assumption_frame = assumptions_frame(cfg, seam_assumptions, frames["quality_issues"])

    thickness_quality = frames["intercepts"][[
        c for c in ("hole_id", "seam", "coal_thickness_m", "parting_thickness_m",
                    "core_loss_counted_as_coal_m", "quality_covered_interval",
                    "quality_coverage_frac", "n_quality_samples")
        if c in frames["intercepts"].columns
    ]] if not frames["intercepts"].empty else pd.DataFrame()

    written.append(write_excel(
        out_dir / "resource_estimate.xlsx", cfg,
        by_seam_class=frames["by_seam_class"], by_seam=frames["by_seam"],
        grand_total=frames["grand_total"], quality_by_seam_class=frames["by_seam_class"],
        intercepts=frames["intercepts"], rd_sensitivity=frames["rd_sensitivity"],
        rpeee_reconciliation=frames["rpeee"], assumptions=assumption_frame,
        audit_findings=frames["audit"], thickness_quality=thickness_quality,
    ))
    written += write_vectors(results.polygons, cfg, out_dir / "vector")

    holes = frames["intercepts"][["east", "north"]].drop_duplicates() \
        if not frames["intercepts"].empty else None
    grids_dir, maps_dir = out_dir / "grids", out_dir / "maps"

    topo = results.surfaces["topo"]["topo"]
    written.append(write_ascii_grid(topo, grids_dir / "topografi.asc"))
    tif = write_geotiff(topo, grids_dir / "topografi.tif")
    if tif:
        written.append(tif)
    written.append(write_grid_sidecar(topo, grids_dir / "topografi.txt", cfg,
                                      "seluruh cakupan DXF"))
    written.append(map_surface(topo, "Topografi", "RL (m)", maps_dir / "topografi.png",
                               holes=holes))

    for seam, surfaces in results.surfaces.items():
        if seam == "topo":
            continue
        for key, label, unit in (
            ("roof", "Struktur roof", "RL (m)"),
            ("floor", "Struktur floor", "RL (m)"),
            ("thickness", "Isopach batubara", "Tebal (m)"),
            ("depth", "Kedalaman di bawah permukaan", "Kedalaman (m)"),
        ):
            surface = surfaces[key]
            written.append(write_ascii_grid(surface, grids_dir / f"{seam}_{key}.asc"))
            tif = write_geotiff(surface, grids_dir / f"{seam}_{key}.tif")
            if tif:
                written.append(tif)
            written.append(write_grid_sidecar(
                surface, grids_dir / f"{seam}_{key}.txt", cfg,
                "di dalam convex hull lubang yang menembus seam ini; di luar itu NaN",
            ))
            written.append(map_surface(surface, f"{label} — {seam}", unit,
                                       maps_dir / f"{seam}_{key}.png", holes=holes))

        # Peta kontur struktur: roof dan floor sebagai garis kontur berlabel,
        # ditumpangi subcrop dan batas blok - bentuk yang dipakai peta struktur
        # batubara, berbeda dari peta permukaan terisi warna di atas.
        from .rpeee import parse_wkt
        boundary = parse_wkt(cfg.block_boundary_wkt, "block_boundary_wkt")
        seam_holes = frames["intercepts"][frames["intercepts"]["seam"] == seam]
        for key, label in (("roof", "roof"), ("floor", "floor")):
            try:
                written.append(map_contours(
                    surfaces[key], f"Peta kontur struktur {label} — seam {seam}",
                    maps_dir / f"{seam}_{key}_kontur.png", cfg,
                    interval_m=cfg.maps.contour_interval_m,
                    index_every=cfg.maps.index_contour_every,
                    label=cfg.maps.label_contours,
                    subcrop=surfaces.get("subcrop"), boundary=boundary,
                    holes=seam_holes[["east", "north"]] if not seam_holes.empty else None,
                    hole_values=seam_holes[f"{key}_rl_m"] if not seam_holes.empty else None,
                    subtitle=f"Interval kontur {cfg.maps.contour_interval_m:,.0f} m · "
                             f"RL meter · {cfg.rpeee_constraints.resource_label}",
                ))
            except Exception as exc:
                log.warning(f"peta kontur {seam}/{key} gagal: {exc}")
        try:
            written.append(map_contours(
                surfaces["thickness"], f"Peta isopach batubara — seam {seam}",
                maps_dir / f"{seam}_isopach_kontur.png", cfg,
                interval_m=max(cfg.maps.contour_interval_m / 5.0, 0.25),
                index_every=cfg.maps.index_contour_every,
                label=cfg.maps.label_contours,
                subcrop=surfaces.get("subcrop"), boundary=boundary,
                holes=seam_holes[["east", "north"]] if not seam_holes.empty else None,
                hole_values=seam_holes["coal_thickness_m"] if not seam_holes.empty else None,
                subtitle="Tebal batubara (m)",
            ))
        except Exception as exc:
            log.warning(f"peta isopach {seam} gagal: {exc}")

        overlays = {"Subcrop": {"geometry": surfaces.get("subcrop"), "colour": "#0b0b0b"}}
        if "depth_limit" in surfaces:
            overlays["Batas kedalaman"] = {"geometry": surfaces["depth_limit"],
                                           "colour": "#eb6834", "linestyle": ":"}
        written.append(map_polygons(results.polygons, seam, cfg,
                                    maps_dir / f"{seam}_klasifikasi.png",
                                    holes=holes, overlays=overlays))

    from .logs import plot_hole
    for wb in inputs.workbooks:
        key = normalise_hole_id(wb.hole_id)
        try:
            written.append(plot_hole(wb, inputs.las_files.get(key),
                                     out_dir / "logs" / f"{key}.png"))
        except Exception as exc:
            log.warning(f"plot log {key} gagal: {exc}")

    hole_reconciliation = pd.DataFrame([
        {"tahap": "workbook dibaca", "lubang": len(inputs.workbooks)},
        {"tahap": "interseksi seam terbentuk",
         "lubang": int(frames["intercepts"]["hole_id"].nunique()) if not frames["intercepts"].empty else 0},
        {"tahap": "masuk estimasi akhir",
         "lubang": int(frames["polygons"]["hole_id"].nunique()) if not frames["polygons"].empty else 0},
    ])
    written.append(write_qaqc_markdown(
        out_dir / "qaqc_report.md", cfg, audit_findings=frames["audit"],
        rpeee_reconciliation=frames["rpeee"], hole_reconciliation=hole_reconciliation,
        exclusions=frames["quality_issues"], assumptions=assumption_frame,
    ))
    written.append(write_run_log(out_dir / "run_log.json", cfg, inputs.digests, Path(config_path)))
    return written
