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
from .estimate import (
    HolePoint, ResourcePolygon, build_polygons, build_polygons_for_unit,
    plan_overlap_report, rd_sensitivity, to_frame,
)
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
from .io.minex import HoleDataset, load_minex
from .seams import (
    SeamIntersection, assumptions, build_intersections,
    build_intersections_from_dataset, to_frame as seams_frame,
)
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
    uncut_surfaces: dict[str, Surface] = field(default_factory=dict)
    quality_surfaces: dict[str, dict[str, Surface]] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)


def gather_minex(cfg: Config) -> tuple[HoleDataset, Inputs]:
    """Muat berkas flat Minex dan bungkus sebagai Inputs untuk hilirnya."""
    dataset = load_minex(cfg)
    digests: dict[str, str] = {}
    spec = cfg.minex
    for path in (spec.survey_file, spec.lithology_file, spec.quality_file,
                 spec.topography_file, spec.faults_file):
        if path is not None and Path(path).exists():
            digests[Path(path).name] = file_digest(Path(path))

    class _TopoShim:
        points = dataset.topo_points

    inputs = Inputs(workbooks=[], las_files={}, quality=None,
                    topo_points=_TopoShim() if dataset.topo_points is not None else None,
                    digests=digests)
    return dataset, inputs


def build_hole_points_minex(
    dataset: HoleDataset, cfg: Config
) -> tuple[dict[str, list[HolePoint]], pd.DataFrame, pd.DataFrame, list[str]]:
    """Titik per seam dari HoleDataset."""
    from .quality import resolve_quality_from_plies

    intersections = build_intersections_from_dataset(dataset, cfg)
    quality_by_key: dict[tuple[str, str], object] = {}
    issues = pd.DataFrame()
    if dataset.quality is not None:
        resolved, issues = resolve_quality_from_plies(
            intersections, dataset.quality, cfg, cfg.minex.quality_rd_basis
        )
        quality_by_key = {(q.hole_id, q.seam): q for q in resolved}

    collars = dataset.collars.set_index("hole_id")
    points: dict[str, list[HolePoint]] = {}
    rows, rd_notes, excluded = [], [], []

    for item in intersections:
        if item.hole_id not in collars.index:
            continue
        collar = collars.loc[item.hole_id]
        east, north, rl = float(collar["east"]), float(collar["north"]), float(collar["rl"])
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
            excluded.append({"hole_id": item.hole_id, "seam": item.seam, "alasan": str(exc)})
            continue

        rd_notes.append(f"{item.hole_id}/{item.seam}: {note}")
        values = dict(q.values) if q else {}
        points.setdefault(item.seam, []).append(HolePoint(
            hole_id=item.hole_id, east=east, north=north,
            coal_thickness_m=item.coal_thickness_m, rd_t_per_m3=rd,
            rd_basis=q.rd_basis if q else "unknown", rd_is_assumed=assumed,
            quality_coverage_frac=q.coverage_frac if q else float("nan"),
            quality=values,
        ))
        rows.append({
            "hole_id": item.hole_id, "seam": item.seam, "east": east, "north": north,
            "collar_rl_m": rl, "roof_m": item.roof_m, "floor_m": item.floor_m,
            "roof_rl_m": rl - item.roof_m, "floor_rl_m": rl - item.floor_m,
            "gross_thickness_m": item.gross_thickness_m,
            "coal_thickness_m": item.coal_thickness_m,
            "parting_thickness_m": item.parting_thickness_m,
            "core_loss_counted_as_coal_m": 0.0,
            "rd_t_per_m3": rd, "rd_basis": q.rd_basis if q else "unknown",
            "rd_is_assumed": assumed,
            "quality_coverage_frac": q.coverage_frac if q else float("nan"),
            "n_quality_samples": q.n_samples if q else 0,
            "quality_covered_interval": q.covered_interval if q else "tidak ada hasil",
            **values,
        })

    if excluded:
        extra = pd.DataFrame(excluded)
        issues = pd.concat([issues, extra], ignore_index=True) if not issues.empty else extra
    return points, pd.DataFrame(rows), issues, rd_notes


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
    if intercepts.empty or "seam" not in intercepts.columns:
        raise MissingDataError(
            "tidak ada satu pun interseksi seam yang lolos ke tahap estimasi. "
            "Penyebab paling lazim: RD tidak dapat diselesaikan untuk seluruh "
            "seam (lihat sheet pengecualian). Isi assumed_rd_t_per_m3, atau "
            "lengkapi kolom moisture yang dibutuhkan konversi basis RD."
        )
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


def build_uncut_thickness(dataset, inputs: Inputs, cfg: Config,
                          reference: Surface) -> dict[str, Surface]:
    """Grid ketebalan UNCUT: seluruh interseksi, tanpa aturan penambangan.

    "Uncut" berarti ketebalan in-situ mentah sebelum aturan penambangan:
    tanpa ketebalan minimum, tanpa pengecualian parting, tanpa dilusi, tanpa
    cutoff. Yang dihasilkan adalah ketebalan GEOLOGI, bukan ketebalan yang
    dapat ditambang.

    Grid ini WAJIB dibangun dari interseksi PRA-cutoff. Memakai interseksi yang
    sudah lolos cutoff menghasilkan berkas yang bernama uncut tetapi isinya cut,
    dan tidak ada cara membedakannya dari berkas yang benar.
    """
    if dataset is not None:
        items = build_intersections_from_dataset(dataset, cfg, apply_cutoffs=False)
        collars = dataset.collars.set_index("hole_id")
        lookup = {h: (float(r["east"]), float(r["north"]), float(r["rl"]))
                  for h, r in collars.iterrows()}
    else:
        items = []
        lookup = {}
        for wb in inputs.workbooks:
            items.extend(build_intersections(wb, cfg, apply_cutoffs=False))
            east, north, rl = _collar(wb, cfg)
            lookup[normalise_hole_id(wb.hole_id)] = (east, north, rl)

    rows = []
    for item in items:
        if item.hole_id not in lookup:
            continue
        east, north, _ = lookup[item.hole_id]
        rows.append({"seam": item.seam, "east": east, "north": north,
                     # Uncut memakai GROSS thickness: amplop roof-floor penuh,
                     # parting belum dikeluarkan.
                     "uncut_m": item.gross_thickness_m})
    frame = pd.DataFrame(rows)

    surfaces: dict[str, Surface] = {}
    for seam, group in frame.groupby("seam", sort=False):
        if len(group) < 3:
            continue
        surface = build_surface(group["east"], group["north"], group["uncut_m"],
                                spacing=reference.spacing, name=f"{seam}_uncut_thickness")
        surfaces[seam] = _align(surface, reference)
    return surfaces


def build_quality_surfaces(intercepts: pd.DataFrame, reference: Surface,
                           attributes: list[str] | None = None) -> dict[str, dict[str, Surface]]:
    """Grid atribut kualitas per seam, dari nilai komposit tiap lubang.

    Grid ini untuk penyajian dan konsumsi hilir. Ia TIDAK menyumbang apa pun ke
    jalur tonase - tonase berasal dari poligon, dan kualitas poligon berasal
    dari lubangnya sendiri.
    """
    from .quality import AVERAGEABLE

    if intercepts.empty:
        return {}
    # RD ikut di-grid: ia faktor ketiga pada tonase = luas x tebal x RD, jadi
    # estimasi cadangan di hilir membutuhkannya bersama grid ketebalan.
    default = [c for c in intercepts.columns if c in AVERAGEABLE]
    if "rd_t_per_m3" in intercepts.columns:
        default.append("rd_t_per_m3")
    candidates = attributes or default
    out: dict[str, dict[str, Surface]] = {}
    for seam, group in intercepts.groupby("seam", sort=False):
        per_attribute: dict[str, Surface] = {}
        for attribute in candidates:
            if attribute not in group:
                continue
            values = group[["east", "north", attribute]].dropna()
            if len(values) < 3:
                continue
            surface = build_surface(values["east"], values["north"], values[attribute],
                                    spacing=reference.spacing,
                                    name=f"{seam}_{attribute}")
            per_attribute[attribute] = _align(surface, reference)
        if per_attribute:
            out[seam] = per_attribute
    return out


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

    if cfg.input_format == "minex_flat":
        from .audit.minex_checks import run_minex_audit
        dataset, inputs = gather_minex(cfg)
        audit = run_minex_audit(dataset, cfg)
    else:
        dataset = None
        inputs = gather(cfg)
        audit = run_audit(inputs.workbooks, inputs.las_files, inputs.quality,
                          inputs.topo_points, cfg)
    if not audit.passed:
        raise MissingDataError(
            f"Phase 0 tidak lulus: {len(audit.stops)} gerbang terbuka. "
            "Jalankan `coalres audit` untuk rinciannya."
        )

    if dataset is not None:
        points, intercepts, quality_issues, rd_notes = build_hole_points_minex(dataset, cfg)
    else:
        points, intercepts, quality_issues, rd_notes = build_hole_points(inputs, cfg)
    topo, surfaces = build_surfaces(inputs, intercepts, cfg)

    # Seam dikelompokkan menurut SATUAN STRATIGRAFI: seam induk dan anaknya
    # dibentuk atas satu tesselasi gabungan, sehingga domainnya tidak dapat
    # saling tumpang tindih. Lihat estimate.build_polygons_for_unit.
    units: dict[str, dict[str, list[HolePoint]]] = {}
    for seam, seam_points in points.items():
        units.setdefault(cfg.parent_seam(seam), {})[seam] = seam_points

    polygons: list[ResourcePolygon] = []
    for unit, members in units.items():
        try:
            if len(members) == 1 and unit in members:
                polygons.extend(build_polygons(unit, members[unit], cfg))
            else:
                log.info(f"satuan '{unit}': {sorted(members)} dibentuk atas satu "
                         "tesselasi gabungan untuk mencegah hitung ganda")
                polygons.extend(build_polygons_for_unit(unit, members, cfg))
        except MissingDataError as exc:
            log.warning(f"satuan {unit} dilewati: {exc}")

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
        "plan_overlap": plan_overlap_report(polygons, cfg),
    }
    reference = topo
    uncut = build_uncut_thickness(dataset, inputs, cfg, reference)
    quality_grids = build_quality_surfaces(
        intercepts, reference,
        cfg.maps.grd_export.quality_attributes or None,
    )
    return Results(config=cfg, audit=audit, polygons=polygons, frames=frames,
                   surfaces={"topo": {"topo": topo}, **surfaces},
                   uncut_surfaces=uncut, quality_surfaces=quality_grids)


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

    # --- Ekspor grid .grd ---------------------------------------------------- #
    grd_cfg = cfg.maps.grd_export
    if grd_cfg.enabled:
        from .grdout import ATTRIBUTE_UNITS, write_grd, write_sidecar

        grd_dir = out_dir / "grd"
        suffix = "grd"
        support = ("di dalam convex hull lubang yang menembus seam ini; "
                   "di luar itu kosong (tidak diekstrapolasi)")

        for seam, surface in results.uncut_surfaces.items():
            if not grd_cfg.write_uncut_thickness:
                break
            written.append(write_grd(surface, grd_dir / f"{seam}_uncut.{suffix}",
                                     fmt=grd_cfg.format))
            written.append(write_sidecar(
                grd_dir / f"{seam}_uncut.txt", name=f"{seam}_uncut",
                description="Ketebalan seam UNCUT - amplop roof ke floor, sebelum "
                            "aturan penambangan apa pun",
                units="meter", surface=surface, support_note=support,
                warnings=[
                    "UNCUT berarti ketebalan geologi in-situ: TANPA ketebalan "
                    "minimum, TANPA pengecualian parting, TANPA dilusi, TANPA cutoff.",
                    "Ini BUKAN ketebalan yang dapat ditambang. Untuk itu pakai "
                    f"{seam}_cut.{suffix}.",
                    "Memakai grid ini langsung untuk estimasi cadangan akan "
                    "MELEBIHKAN hasilnya.",
                ]))

        for seam, surfaces in results.surfaces.items():
            if seam == "topo":
                continue
            if grd_cfg.write_cut_thickness and "thickness" in surfaces:
                surface = surfaces["thickness"]
                written.append(write_grd(surface, grd_dir / f"{seam}_cut.{suffix}",
                                         fmt=grd_cfg.format))
                written.append(write_sidecar(
                    grd_dir / f"{seam}_cut.txt", name=f"{seam}_cut",
                    description="Ketebalan batubara setelah aturan penambangan",
                    units="meter", surface=surface, support_note=support,
                    warnings=[
                        f"Cutoff ketebalan minimum {cfg.cutoffs.min_seam_thickness_m} m "
                        "sudah diterapkan.",
                        f"Parting di atas {cfg.cutoffs.max_parting_thickness_m} m "
                        "memisahkan seam; di bawahnya masuk gross tetapi keluar "
                        "dari tebal batubara.",
                        "Dilusi dan recovery penambangan BELUM diterapkan - keduanya "
                        "milik tahap cadangan.",
                    ]))
            if grd_cfg.write_structure:
                for key, label, units in (("roof", "RL roof seam", "meter RL"),
                                          ("floor", "RL floor seam", "meter RL"),
                                          ("depth", "Kedalaman roof di bawah permukaan", "meter")):
                    if key not in surfaces:
                        continue
                    written.append(write_grd(surfaces[key],
                                             grd_dir / f"{seam}_{key}.{suffix}",
                                             fmt=grd_cfg.format))

        if grd_cfg.write_quality:
            for seam, attributes in results.quality_surfaces.items():
                for attribute, surface in attributes.items():
                    written.append(write_grd(
                        surface, grd_dir / f"{seam}_qual_{attribute}.{suffix}",
                        fmt=grd_cfg.format))
                    written.append(write_sidecar(
                        grd_dir / f"{seam}_qual_{attribute}.txt",
                        name=f"{seam}_qual_{attribute}",
                        description=(
                            "Densitas in-situ per lubang lalu diinterpolasi"
                            if attribute == "rd_t_per_m3" else
                            f"Atribut kualitas {attribute}, komposit terbobot "
                            "massa per lubang lalu diinterpolasi"),
                        units=ATTRIBUTE_UNITS.get(attribute, "lihat sumber data"),
                        surface=surface,
                        support_note=f"{surface.n_points} lubang berdata kualitas "
                                     "untuk seam ini",
                        warnings=[
                            "Grid ini TIDAK menyumbang ke jalur tonase. Tonase "
                            "berasal dari poligon, dan kualitas poligon berasal "
                            "dari lubangnya sendiri.",
                            "Basis analitik mengikuti sumber data - periksa sheet "
                            "Asumsi & Batasan sebelum memakainya.",
                        ]))

    # --- Ekspor kontur DXF --------------------------------------------------- #
    dxf_cfg = cfg.maps.dxf_export
    if dxf_cfg.enabled:
        from .dxfout import export_seam_contours, subcrop_mask

        topo_surface = results.surfaces["topo"]["topo"]
        seam_surfaces = {k: v for k, v in results.surfaces.items() if k != "topo"}
        masks, lines = {}, {}
        for seam, surfaces in seam_surfaces.items():
            if dxf_cfg.clip_to_subcrop:
                masks[seam] = subcrop_mask(topo_surface, surfaces["roof"])
            if dxf_cfg.include_subcrop:
                lines[seam] = surfaces.get("subcrop")

        holes_full = None
        if dxf_cfg.include_boreholes and not frames["intercepts"].empty:
            holes_full = (frames["intercepts"][["hole_id", "east", "north", "collar_rl_m"]]
                          .drop_duplicates("hole_id"))

        dxf_dir = out_dir / "dxf"
        combined, summary = export_seam_contours(
            dxf_dir / "seam_contours.dxf", seam_surfaces,
            interval_m=dxf_cfg.contour_interval_m, index_every=dxf_cfg.index_every,
            layer_template=dxf_cfg.layer_template, index_suffix=dxf_cfg.index_suffix,
            subcrop_masks=masks, holes=holes_full, subcrop_lines=lines,
            topography=topo_surface,
        )
        written.append(combined)
        log.info(f"DXF gabungan: {combined.name} "
                 f"({sum(v.get('polylines', 0) for v in summary.values())} polyline, "
                 f"{len(summary)} layer)")

        if dxf_cfg.per_surface_files:
            from .dxfout import surface_file_stem
            for seam in seam_surfaces:
                for surface_key in ("roof", "floor"):
                    if surface_key not in seam_surfaces[seam]:
                        continue
                    stem = surface_file_stem(seam, surface_key, dxf_cfg.layer_template)
                    path, _ = export_seam_contours(
                        dxf_dir / f"{stem}.dxf", seam_surfaces,
                        interval_m=dxf_cfg.contour_interval_m,
                        index_every=dxf_cfg.index_every,
                        layer_template=dxf_cfg.layer_template,
                        index_suffix=dxf_cfg.index_suffix, subcrop_masks=masks,
                        holes=holes_full, subcrop_lines=lines, topography=topo_surface,
                        seam_filter=[seam], surface_filter=[surface_key],
                    )
                    written.append(path)

        if dxf_cfg.per_seam_files:
            for seam in seam_surfaces:
                path, _ = export_seam_contours(
                    dxf_dir / f"seam_{seam}_contours.dxf", seam_surfaces,
                    interval_m=dxf_cfg.contour_interval_m, index_every=dxf_cfg.index_every,
                    layer_template=dxf_cfg.layer_template,
                    index_suffix=dxf_cfg.index_suffix, subcrop_masks=masks,
                    holes=holes_full, subcrop_lines=lines, topography=topo_surface,
                    seam_filter=[seam],
                )
                written.append(path)

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
