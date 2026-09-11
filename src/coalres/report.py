"""Keluaran: Excel, GeoJSON/Shapefile, grid, peta, QA/QC, run log.

SETIAP deliverable membawa label Sumberdaya atau Inventori Batubara yang
diputuskan aturan 8.4. Tidak ada keluaran tanpa label.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RADII_UNVERIFIED_WARNING
from .logging_setup import get_logger
from .palette import (
    CLASS_COLOURS, CONTOUR_INDEX, CONTOUR_MINOR, GRIDLINE, INK_PRIMARY,
    INK_SECONDARY, SUBCROP_COLOUR, SURFACE, sequential_cmap, style_map_axes,
)
from .topo import Surface

log = get_logger("report")


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def write_excel(
    path: Path, cfg: Config, *,
    by_seam_class: pd.DataFrame, by_seam: pd.DataFrame, grand_total: pd.DataFrame,
    quality_by_seam_class: pd.DataFrame, intercepts: pd.DataFrame,
    rd_sensitivity: pd.DataFrame, rpeee_reconciliation: pd.DataFrame,
    assumptions: pd.DataFrame, audit_findings: pd.DataFrame,
    thickness_quality: pd.DataFrame,
) -> Path:
    label = cfg.rpeee_constraints.resource_label
    path.parent.mkdir(parents=True, exist_ok=True)

    cover = pd.DataFrame({
        "Item": [
            "Klasifikasi keluaran", "Metode estimasi", "Kondisi geologi",
            "Radius measured (m)", "Radius indicated (m)", "Radius inferred (m)",
            "Sumber tebal batubara", "Perlakuan core loss",
            "Sumber koordinat otoritatif", "Batas kedalaman (m)",
            "Dasar batas kedalaman", "Dihasilkan",
        ],
        "Nilai": [
            label, cfg.estimation_method, cfg.geological_condition,
            cfg.radii.measured, cfg.radii.indicated, cfg.radii.inferred,
            cfg.coal_thickness_source, cfg.core_loss_treatment,
            cfg.authoritative_coordinate_source,
            cfg.rpeee_constraints.max_depth_m if cfg.rpeee_constraints.max_depth_m else "TIDAK DITERAPKAN",
            cfg.rpeee_constraints.max_depth_basis or "-",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ],
    })

    sheets = {
        f"Sampul ({label})": cover,
        "Asumsi & Batasan": assumptions,
        "Ringkasan Seam x Kelas": by_seam_class,
        "Ringkasan per Seam": by_seam,
        "Total Keseluruhan": grand_total,
        "Kualitas Seam x Kelas": quality_by_seam_class,
        "Intercept per Lubang": intercepts,
        "Rekonsiliasi RPEEE": rpeee_reconciliation,
        "Rekonsiliasi Tebal-Kualitas": thickness_quality,
        "Sensitivitas RD": rd_sensitivity,
        "Temuan Audit": audit_findings,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe = name[:31]
            body = frame if isinstance(frame, pd.DataFrame) and not frame.empty \
                else pd.DataFrame({"catatan": ["(kosong)"]})
            body.to_excel(writer, sheet_name=safe, index=False)
    log.info(f"workbook ditulis: {path.name} ({label})")
    return path


def assumptions_frame(cfg: Config, extra: list[str], excluded_records: pd.DataFrame) -> pd.DataFrame:
    notes = list(extra)
    notes.append(RADII_UNVERIFIED_WARNING)
    notes.append(
        f"Kondisi geologi '{cfg.geological_condition}' adalah penilaian geolog "
        "yang dimasukkan lewat konfigurasi, bukan kesimpulan program."
    )
    notes.append(f"JUSTIFIKASI KONDISI GEOLOGI (verbatim): {cfg.geological_condition_justification}")
    if cfg.rpeee_constraints.has_economic_constraint:
        notes.append(
            f"Batas kedalaman {cfg.rpeee_constraints.max_depth_m:.0f} m diterapkan pada "
            f"POLIGON lewat grid kedalaman di bawah permukaan, bukan pada titik bor. "
            f"Dasar: {cfg.rpeee_constraints.max_depth_basis}"
        )
    else:
        notes.append(
            "Tidak ada batasan prospek ekonomi yang diterapkan. Seluruh keluaran "
            "berlabel INVENTORI BATUBARA (aturan 8.4), bukan Sumberdaya."
        )
    if cfg.assumed_rd_t_per_m3 is not None:
        notes.append(
            f"RD {cfg.assumed_rd_t_per_m3} t/m3 adalah NILAI ASUMSI dari konfigurasi, "
            "dipakai di mana hasil lab tidak tersedia."
        )
    notes.append(
        "Lubang diperlakukan VERTIKAL: workbook tidak memuat survei downhole. "
        "Tebal yang dilaporkan adalah tebal vertikal."
    )
    notes.append(
        "Volume = luas dalam peta x tebal vertikal. Tidak ada koreksi cos(dip); "
        "koreksi itu hanya berlaku bila luas bidang miring dipasangkan dengan "
        "tebal tegak lurus."
    )
    frame = pd.DataFrame({"No": range(1, len(notes) + 1), "Asumsi / Batasan": notes})
    if not excluded_records.empty:
        excluded_records = excluded_records.copy()
        excluded_records.insert(0, "Jenis", "Record dikeluarkan")
        return pd.concat([frame, excluded_records], axis=0, ignore_index=True)
    return frame


# --------------------------------------------------------------------------- #
# Vektor
# --------------------------------------------------------------------------- #
def write_vectors(polygons, cfg: Config, out_dir: Path, crs: str = "EPSG:32748") -> list[Path]:
    import geopandas as gpd

    if not polygons:
        return []
    label = cfg.rpeee_constraints.resource_label
    records = []
    for p in polygons:
        record = {
            "seam": p.seam, "hole_id": p.hole_id,
            "class": f"{cfg.rpeee_constraints.class_prefix} {p.resource_class.sni_name}",
            "class_sni": p.resource_class.sni_name,
            "label": label,
            "area_m2": p.area_m2, "area_ha": p.area_m2 / 10_000.0,
            "thick_m": p.coal_thickness_m, "rd": p.rd_t_per_m3,
            "rd_assumed": bool(p.rd_is_assumed),
            "qual_cov": p.quality_coverage_frac,
            "tonnes": p.tonnes,
            "geometry": p.geometry,
        }
        record.update({k: v for k, v in p.quality.items()})
        records.append(record)

    frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    geojson = out_dir / "classification_polygons.geojson"
    frame.to_file(geojson, driver="GeoJSON")
    written.append(geojson)
    shapefile = out_dir / "classification_polygons.shp"
    try:
        frame.to_file(shapefile)
        written.append(shapefile)
    except Exception as exc:  # driver shapefile tidak selalu tersedia
        log.warning(f"shapefile tidak ditulis: {exc}")
    return written


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def write_ascii_grid(surface: Surface, path: Path, nodata: float = -9999.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.where(np.isfinite(surface.z), surface.z, nodata)
    header = (
        f"ncols {len(surface.x)}\nnrows {len(surface.y)}\n"
        f"xllcorner {surface.x.min() - surface.spacing / 2:.4f}\n"
        f"yllcorner {surface.y.min() - surface.spacing / 2:.4f}\n"
        f"cellsize {surface.spacing}\nNODATA_value {nodata}\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        for row in z[::-1]:            # ASCII grid ditulis dari baris paling utara
            fh.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    return path


def write_geotiff(surface: Surface, path: Path, crs: str = "EPSG:32748",
                  nodata: float = -9999.0) -> Path | None:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:  # pragma: no cover
        log.warning("rasterio tidak tersedia; GeoTIFF dilewati")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.where(np.isfinite(surface.z), surface.z, nodata).astype("float32")
    transform = from_origin(surface.x.min() - surface.spacing / 2,
                            surface.y.max() + surface.spacing / 2,
                            surface.spacing, surface.spacing)
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1],
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(z[::-1], 1)
    return path


def write_grid_sidecar(surface: Surface, path: Path, cfg: Config,
                       support_note: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Permukaan     : {surface.name}\n"
        f"Label         : {cfg.rpeee_constraints.resource_label}\n"
        f"Interpolasi   : TIN Delaunay, {surface.method}\n"
        f"Spasi grid    : {surface.spacing} m\n"
        f"Titik pendukung: {surface.n_points}\n"
        f"Extent        : {surface.extent}\n"
        f"Dukungan data : {support_note}\n"
        "\n"
        "Permukaan ini adalah GEOMETRI untuk ekspor dan pemotongan, BUKAN model\n"
        "blok dan BUKAN metode estimasi. Tonase berasal dari poligon saja;\n"
        "interpolasi tidak pernah menyumbang tebal ke jalur tonase.\n"
    )
    return path


# --------------------------------------------------------------------------- #
# Peta
# --------------------------------------------------------------------------- #
def map_surface(surface: Surface, title: str, colorbar_label: str, path: Path,
                holes: pd.DataFrame | None = None, subtitle: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    half = surface.spacing / 2
    extent = (surface.x.min() - half, surface.x.max() + half,
              surface.y.min() - half, surface.y.max() + half)
    image = ax.imshow(surface.z, origin="lower", extent=extent,
                      cmap=sequential_cmap(), interpolation="nearest")
    if holes is not None and len(holes):
        ax.scatter(holes["east"], holes["north"], s=10, marker="o", facecolors="none",
                   edgecolors=INK_PRIMARY, linewidths=0.7, zorder=5,
                   label=f"Lubang bor (n={len(holes)})")
        ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor=SURFACE,
                  edgecolor=GRIDLINE, labelcolor=INK_SECONDARY)
    bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    bar.set_label(colorbar_label, color=INK_SECONDARY, fontsize=8.5)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=7.5, length=3, width=0.6)
    bar.outline.set_edgecolor(GRIDLINE)
    style_map_axes(ax, title, subtitle)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def map_contours(
    surface: Surface, title: str, path: Path, cfg: Config, *,
    interval_m: float, index_every: int, label: bool = True,
    subcrop=None, boundary=None, holes: pd.DataFrame | None = None,
    subtitle: str | None = None, hole_values: pd.Series | None = None,
) -> Path:
    """Peta kontur struktur bergaris - roof atau floor seam.

    Bentuk yang dipakai peta struktur batubara: garis kontur pada interval
    tetap, kontur indeks lebih tebal dan berlabel, ditumpangi garis subcrop dan
    batas blok. Berbeda dari peta permukaan terisi warna: pembaca peta struktur
    membaca ANGKA elevasi pada garis, bukan gradasi warna.

    Kontur memakai satu hue - terang untuk kontur biasa, gelap untuk kontur
    indeks - sehingga hierarkinya terbaca tanpa bergantung pada perbedaan warna.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    z = surface.z
    if not np.isfinite(z).any():
        raise ValueError(f"{surface.name}: permukaan seluruhnya kosong")

    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    start = np.floor(zmin / interval_m) * interval_m
    stop = np.ceil(zmax / interval_m) * interval_m
    levels = np.arange(start, stop + interval_m, interval_m)
    if len(levels) < 2:
        levels = np.linspace(zmin, zmax, 3)
    index_levels = [lv for i, lv in enumerate(levels) if i % index_every == 0]
    minor_levels = [lv for lv in levels if lv not in index_levels]

    fig, ax = plt.subplots(figsize=(8.4, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    gx, gy = np.meshgrid(surface.x, surface.y)

    # linestyles="solid" WAJIB: matplotlib menggambar level negatif sebagai garis
    # putus-putus secara bawaan. Pada peta struktur batubara RL negatif itu lazim
    # (seam di bawah muka laut), sementara garis putus-putus di peta geologi
    # berarti "perkiraan" - bawaan itu akan menyesatkan pembaca peta.
    if minor_levels:
        ax.contour(gx, gy, z, levels=minor_levels, colors=CONTOUR_MINOR,
                   linewidths=0.6, linestyles="solid", zorder=2)
    index_set = ax.contour(gx, gy, z, levels=index_levels, colors=CONTOUR_INDEX,
                           linewidths=1.2, linestyles="solid", zorder=3)
    if label and len(index_levels):
        ax.clabel(index_set, fmt=lambda v: f"{v:,.0f}", fontsize=7,
                  colors=INK_PRIMARY, inline=True, inline_spacing=4)

    handles = [
        Line2D([], [], color=CONTOUR_INDEX, linewidth=1.2,
               label=f"Kontur indeks ({interval_m * index_every:,.0f} m)"),
        Line2D([], [], color=CONTOUR_MINOR, linewidth=0.6,
               label=f"Kontur ({interval_m:,.0f} m)"),
    ]

    for geometry, style, name in (
        (subcrop, {"color": SUBCROP_COLOUR, "linewidth": 1.6}, "Subcrop"),
        (boundary, {"color": INK_PRIMARY, "linewidth": 1.2, "linestyle": "--"},
         "Batas blok"),
    ):
        if geometry is None or getattr(geometry, "is_empty", True):
            continue
        for geom in getattr(geometry, "geoms", [geometry]):
            if geom.geom_type != "Polygon":
                continue
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, zorder=5, **style)
        handles.append(Line2D([], [], label=name, **style))

    if holes is not None and len(holes):
        ax.scatter(holes["east"], holes["north"], s=12, marker="o", facecolors=SURFACE,
                   edgecolors=INK_PRIMARY, linewidths=0.8, zorder=6)
        handles.append(Line2D([], [], marker="o", linestyle="none",
                              markerfacecolor=SURFACE, markeredgecolor=INK_PRIMARY,
                              label=f"Lubang bor (n={len(holes)})"))
        if hole_values is not None:
            for (_, row), value in zip(holes.iterrows(), hole_values):
                if np.isfinite(value):
                    ax.annotate(f"{value:,.1f}", (row["east"], row["north"]),
                                textcoords="offset points", xytext=(5, 4),
                                fontsize=6.5, color=INK_SECONDARY, zorder=7)

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=7.5, frameon=True, facecolor=SURFACE, edgecolor=GRIDLINE,
              labelcolor=INK_SECONDARY, borderaxespad=0.0)
    style_map_axes(ax, title, subtitle)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def map_polygons(polygons, seam: str, cfg: Config, path: Path,
                 holes: pd.DataFrame | None = None,
                 overlays: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    subset = [p for p in polygons if p.seam == seam]
    fig, ax = plt.subplots(figsize=(8.4, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    totals: dict[str, float] = {}
    for polygon in sorted(subset, key=lambda p: -int(p.resource_class)):
        name = polygon.resource_class.sni_name
        totals[name] = totals.get(name, 0.0) + polygon.tonnes
        geoms = getattr(polygon.geometry, "geoms", [polygon.geometry])
        for geom in geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, facecolor=CLASS_COLOURS[name], edgecolor=SURFACE,
                    linewidth=0.5, zorder=2)

    for name, style in (overlays or {}).items():
        geometry = style["geometry"]
        if geometry is None or geometry.is_empty:
            continue
        for geom in getattr(geometry, "geoms", [geometry]):
            if geom.geom_type != "Polygon":
                continue
            xs, ys = geom.exterior.xy
            ax.plot(xs, ys, color=style.get("colour", INK_PRIMARY),
                    linewidth=style.get("linewidth", 1.0),
                    linestyle=style.get("linestyle", "--"), zorder=4)

    handles = [Patch(facecolor=CLASS_COLOURS[n], edgecolor=SURFACE,
                     label=f"{cfg.rpeee_constraints.class_prefix} {n} — {t:,.0f} t")
               for n, t in sorted(totals.items(),
                                  key=lambda kv: {"Terukur": 0, "Tertunjuk": 1, "Tereka": 2}[kv[0]])]
    for name, style in (overlays or {}).items():
        handles.append(Patch(facecolor="none", edgecolor=style.get("colour", INK_PRIMARY),
                             label=name))
    if holes is not None and len(holes):
        ax.scatter(holes["east"], holes["north"], s=10, marker="o", facecolors="none",
                   edgecolors=INK_PRIMARY, linewidths=0.7, zorder=6)
        handles.append(Patch(facecolor="none", edgecolor=INK_PRIMARY,
                             label=f"Lubang bor (n={len(holes)})"))
    if handles:
        # Legenda diletakkan DI LUAR area plot: di peta klasifikasi ia menutupi
        # poligon di sudut, dan poligon itu justru yang perlu dilihat.
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=7.5, frameon=True, facecolor=SURFACE,
                  edgecolor=GRIDLINE, labelcolor=INK_SECONDARY, borderaxespad=0.0)

    style_map_axes(
        ax, f"{cfg.rpeee_constraints.resource_label} — seam {seam}",
        f"Metode {cfg.estimation_method} · kondisi geologi {cfg.geological_condition} "
        f"· radius {cfg.radii.measured:.0f}/{cfg.radii.indicated:.0f}/{cfg.radii.inferred:.0f} m",
    )
    ax.autoscale_view()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# QA/QC dan run log
# --------------------------------------------------------------------------- #
def write_qaqc_markdown(
    path: Path, cfg: Config, *, audit_findings: pd.DataFrame,
    rpeee_reconciliation: pd.DataFrame, hole_reconciliation: pd.DataFrame,
    exclusions: pd.DataFrame, assumptions: pd.DataFrame,
) -> Path:
    label = cfg.rpeee_constraints.resource_label
    lines = [
        f"# Laporan QA/QC — {label}",
        "",
        f"Dihasilkan: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"**Seluruh angka dalam laporan ini berlabel {label}.**",
        "",
        "## Justifikasi kondisi geologi (verbatim)",
        "",
        f"Kondisi: `{cfg.geological_condition}` — radius "
        f"{cfg.radii.measured:.0f} / {cfg.radii.indicated:.0f} / {cfg.radii.inferred:.0f} m",
        "",
        "> " + cfg.geological_condition_justification.replace("\n", "\n> "),
        "",
        "## Peringatan radius",
        "",
        RADII_UNVERIFIED_WARNING,
        "",
        "## Rekonsiliasi RPEEE",
        "",
        rpeee_reconciliation.to_markdown(index=False) if not rpeee_reconciliation.empty else "_(kosong)_",
        "",
        "## Rekonsiliasi jumlah lubang",
        "",
        hole_reconciliation.to_markdown(index=False) if not hole_reconciliation.empty else "_(kosong)_",
        "",
        "## Temuan audit",
        "",
        audit_findings.to_markdown(index=False) if not audit_findings.empty else "_(kosong)_",
        "",
        "## Pengecualian",
        "",
        exclusions.to_markdown(index=False) if not exclusions.empty else "_(tidak ada)_",
        "",
        "## Asumsi dan batasan",
        "",
        assumptions.to_markdown(index=False) if not assumptions.empty else "_(kosong)_",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def write_run_log(path: Path, cfg: Config, digests: dict[str, str],
                  config_path: Path) -> Path:
    import importlib.metadata as metadata

    packages = {}
    for name in ("pandas", "numpy", "scipy", "shapely", "geopandas", "ezdxf",
                 "lasio", "matplotlib", "pydantic", "openpyxl", "rasterio"):
        try:
            packages[name] = metadata.version(name)
        except Exception:
            packages[name] = "tidak terpasang"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resource_label": cfg.rpeee_constraints.resource_label,
        "config_path": str(config_path),
        "config": json.loads(cfg.model_dump_json()),
        "input_digests_sha256": digests,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
