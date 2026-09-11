"""Ekspor kontur struktur seam ke DXF.

Menghasilkan garis kontur roof dan floor tiap seam pada interval tetap,
masing-masing di layer sendiri, siap dibuka di AutoCAD, Minex, atau GIS.

Dua keputusan yang menentukan kebenaran keluarannya:

1. KONTUR DIPOTONG OLEH SUBCROP. Permukaan seam di dalam model membentang
   sampai batas dukungan data, termasuk ke area di mana seam berada DI ATAS
   topografi - di sana batubaranya sudah tererosi dan tidak ada. Mengekspor
   kontur di area itu menggambarkan seam yang tidak ada, dan pembaca peta tidak
   punya cara membedakannya.

2. ELEVASI DIBAWA OLEH POLYLINE, bukan hanya oleh nama layer. LWPOLYLINE
   menyimpan elevasi pada atribut `elevation`, sehingga CAD dan GIS dapat
   membaca Z-nya dan memberi label sendiri.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import ezdxf
import numpy as np

from .logging_setup import get_logger
from .topo import Surface

log = get_logger("dxfout")

# Indeks warna AutoCAD (ACI).
ACI = {
    "roof_minor": 151, "roof_index": 150,
    "floor_minor": 31, "floor_index": 30,
    "subcrop": 1, "hole": 7, "hole_label": 8,
}
MIN_VERTICES = 3


@dataclass
class ContourLayerSpec:
    """Nama layer untuk satu permukaan."""

    main: str
    index: str


def layer_names(seam: str, surface: str, template: str, index_suffix: str) -> ContourLayerSpec:
    """Bangun nama layer dari template, mis. 'Seam {seam} {surface}'."""
    label = {"roof": "Roof", "floor": "Floor"}.get(surface, surface.title())
    main = template.format(seam=seam, surface=label)
    return ContourLayerSpec(main=main, index=f"{main}{index_suffix}")


def sanitize_layer(name: str) -> str:
    """Buang karakter yang ditolak DXF pada nama layer.

    Spasi DIBIARKAN: AutoCAD menerimanya, dan pengguna meminta bentuk
    'Seam A Roof'. Karakter yang benar-benar terlarang (< > / \\ " : ; ? * | = `)
    diganti garis bawah.
    """
    return re.sub(r'[<>/\\":;?*|=`]', "_", str(name)).strip() or "LAYER"


def contour_segments(
    surface: Surface, interval_m: float, mask: np.ndarray | None = None
) -> dict[float, list[np.ndarray]]:
    """Hitung jejak kontur per level. Mengembalikan {elevasi: [array (N,2), ...]}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.array(surface.z, dtype=float)
    if mask is not None:
        z = np.where(mask, z, np.nan)
    if not np.isfinite(z).any():
        return {}

    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    start = np.floor(zmin / interval_m) * interval_m
    stop = np.ceil(zmax / interval_m) * interval_m
    levels = np.arange(start, stop + interval_m, interval_m)
    if len(levels) < 2:
        return {}

    gx, gy = np.meshgrid(surface.x, surface.y)
    figure = plt.figure()
    try:
        contours = plt.contour(gx, gy, z, levels=levels)
        out: dict[float, list[np.ndarray]] = {}
        for level, segments in zip(contours.levels, contours.allsegs):
            kept = [np.asarray(s, float) for s in segments if len(s) >= MIN_VERTICES]
            if kept:
                out[float(level)] = kept
        return out
    finally:
        plt.close(figure)


def _add_polylines(msp, segments: dict[float, list[np.ndarray]], layer: str,
                   index_layer: str, index_every: int, interval_m: float) -> int:
    count = 0
    for level, paths in sorted(segments.items()):
        is_index = index_every > 1 and abs(
            round(level / (interval_m * index_every)) * interval_m * index_every - level
        ) < interval_m / 2.0
        target = index_layer if is_index else layer
        for path in paths:
            msp.add_lwpolyline(
                [(float(x), float(y)) for x, y in path],
                dxfattribs={"layer": target, "elevation": float(level)},
            )
            count += 1
    return count


def _ensure_layer(doc, name: str, colour: int) -> None:
    if name not in doc.layers:
        doc.layers.add(name=name, color=colour)


def export_seam_contours(
    path: Path,
    seams: dict[str, dict[str, Surface]],
    *,
    interval_m: float,
    index_every: int,
    layer_template: str,
    index_suffix: str,
    subcrop_masks: dict[str, np.ndarray] | None = None,
    holes=None,
    subcrop_lines: dict[str, object] | None = None,
    topography: Surface | None = None,
    seam_filter: list[str] | None = None,
    surface_filter: list[str] | None = None,
) -> tuple[Path, dict]:
    """Tulis satu DXF berisi kontur untuk seam (dan permukaan) yang diminta.

    `surface_filter` membatasi ke 'roof' saja atau 'floor' saja, dipakai untuk
    menghasilkan berkas terpisah per permukaan.
    """
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6          # meter
    msp = doc.modelspace()
    summary: dict[str, dict] = {}

    for seam, surfaces in seams.items():
        if seam == "topo" or (seam_filter and seam not in seam_filter):
            continue
        mask = (subcrop_masks or {}).get(seam)
        for surface_key in ("roof", "floor"):
            if surface_filter and surface_key not in surface_filter:
                continue
            surface = surfaces.get(surface_key)
            if surface is None:
                continue
            spec = layer_names(seam, surface_key, layer_template, index_suffix)
            main, index = sanitize_layer(spec.main), sanitize_layer(spec.index)
            _ensure_layer(doc, main, ACI[f"{surface_key}_minor"])
            _ensure_layer(doc, index, ACI[f"{surface_key}_index"])

            segments = contour_segments(surface, interval_m, mask=mask)
            n = _add_polylines(msp, segments, main, index, index_every, interval_m)
            summary[main] = {
                "polylines": n,
                "levels": len(segments),
                "z_min": min(segments) if segments else None,
                "z_max": max(segments) if segments else None,
            }

        line = (subcrop_lines or {}).get(seam)
        if line is not None and not getattr(line, "is_empty", True):
            layer = sanitize_layer(f"{layer_template.format(seam=seam, surface='Subcrop')}")
            _ensure_layer(doc, layer, ACI["subcrop"])
            # Subcrop DI-DRAPE ke topografi: menurut definisi ia adalah garis
            # tempat seam memotong permukaan tanah. Menulisnya pada elevasi 0
            # membuatnya melayang di tempat yang salah pada model 3D, dan di
            # sini RL berkisar -100 sampai +60.
            written_lines = 0
            for geom in getattr(line, "geoms", [line]):
                if geom.geom_type != "Polygon":
                    continue
                coords = list(geom.exterior.coords)
                xs = np.array([c[0] for c in coords], float)
                ys = np.array([c[1] for c in coords], float)
                zs = (topography.sample(xs, ys) if topography is not None
                      else np.zeros_like(xs))
                zs = np.where(np.isfinite(zs), zs, 0.0)
                msp.add_polyline3d(
                    [(float(x), float(y), float(z)) for x, y, z in zip(xs, ys, zs)],
                    dxfattribs={"layer": layer},
                )
                written_lines += 1
            summary[layer] = {"polylines": written_lines, "draped_to": "topografi"}

    if holes is not None and len(holes):
        _ensure_layer(doc, "Boreholes", ACI["hole"])
        _ensure_layer(doc, "Borehole Labels", ACI["hole_label"])
        for _, row in holes.iterrows():
            x, y = float(row["east"]), float(row["north"])
            z = float(row.get("collar_rl_m", 0.0) or 0.0)
            msp.add_point((x, y, z), dxfattribs={"layer": "Boreholes"})
            label = str(row.get("hole_id", "")).strip()
            if label:
                msp.add_text(
                    label, height=8.0,
                    dxfattribs={"layer": "Borehole Labels"},
                ).set_placement((x + 6.0, y + 6.0))
        summary["Boreholes"] = {"points": int(len(holes))}

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path, summary


def surface_file_stem(seam: str, surface: str, template: str) -> str:
    """Nama berkas untuk satu permukaan, mis. 'Seam A Roof'."""
    label = {"roof": "Roof", "floor": "Floor"}.get(surface, surface.title())
    return sanitize_filename(template.format(seam=seam, surface=label))


def sanitize_filename(name: str) -> str:
    """Nama berkas yang aman di semua sistem berkas, spasi dipertahankan."""
    return re.sub(r'[<>:"/\\|?*]', "_", str(name)).strip() or "layer"


def subcrop_mask(topo: Surface, roof: Surface) -> np.ndarray:
    """Sel di mana roof seam berada DI BAWAH topografi (batubara masih ada)."""
    return np.isfinite(topo.z) & np.isfinite(roof.z) & (roof.z < topo.z)
