"""Estimasi poligonal: Voronoi (bawaan) dan titik observasi sirkular.

RUMUS TONASE

    Tonnes = luas poligon DALAM PETA (m2) x tebal VERTIKAL batubara (m)
             x RD in-situ (t/m3)

TANPA koreksi cosinus dip. Luas dalam peta dikalikan tebal vertikal sudah
memberi volume prisma yang sebenarnya untuk seam miring berbidang datar: dip
mengecilkan tebal tegak lurus DAN membesarkan luas bidang seam, dan keduanya
saling meniadakan. Menerapkan cosinus sekali di atas luas peta akan MENGURANGI
tonase secara keliru. Jangan "perbaiki" ini.

TEBAL BERASAL DARI LUBANGNYA SENDIRI. Di bawah metode Voronoi, poligon adalah
daerah pengaruh satu lubang; menginterpolasi tebal antar poligon bertentangan
dengan asumsi itu. Permukaan interpolasi hanya boleh MEMOTONG poligon (subcrop,
batas kedalaman), tidak pernah menyumbang tebal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, box
from shapely.ops import unary_union, voronoi_diagram

from .classify import ORDERED, ClassBand, ResourceClass, bands, class_label, disc
from .config import Config
from .errors import MissingDataError
from .logging_setup import get_logger

log = get_logger("estimate")

MIN_POINTS_FOR_VORONOI = 3


@dataclass
class HolePoint:
    """Satu lubang yang menembus satu seam, beserta atribut yang dibawanya."""

    hole_id: str
    east: float
    north: float
    coal_thickness_m: float
    rd_t_per_m3: float
    rd_basis: str
    rd_is_assumed: bool
    quality_coverage_frac: float = float("nan")
    quality: dict[str, float] = field(default_factory=dict)


@dataclass
class ResourcePolygon:
    seam: str
    hole_id: str
    resource_class: ResourceClass
    geometry: Polygon
    coal_thickness_m: float
    rd_t_per_m3: float
    rd_is_assumed: bool
    quality_coverage_frac: float
    quality: dict[str, float] = field(default_factory=dict)

    @property
    def area_m2(self) -> float:
        return float(self.geometry.area)

    @property
    def volume_m3(self) -> float:
        return self.area_m2 * self.coal_thickness_m

    @property
    def tonnes(self) -> float:
        # Tanpa cos(dip): lihat catatan modul.
        return self.volume_m3 * self.rd_t_per_m3


def _as_polygon(geometry) -> Polygon | MultiPolygon | None:
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry if geometry.is_valid else geometry.buffer(0)
    parts = [g for g in getattr(geometry, "geoms", []) if isinstance(g, Polygon)]
    if not parts:
        return None
    merged = unary_union(parts)
    return merged if merged.is_valid else merged.buffer(0)


def voronoi_cells(points: list[HolePoint], margin: float = 5000.0) -> dict[str, Polygon]:
    """Sel Voronoi per lubang, dibatasi amplop yang diperluas.

    Sel dicocokkan ke lubangnya lewat containment, bukan lewat urutan: shapely
    tidak menjamin urutan keluaran sama dengan urutan masukan, dan mengandalkan
    urutan akan menukar tebal antar lubang tanpa gejala yang terlihat.
    """
    if len(points) < MIN_POINTS_FOR_VORONOI:
        raise MissingDataError(
            f"Voronoi memerlukan minimal {MIN_POINTS_FOR_VORONOI} titik tidak "
            f"segaris; diberikan {len(points)}."
        )
    coords = [(p.east, p.north) for p in points]
    if len({(round(x, 6), round(y, 6)) for x, y in coords}) < MIN_POINTS_FOR_VORONOI:
        raise MissingDataError("titik bor berimpit; Voronoi tidak dapat dibentuk.")

    multipoint = MultiPoint(coords)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    envelope = box(min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

    try:
        diagram = voronoi_diagram(multipoint, envelope=envelope)
    except Exception as exc:  # geometri degenerate
        raise MissingDataError(f"diagram Voronoi gagal dibentuk: {exc}") from exc

    cells: dict[str, Polygon] = {}
    unmatched = []
    for cell in diagram.geoms:
        clipped = cell.intersection(envelope)
        if clipped.is_empty:
            continue
        owner = next(
            (p for p in points if clipped.contains(MultiPoint([(p.east, p.north)]).geoms[0])),
            None,
        )
        if owner is None:
            unmatched.append(clipped)
            continue
        cells[owner.hole_id] = clipped

    missing = [p.hole_id for p in points if p.hole_id not in cells]
    if missing:
        raise MissingDataError(
            f"sel Voronoi tidak dapat dicocokkan ke lubang: {missing} "
            f"({len(unmatched)} sel tanpa pemilik)"
        )
    return cells


def build_polygons(
    seam: str, points: list[HolePoint], cfg: Config, method: str | None = None
) -> list[ResourcePolygon]:
    """Bentuk poligon berkelas untuk satu seam, sebelum pemotongan RPEEE."""
    method = method or cfg.estimation_method
    radii = cfg.radii

    if method == "voronoi":
        return _voronoi_polygons(seam, points, radii)
    if method == "circular":
        return _circular_polygons(seam, points, radii)
    raise MissingDataError(f"metode estimasi tidak dikenal: {method}")


def _make(seam: str, point: HolePoint, resource_class: ResourceClass, geometry):
    geometry = _as_polygon(geometry)
    if geometry is None or geometry.area <= 0:
        return None
    return ResourcePolygon(
        seam=seam, hole_id=point.hole_id, resource_class=resource_class,
        geometry=geometry, coal_thickness_m=point.coal_thickness_m,
        rd_t_per_m3=point.rd_t_per_m3, rd_is_assumed=point.rd_is_assumed,
        quality_coverage_frac=point.quality_coverage_frac, quality=dict(point.quality),
    )


def _voronoi_polygons(seam, points, radii) -> list[ResourcePolygon]:
    """Metode A: sel Voronoi dipotong pita radius lubangnya sendiri.

    Sel Voronoi sudah saling eksklusif menurut konstruksi, jadi tidak ada
    tumpang tindih yang perlu diselesaikan.
    """
    cells = voronoi_cells(points)
    out: list[ResourcePolygon] = []
    for point in points:
        cell = cells[point.hole_id]
        for band in bands(radii):
            piece = _make(seam, point, band.resource_class,
                          cell.intersection(band.geometry(point.east, point.north)))
            if piece is not None:
                out.append(piece)
    return out


def _circular_polygons(seam, points, radii) -> list[ResourcePolygon]:
    """Metode B: cakram per lubang, tumpang tindih diberikan kelas TERTINGGI.

    Dikerjakan dari kelas tertinggi ke terendah, dan setiap kelas dikurangi
    gabungan seluruh area yang sudah diklaim kelas di atasnya - sehingga tidak
    ada satu meter persegi pun yang terhitung dua kali.
    """
    out: list[ResourcePolygon] = []
    claimed = Polygon()

    for resource_class in ORDERED:
        discs = {p.hole_id: disc(resource_class, radii, p.east, p.north) for p in points}
        level_union = unary_union(list(discs.values())).difference(claimed)
        if level_union.is_empty:
            continue
        # Di dalam satu kelas, area dibagi menurut lubang TERDEKAT agar tebal
        # dan kualitas yang dipakai berasal dari lubang yang paling relevan.
        try:
            cells = voronoi_cells(points)
        except MissingDataError:
            cells = {p.hole_id: level_union for p in points}
        for point in points:
            share = discs[point.hole_id].intersection(level_union).intersection(
                cells.get(point.hole_id, level_union)
            )
            piece = _make(seam, point, resource_class, share)
            if piece is not None:
                out.append(piece)
        claimed = unary_union([claimed, level_union])
    return out


def clip_polygons(
    polygons: list[ResourcePolygon], mask, reason: str
) -> tuple[list[ResourcePolygon], pd.DataFrame]:
    """Potong seluruh poligon dengan satu mask, catat yang hilang.

    Setiap batasan harus melaporkan luas dan tonase yang ia buang; kalau tidak,
    pengaruh batasan itu terserap ke dalam angka akhir dan tidak dapat ditinjau.
    """
    if mask is None or (hasattr(mask, "is_empty") and mask.is_empty):
        return polygons, pd.DataFrame()

    kept: list[ResourcePolygon] = []
    removed = []
    for polygon in polygons:
        before_area, before_tonnes = polygon.area_m2, polygon.tonnes
        clipped = _as_polygon(polygon.geometry.intersection(mask))
        after_area = float(clipped.area) if clipped is not None else 0.0
        if clipped is not None and after_area > 0:
            polygon.geometry = clipped
            kept.append(polygon)
        after_tonnes = polygon.tonnes if (clipped is not None and after_area > 0) else 0.0
        if before_area - after_area > 1e-9:
            removed.append({
                "reason": reason, "seam": polygon.seam, "hole_id": polygon.hole_id,
                "class": polygon.resource_class.sni_name,
                "area_removed_m2": before_area - after_area,
                "tonnes_removed": before_tonnes - after_tonnes,
            })
    return kept, pd.DataFrame(removed)


def to_frame(polygons: list[ResourcePolygon], cfg: Config) -> pd.DataFrame:
    if not polygons:
        return pd.DataFrame()
    rows = []
    for p in polygons:
        row = {
            "seam": p.seam, "hole_id": p.hole_id,
            "class": class_label(p.resource_class, cfg),
            "class_sni": p.resource_class.sni_name,
            "area_m2": p.area_m2, "area_ha": p.area_m2 / 10_000.0,
            "coal_thickness_m": p.coal_thickness_m,
            "rd_t_per_m3": p.rd_t_per_m3, "rd_is_assumed": p.rd_is_assumed,
            "quality_coverage_frac": p.quality_coverage_frac,
            "volume_m3": p.volume_m3, "tonnes": p.tonnes,
        }
        row.update(p.quality)
        rows.append(row)
    return pd.DataFrame(rows)


def rd_sensitivity(polygons: list[ResourcePolygon], deltas=(-0.10, -0.05, 0.0, 0.05, 0.10)) -> pd.DataFrame:
    """Tonase pada RD terpilih dan pada simpangan tetap.

    Tonase linear terhadap RD, dan di situlah kesalahan senyap terbesar berada.
    """
    base = sum(p.tonnes for p in polygons)
    return pd.DataFrame([
        {"rd_delta_pct": d * 100, "tonnes": base * (1 + d),
         "difference_tonnes": base * d}
        for d in deltas
    ])
