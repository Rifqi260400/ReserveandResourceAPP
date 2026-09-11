"""Batasan prospek ekonomi (RPEEE) dan aturan label inventori.

SNI 5015:2019 mendefinisikan sumberdaya batubara sebagai batubara in-situ yang
punya prospek wajar untuk pada akhirnya ditambang secara ekonomis. Batubara
in-situ yang dilaporkan TANPA batasan itu punya namanya sendiri di standar:
INVENTORI BATUBARA. Melaporkan inventori dengan label sumberdaya adalah
kesalahan klasifikasi, bukan kesalahan aritmetika.

Aturan 8.4 ditegakkan di sini dan TIDAK dapat dilewati setelan lain:
  - max_depth_m kosong  -> seluruh keluaran berlabel Inventori Batubara
  - max_depth_m terisi tanpa dasar -> ditolak saat konfigurasi dimuat
  - max_depth_m terisi dengan dasar -> keluaran berlabel Sumberdaya

Batasan diterapkan berurutan dan SETIAP langkah melaporkan luas dan tonase yang
ia buang, sehingga pengaruhnya terlihat dan bukan terserap ke angka akhir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon

from .config import Config
from .errors import ConfigError
from .estimate import ResourcePolygon, clip_polygons
from .logging_setup import get_logger
from .topo import Surface, depth_limit_extent, subcrop_extent

log = get_logger("rpeee")


@dataclass
class ConstraintStep:
    order: int
    name: str
    applied: bool
    basis: str
    area_ha_before: float
    area_ha_after: float
    tonnes_before: float
    tonnes_after: float
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def tonnes_removed(self) -> float:
        return self.tonnes_before - self.tonnes_after

    @property
    def area_ha_removed(self) -> float:
        return self.area_ha_before - self.area_ha_after


def _totals(polygons: list[ResourcePolygon]) -> tuple[float, float]:
    return (sum(p.area_m2 for p in polygons) / 10_000.0,
            sum(p.tonnes for p in polygons))


def parse_wkt(text: str, label: str) -> Polygon | None:
    if not text or not text.strip():
        return None
    try:
        geometry = wkt.loads(text)
    except Exception as exc:
        raise ConfigError(f"{label}: WKT tidak dapat dibaca: {exc}") from exc
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def apply_constraints(
    polygons: list[ResourcePolygon],
    seam_surfaces: dict[str, dict[str, Surface]],
    cfg: Config,
) -> tuple[list[ResourcePolygon], list[ConstraintStep]]:
    """Terapkan seluruh pemotongan berurutan, mencatat pengaruh tiap langkah.

    Urutan: subcrop -> batas kedalaman -> batas blok -> cutoff kualitas ->
    area terlarang. Cutoff kualitas dikenakan per interseksi seam memakai
    kualitas terbobot tonase interseksi itu, bukan rata-rata seluruh seam.
    """
    steps: list[ConstraintStep] = []
    order = 0

    def record(name: str, basis: str, applied: bool, before, detail=pd.DataFrame()):
        nonlocal order
        order += 1
        area_after, tonnes_after = _totals(polygons)
        steps.append(ConstraintStep(
            order=order, name=name, applied=applied, basis=basis,
            area_ha_before=before[0], area_ha_after=area_after,
            tonnes_before=before[1], tonnes_after=tonnes_after, detail=detail,
        ))

    # --- 1. Subcrop: sumberdaya tidak pernah diekstrapolasi ke atas tanah ---- #
    before = _totals(polygons)
    details = []
    for seam, surfaces in seam_surfaces.items():
        if "subcrop" not in surfaces:
            continue
        subset = [p for p in polygons if p.seam == seam]
        others = [p for p in polygons if p.seam != seam]
        kept, removed = clip_polygons(subset, surfaces["subcrop"], "subcrop")
        polygons = others + kept
        if not removed.empty:
            details.append(removed)
    record("subcrop", "roof seam di bawah topografi", True, before,
           pd.concat(details) if details else pd.DataFrame())

    # --- 2. Batas kedalaman: DIKENAKAN PADA POLIGON, bukan pada lubang ------ #
    constraints = cfg.rpeee_constraints
    before = _totals(polygons)
    if constraints.max_depth_m is not None:
        details = []
        for seam, surfaces in seam_surfaces.items():
            if "depth_limit" not in surfaces:
                continue
            subset = [p for p in polygons if p.seam == seam]
            others = [p for p in polygons if p.seam != seam]
            kept, removed = clip_polygons(subset, surfaces["depth_limit"], "batas kedalaman")
            polygons = others + kept
            if not removed.empty:
                details.append(removed)
        record("batas kedalaman", constraints.max_depth_basis, True, before,
               pd.concat(details) if details else pd.DataFrame())
    else:
        record("batas kedalaman", "TIDAK DITERAPKAN - keluaran berlabel Inventori Batubara",
               False, before)

    # --- 3. Batas blok ------------------------------------------------------ #
    before = _totals(polygons)
    boundary = parse_wkt(cfg.block_boundary_wkt, "block_boundary_wkt")
    if boundary is not None:
        polygons, removed = clip_polygons(polygons, boundary, "batas blok")
        record("batas blok", "block_boundary_wkt", True, before, removed)
    else:
        record("batas blok", "tidak dipasok", False, before)

    # --- 4. Cutoff kualitas ------------------------------------------------- #
    before = _totals(polygons)
    rules = []
    if constraints.min_cv_ar_kcal_kg is not None:
        rules.append(("CV_ar", "min", constraints.min_cv_ar_kcal_kg))
    if constraints.max_ash_adb_pct is not None:
        rules.append(("ASH_adb", "max", constraints.max_ash_adb_pct))

    if rules:
        kept, removed = [], []
        for polygon in polygons:
            fails = []
            for attribute, direction, limit in rules:
                value = polygon.quality.get(attribute)
                if value is None or not np.isfinite(value):
                    continue
                if (direction == "min" and value < limit) or (direction == "max" and value > limit):
                    fails.append(f"{attribute}={value:.1f} melanggar {direction} {limit}")
            if fails:
                removed.append({
                    "reason": "cutoff kualitas", "seam": polygon.seam,
                    "hole_id": polygon.hole_id, "class": polygon.resource_class.sni_name,
                    "area_removed_m2": polygon.area_m2, "tonnes_removed": polygon.tonnes,
                    "detail": "; ".join(fails),
                })
            else:
                kept.append(polygon)
        polygons = kept
        record("cutoff kualitas", "; ".join(f"{a} {d} {v}" for a, d, v in rules),
               True, before, pd.DataFrame(removed))
    else:
        record("cutoff kualitas", "tidak dipasok", False, before)

    # --- 5. Area terlarang -------------------------------------------------- #
    before = _totals(polygons)
    excluded = parse_wkt(constraints.excluded_area_wkt, "excluded_area_wkt")
    if excluded is not None:
        kept, removed = [], []
        for polygon in polygons:
            area_before, tonnes_before = polygon.area_m2, polygon.tonnes
            remainder = polygon.geometry.difference(excluded)
            if remainder.is_empty or remainder.area <= 0:
                removed.append({
                    "reason": "area terlarang", "seam": polygon.seam,
                    "hole_id": polygon.hole_id, "class": polygon.resource_class.sni_name,
                    "area_removed_m2": area_before, "tonnes_removed": tonnes_before,
                })
                continue
            polygon.geometry = remainder if remainder.is_valid else remainder.buffer(0)
            kept.append(polygon)
            if area_before - polygon.area_m2 > 1e-9:
                removed.append({
                    "reason": "area terlarang", "seam": polygon.seam,
                    "hole_id": polygon.hole_id, "class": polygon.resource_class.sni_name,
                    "area_removed_m2": area_before - polygon.area_m2,
                    "tonnes_removed": tonnes_before - polygon.tonnes,
                })
        polygons = kept
        record("area terlarang", constraints.excluded_area_basis, True,
               before, pd.DataFrame(removed))
    else:
        record("area terlarang", "tidak dipasok", False, before)

    return polygons, steps


def reconciliation_table(steps: list[ConstraintStep], cfg: Config) -> pd.DataFrame:
    """Tonase in-situ kotor menurun melewati setiap batasan sampai angka akhir."""
    rows = []
    for step in steps:
        rows.append({
            "urutan": step.order,
            "batasan": step.name,
            "diterapkan": "ya" if step.applied else "tidak",
            "dasar": step.basis,
            "tonnes_sebelum": step.tonnes_before,
            "tonnes_dibuang": step.tonnes_removed,
            "tonnes_sesudah": step.tonnes_after,
            "area_ha_dibuang": step.area_ha_removed,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        label = cfg.rpeee_constraints.resource_label
        frame.attrs["resource_label"] = label
    return frame


def label_warning(cfg: Config) -> str | None:
    """Peringatan yang dicetak di awal DAN akhir run bila tidak ada batasan."""
    if cfg.rpeee_constraints.has_economic_constraint:
        return None
    return (
        "TIDAK ADA BATASAN PROSPEK EKONOMI YANG DITERAPKAN (max_depth_m kosong). "
        "Seluruh keluaran run ini berlabel INVENTORI BATUBARA, bukan Sumberdaya. "
        "Kolom kelas adalah Inventori Terukur / Tertunjuk / Tereka. Melaporkan "
        "angka ini sebagai Sumberdaya adalah kesalahan klasifikasi."
    )
