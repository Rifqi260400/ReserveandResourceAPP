"""Klasifikasi sumberdaya menurut jarak dari titik observasi.

PERINGATAN YANG IKUT KE MANA PUN ANGKA INI DIBAWA. Radius diambil dari
konfigurasi dan BELUM diverifikasi terhadap teks SNI 5015:2019. Kondisi geologi
yang memilih baris radius adalah penilaian geolog, bukan keputusan program:
measured 500 m pada 'sederhana' melawan 100 m pada 'kompleks'.

Radius bersifat PITA KUMULATIF dari titik observasi:
    measured  : cakram dalam,                 0        .. r_measured
    indicated : anulus,                r_measured      .. r_indicated
    inferred  : anulus,                r_indicated     .. r_inferred
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from shapely.geometry import Point, Polygon

from .config import ClassificationRadii, Config

# Jumlah segmen per lingkaran penuh. Luas poligon-n beraturan adalah
# (n/2pi)*sin(2pi/n) kali luas lingkaran sejati, jadi galatnya:
#     n=16  -> -2,58%   (bawaan shapely)
#     n=64  -> -0,16%
#     n=128 -> -0,04%
# Galat ini terbawa LANGSUNG ke tonase, jadi dipakai 128.
CIRCLE_SEGMENTS = 128


class ResourceClass(IntEnum):
    INFERRED = 1
    INDICATED = 2
    MEASURED = 3

    @property
    def sni_name(self) -> str:
        return {1: "Tereka", 2: "Tertunjuk", 3: "Terukur"}[int(self)]


ORDERED = (ResourceClass.MEASURED, ResourceClass.INDICATED, ResourceClass.INFERRED)


def class_label(resource_class: ResourceClass, cfg: Config) -> str:
    """Label lengkap, memuat awalan Sumberdaya/Inventori dari aturan 8.4."""
    return f"{cfg.rpeee_constraints.class_prefix} {resource_class.sni_name}"


@dataclass(frozen=True)
class ClassBand:
    resource_class: ResourceClass
    inner_radius_m: float
    outer_radius_m: float

    def geometry(self, east: float, north: float) -> Polygon:
        centre = Point(east, north)
        outer = centre.buffer(self.outer_radius_m, quad_segs=CIRCLE_SEGMENTS // 4)
        if self.inner_radius_m <= 0:
            return outer
        inner = centre.buffer(self.inner_radius_m, quad_segs=CIRCLE_SEGMENTS // 4)
        return outer.difference(inner)


def bands(radii: ClassificationRadii) -> list[ClassBand]:
    """Pita dari terdalam ke terluar."""
    return [
        ClassBand(ResourceClass.MEASURED, 0.0, radii.measured),
        ClassBand(ResourceClass.INDICATED, radii.measured, radii.indicated),
        ClassBand(ResourceClass.INFERRED, radii.indicated, radii.inferred),
    ]


def disc(resource_class: ResourceClass, radii: ClassificationRadii,
         east: float, north: float) -> Polygon:
    """Cakram penuh sampai radius kelas tersebut (tanpa lubang di tengah).

    Dipakai metode sirkular, di mana tumpang tindih diselesaikan dengan
    memberikan kelas TERTINGGI yang hadir - sehingga tonase tidak pernah
    terhitung ganda.
    """
    radius = {
        ResourceClass.MEASURED: radii.measured,
        ResourceClass.INDICATED: radii.indicated,
        ResourceClass.INFERRED: radii.inferred,
    }[resource_class]
    return Point(east, north).buffer(radius, quad_segs=CIRCLE_SEGMENTS // 4)
