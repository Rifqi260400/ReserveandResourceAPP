"""Permukaan topografi dan permukaan seam.

Permukaan di sini dipakai untuk GEOMETRI: subcrop, overburden, dan grid
kedalaman di bawah permukaan yang memberi makan batas kedalaman RPEEE.

Batas yang harus dijaga: permukaan interpolasi TIDAK PERNAH menyumbang
ketebalan ke jalur tonase. Tebal selalu berasal dari lubangnya sendiri
(lihat estimate.py). Permukaan hanya boleh MEMOTONG poligon - lewat subcrop
dan kontur batas kedalaman - sehingga ia mempengaruhi luas, tidak pernah tebal.
Batas itu dinyatakan di sini karena mudah dilanggar tanpa disadari.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .errors import MissingDataError
from .logging_setup import get_logger

log = get_logger("topo")

MIN_POINTS_FOR_TIN = 3


@dataclass
class Surface:
    """Permukaan hasil interpolasi TIN atas grid reguler."""

    name: str
    x: np.ndarray                  # sumbu grid, 1D
    y: np.ndarray
    z: np.ndarray                  # (ny, nx), NaN di luar dukungan data
    spacing: float
    method: str
    n_points: int
    hull: Polygon | None = None    # convex hull titik pendukung

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (float(self.x.min()), float(self.x.max()),
                float(self.y.min()), float(self.y.max()))

    def sample(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        """Ambil nilai permukaan pada titik sembarang lewat interpolasi bilinear."""
        east = np.atleast_1d(np.asarray(east, float))
        north = np.atleast_1d(np.asarray(north, float))
        ix = np.clip(np.searchsorted(self.x, east) - 1, 0, len(self.x) - 2)
        iy = np.clip(np.searchsorted(self.y, north) - 1, 0, len(self.y) - 2)
        x0, x1 = self.x[ix], self.x[ix + 1]
        y0, y1 = self.y[iy], self.y[iy + 1]
        tx = np.divide(east - x0, x1 - x0, out=np.zeros_like(east), where=(x1 - x0) != 0)
        ty = np.divide(north - y0, y1 - y0, out=np.zeros_like(north), where=(y1 - y0) != 0)
        z00, z10 = self.z[iy, ix], self.z[iy, ix + 1]
        z01, z11 = self.z[iy + 1, ix], self.z[iy + 1, ix + 1]
        return ((1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10
                + (1 - tx) * ty * z01 + tx * ty * z11)

    def contains(self, east: np.ndarray, north: np.ndarray) -> np.ndarray:
        return np.isfinite(self.sample(east, north))


def _grid_axes(x: np.ndarray, y: np.ndarray, spacing: float, margin: float):
    xs = np.arange(x.min() - margin, x.max() + margin + spacing, spacing)
    ys = np.arange(y.min() - margin, y.max() + margin + spacing, spacing)
    return xs, ys


def build_surface(
    east: np.ndarray, north: np.ndarray, value: np.ndarray,
    spacing: float, margin: float = 0.0, name: str = "surface",
    method: str = "linear",
) -> Surface:
    """Interpolasi TIN (Delaunay) titik tersebar menjadi permukaan ber-grid.

    Di luar convex hull titik data hasilnya NaN - tidak diekstrapolasi. Permukaan
    seam yang menjalar jauh dari lubang adalah cara paling mudah memperoleh
    tonase yang tidak pernah dibor.
    """
    east = np.asarray(east, float)
    north = np.asarray(north, float)
    value = np.asarray(value, float)
    ok = np.isfinite(east) & np.isfinite(north) & np.isfinite(value)
    east, north, value = east[ok], north[ok], value[ok]

    if len(east) < MIN_POINTS_FOR_TIN:
        raise MissingDataError(
            f"permukaan '{name}': {len(east)} titik, minimum {MIN_POINTS_FOR_TIN} "
            "untuk triangulasi Delaunay."
        )

    points = np.column_stack([east, north])
    try:
        triangulation = Delaunay(points)
    except QhullError as exc:
        raise MissingDataError(
            f"permukaan '{name}': triangulasi gagal (titik segaris?): {exc}"
        ) from exc

    if method == "linear":
        interpolator = LinearNDInterpolator(triangulation, value)
    elif method == "clough_tocher":
        interpolator = CloughTocher2DInterpolator(triangulation, value)
    elif method == "nearest":
        interpolator = NearestNDInterpolator(points, value)
    else:
        raise MissingDataError(f"metode interpolasi tidak dikenal: {method}")

    xs, ys = _grid_axes(east, north, spacing, margin)
    gx, gy = np.meshgrid(xs, ys)
    z = np.asarray(interpolator(gx, gy), dtype=float)

    hull = Polygon(points[triangulation.convex_hull[:, 0]]).convex_hull if len(points) >= 3 else None
    if hull is not None and not hull.is_valid:
        hull = hull.buffer(0)

    return Surface(name=name, x=xs, y=ys, z=z, spacing=spacing,
                   method=method, n_points=len(east), hull=hull)


def difference(a: Surface, b: Surface, name: str) -> Surface:
    """a - b, pada grid yang sama. Dipakai untuk kedalaman di bawah permukaan."""
    if a.z.shape != b.z.shape or not np.allclose(a.x, b.x) or not np.allclose(a.y, b.y):
        raise MissingDataError(
            f"permukaan '{a.name}' dan '{b.name}' tidak berada pada grid yang sama"
        )
    return Surface(name=name, x=a.x, y=a.y, z=a.z - b.z, spacing=a.spacing,
                   method=a.method, n_points=min(a.n_points, b.n_points), hull=a.hull)


def mask_to_polygon(surface: Surface, mask: np.ndarray, simplify: float = 0.0):
    """Ubah mask boolean pada grid menjadi poligon shapely.

    Dipakai untuk subcrop (seam berada di bawah topografi) dan untuk kontur
    batas kedalaman. Sel dijadikan kotak lalu digabung; sederhana, deterministik,
    dan tidak bergantung pada penanganan kontur terbuka oleh pustaka lain.
    """
    if mask.shape != surface.z.shape:
        raise MissingDataError("bentuk mask tidak sama dengan bentuk grid permukaan")
    if not mask.any():
        return Polygon()

    half = surface.spacing / 2.0
    boxes = []
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        cx, cy = surface.x[c], surface.y[r]
        boxes.append(Polygon([
            (cx - half, cy - half), (cx + half, cy - half),
            (cx + half, cy + half), (cx - half, cy + half),
        ]))
    merged = unary_union(boxes)
    if simplify > 0:
        merged = merged.simplify(simplify, preserve_topology=True)
    if isinstance(merged, Polygon):
        return merged
    return MultiPolygon([g for g in merged.geoms if isinstance(g, Polygon)])


def subcrop_extent(topo: Surface, seam_roof: Surface, simplify: float | None = None):
    """Area di mana roof seam berada DI BAWAH topografi.

    Sumberdaya dipotong pada subcrop dan tidak pernah diekstrapolasi ke atas
    permukaan tanah.

    `simplify` bawaannya satu sel grid: cukup untuk menghilangkan gerigi
    rasterisasi, dan tidak pernah menggeser garis lebih jauh dari resolusi yang
    mendasarinya. Garis subcrop tidak boleh tampak lebih presisi daripada grid
    yang membentuknya.
    """
    valid = np.isfinite(topo.z) & np.isfinite(seam_roof.z)
    tolerance = topo.spacing if simplify is None else simplify
    return mask_to_polygon(topo, valid & (seam_roof.z < topo.z), simplify=tolerance)


def depth_limit_extent(depth_below_surface: Surface, max_depth_m: float,
                       simplify: float | None = None):
    """Area di mana roof seam berada tidak lebih dalam dari batas.

    Batas kedalaman TIDAK boleh diterapkan di titik bor saja: satu poligon
    membentang jauh melewati lubangnya, dan seam yang berada di 80 m pada
    lubang bisa berada di 300 m di seberang poligon.
    """
    z = depth_below_surface.z
    tolerance = depth_below_surface.spacing if simplify is None else simplify
    return mask_to_polygon(depth_below_surface, np.isfinite(z) & (z <= max_depth_m),
                           simplify=tolerance)


def support_extent(surface: Surface) -> Polygon:
    """Area yang benar-benar didukung data bor, bukan hasil ekstrapolasi."""
    return surface.hull if surface.hull is not None else Polygon()
