"""Grid reguler dan interpolasi.

Konvensi: grid disimpan sebagai array 2D dengan indeks [baris, kolom] =
[y, x], baris 0 pada y minimum. Sel yang tidak punya nilai bernilai NaN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Grid:
    """Definisi grid reguler."""

    xmin: float
    ymin: float
    cell_size: float
    ncols: int
    nrows: int

    @classmethod
    def from_extent(
        cls, xmin: float, xmax: float, ymin: float, ymax: float,
        cell_size: float, margin: float = 0.0,
    ) -> "Grid":
        xmin, xmax = xmin - margin, xmax + margin
        ymin, ymax = ymin - margin, ymax + margin
        ncols = int(np.ceil((xmax - xmin) / cell_size)) + 1
        nrows = int(np.ceil((ymax - ymin) / cell_size)) + 1
        return cls(xmin, ymin, cell_size, ncols, nrows)

    @property
    def cell_area(self) -> float:
        return self.cell_size ** 2

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nrows, self.ncols)

    @property
    def x(self) -> np.ndarray:
        return self.xmin + np.arange(self.ncols) * self.cell_size

    @property
    def y(self) -> np.ndarray:
        return self.ymin + np.arange(self.nrows) * self.cell_size

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y)

    def empty(self, fill: float = np.nan) -> np.ndarray:
        return np.full(self.shape, fill, dtype=float)


def _idw(
    xi: np.ndarray, yi: np.ndarray,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    power: float, max_points: int, search_radius: float,
) -> np.ndarray:
    """Inverse distance weighting dengan batas jumlah titik dan radius cari."""
    tree = cKDTree(np.column_stack([x, y]))
    k = min(max_points, len(x))
    dist, idx = tree.query(np.column_stack([xi, yi]), k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]

    out = np.full(len(xi), np.nan)
    within = dist <= search_radius
    has_any = within.any(axis=1)

    # Titik yang berimpit dengan simpul data: ambil nilainya langsung.
    exact = dist[:, 0] < 1e-9
    out[exact] = z[idx[exact, 0]]

    work = has_any & ~exact
    if work.any():
        d = np.where(within[work], dist[work], np.inf)
        w = 1.0 / np.power(d, power)
        vals = z[idx[work]]
        out[work] = np.sum(w * vals, axis=1) / np.sum(w, axis=1)
    return out


def interpolate_to_grid(
    grid: Grid,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    method: str = "idw",
    power: float = 2.0,
    max_points: int = 12,
    search_radius: float = 1500.0,
    max_extrapolation: float | None = 500.0,
) -> np.ndarray:
    """Interpolasikan titik tersebar ke grid.

    ``max_extrapolation`` membatasi hasil agar tidak menjalar jauh dari data.
    Tanpa batas ini, seam bisa "terbang" kilometer dari titik bor terdekat dan
    tetap ikut terhitung sebagai tonase.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) == 0:
        return grid.empty()

    gx, gy = grid.meshgrid()
    xi, yi = gx.ravel(), gy.ravel()

    if len(x) == 1:
        values = np.full(xi.shape, z[0])
    elif method == "idw":
        values = _idw(xi, yi, x, y, z, power, max_points, search_radius)
    elif method == "nearest":
        values = griddata((x, y), z, (xi, yi), method="nearest")
    elif method == "linear":
        values = griddata((x, y), z, (xi, yi), method="linear")
        gap = np.isnan(values)
        if gap.any():  # isi tepi dengan nearest agar DTM tidak berlubang
            values[gap] = griddata((x, y), z, (xi[gap], yi[gap]), method="nearest")
    else:
        raise ValueError(f"metode interpolasi tidak dikenal: {method}")

    if max_extrapolation is not None:
        tree = cKDTree(np.column_stack([x, y]))
        dist, _ = tree.query(np.column_stack([xi, yi]), k=1)
        values = np.where(dist <= max_extrapolation, values, np.nan)

    return np.asarray(values, float).reshape(grid.shape)


def distance_to_data(grid: Grid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Jarak tiap sel ke titik data terdekat."""
    tree = cKDTree(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    gx, gy = grid.meshgrid()
    dist, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    return dist.reshape(grid.shape)


def count_within_radius(
    grid: Grid, x: np.ndarray, y: np.ndarray, radius: float
) -> np.ndarray:
    """Jumlah titik data dalam radius tiap sel.

    Dipakai untuk memberlakukan kriteria SPASI pada klasifikasi: satu lubang
    terisolasi tidak membuktikan kontinuitas, jadi kelas tertinggi menuntut
    beberapa titik yang saling mendukung, bukan sekadar satu titik terdekat.
    """
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    tree = cKDTree(pts)
    gx, gy = grid.meshgrid()
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    counts = tree.query_ball_point(cells, r=radius, return_length=True)
    return np.asarray(counts, float).reshape(grid.shape)


def surface_dip_degrees(surface: np.ndarray, cell_size: float) -> np.ndarray:
    """Dip tiap sel dari gradien permukaan, dalam derajat.

    Dipakai untuk mengubah ketebalan vertikal menjadi true thickness saat
    menerapkan cutoff, dan sebagai pembanding independen terhadap dip yang
    diukur di core.
    """
    dz_dy, dz_dx = np.gradient(surface, cell_size, cell_size)
    slope = np.hypot(dz_dx, dz_dy)
    return np.degrees(np.arctan(slope))
