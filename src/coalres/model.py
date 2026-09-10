"""Pembangunan model stratigrafi ber-grid.

Strategi: STRUCTURE + ISOPACH, bukan interpolasi roof dan floor secara
independen. Floor tiap seam di-grid sebagai permukaan struktur, ketebalan
vertikal di-grid sebagai isopach, lalu roof diturunkan (roof = floor + isopach).

Alasannya: isopach selalu >= 0 dan lebih halus daripada struktur, sehingga
lebih stabil untuk diinterpolasi. Menginterpolasi roof dan floor secara
terpisah kerap menghasilkan ketebalan negatif atau permukaan yang saling
memotong di area ekstrapolasi.

Konvensi volume - ini menentukan kebenaran seluruh angka tonase:

    Volume = luas sel DALAM PETA x ketebalan VERTIKAL (roof RL - floor RL)

Prisma vertikal mengisi ruang antar dua permukaan secara persis, berapa pun
dip-nya, dan lubang vertikal mengukur tepat besaran itu. TIDAK ADA koreksi
cos(dip) pada volume. Menerapkannya justru MENGURANGI tonase secara keliru.

True thickness (= tebal vertikal x cos dip) dipakai hanya untuk hal yang
memang bergantung pada tebal tegak lurus: cutoff ketebalan minimum, parameter
penambangan, dan interpolasi isopach saat dip bervariasi kuat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .grid import Grid, interpolate_to_grid, surface_dip_degrees
from .quality import composite_seam


@dataclass
class SeamModel:
    """Model ber-grid untuk satu seam."""

    name: str
    floor_rl: np.ndarray
    roof_rl: np.ndarray
    thickness_vertical: np.ndarray
    thickness_true: np.ndarray
    dip_degrees: np.ndarray
    depth_to_roof: np.ndarray
    quality: dict[str, np.ndarray] = field(default_factory=dict)
    n_holes: int = 0
    subcropped_cells: int = 0
    eroded_cells: int = 0

    @property
    def present(self) -> np.ndarray:
        """Sel yang punya batubara dengan ketebalan positif."""
        return np.isfinite(self.thickness_vertical) & (self.thickness_vertical > 0)


@dataclass
class StratModel:
    grid: Grid
    topo: np.ndarray
    seams: dict[str, SeamModel] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    hole_points: pd.DataFrame | None = None

    def seam(self, name: str) -> SeamModel:
        return self.seams[name]


def collapse_plies(seam: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan ply menjadi satu baris per (lubang, seam).

    Roof diambil dari ply teratas dan floor dari ply terbawah, sehingga
    ketebalan yang dihasilkan mencakup parting di dalam amplop seam. Parting
    disaring lebih dulu oleh aturan seam (max_parting_thickness) di hulu.
    """
    rows = []
    for (hid, sname), group in seam.groupby(["hole_id", "seam"], sort=False):
        group = group.sort_values("depth_from")
        composited = composite_seam(group)
        row = {
            "hole_id": hid,
            "seam": sname,
            "depth_from": float(group["depth_from"].min()),
            "depth_to": float(group["depth_to"].max()),
            "n_plies": len(group),
            "coal_thickness": float(composited["thickness_vertical"]),
        }
        row["gross_thickness"] = row["depth_to"] - row["depth_from"]
        row["parting_thickness"] = row["gross_thickness"] - row["coal_thickness"]
        for key, value in composited.items():
            if key not in ("thickness_vertical", "mass_weight"):
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def seam_points(collapsed: pd.DataFrame, collar: pd.DataFrame) -> pd.DataFrame:
    """Ubah kedalaman menjadi RL memakai collar.

    Untuk lubang vertikal, RL roof/floor benar tanpa perlu tahu dip: ia hanya
    RL collar dikurangi kedalaman. Karena itu grid struktur dapat dibangun
    lebih dulu, lalu dip diturunkan darinya - satu arah, tanpa iterasi.
    """
    merged = collapsed.merge(
        collar[["hole_id", "east", "north", "rl"]], on="hole_id", how="inner"
    )
    merged["roof_rl"] = merged["rl"] - merged["depth_from"]
    merged["floor_rl"] = merged["rl"] - merged["depth_to"]
    merged["thickness_vertical"] = merged["roof_rl"] - merged["floor_rl"]
    return merged


def build_stratmodel(
    collapsed: pd.DataFrame,
    collar: pd.DataFrame,
    topo: np.ndarray,
    grid: Grid,
    order: list[str],
    cfg,
    quality_attributes: list[str] | None = None,
) -> StratModel:
    """Bangun model seluruh seam dan tegakkan konsistensi stratigrafi."""
    points = seam_points(collapsed, collar)
    method = cfg["grid.method"]
    kwargs = dict(
        method=method,
        power=cfg["grid.idw_power"],
        max_points=cfg["grid.idw_max_points"],
        search_radius=cfg["grid.idw_search_radius_m"],
        max_extrapolation=cfg["grid.max_extrapolation_m"],
    )

    if quality_attributes is None:
        quality_attributes = [
            c for c in ("ash_adb", "vm_adb", "fc_adb", "ts_adb", "cv_adb",
                        "im_adb", "tm_ar", "ard_insitu")
            if c in points.columns
        ]

    present_order = [s for s in order if s in set(points["seam"])]
    if not present_order:  # skema stratigrafi kosong: pakai urutan kedalaman rerata
        present_order = (
            points.groupby("seam")["depth_from"].mean().sort_values().index.tolist()
        )

    model = StratModel(grid=grid, topo=topo, order=present_order, hole_points=points)

    previous_floor: np.ndarray | None = None
    for name in present_order:
        sub = points[points["seam"] == name]

        floor_rl = interpolate_to_grid(grid, sub["east"], sub["north"], sub["floor_rl"], **kwargs)
        isopach = interpolate_to_grid(
            grid, sub["east"], sub["north"], sub["thickness_vertical"], **kwargs
        )
        isopach = np.where(np.isfinite(isopach), np.maximum(isopach, 0.0), np.nan)
        roof_rl = floor_rl + isopach

        # Konsistensi stratigrafi: roof seam ini tidak boleh menembus floor
        # seam di atasnya. Bila terjadi, seluruh amplop diturunkan.
        if previous_floor is not None:
            overlap = np.isfinite(roof_rl) & np.isfinite(previous_floor) & (roof_rl > previous_floor)
            if overlap.any():
                shift = np.where(overlap, roof_rl - previous_floor, 0.0)
                roof_rl = roof_rl - shift
                floor_rl = floor_rl - shift

        # Pemotongan oleh topografi menghasilkan subcrop.
        eroded = np.isfinite(floor_rl) & (floor_rl >= topo)
        partial = np.isfinite(roof_rl) & (roof_rl > topo) & ~eroded
        roof_rl = np.where(partial, topo, roof_rl)
        thickness = np.where(eroded, 0.0, roof_rl - floor_rl)
        thickness = np.where(np.isfinite(thickness), np.maximum(thickness, 0.0), np.nan)

        dip = surface_dip_degrees(np.where(np.isfinite(floor_rl), floor_rl, np.nan),
                                  grid.cell_size)
        thickness_true = thickness * np.cos(np.radians(dip))

        quality: dict[str, np.ndarray] = {}
        for attr in quality_attributes:
            values = sub[["east", "north", attr]].dropna() if attr in sub else None
            if values is None or len(values) == 0:
                continue
            quality[attr] = interpolate_to_grid(
                grid, values["east"], values["north"], values[attr], **kwargs
            )

        model.seams[name] = SeamModel(
            name=name,
            floor_rl=floor_rl,
            roof_rl=roof_rl,
            thickness_vertical=thickness,
            thickness_true=thickness_true,
            dip_degrees=dip,
            depth_to_roof=topo - roof_rl,
            quality=quality,
            n_holes=len(sub),
            subcropped_cells=int(partial.sum()),
            eroded_cells=int(eroded.sum()),
        )
        previous_floor = floor_rl

    return model


def interburden(model: StratModel, upper: str, lower: str) -> np.ndarray:
    """Ketebalan interburden antara floor seam atas dan roof seam bawah."""
    return model.seam(upper).floor_rl - model.seam(lower).roof_rl


def structural_residuals(model: StratModel, seam_name: str) -> pd.DataFrame:
    """Selisih RL floor hasil model terhadap data di titik bor.

    Residual besar yang mengelompok secara spasial adalah petunjuk sesar atau
    salah korelasi - dua hal yang tidak terlihat dari kehalusan grid.
    """
    points = model.hole_points
    sub = points[points["seam"] == seam_name].copy()
    surface = model.seam(seam_name).floor_rl
    grid = model.grid

    col = np.clip(((sub["east"] - grid.xmin) / grid.cell_size).round().astype(int), 0, grid.ncols - 1)
    row = np.clip(((sub["north"] - grid.ymin) / grid.cell_size).round().astype(int), 0, grid.nrows - 1)
    sub["model_floor_rl"] = surface[row, col]
    sub["residual"] = sub["floor_rl"] - sub["model_floor_rl"]
    return sub[["hole_id", "seam", "east", "north", "floor_rl", "model_floor_rl", "residual"]]
