"""Perhitungan volume, tonase, dan kualitas rata-rata terbobot.

Rumus tonase:
    Volume (m3) = luas sel dalam peta x ketebalan vertikal
    Tonase (t)  = Volume x ARD IN-SITU

ARD in-situ, bukan ARD lab. Untuk batubara peringkat rendah dengan TM jauh di
atas IM, memakai ARD lab melebihkan tonase sekitar 10%. Lihat quality.py.

Basis moisture tonase harus dinyatakan dan konsisten dengan basis kualitas
yang dilaporkan. Karena ARD in-situ dipakai, tonase di sini adalah tonase
in-situ pada total moisture, dan pasangannya adalah kualitas basis ar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import CLASS_NAMES, UNCLASSIFIED
from .grid import Grid
from .model import SeamModel, StratModel
from .quality import adb_to_ar


def apply_cutoffs(seam: SeamModel, cfg) -> tuple[np.ndarray, dict[str, int]]:
    """Terapkan cutoff RPEEE dan cutoff penambangan.

    Cutoff ketebalan dikenakan pada TRUE thickness, karena itu yang menentukan
    apakah seam layak ditambang. Ketebalan vertikal selalu lebih besar, jadi
    memakainya akan meloloskan seam yang sebenarnya terlalu tipis.
    """
    mask = seam.present.copy()
    rejected: dict[str, int] = {}

    min_t = cfg.get("seam_rules.min_seam_thickness_m")
    if min_t is not None:
        thin = mask & (seam.thickness_true < float(min_t))
        rejected["below_min_thickness"] = int(thin.sum())
        mask &= ~thin

    max_d = cfg.get("seam_rules.max_depth_m")
    if max_d is not None:
        deep = mask & np.isfinite(seam.depth_to_roof) & (seam.depth_to_roof > float(max_d))
        rejected["beyond_max_depth"] = int(deep.sum())
        mask &= ~deep

    return mask, rejected


def stripping_ratio(model: StratModel, seam_name: str, cfg) -> np.ndarray:
    """SR sederhana untuk satu seam: overburden bcm per ton batubara.

    SR adalah rasio geometri; ia tidak butuh data kualitas sama sekali dan
    dapat dihitung jauh sebelum sumberdaya dapat diklasifikasikan.
    """
    seam = model.seam(seam_name)
    ard = seam.quality.get("ard_insitu")
    if ard is None:
        return np.full(model.grid.shape, np.nan)
    overburden_volume = seam.depth_to_roof * model.grid.cell_area
    coal_tonnes = seam.thickness_vertical * model.grid.cell_area * ard
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(coal_tonnes > 0, overburden_volume / coal_tonnes, np.nan)


def _weighted(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(values[ok] * weights[ok]) / np.sum(weights[ok]))


def summarise_seam(
    model: StratModel,
    seam_name: str,
    class_grid: np.ndarray,
    cfg,
) -> pd.DataFrame:
    """Tabel sumberdaya satu seam, dipecah menurut kelas."""
    seam = model.seam(seam_name)
    grid: Grid = model.grid
    cell_area = grid.cell_area

    mask, _ = apply_cutoffs(seam, cfg)

    ard = seam.quality.get("ard_insitu")
    if ard is None:
        fallback = cfg.get("density.fixed_insitu_ard")
        ard = np.full(grid.shape, float(fallback) if fallback else np.nan)

    volume = np.where(mask, seam.thickness_vertical * cell_area, 0.0)
    tonnes = np.where(mask & np.isfinite(ard), volume * ard, 0.0)

    rows = []
    for level, label in CLASS_NAMES.items():
        if level == UNCLASSIFIED:
            continue
        sel = mask & (class_grid == level)
        if not sel.any():
            continue
        w = tonnes * sel
        row = {
            "seam": seam_name,
            "class": label,
            "cells": int(sel.sum()),
            "area_ha": float(sel.sum() * cell_area / 10_000.0),
            "thickness_mean_m": _weighted(seam.thickness_true, sel * cell_area),
            "volume_m3": float(volume[sel].sum()),
            "tonnes": float(tonnes[sel].sum()),
            "ard_insitu": _weighted(ard, sel * cell_area),
            "depth_mean_m": _weighted(seam.depth_to_roof, w),
        }
        for attr in ("ash_adb", "vm_adb", "fc_adb", "ts_adb", "cv_adb", "im_adb", "tm_ar"):
            if attr in seam.quality:
                row[attr] = _weighted(seam.quality[attr], w)

        # CV(ar) dihitung ulang dari grid, bukan diinterpolasi langsung:
        # interpolasi CV(ar) mencampur variasi kualitas dengan variasi moisture.
        if {"cv_adb", "tm_ar", "im_adb"} <= set(seam.quality):
            cv_ar_grid = adb_to_ar(
                seam.quality["cv_adb"], seam.quality["tm_ar"], seam.quality["im_adb"]
            )
            row["cv_ar"] = _weighted(cv_ar_grid, w)
        if {"ash_adb", "tm_ar", "im_adb"} <= set(seam.quality):
            ash_ar_grid = adb_to_ar(
                seam.quality["ash_adb"], seam.quality["tm_ar"], seam.quality["im_adb"]
            )
            row["ash_ar"] = _weighted(ash_ar_grid, w)
        rows.append(row)

    return pd.DataFrame(rows)


def summarise_model(model: StratModel, class_grids: dict[str, np.ndarray], cfg) -> pd.DataFrame:
    frames = [
        summarise_seam(model, name, class_grids[name], cfg)
        for name in model.order
        if name in class_grids
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def totals_by_class(summary: pd.DataFrame) -> pd.DataFrame:
    """Total lintas seam per kelas, dengan kualitas terbobot tonase."""
    if summary.empty:
        return summary
    rows = []
    for label, group in summary.groupby("class", sort=False):
        w = group["tonnes"].to_numpy(float)
        row = {
            "class": label,
            "seams": group["seam"].nunique(),
            "area_ha": float(group["area_ha"].sum()),
            "volume_m3": float(group["volume_m3"].sum()),
            "tonnes": float(group["tonnes"].sum()),
        }
        for attr in ("thickness_mean_m", "ard_insitu", "ash_adb", "ts_adb",
                     "cv_adb", "cv_ar", "ash_ar", "tm_ar"):
            if attr in group:
                row[attr] = _weighted(group[attr].to_numpy(float), w)
        rows.append(row)
    order = {"Terukur": 0, "Tertunjuk": 1, "Tereka": 2}
    return (
        pd.DataFrame(rows)
        .sort_values("class", key=lambda s: s.map(order).fillna(9))
        .reset_index(drop=True)
    )
