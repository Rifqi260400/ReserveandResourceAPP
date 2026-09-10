"""Klasifikasi sumberdaya.

PERINGATAN YANG HARUS IKUT KE MANA PUN ANGKA INI DIBAWA:

KCMI tidak memuat tabel radius. Ia kode PELAPORAN berbasis prinsip, satu
keluarga dengan JORC, dan menuntut klasifikasi dijustifikasi serta diungkapkan
dasarnya oleh Competent Person. Angka jarak yang lazim dipakai berasal dari
SNI 5015, yang berstatus pedoman - dan harus diverifikasi ke dokumen standar
versi terkini sebelum dipakai untuk pelaporan.

Keluaran modul ini adalah TITIK AWAL untuk penilaian Competent Person, bukan
klasifikasi final. Ia tidak dapat menilai kualitas korelasi, kerapatan
struktur, atau kecukupan QAQC - dan ketiganya adalah bagian dari klasifikasi.

Dua pengaman ditegakkan di sini karena keduanya kerap dilanggar oleh
implementasi berbasis buffer:

1. KRITERIA SPASI, BUKAN BUFFER. Sel hanya naik kelas bila ada CUKUP BANYAK
   titik dalam radius, bukan sekadar satu titik terdekat. Satu lubang
   terisolasi tidak membuktikan kontinuitas apa pun.

2. KLASIFIKASI PER SEAM. Seam tipis yang sulit dikorelasi tidak boleh
   mewarisi kelas dari seam utama di lubang yang sama.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import Grid, count_within_radius, distance_to_data

MEASURED, INDICATED, INFERRED, UNCLASSIFIED = 3, 2, 1, 0
CLASS_NAMES = {
    MEASURED: "Terukur",
    INDICATED: "Tertunjuk",
    INFERRED: "Tereka",
    UNCLASSIFIED: "Tidak Terklasifikasi",
}


def classify_seam(
    grid: Grid,
    points: pd.DataFrame,
    cfg,
    present_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grid kelas sumberdaya untuk satu seam.

    ``points`` adalah lubang yang MENEMBUS seam ini, dengan kolom east, north,
    dan (opsional) has_quality.
    """
    condition = cfg["classification.geological_condition"]
    distances = cfg[f"classification.distances.{condition}"]
    min_points = cfg["classification.min_points"]

    result = np.full(grid.shape, UNCLASSIFIED, dtype=int)
    if len(points) == 0:
        return result

    x_all = points["east"].to_numpy(float)
    y_all = points["north"].to_numpy(float)

    if "has_quality" in points.columns:
        q = points[points["has_quality"].astype(bool)]
    else:
        q = points
    x_q = q["east"].to_numpy(float)
    y_q = q["north"].to_numpy(float)

    require_q = {
        MEASURED: bool(cfg["classification.require_quality_for_measured"]),
        INDICATED: bool(cfg["classification.require_quality_for_indicated"]),
        INFERRED: False,
    }
    radius = {
        MEASURED: float(distances["measured"]),
        INDICATED: float(distances["indicated"]),
        INFERRED: float(distances["inferred"]),
    }
    needed = {
        MEASURED: int(min_points["measured"]),
        INDICATED: int(min_points["indicated"]),
        INFERRED: int(min_points["inferred"]),
    }

    # Kelas terendah dulu, lalu ditimpa kelas yang lebih tinggi.
    for level in (INFERRED, INDICATED, MEASURED):
        if require_q[level]:
            if len(x_q) == 0:
                continue
            xs, ys = x_q, y_q
        else:
            xs, ys = x_all, y_all

        counts = count_within_radius(grid, xs, ys, radius[level])
        qualifies = counts >= needed[level]
        result = np.where(qualifies, level, result)

    if present_mask is not None:
        result = np.where(present_mask, result, UNCLASSIFIED)
    return result


def classification_frame(class_grid: np.ndarray, grid: Grid) -> pd.DataFrame:
    """Ringkasan luas per kelas."""
    rows = []
    for level, name in CLASS_NAMES.items():
        cells = int((class_grid == level).sum())
        rows.append(
            {"class_code": level, "class": name, "cells": cells,
             "area_ha": cells * grid.cell_area / 10_000.0}
        )
    return pd.DataFrame(rows).sort_values("class_code", ascending=False).reset_index(drop=True)


def spacing_statistics(points: pd.DataFrame) -> dict:
    """Statistik spasi bor nyata - dasar memilih kondisi geologi dan grid.

    Spasi yang sebenarnya harus menentukan ukuran sel dan menjadi pemeriksa
    kewajaran terhadap kelas yang diklaim. Grid jauh lebih halus daripada
    spasi bor hanya menciptakan ilusi presisi.
    """
    from scipy.spatial import cKDTree

    if len(points) < 2:
        return {"n_points": len(points)}
    coords = points[["east", "north"]].to_numpy(float)
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=2)
    nearest = dist[:, 1]
    return {
        "n_points": len(points),
        "nearest_min": float(nearest.min()),
        "nearest_median": float(np.median(nearest)),
        "nearest_mean": float(nearest.mean()),
        "nearest_max": float(nearest.max()),
        "suggested_cell_size": float(np.median(nearest) / 4.0),
    }
