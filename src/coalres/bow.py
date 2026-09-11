"""Base of Weathering (BOW) dan hubungannya dengan tiap seam.

Berkas `lit` Minex memuat baris penanda berketebalan nol (kode 'W' pada dataset
ini) yang menandai BATAS BAWAH PELAPUKAN. Batubara di atas horizon itu sudah
lapuk: nilai kalorinya turun, moisture-nya naik, dan pada umumnya ia dikeluarkan
dari sumberdaya.

Modul ini menyandingkan BOW dengan tiap interseksi seam dan melaporkan berapa
bagian seam yang berada di atasnya. BOW adalah SATU horizon, bukan satu per
seam - yang dilaporkan per seam adalah posisi seam itu terhadap BOW.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .logging_setup import get_logger

log = get_logger("bow")

ABOVE = "seluruhnya lapuk"
BELOW = "seluruhnya segar"
STRADDLE = "terpotong BOW"
NO_BOW = "BOW tidak tercatat"


@dataclass
class BowRecord:
    hole_id: str
    bow_depth_m: float
    bow_rl_m: float
    collar_rl_m: float


def extract_bow(dataset, cfg) -> pd.DataFrame:
    """Kedalaman dan RL base of weathering per lubang."""
    intervals = dataset.intervals
    if "is_marker" not in intervals:
        return pd.DataFrame(columns=["hole_id", "bow_depth_m", "bow_rl_m", "collar_rl_m"])

    markers = intervals[intervals["is_marker"]].copy()
    if markers.empty:
        return pd.DataFrame(columns=["hole_id", "bow_depth_m", "bow_rl_m", "collar_rl_m"])

    # Penanda berketebalan nol: from dan to sama. Yang dipakai adalah 'to',
    # yaitu kedalaman batas bawah pelapukan.
    markers["bow_depth_m"] = markers[["depth_from", "depth_to"]].max(axis=1)
    collars = dataset.collars.set_index("hole_id")
    rows = []
    for hole, group in markers.groupby("hole_id"):
        if hole not in collars.index:
            continue
        collar_rl = float(collars.loc[hole, "rl"])
        depth = float(group["bow_depth_m"].max())
        rows.append({"hole_id": hole, "bow_depth_m": depth,
                     "bow_rl_m": collar_rl - depth, "collar_rl_m": collar_rl})
    return pd.DataFrame(rows).sort_values("hole_id").reset_index(drop=True)


def seam_vs_bow(intercepts: pd.DataFrame, bow: pd.DataFrame) -> pd.DataFrame:
    """Posisi tiap interseksi seam terhadap BOW, beserta tebal yang lapuk."""
    if intercepts.empty:
        return pd.DataFrame()

    lookup = bow.set_index("hole_id")["bow_depth_m"].to_dict() if not bow.empty else {}
    rows = []
    for _, row in intercepts.iterrows():
        depth = lookup.get(row["hole_id"], np.nan)
        roof, floor = float(row["roof_m"]), float(row["floor_m"])
        thickness = max(floor - roof, 0.0)

        if not np.isfinite(depth):
            status, weathered = NO_BOW, np.nan
        elif floor <= depth:
            status, weathered = ABOVE, thickness
        elif roof >= depth:
            status, weathered = BELOW, 0.0
        else:
            status, weathered = STRADDLE, depth - roof

        rows.append({
            "hole_id": row["hole_id"], "seam": row["seam"],
            "collar_rl_m": row.get("collar_rl_m"),
            "bow_depth_m": depth,
            "bow_rl_m": (row.get("collar_rl_m") - depth
                         if np.isfinite(depth) and pd.notna(row.get("collar_rl_m")) else np.nan),
            "seam_roof_m": roof, "seam_floor_m": floor,
            "seam_roof_rl_m": row.get("roof_rl_m"), "seam_floor_rl_m": row.get("floor_rl_m"),
            "coal_thickness_m": float(row.get("coal_thickness_m", thickness)),
            "weathered_thickness_m": weathered,
            "fresh_thickness_m": (float(row.get("coal_thickness_m", thickness)) - weathered
                                  if np.isfinite(weathered) else np.nan),
            "cover_below_bow_m": roof - depth if np.isfinite(depth) else np.nan,
            "status": status,
        })
    return pd.DataFrame(rows)


def summarise_by_seam(detail: pd.DataFrame) -> pd.DataFrame:
    """Rekap per seam: berapa lubang lapuk, terpotong, dan segar."""
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for seam, group in detail.groupby("seam", sort=False):
        counts = group["status"].value_counts()
        rows.append({
            "seam": seam,
            "n_intersections": len(group),
            "bow_depth_min_m": group["bow_depth_m"].min(),
            "bow_depth_median_m": group["bow_depth_m"].median(),
            "bow_depth_max_m": group["bow_depth_m"].max(),
            "bow_rl_median_m": group["bow_rl_m"].median(),
            "n_seluruhnya_lapuk": int(counts.get(ABOVE, 0)),
            "n_terpotong_bow": int(counts.get(STRADDLE, 0)),
            "n_seluruhnya_segar": int(counts.get(BELOW, 0)),
            "n_tanpa_bow": int(counts.get(NO_BOW, 0)),
            "tebal_lapuk_total_m": group["weathered_thickness_m"].sum(skipna=True),
            "tebal_segar_total_m": group["fresh_thickness_m"].sum(skipna=True),
            "cover_di_bawah_bow_median_m": group["cover_below_bow_m"].median(),
        })
    return pd.DataFrame(rows)


def write_bow_excel(path, bow: pd.DataFrame, detail: pd.DataFrame,
                    summary: pd.DataFrame, cfg) -> "Path":
    """Rekap BOW sebagai workbook Excel."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    notes = pd.DataFrame({"No": range(1, 6), "Catatan": [
        "BOW adalah SATU horizon per lubang, bukan satu per seam. Yang dilaporkan "
        "per seam adalah posisi seam itu terhadap BOW.",
        f"Penanda BOW dibaca dari baris seam {cfg.minex.marker_seams if cfg.minex else '[W]'} "
        "pada berkas litologi; baris ini berketebalan nol dan menandai batas bawah "
        "pelapukan.",
        "Batubara di atas BOW sudah lapuk: CV turun, moisture naik. Apakah ia "
        "dikeluarkan dari sumberdaya adalah keputusan pelaporan, dan tool ini TIDAK "
        "mengeluarkannya secara otomatis.",
        "tebal_lapuk_total_m dan tebal_segar_total_m adalah penjumlahan lintas "
        "lubang, bukan tonase. Keduanya untuk menilai seberapa besar pengaruhnya.",
        "cover_below_bow_m negatif berarti roof seam berada DI ATAS BOW.",
    ]})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Catatan", index=False)
        summary.to_excel(writer, sheet_name="Rekap per Seam", index=False)
        detail.to_excel(writer, sheet_name="Seam vs BOW", index=False)
        bow.to_excel(writer, sheet_name="BOW per Lubang", index=False)
    return path
