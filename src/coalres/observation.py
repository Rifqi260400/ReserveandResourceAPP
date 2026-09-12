"""Tahap 6: titik observasi.

Klasifikasi bersandar pada titik observasi, bukan pada lubang bor. Bedanya
menggerakkan kelas, bukan sekadar penamaan: bila kualitas tidak dituntut,
populasi membengkak dan area naik kelas atas dasar geometri semata.

Definisinya diputuskan manusia di `observation_point.requires_quality` dan
gerbang G6 tidak akan lulus selama belum diputuskan. Modul ini hanya
MENERAPKAN keputusan itu, per seam, beserta alasan tiap lubang yang gugur.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .logging_setup import get_logger

log = get_logger("observation")

# Alasan gugur, dipakai apa adanya pada keluaran.
NO_INTERSECTION = "tidak menembus seam"
BELOW_CUTOFF = "tebal di bawah cutoff"
NO_QUALITY = "tidak ada data kualitas untuk seam ini"
THIN_QUALITY = "cakupan kualitas di bawah ambang"


@dataclass
class ObservationPoints:
    """Titik observasi per seam, beserta lubang yang gugur dan alasannya."""

    frame: pd.DataFrame          # hole_id, seam, east, north, qualifies, reason
    requires_quality: bool
    basis: str

    def for_seam(self, seam: str) -> pd.DataFrame:
        sub = self.frame[(self.frame["seam"] == seam) & self.frame["qualifies"]]
        return sub[["hole_id", "east", "north"]].reset_index(drop=True)

    @property
    def qualifying(self) -> pd.DataFrame:
        return self.frame[self.frame["qualifies"]]

    def summary(self) -> pd.DataFrame:
        rows = []
        for seam, group in self.frame.groupby("seam"):
            reasons = group[~group["qualifies"]]["reason"].value_counts().to_dict()
            rows.append({
                "seam": seam,
                "menembus_seam": int(len(group)),
                "titik_observasi": int(group["qualifies"].sum()),
                "gugur": int((~group["qualifies"]).sum()),
                **{f"gugur: {k}": v for k, v in reasons.items()},
            })
        return pd.DataFrame(rows).fillna(0)


def _quality_coverage(quality: pd.DataFrame | None, hole: str, seam: str,
                      roof_m: float, floor_m: float) -> float:
    """Fraksi tebal seam yang tertutup interval kualitas yang sah.

    Diukur terhadap tebal seam, bukan terhadap jumlah sampel: sepuluh sampel
    pendek di satu ujung seam tebal bukan cakupan, meski jumlahnya banyak.
    """
    if quality is None or not np.isfinite(roof_m) or floor_m <= roof_m:
        return 0.0
    rows = quality[(quality["hole_id"] == hole) & (quality["seam"] == seam)]
    if rows.empty:
        return 0.0
    covered = 0.0
    for _, row in rows.iterrows():
        top, base = float(row["depth_from"]), float(row["depth_to"])
        if not (np.isfinite(top) and np.isfinite(base)) or base <= top:
            continue
        covered += max(0.0, min(base, floor_m) - max(top, roof_m))
    return covered / (floor_m - roof_m)


def build(intersections: pd.DataFrame, dataset, cfg: Config) -> ObservationPoints:
    """Tentukan titik observasi per seam dari interseksi seam yang sudah dibangun."""
    spec = cfg.observation_point
    if spec.requires_quality is None:
        raise ValueError(
            "observation_point.requires_quality belum diputuskan. Modul ini "
            "menerapkan keputusan, tidak membuatnya - lihat gerbang G6."
        )

    collars = dataset.collars.set_index("hole_id")
    minimum = cfg.cutoffs.min_seam_thickness_m
    threshold = cfg.cutoffs.min_quality_coverage_frac
    rows = []

    for _, row in intersections.iterrows():
        hole, seam = str(row["hole_id"]), str(row["seam"])
        thickness = float(row.get("coal_thickness_m", float("nan")))
        qualifies, reason, coverage = True, "", float("nan")

        if not np.isfinite(thickness) or thickness <= 0:
            qualifies, reason = False, NO_INTERSECTION
        elif thickness < minimum:
            qualifies, reason = False, BELOW_CUTOFF
        elif spec.requires_quality:
            coverage = _quality_coverage(dataset.quality, hole, seam,
                                         float(row["roof_m"]), float(row["floor_m"]))
            if coverage <= 0.0:
                qualifies, reason = False, NO_QUALITY
            elif coverage < threshold:
                qualifies, reason = False, THIN_QUALITY

        collar = collars.loc[hole] if hole in collars.index else None
        rows.append({
            "hole_id": hole, "seam": seam,
            "east": float(collar["east"]) if collar is not None else float("nan"),
            "north": float(collar["north"]) if collar is not None else float("nan"),
            "coal_thickness_m": thickness,
            "quality_coverage_frac": coverage,
            "qualifies": qualifies, "reason": reason,
        })

    frame = pd.DataFrame(rows)
    points = ObservationPoints(frame=frame, requires_quality=spec.requires_quality,
                               basis=spec.basis)
    total = int(frame["qualifies"].sum())
    log.info(
        f"titik observasi: {total} dari {len(frame)} interseksi seam lolos "
        f"(kualitas {'DITUNTUT' if spec.requires_quality else 'tidak dituntut'})"
    )
    return points


def compare_populations(intersections: pd.DataFrame, dataset, cfg: Config
                        ) -> pd.DataFrame:
    """Bandingkan kedua definisi titik observasi berdampingan.

    Dicetak pada keluaran apa pun pilihannya, supaya pembaca melihat berapa
    yang dikorbankan - atau diperoleh - oleh definisi yang dipakai.
    """
    frames = {}
    for requires in (False, True):
        variant = cfg.model_copy(update={
            "observation_point": cfg.observation_point.model_copy(
                update={"requires_quality": requires,
                        "basis": cfg.observation_point.basis or "pembanding"})})
        result = build(intersections, dataset, variant)
        counts = result.qualifying.groupby("seam")["hole_id"].nunique()
        frames["menuntut kualitas" if requires else "cukup ketebalan"] = counts

    out = pd.DataFrame(frames).fillna(0).astype(int)
    out["selisih"] = out["menuntut kualitas"] - out["cukup ketebalan"]
    out["dipakai"] = ("menuntut kualitas" if cfg.observation_point.requires_quality
                      else "cukup ketebalan")
    return out.reset_index()
