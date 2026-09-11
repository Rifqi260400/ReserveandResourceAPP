"""Kualitas: rekonstruksi interval, cakupan, dan rata-rata terbobot tonase.

Aturan yang ditegakkan modul ini:

1. Rata-rata dibobot TONASE, bukan jumlah lubang dan bukan tebal saja.
2. TIDAK PERNAH merata-ratakan lintas basis analitik. Kalau satu seam mencampur
   basis, modul berhenti dan melaporkannya.
3. Jumlah sampel di balik setiap angka rata-rata ikut dilaporkan. Rata-rata dari
   dua sampel tidak boleh terlihat sama dengan rata-rata dari empat puluh.
4. Interval yang benar-benar tercakup hasil lab dilaporkan bersama nilainya.
   Komposit tidak boleh disajikan seolah mewakili tebal yang tidak ia sampel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .errors import MissingDataError
from .io.excel import Workbook, normalise_hole_id
from .io.quality_table import QualityTable
from .logging_setup import get_logger
from .seams import SeamIntersection

log = get_logger("quality")

# Atribut yang boleh dirata-rata linear pada basis yang sama.
#
# Nama BERSUFIKS BASIS (ASH_adb) dipakai sumber yang menyatakan basisnya.
# Nama POLOS (ASH) dipakai sumber yang tidak - berkas flat Minex termasuk di
# situ. Keduanya harus dikenali: tanpa nama polos, kualitas terbaca dari berkas
# tetapi tidak pernah sampai ke tabel keluaran, dan laporan tampak seolah tidak
# ada data kualitas sama sekali.
AVERAGEABLE = (
    "TM_ar", "M_adb", "ASH_adb", "VM_adb", "FC_adb", "TS_adb",
    "CV_adb", "CV_ar", "CV_daf",
    "MOISTURE", "ASH", "VM", "FC", "TS", "CV",
)

# Atribut non-additive: merata-ratakannya secara aritmetik salah secara fisik.
NON_ADDITIVE = ("HGI", "AFT", "AFT_ID", "AFT_ST", "AFT_HT", "AFT_FT")

BASIS_SUFFIX = {"_ar": "ar", "_adb": "adb", "_daf": "daf", "_db": "db"}


def attribute_basis(column: str) -> str:
    for suffix, basis in BASIS_SUFFIX.items():
        if column.endswith(suffix):
            return basis
    return "unspecified"


@dataclass
class SeamQuality:
    """Hasil kualitas untuk satu interseksi seam."""

    hole_id: str
    seam: str
    values: dict[str, float]
    n_samples: int
    covered_from_m: float
    covered_to_m: float
    covered_m: float
    coverage_frac: float
    rd_t_per_m3: float
    rd_basis: str
    report_no: str = ""

    @property
    def covered_interval(self) -> str:
        if not np.isfinite(self.covered_from_m):
            return "tidak tercakup"
        return f"{self.covered_from_m:.3f}-{self.covered_to_m:.3f} m"


def sample_intervals(wb: Workbook) -> pd.DataFrame:
    """Interval sampel pada basis ADJUSTED DEPTH, bukan drilling depth."""
    table = wb.sheets.get("Sampling")
    if table is None or table.frame.empty:
        return pd.DataFrame(columns=["sample_number", "adjusted_from", "adjusted_to"])
    frame = table.frame.copy()
    for column in ("adjusted_from", "adjusted_to"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["sample_number"] = frame["sample_number"].astype(str).str.strip()
    frame["hole_id"] = normalise_hole_id(wb.hole_id)
    return frame[np.isfinite(frame["adjusted_from"]) & np.isfinite(frame["adjusted_to"])]


def resolve_quality(
    intersections: list[SeamIntersection],
    quality: QualityTable,
    workbooks: dict[str, Workbook],
    cfg: Config,
) -> tuple[list[SeamQuality], pd.DataFrame]:
    """Padankan hasil lab ke interseksi seam dan hitung cakupannya.

    Interval yang diwakili sebuah hasil direkonstruksi lewat `composite_of`
    yang di-join balik ke sheet `Sampling` - bukan diasumsikan sama dengan
    amplop seam.
    """
    frame = quality.frame.copy()
    frame["hole_key"] = frame["hole_id"].map(normalise_hole_id)
    frame["seam_key"] = frame["seam"].astype(str).str.strip()

    results: list[SeamQuality] = []
    issues: list[dict] = []

    for item in intersections:
        wb = workbooks.get(item.hole_id)
        samples = sample_intervals(wb) if wb is not None else pd.DataFrame()
        base_seam = item.split_from or item.seam
        matched = frame[(frame["hole_key"] == item.hole_id)
                        & (frame["seam_key"] == base_seam)]

        if matched.empty:
            issues.append({"hole_id": item.hole_id, "seam": item.seam,
                           "issue": "tidak ada hasil kualitas", "detail": ""})
            continue

        bases = sorted({str(b).strip() for b in matched["RD_basis"].dropna()})
        if len(bases) > 1:
            raise MissingDataError(
                f"{item.hole_id}/{item.seam}: hasil kualitas mencampur RD_basis "
                f"{bases}. Nilai lintas basis tidak boleh dirata-ratakan."
            )
        rd_basis = bases[0] if bases else "unknown"

        members: list[str] = []
        for sample_id in matched["sample_id"].astype(str):
            members.extend(quality.composite_members(sample_id))

        covered_from, covered_to, covered = np.nan, np.nan, 0.0
        if members and not samples.empty:
            hit = samples[samples["sample_number"].isin(members)]
            if not hit.empty:
                covered_from = float(hit["adjusted_from"].min())
                covered_to = float(hit["adjusted_to"].max())
                covered = float((hit["adjusted_to"] - hit["adjusted_from"]).clip(lower=0).sum())
        if not members:
            issues.append({"hole_id": item.hole_id, "seam": item.seam,
                           "issue": "composite_of kosong", "detail": ""})

        thickness = item.coal_thickness_m
        fraction = covered / thickness if thickness > 0 else np.nan
        if np.isfinite(fraction) and fraction < cfg.cutoffs.min_quality_coverage_frac:
            issues.append({
                "hole_id": item.hole_id, "seam": item.seam,
                "issue": "cakupan kualitas di bawah ambang",
                "detail": f"{covered:.3f} m dari {thickness:.3f} m ({fraction:.1%})",
            })

        values: dict[str, float] = {}
        for column in AVERAGEABLE:
            if column not in matched:
                continue
            series = pd.to_numeric(matched[column], errors="coerce").dropna()
            if series.empty:
                continue
            # Beberapa hasil untuk satu seam: dibobot panjang komposit masing-masing.
            weights = []
            for sample_id in matched.loc[series.index, "sample_id"].astype(str):
                ids = quality.composite_members(sample_id)
                span = samples[samples["sample_number"].isin(ids)] if not samples.empty else pd.DataFrame()
                weights.append(float((span["adjusted_to"] - span["adjusted_from"]).sum())
                               if not span.empty else 1.0)
            weight = np.asarray(weights, float)
            values[column] = float(np.average(series.to_numpy(float), weights=weight)) \
                if weight.sum() > 0 else float(series.mean())

        for column in NON_ADDITIVE:
            if column in matched:
                series = pd.to_numeric(matched[column], errors="coerce").dropna()
                if not series.empty:
                    # Ditandai secara eksplisit: atribut ini tidak linear.
                    values[f"{column}_mean_nonadditive"] = float(series.mean())

        rd_series = pd.to_numeric(matched.get("RD"), errors="coerce").dropna()
        results.append(SeamQuality(
            hole_id=item.hole_id, seam=item.seam, values=values,
            n_samples=len(matched), covered_from_m=covered_from, covered_to_m=covered_to,
            covered_m=covered, coverage_frac=fraction,
            rd_t_per_m3=float(rd_series.mean()) if not rd_series.empty else np.nan,
            rd_basis=rd_basis,
            report_no=str(matched.iloc[0].get("report_no", "")),
        ))

    return results, pd.DataFrame(issues)


def weighted_average(frame: pd.DataFrame, column: str, weight_column: str = "tonnes") -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return float("nan")
    return float(np.average(values[ok].to_numpy(float), weights=weights[ok].to_numpy(float)))


def summarise(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Rata-rata kualitas terbobot tonase, per kelompok, dengan jumlah sampel.

    Kolom yang basisnya berbeda tidak pernah digabung: nama kolom membawa
    basisnya, dan setiap kolom dirata-rata sendiri.
    """
    if frame.empty:
        return pd.DataFrame()

    attributes = [c for c in frame.columns if c in AVERAGEABLE]
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["polygons"] = len(group)
        row["area_ha"] = float(group["area_ha"].sum())
        row["tonnes"] = float(group["tonnes"].sum())
        row["coal_thickness_m"] = weighted_average(group, "coal_thickness_m")
        row["rd_t_per_m3"] = weighted_average(group, "rd_t_per_m3")
        row["rd_assumed"] = bool(group["rd_is_assumed"].any())
        row["holes"] = int(group["hole_id"].nunique())
        for attribute in attributes:
            row[attribute] = weighted_average(group, attribute)
            row[f"{attribute}_n"] = int(pd.to_numeric(group[attribute], errors="coerce").notna().sum())
        row["quality_coverage_frac"] = weighted_average(group, "quality_coverage_frac")
        rows.append(row)
    return pd.DataFrame(rows)


def basis_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Daftar basis analitik yang hadir, untuk dicantumkan di laporan."""
    attributes = [c for c in frame.columns if c in AVERAGEABLE]
    return pd.DataFrame([
        {"attribute": a, "basis": attribute_basis(a),
         "n_values": int(pd.to_numeric(frame[a], errors="coerce").notna().sum())}
        for a in attributes
    ])


def resolve_quality_from_plies(
    intersections: list[SeamIntersection],
    plies: pd.DataFrame,
    cfg: Config,
    rd_basis: str,
    rd_column: str = "RD",
) -> tuple[list[SeamQuality], pd.DataFrame]:
    """Compositing kualitas dari hasil PER PLY (jalur flat file Minex).

    Lebih baik daripada jalur komposit lab: interval yang diwakili tiap nilai
    diketahui persis, sehingga cakupan dapat dihitung dan pembobotan memakai
    MASSA (panjang x RD), bukan panjang saja. Pembobotan panjang saja bias untuk
    seam yang RD antar-ply-nya berbeda jauh - persis kasus ply berash tinggi.
    """
    frame = plies.copy()
    frame["length"] = frame["depth_to"] - frame["depth_from"]
    # Interval terbalik tidak boleh diam-diam menjadi bobot negatif.
    invalid = frame[frame["length"] <= 0]
    frame = frame[frame["length"] > 0]

    results: list[SeamQuality] = []
    issues: list[dict] = []
    for _, row in invalid.iterrows():
        issues.append({
            "hole_id": row["hole_id"], "seam": row["seam"],
            "issue": "interval kualitas terbalik atau nol",
            "detail": f"from={row['depth_from']} to={row['depth_to']}",
        })

    attributes = [c for c in frame.columns if c in AVERAGEABLE]
    for item in intersections:
        base_seam = item.split_from or item.seam
        matched = frame[(frame["hole_id"] == item.hole_id)
                        & (frame["seam"].astype(str).str.strip() == base_seam)]
        if matched.empty:
            issues.append({"hole_id": item.hole_id, "seam": item.seam,
                           "issue": "tidak ada hasil kualitas", "detail": ""})
            continue

        length = matched["length"].to_numpy(float)
        density = pd.to_numeric(matched.get(rd_column), errors="coerce").to_numpy(float)
        mass = length * np.where(np.isfinite(density), density, 1.0)

        values: dict[str, float] = {}
        for attribute in attributes:
            series = pd.to_numeric(matched[attribute], errors="coerce").to_numpy(float)
            ok = np.isfinite(series) & np.isfinite(mass) & (mass > 0)
            if ok.any():
                values[attribute] = float(np.average(series[ok], weights=mass[ok]))

        # RD dibobot VOLUME (panjang), bukan massa: membobot densitas dengan
        # massa yang dihitung dari densitas itu sendiri akan melebihkan ply padat.
        rd_ok = np.isfinite(density) & (length > 0)
        rd_value = float(np.average(density[rd_ok], weights=length[rd_ok])) if rd_ok.any() else np.nan

        covered = float(length.sum())
        fraction = covered / item.coal_thickness_m if item.coal_thickness_m > 0 else np.nan
        if np.isfinite(fraction) and fraction < cfg.cutoffs.min_quality_coverage_frac:
            issues.append({
                "hole_id": item.hole_id, "seam": item.seam,
                "issue": "cakupan kualitas di bawah ambang",
                "detail": f"{covered:.3f} m dari {item.coal_thickness_m:.3f} m ({fraction:.1%})",
            })

        results.append(SeamQuality(
            hole_id=item.hole_id, seam=item.seam, values=values,
            n_samples=len(matched),
            covered_from_m=float(matched["depth_from"].min()),
            covered_to_m=float(matched["depth_to"].max()),
            covered_m=covered, coverage_frac=fraction,
            rd_t_per_m3=rd_value, rd_basis=rd_basis,
        ))
    return results, pd.DataFrame(issues)
