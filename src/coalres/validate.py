"""Validasi data lubang bor.

Menghasilkan laporan exception per lubang, bukan exception Python: pipeline
harus tetap berjalan agar seluruh masalah terlihat sekaligus, bukan satu per
satu setiap kali dijalankan ulang.

Severity:
    ERROR   data tidak dapat dipakai apa adanya; perbaiki sebelum modelling
    WARNING mungkin benar, tapi harus dilihat manusia
    INFO    catatan yang perlu didokumentasikan
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .quality import adb_to_ar, adb_to_daf, dry_matter_ard


@dataclass
class Finding:
    severity: str
    code: str
    hole_id: str | None
    seam: str | None
    message: str
    value: float | None = None


class ValidationReport:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, severity, code, message, hole_id=None, seam=None, value=None) -> None:
        self.findings.append(Finding(severity, code, hole_id, seam, message, value))

    def to_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(
                columns=["severity", "code", "hole_id", "seam", "message", "value"]
            )
        df = pd.DataFrame([asdict(f) for f in self.findings])
        order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        return df.sort_values(
            ["severity", "code", "hole_id"], key=lambda s: s.map(order).fillna(9)
            if s.name == "severity" else s
        ).reset_index(drop=True)

    def counts(self) -> dict[str, int]:
        df = self.to_frame()
        return {} if df.empty else df["severity"].value_counts().to_dict()

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "ERROR")

    def summary(self) -> str:
        counts = self.counts()
        if not counts:
            return "Validasi: tidak ada temuan."
        parts = [f"{counts.get(s, 0)} {s}" for s in ("ERROR", "WARNING", "INFO") if counts.get(s)]
        return "Validasi: " + ", ".join(parts)


# --------------------------------------------------------------------------- #
# Cek collar
# --------------------------------------------------------------------------- #
def check_collar(collar: pd.DataFrame, report: ValidationReport, cfg) -> None:
    dupes = collar[collar.duplicated("hole_id", keep=False)]
    for hid in dupes["hole_id"].unique():
        report.add("ERROR", "COLLAR_DUPLICATE", f"hole_id muncul {int((collar['hole_id'] == hid).sum())} kali", hole_id=hid)

    for col in ("east", "north", "rl"):
        missing = collar[collar[col].isna()]
        for hid in missing["hole_id"]:
            report.add("ERROR", "COLLAR_MISSING", f"kolom '{col}' kosong", hole_id=hid)

    # Koordinat yang jauh dari pusat sebaran hampir selalu salah ketik atau
    # tertukar X/Y, bukan lubang yang benar-benar terpencil.
    for col in ("east", "north"):
        values = collar[col].dropna()
        if len(values) < 5:
            continue
        med, spread = values.median(), values.std()
        if spread <= 0:
            continue
        outliers = collar[(collar[col] - med).abs() > 10 * spread]
        for _, row in outliers.iterrows():
            report.add(
                "WARNING", "COLLAR_OUTLIER",
                f"{col}={row[col]:,.1f} jauh dari sebaran (median {med:,.1f})",
                hole_id=row["hole_id"], value=float(row[col]),
            )


def check_collar_vs_topo(
    collar: pd.DataFrame, topo_at_collar: np.ndarray, report: ValidationReport, cfg
) -> pd.DataFrame:
    """Bandingkan RL collar terhadap permukaan topo.

    Ini uji termurah dan paling berdaya di seluruh pipeline. Offset sistematis
    menandakan masalah datum (mis. RL ortometrik vs elipsoidal, atau geoid yang
    tidak diterapkan) yang menggeser SELURUH seam secara seragam - kesalahan
    yang tidak akan tertangkap oleh QC mana pun di hilir karena peta isopach
    tetap terlihat sempurna.
    """
    tol = float(cfg["validation.collar_vs_topo_tolerance_m"])
    residual = collar["rl"].to_numpy(float) - np.asarray(topo_at_collar, float)
    out = collar[["hole_id", "rl"]].copy()
    out["topo_rl"] = topo_at_collar
    out["residual"] = residual

    valid = residual[np.isfinite(residual)]
    if len(valid) >= 5:
        bias = float(np.median(valid))
        if abs(bias) > tol:
            report.add(
                "ERROR", "TOPO_DATUM_BIAS",
                f"median selisih RL collar - topo = {bias:+.2f} m pada {len(valid)} lubang. "
                "Offset sistematis menandakan masalah datum, bukan collar per lubang.",
                value=bias,
            )
    for _, row in out.iterrows():
        r = row["residual"]
        if np.isfinite(r) and abs(r) > tol:
            report.add(
                "WARNING", "COLLAR_VS_TOPO",
                f"RL collar {row['rl']:.2f} vs topo {row['topo_rl']:.2f} (selisih {r:+.2f} m)",
                hole_id=row["hole_id"], value=float(r),
            )
        elif not np.isfinite(r):
            report.add(
                "WARNING", "COLLAR_OUTSIDE_TOPO",
                "collar berada di luar cakupan data topo",
                hole_id=row["hole_id"],
            )
    return out


# --------------------------------------------------------------------------- #
# Cek interval seam
# --------------------------------------------------------------------------- #
def check_intervals(
    collar: pd.DataFrame, seam: pd.DataFrame, report: ValidationReport, cfg
) -> None:
    tol = float(cfg["validation.interval_tolerance_m"])
    td_by_hole = collar.set_index("hole_id")["td"].to_dict() if "td" in collar else {}
    known = set(collar["hole_id"])

    for hid in sorted(set(seam["hole_id"]) - known):
        report.add("ERROR", "SEAM_ORPHAN", "interval seam tanpa collar", hole_id=hid)

    for hid, group in seam.groupby("hole_id"):
        group = group.sort_values("depth_from")

        bad = group[group["depth_to"] <= group["depth_from"]]
        for _, row in bad.iterrows():
            report.add(
                "ERROR", "INTERVAL_INVALID",
                f"from={row['depth_from']} >= to={row['depth_to']}",
                hole_id=hid, seam=row["seam"],
            )

        prev_to, prev_seam = None, None
        for _, row in group.iterrows():
            if prev_to is not None and row["depth_from"] < prev_to - tol:
                report.add(
                    "ERROR", "INTERVAL_OVERLAP",
                    f"seam '{row['seam']}' mulai {row['depth_from']:.2f} "
                    f"sebelum '{prev_seam}' berakhir {prev_to:.2f}",
                    hole_id=hid, seam=row["seam"],
                )
            prev_to, prev_seam = row["depth_to"], row["seam"]

        td = td_by_hole.get(hid)
        if td is not None and np.isfinite(td):
            deeper = group[group["depth_to"] > td + tol]
            for _, row in deeper.iterrows():
                report.add(
                    "ERROR", "INTERVAL_BEYOND_TD",
                    f"floor {row['depth_to']:.2f} melebihi total depth {td:.2f}",
                    hole_id=hid, seam=row["seam"],
                )

        dupe_seams = group["seam"].value_counts()
        for s, n in dupe_seams[dupe_seams > 1].items():
            report.add(
                "INFO", "SEAM_MULTI_PLY",
                f"seam '{s}' muncul {n} interval - akan dikompositkan sebagai ply",
                hole_id=hid, seam=s,
            )


def check_stratigraphic_order(
    seam: pd.DataFrame, order: list[str], report: ValidationReport
) -> None:
    """Pastikan urutan seam per lubang konsisten dengan skema stratigrafi.

    Seam yang urutannya terbalik di satu lubang hampir selalu berarti salah
    korelasi - dan salah korelasi tidak akan pernah terlihat dari statistik
    model, karena grid tetap mulus dan masuk akal.
    """
    if not order:
        report.add(
            "WARNING", "STRAT_UNDEFINED",
            "stratigraphy kosong di konfigurasi; urutan seam tidak dapat divalidasi",
        )
        return

    rank = {s: i for i, s in enumerate(order)}
    unknown = sorted(set(seam["seam"]) - set(rank))
    for s in unknown:
        report.add("ERROR", "SEAM_UNKNOWN", f"seam '{s}' tidak ada di skema stratigrafi", seam=s)

    for hid, group in seam.groupby("hole_id"):
        group = group[group["seam"].isin(rank)].sort_values("depth_from")
        ranks = [rank[s] for s in group["seam"]]
        # Ply berulang boleh; yang tidak boleh adalah urutan yang mundur.
        for i in range(1, len(ranks)):
            if ranks[i] < ranks[i - 1]:
                report.add(
                    "ERROR", "STRAT_OUT_OF_ORDER",
                    f"'{group.iloc[i]['seam']}' muncul di bawah "
                    f"'{group.iloc[i-1]['seam']}', bertentangan dengan skema stratigrafi",
                    hole_id=hid, seam=group.iloc[i]["seam"],
                )
                break


# --------------------------------------------------------------------------- #
# Cek kualitas
# --------------------------------------------------------------------------- #
def check_quality(seam: pd.DataFrame, report: ValidationReport, cfg) -> None:
    mb_tol = float(cfg["validation.mass_balance_tolerance_pct"])
    cv_tol = float(cfg["validation.cv_conversion_tolerance"])
    z_thresh = float(cfg["validation.rd_ash_outlier_z"])

    has = set(seam.columns)

    if {"im_adb", "ash_adb", "vm_adb", "fc_adb"} <= has:
        total = seam[["im_adb", "ash_adb", "vm_adb", "fc_adb"]].sum(axis=1, min_count=4)
        bad = seam[(total - 100.0).abs() > mb_tol]
        for idx, row in bad.iterrows():
            report.add(
                "ERROR", "MASS_BALANCE",
                f"IM+Ash+VM+FC = {total[idx]:.2f} (adb), menyimpang dari 100",
                hole_id=row["hole_id"], seam=row["seam"], value=float(total[idx]),
            )

    if {"cv_adb", "cv_ar", "tm_ar", "im_adb"} <= has:
        expected = adb_to_ar(seam["cv_adb"], seam["tm_ar"], seam["im_adb"])
        diff = seam["cv_ar"].to_numpy(float) - expected
        bad = np.isfinite(diff) & (np.abs(diff) > cv_tol)
        for i in np.flatnonzero(bad):
            row = seam.iloc[i]
            report.add(
                "ERROR", "CV_CONVERSION",
                f"CV(ar) dilaporkan {row['cv_ar']:.0f} vs hitung ulang {expected[i]:.0f} "
                f"dari CV(adb), TM, IM (selisih {diff[i]:+.0f} cal/g)",
                hole_id=row["hole_id"], seam=row["seam"], value=float(diff[i]),
            )

    # ARD: kewajaran nilai, dan uji apakah ia benar-benar apparent density.
    if "rd_adb" in has:
        lo, hi = cfg["density.plausible_range"]
        bad = seam[(seam["rd_adb"] < lo) | (seam["rd_adb"] > hi)]
        for _, row in bad.iterrows():
            report.add(
                "WARNING", "ARD_RANGE",
                f"ARD {row['rd_adb']:.3f} di luar rentang wajar [{lo}, {hi}]",
                hole_id=row["hole_id"], seam=row["seam"], value=float(row["rd_adb"]),
            )

    if {"rd_adb", "im_adb"} <= has:
        dm = dry_matter_ard(seam["rd_adb"], seam["im_adb"])
        suspicious = np.isfinite(dm) & ((dm < 1.30) | (dm > 1.70))
        for i in np.flatnonzero(suspicious):
            row = seam.iloc[i]
            report.add(
                "WARNING", "ARD_NOT_APPARENT",
                f"ARD matriks kering tersirat = {dm[i]:.3f}, di luar 1,30-1,70. "
                "Nilai yang dilaporkan mungkin bukan apparent density (ASTM D167), "
                "atau basisnya bukan air-dried.",
                hole_id=row["hole_id"], seam=row["seam"], value=float(dm[i]),
            )

    # Regresi ARD terhadap ash: hubungannya harus positif dan rapat.
    if {"rd_adb", "ash_adb"} <= has:
        sub = seam[["hole_id", "seam", "rd_adb", "ash_adb"]].dropna()
        if len(sub) >= 10:
            slope, intercept = np.polyfit(sub["ash_adb"], sub["rd_adb"], 1)
            resid = sub["rd_adb"] - (slope * sub["ash_adb"] + intercept)
            sd = resid.std()
            if slope <= 0:
                report.add(
                    "WARNING", "ARD_ASH_SLOPE",
                    f"regresi ARD terhadap Ash bergradien {slope:+.5f} (semestinya positif)",
                    value=float(slope),
                )
            if sd > 0:
                for idx, r in resid.items():
                    if abs(r / sd) > z_thresh:
                        row = sub.loc[idx]
                        report.add(
                            "WARNING", "ARD_ASH_OUTLIER",
                            f"ARD {row['rd_adb']:.3f} menyimpang {r/sd:+.1f} sigma "
                            f"dari tren ARD-Ash",
                            hole_id=row["hole_id"], seam=row["seam"], value=float(r / sd),
                        )

    # Cakupan kualitas: ini yang menentukan plafon klasifikasi, bukan spasi bor.
    qual_cols = [c for c in ("ash_adb", "cv_adb", "rd_adb") if c in has]
    if qual_cols:
        for s, group in seam.groupby("seam"):
            n_total = len(group)
            n_qual = int(group[qual_cols].notna().any(axis=1).sum())
            pct = 100.0 * n_qual / n_total if n_total else 0.0
            severity = "ERROR" if n_qual == 0 else ("WARNING" if pct < 25 else "INFO")
            report.add(
                severity, "QUALITY_COVERAGE",
                f"seam '{s}': {n_qual}/{n_total} interval punya data kualitas ({pct:.0f}%)",
                seam=s, value=pct,
            )


def check_qaqc_presence(seam: pd.DataFrame, report: ValidationReport) -> None:
    """Ketiadaan QAQC adalah temuan, bukan kekosongan yang boleh didiamkan."""
    markers = [c for c in seam.columns if any(
        k in c.lower() for k in ("duplicate", "duplikat", "crm", "standard", "umpire", "qaqc", "blank")
    )]
    if not markers:
        report.add(
            "WARNING", "NO_QAQC",
            "tidak ada kolom duplikat/CRM/umpire/blank di data. Tanpa QAQC yang "
            "dapat diungkapkan, klasifikasi tertinggi yang dapat dipertahankan "
            "umumnya terbatas - berapa pun rapat spasi bornya.",
        )


def run_all(
    data, report: ValidationReport, cfg, topo_at_collar: np.ndarray | None = None
) -> ValidationReport:
    check_collar(data.collar, report, cfg)
    check_intervals(data.collar, data.seam, report, cfg)
    check_stratigraphic_order(data.seam, cfg.get("stratigraphy") or [], report)
    check_quality(data.seam, report, cfg)
    check_qaqc_presence(data.seam, report)
    if topo_at_collar is not None:
        check_collar_vs_topo(data.collar, topo_at_collar, report, cfg)
    return report
