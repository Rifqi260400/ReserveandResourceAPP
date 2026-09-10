"""Phase 0: audit data. Gerbang keras sebelum logika estimasi apa pun.

Setiap temuan bertingkat: INFO, WARN, atau STOP. Temuan STOP menandai kondisi
yang tidak boleh diselesaikan oleh kode - hanya manusia yang boleh memutuskan,
dan keputusannya harus masuk ke konfigurasi agar ikut tercatat pada keluaran.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, RADII_UNVERIFIED_WARNING
from ..io.excel import Workbook, normalise_hole_id
from ..io.las import LasFile
from ..io.quality_table import QualityTable
from ..logging_setup import get_logger

log = get_logger("audit")

MIN_HOLES_FOR_VORONOI = 3
DEPTH_OFFSET_WARN_M = 0.50
DEPTH_OFFSET_WARN_FRAC = 0.20


class Severity(IntEnum):
    INFO = 0
    WARN = 1
    STOP = 2

    def __str__(self) -> str:
        return {0: "INFO", 1: "WARN", 2: "STOP"}[int(self)]


@dataclass
class Finding:
    severity: Severity
    check: str
    message: str
    hole_id: str | None = None
    seam: str | None = None
    remedy: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, severity, check, message, **kw) -> None:
        self.findings.append(Finding(severity, check, message, **kw))

    def of(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def stops(self) -> list[Finding]:
        return self.of(Severity.STOP)

    @property
    def passed(self) -> bool:
        return not self.stops

    def to_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["severity", "check", "hole_id", "seam", "message", "remedy"])
        return pd.DataFrame([
            {"severity": str(f.severity), "check": f.check, "hole_id": f.hole_id,
             "seam": f.seam, "message": f.message, "remedy": f.remedy}
            for f in sorted(self.findings, key=lambda f: (-int(f.severity), f.check, f.hole_id or ""))
        ])


# --------------------------------------------------------------------------- #
# Helper bersama
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def unrecovered_codes(library: dict[str, str]) -> set[str]:
    """Kode yang berarti material TIDAK TERAMBIL, bukan batuan yang teramati.

    Core loss di dalam interval sampling batubara adalah masalah yang berbeda
    dari parting: bukan pengotor yang terukur, melainkan panjang yang tidak
    diketahui isinya namun tetap masuk ke interval sampel. Keduanya harus
    dilaporkan terpisah agar tidak tertukar.
    """
    codes: set[str] = set()
    for code, description in library.items():
        text = description.strip().lower()
        if any(k in text for k in ("core loss", "not logged", "not recovery",
                                   "lost circulation", "not cored", "not sample")):
            codes.add(code.upper())
    return codes


def coal_codes(library: dict[str, str]) -> set[str]:
    """Kode litologi yang berarti batubara, dibaca dari 'Library SLL'.

    Kode TIDAK dihardcode. Deskripsi yang mengandung 'coal' atau 'lignite'
    diambil, tetapi 'coaly ...' (coaly claystone, coaly mudstone) dikecualikan
    karena itu batuan karbonan, bukan batubara.
    """
    codes: set[str] = set()
    for code, description in library.items():
        text = description.strip().lower()
        if text.startswith("coaly"):
            continue
        if "coal" in text or "lignite" in text:
            codes.add(code.upper())
    return codes


def seam_intervals(wb: Workbook, sheet: str = "SLL_Reconciled") -> pd.DataFrame:
    """Interval batubara per seam dari satu sheet SLL, tanpa cutoff apa pun.

    Cutoff milik modul seams. Di sini kita hanya butuh geometri mentah untuk
    memeriksa konsistensi antar-sumber.
    """
    table = wb.sheets.get(sheet)
    if table is None:
        return pd.DataFrame()
    df = table.frame.copy()
    df["hole_id"] = normalise_hole_id(wb.hole_id)
    df["depth_from"] = df["depth_from"].map(_num)
    df["depth_to"] = df["depth_to"].map(_num)
    df["lithology"] = df["lithology"].astype(str).str.strip().str.upper()
    df["seam"] = df["seam"].astype(str).str.strip()
    df = df[np.isfinite(df["depth_from"]) & np.isfinite(df["depth_to"])]
    return df.reset_index(drop=True)


def seam_envelopes(intervals: pd.DataFrame, coal: set[str]) -> pd.DataFrame:
    """Amplop roof-floor per seam, plus tebal batubara dan parting."""
    rows = []
    named = intervals[intervals["seam"].notna() & (intervals["seam"] != "nan")
                      & (intervals["seam"] != "") & (intervals["seam"].str.upper() != "PA")]
    for (hole, seam), group in named.groupby(["hole_id", "seam"], sort=False):
        roof, floor = float(group["depth_from"].min()), float(group["depth_to"].max())
        envelope = intervals[(intervals["depth_from"] >= roof - 1e-9)
                             & (intervals["depth_to"] <= floor + 1e-9)]
        coal_t = float(sum(r.depth_to - r.depth_from for r in envelope.itertuples()
                           if r.lithology in coal))
        rows.append({
            "hole_id": hole, "seam": seam, "roof_m": roof, "floor_m": floor,
            "gross_thickness_m": floor - roof, "coal_thickness_m": coal_t,
            "parting_thickness_m": (floor - roof) - coal_t,
            "n_intervals": len(envelope),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Item 1: peta header
# --------------------------------------------------------------------------- #
def check_header_maps(workbooks: list[Workbook], report: AuditReport) -> None:
    rows = []
    for wb in workbooks:
        for name, table in wb.sheets.items():
            h = table.header
            rows.append({
                "file": wb.path.name, "sheet": name,
                "header_rows": ", ".join(str(r + 1) for r in h.header_rows),
                "data_start_row": h.data_start_row + 1,
                "columns_mapped": len(h.canonical),
                "columns_unmapped": len(h.unmapped),
            })
    report.tables["header_maps"] = pd.DataFrame(rows)
    report.add(
        Severity.WARN, "01_header_map",
        f"Peta header teratasi untuk {len(workbooks)} workbook. Peta ini "
        "dibangun dari merged range, bukan nomor baris tetap - PERIKSA dan "
        "konfirmasi sebelum estimasi dijalankan.",
        remedy="Tinjau tabel 'Peta header' dan sampel baris di laporan audit.",
    )


# --------------------------------------------------------------------------- #
# Item 2: konflik koordinat
# --------------------------------------------------------------------------- #
def check_coordinate_conflict(workbooks: list[Workbook], cfg: Config, report: AuditReport) -> None:
    """Collar dan BHC membawa set koordinat dari metode survei berbeda.

    RL collar merambat langsung ke elevasi roof dan floor seam, jadi ini tidak
    boleh diselesaikan diam-diam.
    """
    rows = []
    for wb in workbooks:
        collar = wb.sheets["Collar"].frame.iloc[0]
        bhc = wb.sheets.get("BHC")
        record: dict[str, Any] = {
            "hole_id": wb.hole_id,
            "collar_east": _num(collar.get("east")), "collar_north": _num(collar.get("north")),
            "collar_rl": _num(collar.get("rl")),
            "collar_survey_method": str(collar.get("survey_method", "")).strip(),
        }
        if bhc is not None and not bhc.frame.empty:
            head = bhc.frame.iloc[0]
            for src in ("gps", "ts"):
                for axis in ("east", "north", "rl"):
                    record[f"bhc_{src}_{axis}"] = _num(head.get(f"{src}_{axis}"))
        rows.append(record)

    frame = pd.DataFrame(rows)
    for src in ("ts", "gps"):
        for axis in ("east", "north", "rl"):
            a, b = f"collar_{axis}", f"bhc_{src}_{axis}"
            if a in frame and b in frame:
                frame[f"d{axis}_vs_{src}"] = frame[a] - frame[b]
    report.tables["coordinate_conflict"] = frame

    for _, row in frame.iterrows():
        for src, label in (("ts", "BHC total station"), ("gps", "BHC GPS")):
            deltas = {ax: row.get(f"d{ax}_vs_{src}", math.nan) for ax in ("east", "north", "rl")}
            if all(math.isnan(v) for v in deltas.values()):
                report.add(
                    Severity.INFO, "02_coordinate_conflict",
                    f"{label} tidak terisi - perbandingan tidak dapat dilakukan.",
                    hole_id=row["hole_id"],
                )
                continue
            worst = max((abs(v) for v in deltas.values() if not math.isnan(v)), default=0.0)
            text = ", ".join(f"d{k}={v:+.3f} m" for k, v in deltas.items() if not math.isnan(v))
            severity = Severity.WARN if worst > 0.10 else Severity.INFO
            report.add(
                severity, "02_coordinate_conflict",
                f"Collar vs {label}: {text}", hole_id=row["hole_id"],
                remedy="Sumber otoritatif diambil dari authoritative_coordinate_source.",
            )

    report.add(
        Severity.INFO, "02_coordinate_conflict",
        f"Sumber koordinat otoritatif = '{cfg.authoritative_coordinate_source}' "
        "(dari konfigurasi).",
    )


# --------------------------------------------------------------------------- #
# Item 3: basis kedalaman
# --------------------------------------------------------------------------- #
def check_depth_basis(workbooks: list[Workbook], report: AuditReport) -> None:
    rows = []
    for wb in workbooks:
        recon = wb.sheets.get("SLL_Reconciled")
        wells = wb.sheets.get("SLL_Wellsite")
        if recon is None:
            report.add(
                Severity.STOP, "03_depth_basis",
                "tidak ada sheet SLL_Reconciled. Tool tidak akan mundur ke "
                "kedalaman wellsite.",
                hole_id=wb.hole_id,
                remedy="Sediakan log terekonsiliasi, atau keluarkan lubang ini dari run.",
            )
            continue

        coal = coal_codes(wb.lithology_library)
        env_r = seam_envelopes(seam_intervals(wb, "SLL_Reconciled"), coal)
        env_w = (seam_envelopes(seam_intervals(wb, "SLL_Wellsite"), coal)
                 if wells is not None else pd.DataFrame())
        if env_w.empty:
            report.add(Severity.INFO, "03_depth_basis",
                       "SLL_Wellsite tidak tersedia; offset tidak dapat dihitung.",
                       hole_id=wb.hole_id)
            continue

        merged = env_r.merge(env_w, on=["hole_id", "seam"], suffixes=("_rec", "_well"))
        for _, r in merged.iterrows():
            rows.append({
                "hole_id": r["hole_id"], "seam": r["seam"],
                "roof_reconciled_m": r["roof_m_rec"], "roof_wellsite_m": r["roof_m_well"],
                "roof_offset_m": r["roof_m_rec"] - r["roof_m_well"],
                "floor_reconciled_m": r["floor_m_rec"], "floor_wellsite_m": r["floor_m_well"],
                "floor_offset_m": r["floor_m_rec"] - r["floor_m_well"],
                "thickness_offset_m": r["coal_thickness_m_rec"] - r["coal_thickness_m_well"],
            })

    frame = pd.DataFrame(rows)
    report.tables["depth_basis_offset"] = frame
    if frame.empty:
        return

    worst = frame.reindex(frame["roof_offset_m"].abs().sort_values(ascending=False).index).iloc[0]
    report.add(
        Severity.INFO, "03_depth_basis",
        f"Offset wellsite -> reconciled: median roof "
        f"{frame['roof_offset_m'].median():+.3f} m, terbesar "
        f"{worst['roof_offset_m']:+.3f} m pada {worst['hole_id']}/{worst['seam']}. "
        "Semua kedalaman memakai basis terekonsiliasi.",
    )

    # Offset kecil adalah rekonsiliasi normal terhadap log geofisika. Offset
    # yang sebanding dengan tebal seam itu sendiri menandakan pick yang bergeser
    # jauh, dan pada seam tipis ia mengubah tonase secara proporsional besar.
    for row in frame.itertuples():
        thickness = row.floor_reconciled_m - row.roof_reconciled_m
        relative = abs(row.thickness_offset_m) / thickness if thickness > 0 else 0.0
        if abs(row.roof_offset_m) > DEPTH_OFFSET_WARN_M or relative > DEPTH_OFFSET_WARN_FRAC:
            report.add(
                Severity.WARN, "03_depth_basis",
                f"rekonsiliasi menggeser roof {row.roof_offset_m:+.3f} m dan tebal "
                f"batubara {row.thickness_offset_m:+.3f} m ({relative:.0%} dari tebal "
                f"terekonsiliasi {thickness:.2f} m).",
                hole_id=row.hole_id, seam=row.seam,
                remedy="Periksa pick terhadap kurva LAS di interval ini.",
            )


# --------------------------------------------------------------------------- #
# Item 4: konflik litologi vs sampling
# --------------------------------------------------------------------------- #
def check_lithology_vs_sampling(workbooks: list[Workbook], cfg: Config, report: AuditReport) -> None:
    """Parting tipis di log litologi yang ditembus lurus oleh interval sampling.

    Keduanya menghasilkan tebal batubara yang berbeda untuk seam yang sama.
    """
    rows = []
    for wb in workbooks:
        sampling = wb.sheets.get("Sampling")
        if sampling is None or sampling.frame.empty:
            continue
        coal = coal_codes(wb.lithology_library)
        unrecovered = unrecovered_codes(wb.lithology_library)
        intervals = seam_intervals(wb, "SLL_Reconciled")
        non_coal = intervals[~intervals["lithology"].isin(coal)]

        samples = sampling.frame.copy()
        samples["adjusted_from"] = samples["adjusted_from"].map(_num)
        samples["adjusted_to"] = samples["adjusted_to"].map(_num)
        samples["sample_position"] = samples["sample_position"].astype(str).str.strip().str.upper()
        samples = samples[np.isfinite(samples["adjusted_from"])
                          & np.isfinite(samples["adjusted_to"])
                          & (samples["adjusted_to"] > samples["adjusted_from"])]
        coal_samples = samples[samples["sample_position"].str.contains("COAL", na=False)]

        for s in coal_samples.itertuples():
            inside = non_coal[(non_coal["depth_from"] < s.adjusted_to - 1e-9)
                              & (non_coal["depth_to"] > s.adjusted_from + 1e-9)]
            for p in inside.itertuples():
                overlap = min(p.depth_to, s.adjusted_to) - max(p.depth_from, s.adjusted_from)
                rows.append({
                    "hole_id": wb.hole_id, "seam": str(s.seam).strip(),
                    "sample_number": s.sample_number, "sample_position": s.sample_position,
                    "sample_from_m": s.adjusted_from, "sample_to_m": s.adjusted_to,
                    "lith_code": p.lithology,
                    "lith_meaning": wb.lithology_library.get(p.lithology, "?"),
                    "conflict_type": ("core_loss" if p.lithology in unrecovered else "parting"),
                    "lith_from_m": p.depth_from,
                    "lith_to_m": p.depth_to, "overlap_m": overlap,
                })

    frame = pd.DataFrame(rows)
    report.tables["lithology_vs_sampling"] = frame
    if frame.empty:
        report.add(Severity.INFO, "04_lith_vs_sampling",
                   "Tidak ada konflik litologi-sampling yang terdeteksi.")
        return

    partings = frame[frame["conflict_type"] == "parting"]
    losses = frame[frame["conflict_type"] == "core_loss"]

    if not partings.empty:
        report.add(
            Severity.WARN, "04_lith_vs_sampling",
            f"{len(partings)} interval sampling batubara menembus PARTING di log "
            f"litologi (total {partings['overlap_m'].sum():.2f} m). Kedua sumber "
            "memberi tebal batubara yang berbeda untuk seam yang sama.",
            remedy=f"coal_thickness_source = '{cfg.coal_thickness_source}' akan dipakai.",
        )
    if not losses.empty:
        # Dipisahkan dari parting: ini panjang yang tidak terambil, bukan
        # pengotor yang terukur. Nilai kualitas komposit yang mencakupnya
        # mewakili material yang sebagian tidak pernah ada di tangan.
        report.add(
            Severity.WARN, "04_lith_vs_sampling",
            f"{len(losses)} interval sampling batubara mencakup CORE LOSS "
            f"(total {losses['overlap_m'].sum():.2f} m). Berbeda dari parting: "
            "panjang ini tidak terambil, sehingga sampel dihitung atas material "
            "yang sebagian tidak pernah diperoleh.",
            remedy="Tinjau apakah panjang sampel perlu dikoreksi terhadap core loss.",
        )


# --------------------------------------------------------------------------- #
# Item 5 & 6: cakupan kualitas dan basis RD
# --------------------------------------------------------------------------- #
def check_quality(
    workbooks: list[Workbook], quality: QualityTable | None, cfg: Config, report: AuditReport
) -> None:
    if quality is None:
        report.add(
            Severity.STOP, "05_quality_coverage",
            "Tabel kualitas tidak dipasok. Kualitas tidak ada di dalam workbook; "
            "ia hanya ada di PDF laporan lab, dan tool ini sengaja tidak "
            "mem-parse PDF.",
            remedy="Isi paths.quality_table dengan CSV/Excel sesuai skema "
                   "(lihat io/quality_table.py: QUALITY_COLUMNS).",
        )
        report.add(
            Severity.STOP, "06_rd_basis",
            "Basis RD tidak dapat diperiksa tanpa tabel kualitas.",
            remedy="Kolom RD_basis wajib terisi salah satu dari in_situ, "
                   "air_dried, as_received, unknown. Nilai 'unknown' menghentikan run.",
        )
        return

    frame = quality.frame
    unknown = frame[frame["RD_basis"].astype(str).str.strip() == "unknown"]
    for _, row in unknown.iterrows():
        report.add(
            Severity.STOP, "06_rd_basis",
            f"RD_basis = 'unknown' pada sampel {row['sample_id']}. Basis tidak "
            "boleh disimpulkan dari nilainya.",
            hole_id=str(row.get("hole_id")), seam=str(row.get("seam")),
            remedy="Konfirmasi ke laboratorium metode uji RD (ASTM D167 apparent "
                   "vs piknometer) dan basis moisture saat pengukuran.",
        )

    rows = []
    by_hole = {normalise_hole_id(wb.hole_id): wb for wb in workbooks}
    for wb in workbooks:
        coal = coal_codes(wb.lithology_library)
        envelopes = seam_envelopes(seam_intervals(wb, "SLL_Reconciled"), coal)
        sampling = wb.sheets.get("Sampling")
        samples = sampling.frame.copy() if sampling is not None else pd.DataFrame()
        if not samples.empty:
            samples["adjusted_from"] = samples["adjusted_from"].map(_num)
            samples["adjusted_to"] = samples["adjusted_to"].map(_num)
            samples["sample_number"] = samples["sample_number"].astype(str).str.strip()

        hid = normalise_hole_id(wb.hole_id)
        for env in envelopes.itertuples():
            results = frame[(frame["hole_id"].map(normalise_hole_id) == hid)
                            & (frame["seam"].astype(str).str.strip() == env.seam)]
            covered = 0.0
            members: list[str] = []
            for _, res in results.iterrows():
                members.extend(quality.composite_members(str(res["sample_id"])))
            if members and not samples.empty:
                hit = samples[samples["sample_number"].isin(members)]
                covered = float((hit["adjusted_to"] - hit["adjusted_from"]).clip(lower=0).sum())
            fraction = covered / env.coal_thickness_m if env.coal_thickness_m > 0 else 0.0
            rows.append({
                "hole_id": hid, "seam": env.seam,
                "coal_thickness_m": env.coal_thickness_m,
                "quality_covered_m": covered, "coverage_frac": fraction,
                "n_results": len(results),
            })

    coverage = pd.DataFrame(rows)
    report.tables["quality_coverage"] = coverage
    threshold = cfg.cutoffs.min_quality_coverage_frac
    for row in coverage.itertuples():
        if row.n_results == 0:
            report.add(
                Severity.WARN, "05_quality_coverage",
                f"tidak ada hasil kualitas sama sekali untuk seam ini.",
                hole_id=row.hole_id, seam=row.seam,
                remedy="Seam tanpa kualitas tidak dapat dinilai terhadap cutoff CV/Ash.",
            )
        elif row.coverage_frac < threshold:
            report.add(
                Severity.WARN, "05_quality_coverage",
                f"kualitas hanya mencakup {row.quality_covered_m:.2f} m dari "
                f"{row.coal_thickness_m:.2f} m batubara ({row.coverage_frac:.1%}, "
                f"ambang {threshold:.0%}).",
                hole_id=row.hole_id, seam=row.seam,
                remedy="Komposit tidak boleh disajikan sebagai mewakili tebal yang "
                       "tidak ia sampel.",
            )


# --------------------------------------------------------------------------- #
# Item 7: open hole vs cored
# --------------------------------------------------------------------------- #
def check_core_coverage(workbooks: list[Workbook], cfg: Config, report: AuditReport) -> None:
    rows = []
    for wb in workbooks:
        coal = coal_codes(wb.lithology_library)
        intervals = seam_intervals(wb, "SLL_Reconciled")
        envelopes = seam_envelopes(intervals, coal)
        for env in envelopes.itertuples():
            window = intervals[(intervals["depth_from"] >= env.roof_m - 1e-9)
                               & (intervals["depth_to"] <= env.floor_m + 1e-9)]
            types = sorted({str(t).strip().upper() for t in window["hole_type"].dropna()})
            recovery = pd.to_numeric(window["core_recovery"], errors="coerce").dropna()
            rows.append({
                "hole_id": env.hole_id, "seam": env.seam,
                "hole_types": ",".join(types) or "?",
                "core_recovery_pct": float(recovery.mean()) if len(recovery) else math.nan,
                "open_hole": all(t == "OH" for t in types) if types else True,
            })

    frame = pd.DataFrame(rows)
    report.tables["core_coverage"] = frame
    for row in frame.itertuples():
        if row.open_hole:
            report.add(
                Severity.WARN, "07_open_hole",
                "seam dilog dari cutting open hole. Pick roof dan floor kurang "
                "andal dibanding interval cored, dan tidak ada material untuk "
                "dianalisa.",
                hole_id=row.hole_id, seam=row.seam,
            )
        elif np.isfinite(row.core_recovery_pct) and row.core_recovery_pct < cfg.cutoffs.min_core_recovery_pct:
            report.add(
                Severity.WARN, "07_open_hole",
                f"core recovery {row.core_recovery_pct:.1f}% di bawah ambang "
                f"{cfg.cutoffs.min_core_recovery_pct}%.",
                hole_id=row.hole_id, seam=row.seam,
            )


# --------------------------------------------------------------------------- #
# Item 8: integritas umum + LAS
# --------------------------------------------------------------------------- #
def check_integrity(
    workbooks: list[Workbook], las_files: dict[str, LasFile], report: AuditReport
) -> None:
    seen: dict[str, str] = {}
    for wb in workbooks:
        hid = normalise_hole_id(wb.hole_id)
        if hid in seen:
            report.add(Severity.STOP, "08_integrity",
                       f"hole_id ganda: {wb.path.name} dan {seen[hid]}", hole_id=hid,
                       remedy="Satu lubang harus punya tepat satu workbook.")
        seen[hid] = wb.path.name

        intervals = seam_intervals(wb, "SLL_Reconciled").sort_values("depth_from")
        prev_to, prev_lith = None, None
        for r in intervals.itertuples():
            if r.depth_to <= r.depth_from:
                report.add(Severity.STOP, "08_integrity",
                           f"interval tidak sah: from={r.depth_from} >= to={r.depth_to}",
                           hole_id=hid, remedy="Perbaiki log litologi.")
            if prev_to is not None:
                gap = r.depth_from - prev_to
                if gap > 1e-6:
                    report.add(Severity.WARN, "08_integrity",
                               f"celah {gap:.3f} m pada {prev_to:.3f}-{r.depth_from:.3f} m "
                               f"(setelah {prev_lith})", hole_id=hid)
                elif gap < -1e-6:
                    report.add(Severity.STOP, "08_integrity",
                               f"tumpang tindih {-gap:.3f} m pada {r.depth_from:.3f} m",
                               hole_id=hid, remedy="Perbaiki log litologi.")
            prev_to, prev_lith = r.depth_to, r.lithology

        collar = wb.sheets["Collar"].frame.iloc[0]
        td = _num(collar.get("max_depth_drilling"))
        if np.isfinite(td) and len(intervals):
            deepest = float(intervals["depth_to"].max())
            if deepest > td + 1e-6:
                report.add(Severity.STOP, "08_integrity",
                           f"log mencapai {deepest:.2f} m melebihi total depth {td:.2f} m",
                           hole_id=hid, remedy="Rekonsiliasi total depth dan log.")

        east, north = _num(collar.get("east")), _num(collar.get("north"))
        if np.isfinite(east) and np.isfinite(north):
            zone = int(east // 1_000_000) if east > 1_000_000 else None
            if not (100_000 <= east <= 900_000):
                report.add(Severity.WARN, "08_integrity",
                           f"easting {east:,.1f} di luar rentang UTM lazim", hole_id=hid)
            if not (0 <= north <= 10_000_000):
                report.add(Severity.WARN, "08_integrity",
                           f"northing {north:,.1f} di luar rentang UTM lazim", hole_id=hid)

        # Cakupan LAS terhadap interval seam.
        las = las_files.get(hid)
        if las is None:
            report.add(Severity.INFO, "08_integrity", "tidak ada LAS untuk lubang ini.",
                       hole_id=hid)
            continue
        if las.density_curves_are_raw_counts:
            units = ", ".join(f"{c.mnemonic}={c.unit or '?'}" for c in las.curves.values())
            report.add(
                Severity.INFO, "08_integrity",
                f"LAS tidak memuat kurva densitas terkalibrasi ({units}). Kurva ini "
                "sah untuk verifikasi pick dan rekonsiliasi kedalaman, dan "
                "DIBLOKIR dari jalur tonase.",
                hole_id=hid,
            )
        top, base = las.logged_interval
        coal = coal_codes(wb.lithology_library)
        for env in seam_envelopes(intervals, coal).itertuples():
            if env.floor_m > base + 1e-6 or env.roof_m < top - 1e-6:
                report.add(
                    Severity.WARN, "08_integrity",
                    f"interval seam {env.roof_m:.2f}-{env.floor_m:.2f} m berada di "
                    f"luar cakupan log {top:.2f}-{base:.2f} m; pick terekonsiliasinya "
                    "tidak terverifikasi.",
                    hole_id=hid, seam=env.seam,
                )
        for mnemonic in sorted(las.curves):
            span = las.curve_coverage(mnemonic)
            if span is None:
                report.add(Severity.WARN, "08_integrity",
                           f"kurva {mnemonic} kosong seluruhnya.", hole_id=hid)


# --------------------------------------------------------------------------- #
# Item 9: jumlah lubang per seam
# --------------------------------------------------------------------------- #
def check_hole_counts(workbooks: list[Workbook], report: AuditReport) -> pd.DataFrame:
    rows = []
    for wb in workbooks:
        coal = coal_codes(wb.lithology_library)
        for env in seam_envelopes(seam_intervals(wb, "SLL_Reconciled"), coal).itertuples():
            rows.append({"hole_id": env.hole_id, "seam": env.seam,
                         "coal_thickness_m": env.coal_thickness_m})
    frame = pd.DataFrame(rows)
    if frame.empty:
        report.add(Severity.STOP, "09_hole_count",
                   "tidak ada interseksi seam sama sekali.",
                   remedy="Periksa kolom Seam pada SLL_Reconciled.")
        return frame

    counts = frame.groupby("seam")["hole_id"].nunique().sort_values(ascending=False)
    report.tables["seam_hole_counts"] = counts.rename("n_holes").reset_index()
    for seam, n in counts.items():
        if n < MIN_HOLES_FOR_VORONOI:
            report.add(
                Severity.STOP, "09_hole_count",
                f"seam ditembus {n} lubang; Voronoi memerlukan minimal "
                f"{MIN_HOLES_FOR_VORONOI} titik tidak segaris. Tidak ada estimasi "
                "poligonal yang mungkin untuk seam ini.",
                seam=str(seam),
                remedy="Tambahkan lubang, gabungkan seam, atau keluarkan seam dari run.",
            )
    return frame


# --------------------------------------------------------------------------- #
# Item 10 & 11: kondisi geologi dan RPEEE
# --------------------------------------------------------------------------- #
def check_geological_condition(cfg: Config, report: AuditReport) -> None:
    # Keberadaan dan panjang justifikasi sudah ditegakkan pydantic saat load.
    report.add(
        Severity.INFO, "10_geological_condition",
        f"Kondisi geologi = '{cfg.geological_condition}'. Radius: "
        f"measured {cfg.radii.measured:.0f} m, indicated {cfg.radii.indicated:.0f} m, "
        f"inferred {cfg.radii.inferred:.0f} m.",
    )
    report.add(Severity.WARN, "10_geological_condition", RADII_UNVERIFIED_WARNING,
               remedy="Verifikasi tabel radius terhadap teks SNI 5015:2019.")
    report.context["geological_condition_justification"] = cfg.geological_condition_justification


def check_rpeee(cfg: Config, report: AuditReport) -> None:
    r = cfg.rpeee_constraints
    set_items, null_items = [], []
    for name in ("max_depth_m", "min_cv_ar_kcal_kg", "max_ash_adb_pct"):
        (set_items if getattr(r, name) is not None else null_items).append(name)
    (set_items if r.excluded_area_wkt.strip() else null_items).append("excluded_area_wkt")

    report.add(Severity.INFO, "11_rpeee",
               f"Batasan terisi: {set_items or 'tidak ada'}. Kosong: {null_items or 'tidak ada'}.")
    report.context["resource_label"] = r.resource_label

    if r.has_economic_constraint:
        report.add(Severity.INFO, "11_rpeee",
                   f"Batas kedalaman {r.max_depth_m:.0f} m dengan dasar: "
                   f"\"{r.max_depth_basis}\". Keluaran akan dilabeli SUMBERDAYA.")
    else:
        report.add(
            Severity.WARN, "11_rpeee",
            "max_depth_m kosong: tidak ada batasan prospek ekonomi yang diterapkan. "
            "SETIAP keluaran run ini akan dilabeli INVENTORI BATUBARA, bukan "
            "Sumberdaya, dan kolom kelas menjadi Inventori Terukur / Tertunjuk / "
            "Tereka (aturan 8.4).",
            remedy="Isi max_depth_m beserta max_depth_basis untuk melaporkan Sumberdaya.",
        )


# --------------------------------------------------------------------------- #
def run_audit(
    workbooks: list[Workbook],
    las_files: dict[str, LasFile],
    quality: QualityTable | None,
    topo,
    cfg: Config,
) -> AuditReport:
    report = AuditReport()
    report.context["n_workbooks"] = len(workbooks)
    report.context["n_las"] = len(las_files)

    check_header_maps(workbooks, report)
    check_coordinate_conflict(workbooks, cfg, report)
    check_depth_basis(workbooks, report)
    check_lithology_vs_sampling(workbooks, cfg, report)
    check_quality(workbooks, quality, cfg, report)
    check_core_coverage(workbooks, cfg, report)
    check_integrity(workbooks, las_files, report)
    check_hole_counts(workbooks, report)
    check_geological_condition(cfg, report)
    check_rpeee(cfg, report)

    if topo is None:
        report.add(Severity.STOP, "08_integrity",
                   "topografi DXF tidak dipasok. Subcrop dan batas kedalaman tidak "
                   "dapat dihitung.",
                   remedy="Isi paths.topography_dxf.")
    else:
        report.add(Severity.INFO, "08_integrity",
                   f"Topografi: {len(topo.points):,} simpul, entitas {topo.entity_counts}, "
                   f"RL {topo.z_range[0]:.1f}-{topo.z_range[1]:.1f} m.")
    return report
