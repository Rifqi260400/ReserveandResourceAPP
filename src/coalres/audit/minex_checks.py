"""Audit Phase 0 untuk jalur flat file Minex.

Berkas Minex tidak berheader dan tidak membawa metadata survei, coring, atau
logging seperti workbook BGG. Karena itu sebagian pemeriksaan BGG tidak berlaku,
dan sebagai gantinya arti kolom serta basis RD menjadi gerbang yang lebih keras:
keduanya HANYA ada di konfigurasi, sehingga salah urut kolom tidak akan memberi
gejala apa pun.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..io.minex import HoleDataset
from ..logging_setup import get_logger
from .checks import AuditReport, Severity, check_geological_condition, check_rpeee
from .gate2 import run_gate2

log = get_logger("audit.minex")

MIN_HOLES_FOR_VORONOI = 3


def check_column_declaration(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    spec = cfg.minex
    rows = []
    for label, columns, path in (
        ("survey", spec.survey_columns, spec.survey_file),
        ("lithology", spec.lithology_columns, spec.lithology_file),
        ("quality", spec.quality_columns, spec.quality_file),
        ("faults", spec.fault_columns, spec.faults_file),
    ):
        if path is None:
            continue
        rows.append({"berkas": str(path), "peran": label,
                     "kolom (urut)": ", ".join(columns)})
    report.tables["column_declaration"] = pd.DataFrame(rows)
    report.add(
        Severity.WARN, "01_column_declaration",
        "Berkas flat tidak berheader: arti kolom seluruhnya berasal dari "
        "konfigurasi. Urutan yang salah TIDAK akan memunculkan kesalahan - "
        "angka tetap terbaca wajar. PERIKSA tabel deklarasi kolom terhadap isi "
        "berkas sebelum estimasi dijalankan.",
        remedy="Cocokkan dengan beberapa baris pertama tiap berkas.",
    )


def check_duplicates(dataset: HoleDataset, report: AuditReport) -> None:
    removed = dataset.provenance.get("duplicate_rows_removed", {})
    for label, count in removed.items():
        if count:
            report.add(
                Severity.WARN, "02_duplicates",
                f"{label}: {count} baris duplikat identik dibuang saat pembacaan. "
                "Dibiarkan, duplikat ini akan melipatgandakan tonase tanpa gejala.",
                remedy="Periksa proses ekspor yang menghasilkan berkas ini.",
            )


def check_cross_file_coverage(dataset: HoleDataset, report: AuditReport) -> None:
    collars = set(dataset.collars["hole_id"])
    intervals = set(dataset.intervals["hole_id"])
    for hole in sorted(intervals - collars):
        report.add(Severity.STOP, "03_coverage", "interval tanpa collar",
                   hole_id=hole, remedy="Lengkapi berkas survey.")
    for hole in sorted(collars - intervals):
        report.add(Severity.INFO, "03_coverage", "collar tanpa interval seam", hole_id=hole)

    if dataset.quality is not None:
        qual = set(dataset.quality["hole_id"])
        for hole in sorted(qual - collars):
            report.add(Severity.STOP, "03_coverage", "kualitas tanpa collar",
                       hole_id=hole, remedy="Lengkapi berkas survey.")
        pct = 100.0 * len(qual & collars) / max(len(collars), 1)
        report.add(
            Severity.INFO if pct >= 25 else Severity.WARN, "03_coverage",
            f"{len(qual & collars)}/{len(collars)} lubang punya data kualitas ({pct:.0f}%). "
            "Cakupan kualitas, bukan spasi bor, yang menentukan plafon klasifikasi.",
        )


def check_intervals(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    body = dataset.intervals
    if "is_marker" in body:
        markers = body[body["is_marker"]]
        if len(markers):
            zero = int((markers["depth_from"] == markers["depth_to"]).sum())
            report.add(
                Severity.INFO, "04_intervals",
                f"{len(markers)} baris penanda seam {cfg.minex.marker_seams} "
                f"({zero} berketebalan nol). Diperlakukan sebagai horizon, bukan seam.",
            )
        body = body[~body["is_marker"]]

    bad = body[body["depth_to"] <= body["depth_from"]]
    for _, row in bad.iterrows():
        report.add(Severity.STOP, "04_intervals",
                   f"interval tidak sah: from={row['depth_from']} >= to={row['depth_to']}",
                   hole_id=row["hole_id"], seam=row["seam"],
                   remedy="Perbaiki berkas lit.")

    depth = dataset.collars.set_index("hole_id").get("total_depth")
    if depth is not None:
        for _, row in body.iterrows():
            td = depth.get(row["hole_id"])
            if td is not None and np.isfinite(td) and row["depth_to"] > td + 1e-6:
                report.add(Severity.STOP, "04_intervals",
                           f"floor {row['depth_to']:.2f} m melebihi total depth {td:.2f} m",
                           hole_id=row["hole_id"], seam=row["seam"],
                           remedy="Rekonsiliasi berkas surv dan lit.")

    for hole, group in body.sort_values("depth_from").groupby("hole_id"):
        previous_to, previous_seam = None, None
        for _, row in group.iterrows():
            if previous_to is not None and row["depth_from"] < previous_to - 1e-6:
                report.add(Severity.STOP, "04_intervals",
                           f"seam '{row['seam']}' mulai {row['depth_from']:.2f} m sebelum "
                           f"'{previous_seam}' berakhir {previous_to:.2f} m",
                           hole_id=hole, seam=row["seam"], remedy="Perbaiki berkas lit.")
            previous_to, previous_seam = row["depth_to"], row["seam"]


def check_stratigraphic_order(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    rank = cfg.stratigraphic_rank()
    if not rank:
        report.add(Severity.WARN, "05_stratigraphy",
                   "stratigraphy kosong di konfigurasi; urutan seam tidak dapat divalidasi.",
                   remedy="Isi stratigraphy (muda -> tua) dan seam_splits.")
        return

    body = dataset.intervals
    body = body[~body["is_marker"]] if "is_marker" in body else body
    unknown = sorted(set(body["seam"]) - set(rank))
    for seam in unknown:
        report.add(Severity.STOP, "05_stratigraphy",
                   f"seam '{seam}' tidak ada di skema stratigrafi",
                   seam=seam, remedy="Tambahkan ke stratigraphy atau seam_splits.")

    for hole, group in body.sort_values("depth_from").groupby("hole_id"):
        ranks = [rank[s] for s in group["seam"] if s in rank]
        for index in range(1, len(ranks)):
            if ranks[index] < ranks[index - 1]:
                report.add(Severity.STOP, "05_stratigraphy",
                           f"urutan seam terbalik terhadap skema stratigrafi",
                           hole_id=hole, seam=str(group.iloc[index]["seam"]),
                           remedy="Periksa korelasi seam pada lubang ini.")
                break

    # Konsistensi split: lubang tidak boleh memuat induk DAN anaknya sekaligus.
    for parent, children in cfg.seam_splits.items():
        for hole, group in body.groupby("hole_id"):
            present = set(group["seam"])
            if parent in present and present & set(children):
                report.add(
                    Severity.STOP, "05_stratigraphy",
                    f"lubang memuat seam induk '{parent}' DAN anaknya "
                    f"{sorted(present & set(children))} sekaligus - tonase akan "
                    "terhitung dua kali.",
                    hole_id=hole, seam=parent,
                    remedy="Pilih satu representasi per lubang.",
                )


def check_quality(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    basis = cfg.minex.quality_rd_basis
    if basis == "unknown":
        report.add(
            Severity.STOP, "06_rd_basis",
            "minex.quality_rd_basis = 'unknown'. Basis RD tidak boleh "
            "disimpulkan dari nilainya; ia menentukan apakah konversi "
            "Preston & Sanders diterapkan, dan itu menggerakkan tonase sekitar 10%.",
            remedy="Konfirmasi ke laboratorium/sumber data: apparent density "
                   "(ASTM D167) pada kondisi in_situ, air_dried, atau as_received.",
        )
    else:
        report.add(Severity.INFO, "06_rd_basis",
                   f"Basis RD dinyatakan konfigurasi: '{basis}'.")
        if basis == "in_situ":
            report.add(
                Severity.INFO, "06_rd_basis",
                "RD sudah in-situ: konversi Preston & Sanders tidak diterapkan, "
                "dan basis kolom moisture tidak menyentuh tonase.",
            )

    moisture = cfg.minex.quality_moisture_basis
    if moisture == "unknown":
        severity = Severity.INFO if basis == "in_situ" else Severity.STOP
        report.add(
            severity, "06_rd_basis",
            "minex.quality_moisture_basis = 'unknown'. "
            + ("Karena RD sudah in-situ, ini tidak menyentuh tonase; ia hanya "
               "menentukan label basis pada kualitas yang dilaporkan."
               if basis == "in_situ" else
               "Konversi basis RD menuntut TM dan IM; keduanya tidak dapat "
               "dibedakan selama basis moisture belum dinyatakan."),
            remedy="Nyatakan apakah kolom moisture adalah IM (adb) atau TM (ar).",
        )
    else:
        report.add(Severity.INFO, "06_rd_basis",
                   f"Basis moisture: '{moisture}'. Satuan CV: "
                   f"'{cfg.minex.quality_cv_unit}'.")

    quality = dataset.quality
    if quality is None:
        report.add(Severity.STOP, "07_quality", "berkas kualitas tidak dipasok.",
                   remedy="Isi minex.quality_file.")
        return

    bad = quality[quality["depth_to"] <= quality["depth_from"]]
    policy = cfg.minex.on_invalid_quality_interval
    severity = Severity.STOP if policy == "stop" else Severity.WARN
    for _, row in bad.iterrows():
        report.add(severity, "07_quality",
                   f"interval kualitas terbalik: from={row['depth_from']} to={row['depth_to']}"
                   + ("" if policy == "stop" else " - baris dikeluarkan"),
                   hole_id=row["hole_id"], seam=row["seam"],
                   remedy="Perbaiki berkas qual. Menukar from dan to di dalam kode berarti "
                          "menebak niat penulisnya, jadi itu tidak dilakukan.")

    components = [c for c in ("MOISTURE", "M_adb", "ASH", "ASH_adb", "VM", "VM_adb",
                              "FC", "FC_adb") if c in quality]
    components = components[:4] if len(components) >= 4 else []
    if len(components) == 4:
        total = quality[components].sum(axis=1, min_count=4)
        off = (total - 100.0).abs() > cfg.validation.mass_balance_tolerance_pct
        if off.any():
            report.add(
                Severity.WARN if off.all() else Severity.STOP, "07_quality",
                f"mass balance {'+'.join(components)} tidak menutup pada "
                f"{int(off.sum())}/{len(quality)} baris "
                f"(median {total.median():.2f}, rentang {total.min():.1f}-{total.max():.1f}). "
                "Bila SELURUH baris meleset seragam, kemungkinan besar penamaan atau "
                "basis kolom yang keliru, bukan kesalahan per sampel.",
                remedy="Konfirmasi arti dan basis tiap kolom di berkas qual.",
            )

    ash_column = next((c for c in ("ASH", "ASH_adb") if c in quality.columns), None)
    if ash_column and "RD" in quality.columns:
        sub = quality[["RD", ash_column]].dropna()
        if len(sub) >= 10:
            r = float(np.corrcoef(sub["RD"], sub[ash_column])[0, 1])
            severity = Severity.INFO if r > 0.5 else Severity.WARN
            report.add(severity, "07_quality",
                       f"korelasi RD terhadap ASH = {r:+.3f} pada {len(sub)} sampel. "
                       "Hubungan yang kuat dan positif mendukung penamaan kedua kolom itu.")


def check_hole_counts(dataset: HoleDataset, cfg: Config, report: AuditReport) -> pd.DataFrame:
    body = dataset.intervals
    body = body[~body["is_marker"]] if "is_marker" in body else body
    counts = body.groupby("seam")["hole_id"].nunique().sort_values(ascending=False)
    report.tables["seam_hole_counts"] = counts.rename("n_holes").reset_index()
    for seam, n in counts.items():
        if n < MIN_HOLES_FOR_VORONOI:
            report.add(Severity.STOP, "08_hole_count",
                       f"seam ditembus {n} lubang; Voronoi memerlukan minimal "
                       f"{MIN_HOLES_FOR_VORONOI} titik tidak segaris.",
                       seam=str(seam),
                       remedy="Tambahkan lubang, gabungkan seam, atau keluarkan dari run.")
    return counts.reset_index()


def check_collar_vs_topo(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    if dataset.topo_points is None:
        report.add(Severity.STOP, "09_topo", "topografi tidak dipasok.",
                   remedy="Isi minex.topography_file.")
        return

    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    points = dataset.topo_points
    collars = dataset.collars
    try:
        interpolator = LinearNDInterpolator(Delaunay(points[:, :2]), points[:, 2])
    except Exception as exc:
        report.add(Severity.WARN, "09_topo", f"triangulasi topografi gagal: {exc}")
        return

    sampled = interpolator(collars["east"].to_numpy(float), collars["north"].to_numpy(float))
    residual = collars["rl"].to_numpy(float) - sampled
    frame = collars[["hole_id", "rl"]].copy()
    frame["topo_rl"] = sampled
    frame["residual"] = residual
    report.tables["collar_vs_topo"] = frame

    tolerance = cfg.validation.collar_vs_topo_tolerance_m
    valid = residual[np.isfinite(residual)]
    if len(valid) >= 5:
        bias = float(np.median(valid))
        if abs(bias) > tolerance:
            report.add(
                Severity.STOP, "09_topo",
                f"median selisih RL collar - topo = {bias:+.2f} m pada {len(valid)} lubang. "
                "Offset sistematis menandakan masalah datum, bukan collar per lubang, "
                "dan ia menggeser SELURUH seam secara seragam.",
                remedy="Rekonsiliasi datum collar dan topografi.",
            )
    outside = int(np.isnan(sampled).sum())
    if outside:
        report.add(Severity.WARN, "09_topo",
                   f"{outside} collar berada di luar cakupan topografi.")
    for _, row in frame.iterrows():
        if np.isfinite(row["residual"]) and abs(row["residual"]) > tolerance:
            report.add(Severity.WARN, "09_topo",
                       f"RL collar {row['rl']:.2f} vs topo {row['topo_rl']:.2f} "
                       f"(selisih {row['residual']:+.2f} m)", hole_id=row["hole_id"])


def check_faults(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    if not dataset.faults:
        report.add(Severity.INFO, "10_faults", "tidak ada berkas sesar dipasok.")
        return
    for fault in dataset.faults:
        z = fault.points[:, 2]
        report.add(
            Severity.WARN, "10_faults",
            f"sesar '{fault.name}': {len(fault.points)} titik, Z {z.min():.1f} "
            f"sampai {z.max():.1f}. Interpolasi belum dipisah per blok sesar, "
            "sehingga offset akan tersebar mulus menjadi dip palsu.",
            remedy="Nyatakan arti kolom sesar dan tentukan apakah domain "
                   "interpolasi perlu dipisah.",
        )


def run_minex_audit(dataset: HoleDataset, cfg: Config) -> AuditReport:
    report = AuditReport()
    report.context["n_holes"] = len(dataset.collars)
    check_column_declaration(dataset, cfg, report)
    check_cross_file_coverage(dataset, report)
    check_intervals(dataset, cfg, report)
    check_stratigraphic_order(dataset, cfg, report)
    check_quality(dataset, cfg, report)
    check_hole_counts(dataset, cfg, report)
    check_collar_vs_topo(dataset, cfg, report)
    check_faults(dataset, cfg, report)
    check_geological_condition(cfg, report)
    check_rpeee(cfg, report)
    # Gerbang Tahap 2: kekeliruan yang tidak bergejala.
    run_gate2(dataset, cfg, report)
    return report
