"""Gerbang Tahap 2: pemeriksaan yang menghentikan run sebelum pemodelan.

Berbeda dari pemeriksaan bentuk data di `minex_checks`, yang di sini menyasar
kekeliruan yang TIDAK bergejala: tiap satunya membaca wajar per baris, per
lubang, dan per berkas, tapi menggeser tonase akhir puluhan persen. Karena tidak
bergejala, tidak satu pun boleh diselesaikan dengan default di dalam kode -
semuanya menuntut keputusan manusia yang tercatat di konfigurasi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..io.minex import HoleDataset
from ..logging_setup import get_logger
from .checks import AuditReport, Severity

log = get_logger("audit.gate2")

# Pada basis adb, abu mengencerkan kalori, jadi korelasinya kuat dan negatif.
# Pada basis daf abu sudah dikeluarkan, jadi korelasinya lemah.
ASH_CV_STRONG = 0.70
ASH_CV_WEAK = 0.50
MIN_SAMPLES_FOR_CORRELATION = 10

# Nilai BOW yang berulang pada mayoritas lubang adalah konstanta, bukan bacaan.
WEATHERING_CONSTANT_FRAC = 0.60


def _key_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ("hole_id", "seam", "depth_from", "depth_to") if c in frame.columns]


def check_seam_collision(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    """Tabrakan induk-anak seam.

    Deteksinya bukan "satu lubang memuat A dan A1 sekaligus" - itu justru kasus
    yang mudah, dan pada data ini tidak pernah terjadi. Yang berbahaya adalah
    sebaran: A di satu kelompok lubang, A1/A2 di kelompok lain, saling lepas.
    Tesselasi terpisah untuk keduanya masing-masing menutup seluruh area.
    """
    body = dataset.intervals
    body = body[~body["is_marker"]] if "is_marker" in body else body
    present = set(body["seam"])
    rows = []
    collided = False

    for parent, children in cfg.seam_splits.items():
        child_set = set(children) & present
        if parent not in present or not child_set:
            continue
        collided = True
        per_hole = body.groupby("hole_id")["seam"].apply(set)
        parent_holes = {h for h, s in per_hole.items() if parent in s}
        child_holes = {h for h, s in per_hole.items() if s & child_set}
        both = parent_holes & child_holes

        px = dataset.collars.set_index("hole_id")[["east", "north"]]
        overlap_area_ha = _overlap_hull_ha(px, parent_holes, child_holes)
        declared = cfg.seam_policy.collision_resolution != "undeclared"

        rows.append({
            "induk": parent, "anak": ", ".join(sorted(child_set)),
            "lubang_induk": len(parent_holes), "lubang_anak": len(child_holes),
            "lubang_keduanya": len(both),
            "luas_hull_bertumpang_ha": round(overlap_area_ha, 1),
        })
        report.add(
            Severity.INFO if declared else Severity.STOP, "G1_seam_collision",
            f"seam induk '{parent}' hadir di {len(parent_holes)} lubang dan anaknya "
            f"{sorted(child_set)} di {len(child_holes)} lubang, dengan "
            f"{len(both)} lubang memuat keduanya. Hull keduanya bertumpang "
            f"{overlap_area_ha:.1f} ha. Tanpa keputusan, '{parent}' dan "
            f"{sorted(child_set)} akan menghasilkan dua tesselasi yang sama-sama "
            "menutup area itu dan tonasenya terhitung DUA KALI. Tidak ada gejala "
            "per lubang: tiap lubang tampak konsisten.",
            seam=parent,
            remedy="Isi seam_policy.collision_resolution: 'merge_children' "
                   "(A1+A2 dikembalikan menjadi A), 'split_parent' (A dipecah "
                   "mengikuti pola tetangga), atau 'treat_as_distinct' (keduanya "
                   "seam berbeda yang tidak menempati ruang yang sama). Sertakan "
                   "collision_basis.",
            data={"parent": parent, "children": sorted(child_set)},
        )

    if rows:
        report.tables["seam_collision"] = pd.DataFrame(rows)

    if not collided:
        report.add(Severity.INFO, "G1_seam_collision",
                   "tidak ada seam induk yang hadir bersama anaknya dalam satu dataset.")
        return

    resolution = cfg.seam_policy.collision_resolution
    if resolution == "undeclared":
        report.add(
            Severity.STOP, "G1_seam_collision",
            "seam_policy.collision_resolution = 'undeclared'. Kode tidak boleh "
            "memilihkan: ketiga pilihan sah secara geologi dan menghasilkan "
            "tonase yang berbeda jauh.",
            remedy="Nyatakan penyelesaiannya di konfigurasi.",
        )
    else:
        report.add(Severity.INFO, "G1_seam_collision",
                   f"penyelesaian dinyatakan: '{resolution}'. "
                   f"Dasar: {cfg.seam_policy.collision_basis[:200]}")


def _overlap_hull_ha(coords: pd.DataFrame, a: set[str], b: set[str]) -> float:
    """Luas tumpang tindih convex hull dua kelompok lubang, dalam hektar."""
    try:
        from shapely.geometry import MultiPoint
    except Exception:
        return float("nan")
    def hull(holes):
        pts = coords.reindex(sorted(holes)).dropna()
        if len(pts) < 3:
            return None
        return MultiPoint(pts.to_numpy(float)).convex_hull
    ha, hb = hull(a), hull(b)
    if ha is None or hb is None:
        return float("nan")
    return ha.intersection(hb).area / 10_000.0


def check_quality_basis(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    """Basis tiap kolom kualitas, diuji terhadap bukti abu-kalori."""
    quality = dataset.quality
    if quality is None:
        return

    declared = cfg.minex.quality_column_basis
    reported = [c for c in cfg.minex.quality_columns
                if c not in ("hole_id", "seam", "depth_from", "depth_to")]
    rows = [{"kolom": c, "basis_dinyatakan": declared.get(c, "unknown")} for c in reported]
    report.tables["quality_basis"] = pd.DataFrame(rows)

    undeclared = [c for c in reported if declared.get(c, "unknown") == "unknown"]
    if undeclared:
        report.add(
            Severity.STOP, "G2_quality_basis",
            f"basis belum dinyatakan untuk {len(undeclared)} kolom kualitas: "
            f"{undeclared}. Basis tidak terbaca dari angkanya - abu 6,5% wajar "
            "pada adb maupun daf - dan ia menentukan apakah nilai itu boleh "
            "dibandingkan dengan cutoff, dirata-rata bersama, atau dilaporkan "
            "apa adanya.",
            remedy="Isi minex.quality_column_basis untuk tiap kolom.",
        )

    ash = next((c for c in ("ASH", "ASH_adb") if c in quality.columns), None)
    cv = next((c for c in ("CV", "CV_adb") if c in quality.columns), None)
    moisture = next((c for c in ("MOISTURE", "M_adb", "IM") if c in quality.columns), None)
    if not (ash and cv and moisture):
        report.add(Severity.WARN, "G2_quality_basis",
                   "uji korelasi abu-kalori dilewati: kolom ASH, CV, atau "
                   "moisture tidak dikenali.")
        return

    sub = quality[[ash, cv, moisture]].dropna()
    if len(sub) < MIN_SAMPLES_FOR_CORRELATION:
        report.add(Severity.WARN, "G2_quality_basis",
                   f"uji korelasi abu-kalori dilewati: hanya {len(sub)} sampel lengkap.")
        return

    r_raw = float(np.corrcoef(sub[ash], sub[cv])[0, 1])
    # Bila kolom CV sebenarnya daf, maka nilai adb-nya = CV x (100 - M - ASH)/100.
    dry_ash_free = (100.0 - sub[moisture] - sub[ash]) / 100.0
    valid = dry_ash_free > 0
    r_as_daf = float(np.corrcoef(sub.loc[valid, ash],
                                 (sub.loc[valid, cv] * dry_ash_free[valid]))[0, 1])
    declared_cv = declared.get(cv, "unknown")

    report.tables["ash_cv_test"] = pd.DataFrame([{
        "n_sampel": len(sub), "basis_dinyatakan": declared_cv,
        "r(ASH, CV apa adanya)": round(r_raw, 4),
        "r(ASH, CV jika daf -> adb)": round(r_as_daf, 4),
    }])

    verdict = (
        "daf" if (abs(r_raw) < ASH_CV_WEAK and abs(r_as_daf) > ASH_CV_STRONG and r_as_daf < 0)
        else "adb" if (r_raw < -ASH_CV_STRONG)
        else "tidak konklusif"
    )
    message = (
        f"korelasi abu terhadap kalori pada {len(sub)} sampel: apa adanya "
        f"r = {r_raw:+.3f}; setelah kolom itu diperlakukan sebagai daf lalu "
        f"dikonversi ke adb, r = {r_as_daf:+.3f}. Pada basis adb abu mengencerkan "
        "kalori sehingga korelasinya kuat dan negatif; pada daf abu sudah "
        f"dikeluarkan sehingga korelasinya lemah. Bukti menunjuk basis '{verdict}'."
    )

    if verdict == "tidak konklusif":
        report.add(Severity.WARN, "G2_quality_basis", message,
                   remedy="Konfirmasi basis ke sertifikat laboratorium.")
    elif declared_cv == "unknown":
        report.add(Severity.STOP, "G2_quality_basis",
                   message + f" Basis kolom '{cv}' belum dinyatakan di konfigurasi.",
                   remedy=f"Nyatakan minex.quality_column_basis['{cv}'].")
    elif declared_cv != verdict:
        override = cfg.minex.quality_basis_override_basis.strip()
        report.add(
            Severity.WARN if override else Severity.STOP, "G2_quality_basis",
            message + f" Ini BERTENTANGAN dengan basis yang dinyatakan "
            f"('{declared_cv}'). Melaporkan nilai daf sebagai adb "
            "melebih-lebihkan kalori jual, dan rata-rata tertimbangnya tidak sah."
            + (f" DITIMPA oleh keputusan manusia: {override[:300]}" if override
               else ""),
            remedy=f"Perbaiki minex.quality_column_basis['{cv}'] menjadi "
                   f"'{verdict}', tunjukkan sertifikat yang menyanggah, atau isi "
                   "minex.quality_basis_override_basis.",
        )
    else:
        report.add(Severity.INFO, "G2_quality_basis",
                   message + f" Sesuai dengan basis yang dinyatakan ('{declared_cv}').")


def check_proximate_closure(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    """M + ASH + VM + FC harus menutup 100%, pada basis apa pun.

    Pemeriksaan ini TIDAK bergantung pada basis. Apa pun jawabannya untuk adb
    melawan daf, analisis proksimat pada satu basis selalu berjumlah 100%. Bila
    tidak, salah satu kolom bukan yang tertulis di namanya - dan itu pertanyaan
    yang harus dijawab lebih dulu daripada pertanyaan basis.
    """
    quality = dataset.quality
    if quality is None:
        return
    names = {"moisture": ("MOISTURE", "M_adb", "IM"), "ash": ("ASH", "ASH_adb"),
             "vm": ("VM", "VM_adb"), "fc": ("FC", "FC_adb")}
    picked = {}
    for role, options in names.items():
        column = next((c for c in options if c in quality.columns), None)
        if column is None:
            report.add(Severity.WARN, "G7_proximate",
                       f"uji penutupan proksimat dilewati: kolom {role} tidak dikenali.")
            return
        picked[role] = column

    total = quality[list(picked.values())].sum(axis=1, min_count=4).dropna()
    if total.empty:
        return
    gap = 100.0 - total
    report.tables["proximate_closure"] = pd.DataFrame([{
        "kolom": " + ".join(picked.values()), "n_sampel": len(total),
        "jumlah_median_pct": round(float(total.median()), 2),
        "jumlah_min_pct": round(float(total.min()), 2),
        "jumlah_maks_pct": round(float(total.max()), 2),
        "kekurangan_median_pct": round(float(gap.median()), 2),
    }])

    tolerance = cfg.validation.mass_balance_tolerance_pct
    off = (total - 100.0).abs() > tolerance
    if not off.any():
        report.add(Severity.INFO, "G7_proximate",
                   f"proksimat menutup pada {len(total)} sampel "
                   f"(median {total.median():.2f}%).")
        return

    waiver = cfg.validation.proximate_closure_waiver_basis.strip()
    report.add(
        Severity.WARN if waiver else Severity.STOP, "G7_proximate",
        f"{' + '.join(picked.values())} berjumlah median {total.median():.2f}% "
        f"pada {len(total)} sampel, bukan 100% - kekurangan median "
        f"{gap.median():.2f}%. Seluruh {int(off.sum())} sampel meleset, dan "
        "kekurangan yang SERAGAM seperti ini bukan galat per sampel: ia berarti "
        "salah satu kolom bukan yang tertulis di namanya, atau ada komponen yang "
        "tidak ikut terbaca. Pertanyaan ini mendahului pertanyaan basis - selama "
        "proksimat tidak menutup, konversi basis apa pun bertumpu pada kolom yang "
        "belum tentu benar."
        + (f" DITIMPA oleh keputusan manusia: {waiver[:300]}" if waiver else ""),
        remedy="Konfirmasi arti kolom 6-9 ke sertifikat laboratorium. Bila data "
               "ini sintetis dan memang tidak konsisten secara fisik, nyatakan itu "
               "agar tidak dibaca sebagai temuan.",
    )


def check_duplicate_records(dataset: HoleDataset, report: AuditReport) -> None:
    """Duplikat yang identik dilaporkan; duplikat yang BERBEDA menghentikan run."""
    frames = {"survey": dataset.collars, "lithology": dataset.intervals}
    if dataset.quality is not None:
        frames["quality"] = dataset.quality

    rows = []
    for label, frame in frames.items():
        removed = dataset.provenance.get("duplicate_rows_removed", {}).get(label, 0)
        keys = _key_columns(frame) or (["hole_id"] if "hole_id" in frame else [])
        if label == "survey":
            keys = ["hole_id"]
        conflicting = 0
        if keys:
            grouped = frame.groupby(keys, dropna=False)
            for key, group in grouped:
                if len(group) > 1 and len(group.drop_duplicates()) > 1:
                    conflicting += 1
                    report.add(
                        Severity.STOP, "G3_duplicates",
                        f"{label}: kunci {dict(zip(keys, key if isinstance(key, tuple) else (key,)))} "
                        f"muncul {len(group)} kali dengan nilai BERBEDA. Duplikat "
                        "identik dapat dibuang; duplikat yang bertentangan tidak - "
                        "memilih salah satunya berarti menebak mana yang benar.",
                        hole_id=str(group.iloc[0].get("hole_id", "")),
                        remedy="Rekonsiliasi di sumber data.",
                    )
        rows.append({"berkas": label, "baris_akhir": len(frame),
                     "duplikat_identik_dibuang": removed,
                     "kunci_bertentangan": conflicting})
        if removed:
            report.add(
                Severity.WARN, "G3_duplicates",
                f"{label}: {removed} baris duplikat identik dibuang saat pembacaan "
                f"({removed + len(frame)} -> {len(frame)}). Dibiarkan, tiap duplikat "
                "melipatgandakan sumbangan lubang itu tanpa gejala.",
                remedy="Periksa proses ekspor yang menghasilkan berkas ini.",
            )
    report.tables["duplicate_summary"] = pd.DataFrame(rows)


def check_weathering(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    """Asal angka zona pelapukan."""
    spec = cfg.weathering
    body = dataset.intervals
    if "is_marker" not in body:
        report.add(Severity.STOP, "G4_weathering",
                   "baris penanda pelapukan tidak dikenali; marker_seams kosong.",
                   remedy=f"Tambahkan '{spec.marker_seam}' ke minex.marker_seams.")
        return

    markers = body[body["is_marker"] &
                   (body["seam"].str.upper() == spec.marker_seam.strip().upper())]
    holes = set(dataset.collars["hole_id"])
    with_marker = set(markers["hole_id"])
    missing = sorted(holes - with_marker)

    if markers.empty:
        report.add(Severity.STOP, "G4_weathering",
                   f"tidak ada satu pun baris penanda '{spec.marker_seam}'.",
                   remedy="Pasok kedalaman pelapukan atau isi weathering.constant_depth_m.")
        return

    depths = markers["depth_to"].astype(float)
    counts = depths.value_counts()
    top_value = float(counts.index[0])
    top_frac = float(counts.iloc[0]) / len(depths)
    report.tables["weathering_depths"] = (
        counts.rename("n_lubang").rename_axis("kedalaman_m").reset_index()
    )

    if missing:
        if spec.constant_depth_m is None:
            report.add(
                Severity.STOP, "G4_weathering",
                f"{len(missing)} lubang tanpa baris '{spec.marker_seam}' dan "
                "weathering.constant_depth_m kosong. Memakai nol berarti "
                "menganggap seluruh batubara di lubang itu segar - asumsi paling "
                "murah hati yang tersedia.",
                remedy="Pasok bacaannya, atau nyatakan konstanta beserta dasarnya.",
                data={"holes": missing[:20]},
            )
        else:
            report.add(Severity.WARN, "G4_weathering",
                       f"{len(missing)} lubang memakai konstanta "
                       f"{spec.constant_depth_m} m karena tidak punya bacaan.",
                       data={"holes": missing[:20]})

    if top_frac >= WEATHERING_CONSTANT_FRAC and spec.provenance != "assumed":
        report.add(
            Severity.STOP if spec.provenance == "unknown" else Severity.WARN,
            "G4_weathering",
            f"kedalaman {top_value:g} m berulang pada {counts.iloc[0]} dari "
            f"{len(depths)} lubang ({100 * top_frac:.0f}%). Sebaran seperti ini "
            "adalah ciri konstanta yang diketikkan, bukan pembacaan log per "
            f"lubang. Provenance dinyatakan '{spec.provenance}'. Bila sebenarnya "
            "asumsi, ia wajib muncul sebagai asumsi pemodelan pada keluaran, "
            "bukan sebagai data terukur.",
            remedy="Nyatakan weathering.provenance = 'measured' (dengan bukti log) "
                   "atau 'assumed' (dengan dasarnya).",
        )
    elif spec.provenance == "unknown":
        report.add(Severity.STOP, "G4_weathering",
                   "weathering.provenance = 'unknown'.",
                   remedy="Nyatakan 'measured' atau 'assumed' beserta dasarnya.")
    else:
        report.add(Severity.INFO, "G4_weathering",
                   f"provenance '{spec.provenance}'; kedalaman {depths.min():g}-"
                   f"{depths.max():g} m pada {len(depths)} lubang.")


def check_barren_holes(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    """Lubang tanpa batubara adalah bukti ketiadaan, bukan ketiadaan bukti."""
    body = dataset.intervals
    body = body[~body["is_marker"]] if "is_marker" in body else body
    collars = dataset.collars.set_index("hole_id")
    barren = sorted(set(collars.index) - set(body["hole_id"]))
    if not barren:
        report.add(Severity.INFO, "G5_barren", "tidak ada lubang tanpa interseksi seam.")
        return

    rows = []
    for hole in barren:
        td = float(collars.loc[hole].get("total_depth", float("nan")))
        rows.append({"hole_id": hole, "total_depth_m": td,
                     "east": float(collars.loc[hole]["east"]),
                     "north": float(collars.loc[hole]["north"])})
        report.add(
            Severity.WARN, "G5_barren",
            f"lubang menembus {td:.1f} m tanpa satu pun interseksi seam. Ini BUKTI "
            "KETIADAAN, dan harus ikut membatasi model: bila roof hasil pemodelan "
            f"jatuh di atas {td:.1f} m pada titik ini, model bertentangan dengan "
            "pengamatan. Lubang tanpa interval kerap hilang begitu saja dari "
            "interpolasi, sehingga seam dimodelkan menembusnya.",
            hole_id=hole,
            remedy="Bawa lubang ini ke tahap validasi model sebagai batasan.",
        )
    report.tables["barren_holes"] = pd.DataFrame(rows)


def check_hole_populations(dataset: HoleDataset, cfg: Config, report: AuditReport) -> None:
    """Dua populasi lubang: seluruh lubang, dan lubang yang punya kualitas.

    Klasifikasi bersandar pada titik observasi. Bila titik observasi menuntut
    kualitas, populasinya menyusut drastis dan plafon kelas ikut turun. Kedua
    hasil ditunjukkan agar pilihannya dibuat sadar.
    """
    if dataset.quality is None:
        return
    try:
        from shapely.geometry import MultiPoint, Point
        from shapely.ops import unary_union
    except Exception:
        report.add(Severity.WARN, "G6_populations", "shapely tidak tersedia.")
        return

    body = dataset.intervals
    body = body[~body["is_marker"]] if "is_marker" in body else body
    coords = dataset.collars.set_index("hole_id")[["east", "north"]]
    # Kualitas diukur PER SEAM, bukan per lubang. Sebuah lubang yang punya
    # kualitas pada seam B tidak menjadikannya titik observasi untuk seam A1.
    # Mengukurnya per lubang melebih-lebihkan populasi secara besar-besaran.
    quality_pairs = set(zip(dataset.quality["hole_id"], dataset.quality["seam"]))
    quality_holes = set(dataset.quality["hole_id"])
    radii = cfg.radii

    rows = []
    for seam, group in body.groupby("seam"):
        seam_holes = set(group["hole_id"])
        extent = coords.reindex(sorted(seam_holes)).dropna()
        if len(extent) < 3:
            continue
        hull = MultiPoint(extent.to_numpy(float)).convex_hull
        with_quality = {h for h in seam_holes if (h, seam) in quality_pairs}
        for label, population in (("seluruh lubang", seam_holes),
                                  ("lubang berkualitas", with_quality)):
            points = coords.reindex(sorted(population)).dropna()
            entry = {"seam": seam, "populasi": label, "n_lubang": len(points)}
            if len(points) == 0:
                entry.update({"terukur_ha": 0.0, "tertunjuk_ha": 0.0, "tereka_ha": 0.0})
                rows.append(entry)
                continue
            geoms = [Point(xy) for xy in points.to_numpy(float)]
            discs = {}
            for name, radius in (("terukur", radii.measured),
                                 ("tertunjuk", radii.indicated),
                                 ("tereka", radii.inferred)):
                discs[name] = unary_union([g.buffer(radius, quad_segs=32)
                                           for g in geoms]).intersection(hull)
            entry["terukur_ha"] = round(discs["terukur"].area / 1e4, 1)
            entry["tertunjuk_ha"] = round(
                discs["tertunjuk"].difference(discs["terukur"]).area / 1e4, 1)
            entry["tereka_ha"] = round(
                discs["tereka"].difference(discs["tertunjuk"]).area / 1e4, 1)
            rows.append(entry)

    frame = pd.DataFrame(rows)
    report.tables["hole_populations"] = frame

    spec = cfg.observation_point
    all_holes = len(set(dataset.collars["hole_id"]))
    n_quality = len(quality_holes & set(dataset.collars["hole_id"]))
    n_pairs = len(body.groupby(["hole_id", "seam"]).size())
    n_pairs_quality = sum(1 for key in body.groupby(["hole_id", "seam"]).groups
                          if key in quality_pairs)
    totals = frame.groupby("populasi")[["terukur_ha", "tertunjuk_ha", "tereka_ha"]].sum()
    measured_all = float(totals.loc["seluruh lubang", "terukur_ha"])
    measured_q = float(totals.loc["lubang berkualitas", "terukur_ha"])
    shrink = 100.0 * (1 - measured_q / measured_all) if measured_all else float("nan")

    report.add(
        Severity.STOP if spec.requires_quality is None else Severity.INFO,
        "G6_populations",
        f"dua populasi titik observasi: {all_holes} lubang seluruhnya, "
        f"{n_quality} di antaranya punya kualitas ({100 * n_quality / max(all_holes, 1):.0f}%). "
        f"Diukur PER SEAM - dan itu yang berlaku - hanya {n_pairs_quality} dari "
        f"{n_pairs} pasangan lubang-seam yang punya kualitas "
        f"({100 * n_pairs_quality / max(n_pairs, 1):.0f}%): sebuah lubang yang "
        "punya kualitas pada satu seam tidak menjadikannya titik observasi bagi "
        "seam lain di lubang yang sama. "
        f"Luas Terukur - DIJUMLAHKAN antar seam, jadi bukan luas bidang datar - "
        f"turun dari {measured_all:.0f} ha menjadi {measured_q:.0f} ha "
        f"({shrink:.0f}% lebih kecil) bila titik observasi menuntut kualitas. "
        "SNI menuntut ketebalan DAN kualitas pada titik observasi, jadi memakai "
        "seluruh lubang menaikkan kelas atas dasar geometri semata."
        + ("" if spec.requires_quality is None else
           f" Dinyatakan: titik observasi "
           f"{'MENUNTUT' if spec.requires_quality else 'TIDAK menuntut'} kualitas."),
        remedy="Nyatakan observation_point.requires_quality beserta basisnya, dan "
               "laporkan yang mana pada keluaran. Lihat tabel hole_populations.",
    )


def run_gate2(dataset: HoleDataset, cfg: Config, report: AuditReport) -> AuditReport:
    check_seam_collision(dataset, cfg, report)
    check_quality_basis(dataset, cfg, report)
    check_proximate_closure(dataset, cfg, report)
    check_duplicate_records(dataset, report)
    check_weathering(dataset, cfg, report)
    check_barren_holes(dataset, cfg, report)
    check_hole_populations(dataset, cfg, report)
    return report
