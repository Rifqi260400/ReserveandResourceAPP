"""Tahap 7: penilaian kompleksitas geologi (formulir Tabel 5-32).

Program MENILAI lebih dulu dari data, lalu hasilnya menentukan radius mana dari
tiga yang dipakai di tahap 8. Penilaian itu dapat ditimpa manusia, dan
penimpaannya wajib berdasar - tetapi yang ditimpa adalah SKOR PARAMETER atau
KELAS AKHIR, bukan angka radiusnya.

Penilaian otomatis di sini adalah USULAN berbasis ambang heuristik, bukan
penilaian geolog. Ambangnya dicetak bersama tiap usulan supaya dapat dibantah.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from . import radius as radius_module
from .logging_setup import get_logger

log = get_logger("complexity")

Score = Literal["sederhana", "moderat", "kompleks"]
SCORES: tuple[Score, ...] = ("sederhana", "moderat", "kompleks")
JUSTIFICATION_MIN_CHARS = 40

# Penilaian subaspek adalah CEKLIS - satu kolom ditandai, bukan angka diisi.
# Ordinal di bawah hanya alat hitung internal; ia tidak pernah jadi masukan.
ORDINAL: dict[Score, int] = {"sederhana": 1, "moderat": 2, "kompleks": 3}

# Batas pita saat nilai rata-rata dikembalikan menjadi kelas.
BOUNDARIES = (1.5, 2.5)

# Nilai yang jatuh sedekat ini ke batas pita diperlakukan sebagai mendua.
BOUNDARY_EPSILON = 1e-9

# Formulir Tabel 5-32: kelompok -> parameter -> uraian tiap skor.
FORM: dict[str, dict[str, dict[Score, str]]] = {
    "sedimentasi": {
        "variasi": {"sederhana": "sedikit variasi", "moderat": "bervariasi",
                    "kompleks": "sangat bervariasi"},
        "kesinambungan": {"sederhana": "ribuan meter", "moderat": "ratusan meter",
                          "kompleks": "puluhan meter"},
        "percabangan": {"sederhana": "hampir tidak ada", "moderat": "beberapa",
                        "kompleks": "banyak"},
    },
    "tektonik": {
        "sesar": {"sederhana": "tidak ada", "moderat": "jarang", "kompleks": "rapat"},
        "lipatan": {"sederhana": "ada, landai", "moderat": "terlipat sedang",
                    "kompleks": "terlipat kuat"},
        "intrusi": {"sederhana": "tidak ada", "moderat": "berpengaruh",
                    "kompleks": "sangat berpengaruh"},
        "kemiringan": {"sederhana": "landai", "moderat": "sedang", "kompleks": "terjal"},
    },
    "kualitas": {
        "variasi_kualitas": {"sederhana": "sedikit variasi", "moderat": "bervariasi",
                             "kompleks": "sangat bervariasi"},
    },
}

# Ambang heuristik penilaian otomatis. BUKAN isi SNI - dipakai hanya untuk
# mengusulkan, dan dicetak bersama usulannya supaya dapat dibantah.
THRESHOLDS = {
    "variasi_cv": (0.20, 0.40),            # koefisien variasi tebal
    "kesinambungan_m": (1000.0, 100.0),    # bentang menerus: >1000 m, >100 m
    "percabangan_n": (0, 2),               # jumlah seam yang memecah
    "sesar_per_km2": (0.0, 0.5),           # rapat jejak sesar
    "lipatan_residual_m": (2.0, 8.0),      # simpangan roof dari bidang datar
    "kemiringan_deg": (15.0, 30.0),        # dip rata-rata
    "kualitas_cv": (0.15, 0.35),           # koefisien variasi abu
}


@dataclass
class Suggestion:
    parameter: str
    group: str
    score: Score | None            # None = tidak dapat dinilai dari data
    evidence: str
    threshold: str

    @property
    def assessable(self) -> bool:
        return self.score is not None


@dataclass
class Entry:
    """Satu baris formulir yang sudah diputuskan."""

    parameter: str
    group: str
    score: Score
    justification: str
    source: Literal["otomatis", "manual"] = "otomatis"


class TiedAssessment(ValueError):
    """Skor seri. Tidak boleh diputus kode - lihat tally_and_warn."""


@dataclass
class Assessment:
    entries: list[Entry]
    suggestions: list[Suggestion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tally: dict[Score, int] = field(default_factory=dict)
    aspect_values: dict[str, float] = field(default_factory=dict)
    final_value: float = float("nan")
    condition: Score | None = None
    margin: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"aspek": e.group, "subaspek": e.parameter, "skor": e.score,
             "uraian": FORM[e.group][e.parameter][e.score],
             "sumber": e.source, "justifikasi": e.justification}
            for e in self.entries
        ])

    def to_checklist_frame(self) -> pd.DataFrame:
        """Bentuk Tabel 5-32: satu kolom diceklis per subaspek."""
        rows = []
        for entry in self.entries:
            row = {"aspek": entry.group, "subaspek": entry.parameter}
            for score in SCORES:
                row[score] = "v" if entry.score == score else ""
            rows.append(row)
        for group, value in self.aspect_values.items():
            rows.append({"aspek": group, "subaspek": f"-- rata-rata {group} --",
                         **{s: "" for s in SCORES},
                         "nilai": round(value, 3), "kelas": classify(value)})
        rows.append({"aspek": "", "subaspek": "== rata-rata tiga aspek ==",
                     **{s: "" for s in SCORES},
                     "nilai": round(self.final_value, 3),
                     "kelas": self.condition or ""})
        return pd.DataFrame(rows)


def classify(value: float) -> Score:
    """Kembalikan nilai rata-rata menjadi kelas."""
    low, high = BOUNDARIES
    return "sederhana" if value < low else "moderat" if value < high else "kompleks"


def aspect_value(entries: list[Entry], group: str) -> float:
    """Rata-rata ordinal subaspek dalam satu aspek."""
    scores = [ORDINAL[e.score] for e in entries if e.group == group]
    if not scores:
        raise ValueError(f"aspek '{group}' tidak punya subaspek yang dinilai.")
    return float(np.mean(scores))


def _bucket(value: float, low: float, high: float, *, ascending: bool) -> Score:
    """Petakan nilai ke skor. `ascending` = makin besar makin kompleks."""
    if ascending:
        return "sederhana" if value <= low else "moderat" if value <= high else "kompleks"
    return "sederhana" if value >= low else "moderat" if value >= high else "kompleks"


def _dip_and_fold(intervals: pd.DataFrame, collars: pd.DataFrame) -> tuple[float, float]:
    """Dip rata-rata (derajat) dan simpangan roof dari bidang datar (meter).

    Bidang datar dicocokkan ke RL roof tiap seam. Gradiennya memberi dip;
    sisaannya memberi ukuran seberapa terlipat permukaannya.
    """
    dips, residuals = [], []
    coords = collars.set_index("hole_id")
    for _, group in intervals.groupby("seam"):
        rows = []
        for _, row in group.iterrows():
            hole = coords.loc[row["hole_id"]] if row["hole_id"] in coords.index else None
            if hole is None:
                continue
            rows.append((float(hole["east"]), float(hole["north"]),
                         float(hole["rl"]) - float(row["depth_from"])))
        if len(rows) < 4:
            continue
        arr = np.asarray(rows, float)
        design = np.column_stack([arr[:, 0], arr[:, 1], np.ones(len(arr))])
        coeff, *_ = np.linalg.lstsq(design, arr[:, 2], rcond=None)
        dips.append(float(np.degrees(np.arctan(np.hypot(coeff[0], coeff[1])))))
        residuals.append(float(np.std(arr[:, 2] - design @ coeff)))
    return (float(np.mean(dips)) if dips else float("nan"),
            float(np.mean(residuals)) if residuals else float("nan"))


def suggest(dataset, cfg) -> list[Suggestion]:
    """Usulkan skor tiap parameter dari data. Tujuh dari delapan dapat dinilai."""
    body = dataset.intervals
    body = body[~body["is_marker"]] if "is_marker" in body else body
    collars = dataset.collars
    out: list[Suggestion] = []

    def add(parameter, group, score, evidence, threshold):
        out.append(Suggestion(parameter, group, score, evidence, threshold))

    # 1. Variasi ketebalan.
    low, high = THRESHOLDS["variasi_cv"]
    thickness = body.assign(t=body["depth_to"] - body["depth_from"])
    per_seam = thickness.groupby("seam")["t"]
    cv = float((per_seam.std() / per_seam.mean()).mean())
    add("variasi", "sedimentasi", _bucket(cv, low, high, ascending=True),
        f"koefisien variasi tebal rata-rata {cv:.2f} pada {body['seam'].nunique()} seam",
        f"<={low:.2f} sederhana, <={high:.2f} moderat")

    # 2. Kesinambungan lateral: bentang seam paling menerus.
    low, high = THRESHOLDS["kesinambungan_m"]
    coords = collars.set_index("hole_id")[["east", "north"]]
    spans = []
    for seam, group in body.groupby("seam"):
        points = coords.reindex(group["hole_id"].unique()).dropna().to_numpy(float)
        if len(points) >= 2:
            spans.append(float(np.hypot(*(points.max(0) - points.min(0)))))
    span = max(spans) if spans else 0.0
    add("kesinambungan", "sedimentasi", _bucket(span, low, high, ascending=False),
        f"bentang seam terluas {span:.0f} m",
        f">={low:.0f} m sederhana, >={high:.0f} m moderat")

    # 3. Percabangan: seam yang memecah (terdeteksi dari alias atau seam_splits).
    low, high = THRESHOLDS["percabangan_n"]
    splits = len(cfg.seam_splits) + len(getattr(cfg.minex, "seam_aliases", {}) or {})
    add("percabangan", "sedimentasi", _bucket(splits, low, high, ascending=True),
        f"{splits} seam tercatat memecah atau dinamai ulang akibat percabangan",
        f"<={low} sederhana, <={high} moderat")

    # 4. Sesar: rapat jejak per km2.
    low, high = THRESHOLDS["sesar_per_km2"]
    points = coords.dropna().to_numpy(float)
    area_km2 = float(np.prod(points.max(0) - points.min(0))) / 1e6 if len(points) else 0.0
    density = len(dataset.faults) / area_km2 if area_km2 > 0 else 0.0
    add("sesar", "tektonik", _bucket(density, low, high, ascending=True),
        f"{len(dataset.faults)} jejak sesar pada {area_km2:.2f} km2 "
        f"({density:.2f} per km2)",
        f"<={low:.1f} sederhana, <={high:.1f} moderat")

    dip, fold = _dip_and_fold(body, collars)

    # 5. Lipatan: simpangan roof dari bidang datar.
    low, high = THRESHOLDS["lipatan_residual_m"]
    add("lipatan", "tektonik",
        _bucket(fold, low, high, ascending=True) if np.isfinite(fold) else None,
        f"simpangan baku roof terhadap bidang datar {fold:.2f} m"
        if np.isfinite(fold) else "tidak dapat dinilai",
        f"<={low:.0f} m sederhana, <={high:.0f} m moderat")

    # 6. Intrusi: TIDAK dapat dinilai - berkas lit hanya memuat interval seam.
    add("intrusi", "tektonik", None,
        "tidak dapat dinilai dari data: berkas litologi hanya memuat interval "
        "seam, tanpa kode batuan beku",
        "wajib dinyatakan manusia")

    # 7. Kemiringan.
    low, high = THRESHOLDS["kemiringan_deg"]
    add("kemiringan", "tektonik",
        _bucket(dip, low, high, ascending=True) if np.isfinite(dip) else None,
        f"dip rata-rata {dip:.1f} derajat dari pencocokan bidang datar"
        if np.isfinite(dip) else "tidak dapat dinilai",
        f"<={low:.0f} derajat landai, <={high:.0f} derajat sedang")

    # 8. Variasi kualitas: koefisien variasi abu.
    low, high = THRESHOLDS["kualitas_cv"]
    quality = dataset.quality
    ash = next((c for c in ("ASH", "ASH_adb") if quality is not None
                and c in quality.columns), None)
    if ash:
        values = quality[ash].dropna()
        qcv = float(values.std() / values.mean()) if len(values) > 2 else float("nan")
        add("variasi_kualitas", "kualitas",
            _bucket(qcv, low, high, ascending=True) if np.isfinite(qcv) else None,
            f"koefisien variasi abu {qcv:.2f} pada {len(values)} sampel",
            f"<={low:.2f} sederhana, <={high:.2f} moderat")
    else:
        add("variasi_kualitas", "kualitas", None,
            "tidak dapat dinilai: kolom abu tidak dikenali", "wajib dinyatakan manusia")
    return out


def tally_and_warn(entries: list[Entry]) -> Assessment:
    """Hitung kelas akhir dan cetak dua peringatan WAJIB."""
    tally = {score: sum(1 for e in entries if e.score == score) for score in SCORES}
    values = {group: aspect_value(entries, group) for group in FORM}
    final = float(np.mean(list(values.values())))

    # Nilai yang jatuh PERSIS di batas pita mendua. Membulatkannya di dalam kode
    # berarti memilihkan kelas - dan pembulatan ke bawah selalu memilih kelas
    # dengan radius lebih besar, arah yang menaikkan sumberdaya tanpa dasar.
    for boundary in BOUNDARIES:
        if abs(final - boundary) <= BOUNDARY_EPSILON:
            below, above = classify(boundary - 1.0), classify(boundary + 1.0)
            raise TiedAssessment(
                f"rata-rata tiga aspek jatuh PERSIS di batas pita ({final:.3f}). "
                f"Kelasnya mendua antara '{below}' dan '{above}', dan kode tidak "
                "memilihkan: membulatkan ke bawah selalu memenangkan radius yang "
                "lebih besar. Selisihnya nyata - radius terukur "
                f"{radius_module.radius_m(below, 'terukur'):.0f} m lawan "
                f"{radius_module.radius_m(above, 'terukur'):.0f} m. Putuskan "
                "dengan condition_override beserta alasannya, atau tinjau ulang "
                "ceklis subaspek yang masih dapat dibantah."
            )

    condition = classify(final)
    margin = min(abs(final - boundary) for boundary in BOUNDARIES)

    counts = {group: len(params) for group, params in FORM.items()}
    warnings = [
        # 1. Cara pembobotan, dan bedanya dari hitungan mentah di laporan rujukan.
        "PERINGATAN WAJIB - cara pembobotan: nilai dihitung dua tingkat. "
        "Subaspek dirata-ratakan menjadi nilai aspek, lalu ketiga aspek "
        f"dirata-ratakan ({', '.join(f'{g}={values[g]:.3f}' for g in FORM)} "
        f"-> {final:.3f}). Cara ini memberi bobot SAMA kepada ketiga aspek "
        f"meski jumlah subaspeknya timpang ({', '.join(f'{g} {n}' for g, n in counts.items())}), "
        "sehingga Variasi Kualitas yang hanya satu subaspek tetap bernilai "
        "sepertiga. Perhatikan bahwa Tabel 5-32 pada laporan rujukan justru "
        "MENJUMLAH ceklis mentah (6 lawan 2); kedua cara itu dapat menghasilkan "
        "kelas yang berbeda.",
        # 2. Margin tepi jurang.
        f"PERINGATAN WAJIB - margin tepi jurang: nilai akhir {final:.3f}, kelas "
        f"'{condition}', jarak ke batas pita terdekat {margin:.3f}. "
        + ("Margin ini TIPIS: satu ceklis subaspek yang berubah dapat membalik "
           "kelas." if margin < 0.2 else
           "Margin ini cukup lebar terhadap satu perubahan ceklis.")
        + " Kelas menggerakkan SELURUH radius klasifikasi, jadi tiap ceklis "
        "subaspek berdampak pada angka sumberdaya akhir.",
    ]
    for line in warnings:
        log.warning(line)
    return Assessment(entries=entries, warnings=warnings, tally=tally,
                      aspect_values=values, final_value=final,
                      condition=condition, margin=margin)


def assess(dataset, cfg, overrides: dict[str, tuple[Score, str]] | None = None,
           condition_override: tuple[Score, str] | None = None) -> Assessment:
    """Nilai kompleksitas dari data, dengan penimpaan manual opsional.

    `overrides` berbentuk {parameter: (skor, justifikasi)}. Parameter yang tidak
    dapat dinilai otomatis WAJIB ada di sini, dan tiap justifikasi wajib
    sekurangnya JUSTIFICATION_MIN_CHARS karakter.
    """
    overrides = overrides or {}
    suggestions = suggest(dataset, cfg)
    groups = {parameter: group for group, params in FORM.items() for parameter in params}

    entries, missing, thin = [], [], []
    for suggestion in suggestions:
        if suggestion.parameter in overrides:
            score, justification = overrides[suggestion.parameter]
            if len(justification.strip()) < JUSTIFICATION_MIN_CHARS:
                thin.append(suggestion.parameter)
            entries.append(Entry(suggestion.parameter, suggestion.group, score,
                                 justification.strip(), "manual"))
        elif suggestion.assessable:
            entries.append(Entry(
                suggestion.parameter, suggestion.group, suggestion.score,
                f"Usulan otomatis. Bukti: {suggestion.evidence}. "
                f"Ambang: {suggestion.threshold}.", "otomatis"))
        else:
            missing.append(suggestion.parameter)

    if missing:
        raise ValueError(
            f"parameter {missing} tidak dapat dinilai dari data dan belum "
            "dinyatakan manusia. Memberi nilai bawaan berarti menebak kondisi "
            "geologi, dan kondisi geologi menggerakkan seluruh radius."
        )
    if thin:
        raise ValueError(
            f"justifikasi terlalu pendek untuk {thin}: minimum "
            f"{JUSTIFICATION_MIN_CHARS} karakter."
        )

    try:
        assessment = tally_and_warn(entries)
    except TiedAssessment:
        if condition_override is None:
            raise
        values = {group: aspect_value(entries, group) for group in FORM}
        assessment = Assessment(
            entries=entries,
            tally={score: sum(1 for e in entries if e.score == score) for score in SCORES},
            aspect_values=values, final_value=float(np.mean(list(values.values()))))
    if condition_override is not None:
        chosen, reason = condition_override
        if len(reason.strip()) < JUSTIFICATION_MIN_CHARS:
            raise ValueError(
                f"alasan condition_override terlalu pendek: minimum "
                f"{JUSTIFICATION_MIN_CHARS} karakter.")
        if assessment.condition is not None and assessment.condition != chosen:
            assessment.warnings.append(
                f"PERINGATAN WAJIB - kelas ditimpa manusia: penilaian "
                f"menghasilkan '{assessment.condition}', dipakai '{chosen}'. "
                f"Radius terukur berubah "
                f"{radius_module.radius_m(assessment.condition, 'terukur'):.0f} m -> "
                f"{radius_module.radius_m(chosen, 'terukur'):.0f} m. Alasan: {reason.strip()}")
        else:
            assessment.warnings.append(
                f"PERINGATAN WAJIB - seri diputus manusia menjadi '{chosen}'. "
                f"Alasan: {reason.strip()}")
        assessment.condition = chosen
        assessment.margin = 0.0
        for line in assessment.warnings[-1:]:
            log.warning(line)
    assessment.suggestions = suggestions
    for parameter in overrides:
        if parameter not in groups:
            raise ValueError(f"parameter tidak dikenal di penimpaan: '{parameter}'")
    return assessment
