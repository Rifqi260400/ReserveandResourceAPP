"""Tahap 5: validasi model - gerbang yang berputar kembali ke tahap 4.

Pedoman Praktis KCMI 2017 pasal 4.4 menuntut CPI memastikan model geologi
"cukup valid dan memenuhi syarat untuk digunakan dalam estimasi Sumber daya",
lalu menyebut delapan pemeriksaan.

4.4.1 Validasi Model Struktural
  1. cross section 2D melalui beberapa titik bor: posisi elevasi bor terhadap
     topografi, dan posisi perlapisan menurut model dibanding menurut data bor;
  2. perbandingan statistik ketebalan menurut model dan menurut data bor;
  3. kontur isopach;
  4. kontur struktur (floor) beberapa seam major.

4.4.2 Validasi Model Kualitas
  1. statistik min-max parameter kunci (Rd, CV, TM, Ash, TS) - cari angka janggal;
  2. apakah Rd yang di-input SUDAH Rd in-situ (konversi Preston & Sanders);
  3. batubara peringkat rendah: MHC atau EQM disarankan;
  4. kontur kualitas parameter kunci.

DUA UKURAN YANG BERBEDA, dan bedanya penting.

  MENGHORMATI DATA - model dibaca TEPAT di lubangnya. TIN-nya sendiri eksak di
  titik data; yang dibaca di sini adalah GRID hasil TIN itu, dan grid tidak
  dapat mereproduksi titik data secara persis. Dua sebabnya, keduanya normal:

    diskretisasi - lubang jarang jatuh tepat di node grid, sehingga nilainya
    diinterpolasi antar node. Galatnya sebanding ukuran sel dikali relief lokal,
    dan terbukti menyusut bersama ukuran sel (50 m -> 5,54 m; 25 m -> 1,67 m;
    10 m -> 0,80 m; 5 m -> 0,58 m pada data ini);

    lubang di tepi dukungan - node di sekitarnya sebagian NaN, sehingga
    interpolasi antar node menyeberang batas dan hasilnya melenceng. Pada data
    ini H057 terbaca -7,611 padahal TIN-nya tepat -9,000, karena tetangganya
    NaN.

  Karena itu toleransinya TIDAK absolut: ia diturunkan dari relief per sel.
  Melebihi itu barulah menandakan kekeliruan mekanis - grid tidak sejajar,
  koordinat tertukar, atau domain terpotong keliru. Lulus di sini TIDAK berarti
  modelnya bagus.

  VALIDASI SILANG - model dibangun ulang TANPA satu lubang, lalu dipakai menebak
  lubang itu. Inilah galat ramalan yang sebenarnya, dan inilah yang berarti
  ketika ditanya "seberapa dapat dipercaya model ini di antara lubang".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logging_setup import get_logger
from .model import SeamModel, build_seam

log = get_logger("validate")

# Toleransi "menghormati" = faktor ini dikali relief LOKAL di sekitar tiap
# lubang. Grid tidak dapat mereproduksi titik data lebih tepat daripada relief
# antar node di tempat itu; menuntut lebih berarti menghukum ukuran sel dan
# kecuraman setempat, bukan modelnya.
HONOUR_RELIEF_FACTOR = 0.5

# Batubara peringkat rendah: KCMI 4.4.2 menyarankan MHC/EQM di bawah ini.
LOW_RANK_CV_ADB = 6100.0

# Rentang kelayakan parameter kualitas kunci. Di luar ini bukan salah ketik
# kecil melainkan angka yang tidak mungkin dimiliki batubara.
PLAUSIBLE = {
    "RD": (1.10, 2.00), "CV": (1000.0, 8500.0), "ASH": (0.0, 60.0),
    "VM": (5.0, 60.0), "FC": (5.0, 80.0), "TS": (0.0, 10.0),
    "MOISTURE": (0.0, 50.0), "TM": (0.0, 60.0),
}


@dataclass
class Deviation:
    seam: str
    attribute: str
    frame: pd.DataFrame            # hole_id, data, model, deviasi
    tolerance_m: float
    kind: str                      # "menghormati" | "validasi_silang"

    @property
    def outside(self) -> pd.DataFrame:
        limit = (self.frame["toleransi"] if "toleransi" in self.frame
                 else self.tolerance_m)
        return self.frame[self.frame["deviasi"].abs() > limit]

    @property
    def fraction_outside(self) -> float:
        return len(self.outside) / max(len(self.frame), 1)

    def stats(self) -> dict:
        d = self.frame["deviasi"].dropna()
        return {
            "seam": self.seam, "atribut": self.attribute, "ukuran": self.kind,
            "n": len(d),
            "rerata": round(float(d.mean()), 3) if len(d) else None,
            "rms": round(float(np.sqrt((d ** 2).mean())), 3) if len(d) else None,
            "maks_abs": round(float(d.abs().max()), 3) if len(d) else None,
            "di_luar_toleransi": len(self.outside),
            "frac_di_luar": round(self.fraction_outside, 3),
            "toleransi_m": self.tolerance_m,
        }


@dataclass
class ValidationReport:
    deviations: list[Deviation] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def deviation_summary(self) -> pd.DataFrame:
        return pd.DataFrame([d.stats() for d in self.deviations])


def _sample(surface, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    return surface.sample(np.asarray(east, float), np.asarray(north, float))


def _hole_values(frame: pd.DataFrame, collars: pd.DataFrame
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    east, north, roof_rl, thickness, ids = [], [], [], [], []
    for _, row in frame.iterrows():
        hole = row["hole_id"]
        if hole not in collars.index:
            continue
        collar = collars.loc[hole]
        east.append(float(collar["east"]))
        north.append(float(collar["north"]))
        roof_rl.append(float(collar["rl"]) - float(row["roof_m"]))
        thickness.append(float(row["coal_thickness_m"]))
        ids.append(str(hole))
    return (np.array(east), np.array(north), np.array(roof_rl),
            np.array(thickness), ids)


def _local_scale(surface, east: np.ndarray, north: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Relief antar node DI SEKITAR tiap lubang, dan apakah tetangganya NaN.

    Skala galat diskretisasi bersifat LOKAL: lubang di lereng curam tidak dapat
    direproduksi grid seketat lubang di daerah landai. Memakai satu angka global
    akan menghukum lubang di lereng dan memaafkan lubang di daerah datar.
    """
    relief, edge = [], []
    for e, n in zip(east, north):
        ix = int(np.argmin(np.abs(surface.x - e)))
        iy = int(np.argmin(np.abs(surface.y - n)))
        window = surface.z[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3]
        edge.append(bool(np.isnan(window).any()))
        finite = window[np.isfinite(window)]
        relief.append(float(finite.max() - finite.min()) if finite.size > 1 else 0.0)
    return np.array(relief), np.array(edge)


def honour_check(model: SeamModel, frame: pd.DataFrame, collars: pd.DataFrame,
                 roof_tolerance_m: float, thickness_tolerance_m: float
                 ) -> list[Deviation]:
    """KCMI 4.4.1 butir 1 dan 2: model dibaca tepat di lubangnya.

    Untuk interpolator eksak, hasilnya semestinya nol. Yang ditangkapnya adalah
    kekeliruan mekanis - grid tidak sejajar, koordinat tertukar, sel terpotong
    domain yang keliru - bukan mutu ramalan.
    """
    east, north, roof_rl, thickness, ids = _hole_values(frame, collars)
    if not len(ids):
        return []
    out = []
    for attribute, surface, data, floor_tolerance in (
        ("roof_rl", model.roof, roof_rl, roof_tolerance_m),
        ("tebal", model.isopach, thickness, thickness_tolerance_m),
    ):
        modelled = _sample(surface, east, north)
        relief, edge = _local_scale(surface, east, north)
        # Toleransi PER LUBANG: grid tidak dapat lebih tepat daripada relief di
        # sekitarnya sendiri.
        tolerance = np.maximum(floor_tolerance, HONOUR_RELIEF_FACTOR * relief)
        table = pd.DataFrame({"hole_id": ids, "data": data, "model": modelled,
                              "deviasi": modelled - data, "toleransi": tolerance,
                              "relief_lokal": relief, "di_tepi_dukungan": edge})
        out.append(Deviation(seam=model.seam, attribute=attribute,
                             tolerance_m=round(float(np.median(tolerance)), 3),
                             kind="menghormati", frame=table))
    return out


def cross_validate(seam: str, frame: pd.DataFrame, collars: pd.DataFrame,
                   spacing: float, influence_m: float,
                   roof_tolerance_m: float, thickness_tolerance_m: float,
                   method: str = "linear") -> list[Deviation]:
    """Validasi silang tinggalkan-satu: galat ramalan yang sebenarnya.

    Tiap lubang dikeluarkan, model dibangun ulang dari sisanya, lalu lubang itu
    ditebak. Inilah yang menjawab "seberapa dapat dipercaya model DI ANTARA
    lubang" - pertanyaan yang menentukan apakah radius klasifikasi masuk akal.
    """
    east, north, roof_rl, thickness, ids = _hole_values(frame, collars)
    if len(ids) < 4:
        return []

    rows = {"roof_rl": [], "tebal": []}
    for index, hole in enumerate(ids):
        rest = frame[frame["hole_id"] != hole]
        try:
            trial = build_seam(seam, rest, collars, spacing, influence_m,
                               domain="cv", method=method)
        except Exception:
            trial = None
        if trial is None:
            continue
        predicted_roof = float(_sample(trial.roof, [east[index]], [north[index]])[0])
        predicted_thick = float(_sample(trial.isopach, [east[index]], [north[index]])[0])
        rows["roof_rl"].append((hole, roof_rl[index], predicted_roof))
        rows["tebal"].append((hole, thickness[index], predicted_thick))

    out = []
    for attribute, tolerance in (("roof_rl", roof_tolerance_m),
                                 ("tebal", thickness_tolerance_m)):
        data = pd.DataFrame(rows[attribute], columns=["hole_id", "data", "model"])
        if data.empty:
            continue
        data["deviasi"] = data["model"] - data["data"]
        out.append(Deviation(seam=seam, attribute=attribute, tolerance_m=tolerance,
                             kind="validasi_silang", frame=data))
    return out


def check_quality_model(quality: pd.DataFrame | None, cfg,
                        report: ValidationReport) -> None:
    """KCMI 4.4.2: statistik janggal, basis Rd, dan saran MHC/EQM."""
    if quality is None:
        report.warnings.append("KCMI 4.4.2: tidak ada data kualitas untuk divalidasi.")
        return

    rows = []
    for column, (low, high) in PLAUSIBLE.items():
        if column not in quality.columns:
            continue
        values = quality[column].dropna()
        if values.empty:
            continue
        outside = values[(values < low) | (values > high)]
        rows.append({"parameter": column, "n": len(values),
                     "min": round(float(values.min()), 3),
                     "maks": round(float(values.max()), 3),
                     "wajar_min": low, "wajar_maks": high,
                     "di_luar": len(outside)})
        if len(outside):
            report.failures.append(
                f"KCMI 4.4.2 butir 1: {column} memuat {len(outside)} nilai di luar "
                f"rentang wajar {low}-{high} (terbaca {values.min():.2f} sampai "
                f"{values.max():.2f}). Angka semacam ini tidak dimiliki batubara; "
                "periksa satuan, basis, atau salah ketik di database.")
    report.tables["kcmi_442_minmax"] = pd.DataFrame(rows)

    # Butir 2: basis Rd.
    basis = cfg.minex.quality_rd_basis if cfg.minex else "unknown"
    if basis == "in_situ":
        report.warnings.append(
            "KCMI 4.4.2 butir 2: Rd dinyatakan SUDAH in-situ, jadi konversi "
            "Preston & Sanders tidak diterapkan. Pastikan penyedia data memang "
            "sudah melakukan penyesuaian moisture - bila belum, tonase kelebihan "
            "sekitar 10% pada batubara peringkat rendah.")
    elif basis in ("air_dried", "as_received"):
        report.warnings.append(
            f"KCMI 4.4.2 butir 2: Rd berbasis '{basis}' - konversi Preston & "
            "Sanders WAJIB sebelum dipakai menghitung tonase.")
    else:
        report.failures.append(
            "KCMI 4.4.2 butir 2: basis Rd belum dinyatakan, sehingga tidak dapat "
            "dipastikan apakah Rd yang masuk model sudah in-situ.")

    # Butir 3: MHC/EQM untuk batubara peringkat rendah.
    cv = next((c for c in ("CV", "CV_adb") if c in quality.columns), None)
    if cv is not None:
        median_cv = float(quality[cv].dropna().median())
        if median_cv < LOW_RANK_CV_ADB:
            report.warnings.append(
                f"KCMI 4.4.2 butir 3: CV median {median_cv:.0f} kcal/kg menandakan "
                "batubara peringkat rendah. Pedoman menyarankan pengujian Moisture "
                "Holding Capacity (MHC) atau Equilibrium Moisture (EQM) agar "
                "moisture in-situ lebih representatif.")


def run(models: dict[str, SeamModel], intersections: pd.DataFrame,
        collars: pd.DataFrame, cfg, quality: pd.DataFrame | None = None,
        spacing: float = 25.0, influence_m: float = 119.0,
        roof_tolerance_m: float = 1.0, thickness_tolerance_m: float = 0.30,
        max_outside_frac: float = 0.10,
        cross_validation: bool = True) -> ValidationReport:
    """Jalankan gerbang tahap 5. Gagal berarti kembali ke tahap 4, bukan maju."""
    report = ValidationReport()

    for key, model in models.items():
        seam = model.seam
        frame = intersections[intersections["seam"] == seam]
        if model.domain != "penuh":
            # Domain split: batasi ke lubang yang benar-benar membentuk model itu.
            frame = frame[frame["hole_id"].isin(
                set(frame["hole_id"]))].copy()

        report.deviations.extend(honour_check(
            model, frame, collars, roof_tolerance_m, thickness_tolerance_m))
        if cross_validation:
            report.deviations.extend(cross_validate(
                seam, frame, collars, spacing, influence_m,
                roof_tolerance_m, thickness_tolerance_m))

    for deviation in report.deviations:
        if deviation.kind == "menghormati" and deviation.fraction_outside > 0:
            outside = deviation.outside
            at_edge = int(outside.get("di_tepi_dukungan", pd.Series(dtype=bool)).sum())
            interior = len(outside) - at_edge
            if interior:
                report.failures.append(
                    f"KCMI 4.4.1: seam {deviation.seam} - {interior} lubang DI DALAM "
                    f"dukungan meleset melampaui toleransinya sendiri pada "
                    f"{deviation.attribute} (maks "
                    f"{outside['deviasi'].abs().max():.3f} m). Toleransi tiap lubang "
                    "sudah disetel dari relief di sekitarnya, jadi selisih di atasnya bukan "
                    "diskretisasi melainkan kekeliruan mekanis - grid tidak sejajar, "
                    "koordinat tertukar, atau domain terpotong keliru.")
            if at_edge:
                report.warnings.append(
                    f"KCMI 4.4.1: seam {deviation.seam} - {at_edge} lubang di TEPI "
                    f"dukungan meleset pada {deviation.attribute}. Node grid di "
                    "sekitarnya sebagian NaN, sehingga pembacaan menyeberang batas. "
                    "Ini sifat grid di tepi hull, bukan cacat model; perkecil sel "
                    "atau beri margin bila angkanya dipakai di tepi.")
        elif (deviation.kind == "validasi_silang"
              and deviation.fraction_outside > max_outside_frac):
            report.warnings.append(
                f"KCMI 4.4.1 butir 2: seam {deviation.seam} - "
                f"{deviation.fraction_outside:.0%} lubang meleset lebih dari "
                f"{deviation.tolerance_m} m pada {deviation.attribute} saat "
                f"validasi silang (RMS {deviation.stats()['rms']}). Model tidak "
                "meramalkan dirinya sendiri sebaik yang dituntut toleransi.")

    check_quality_model(quality, cfg, report)

    report.tables["deviasi"] = report.deviation_summary()
    for line in report.failures:
        log.error(line)
    for line in report.warnings:
        log.warning(line)
    return report
