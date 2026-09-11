"""Skema konfigurasi, divalidasi dengan pydantic.

Setiap gerbang yang disebut "stop" di spesifikasi Phase 0 ditegakkan di sini
atau di modul audit. Konfigurasi tidak punya default di dalam kode: nilai harus
ada di file YAML, sehingga apa yang dipakai selalu terlihat dan dapat ditelusuri.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError

GeologicalCondition = Literal["sederhana", "moderat", "kompleks"]
CoordinateSource = Literal["collar_ts", "bhc_gps"]
ThicknessSource = Literal["lithology", "sampling"]
CoreLossTreatment = Literal["as_coal", "as_waste", "exclude"]
RDBasis = Literal["in_situ", "air_dried", "as_received", "unknown"]
EstimationMethod = Literal["voronoi", "circular"]

JUSTIFICATION_MIN_CHARS = 100

# Nilai radius di template konfigurasi BELUM diverifikasi terhadap teks
# SNI 5015:2019. Peringatan ini dicetak pada setiap run.
RADII_UNVERIFIED_WARNING = (
    "Tabel radius klasifikasi di konfigurasi BELUM diverifikasi terhadap teks "
    "SNI 5015:2019. Angka ini tidak boleh diperlakukan sebagai otoritatif. "
    "Verifikasi ke dokumen standar sebelum hasil dipakai untuk RKAB atau laporan "
    "Competent Person."
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ClassificationRadii(_Strict):
    """Radius kumulatif dari titik observasi, dalam meter.

    Bersifat pita: measured adalah cakram dalam, indicated anulus sampai radius
    berikutnya, inferred anulus terluar.
    """

    measured: float = Field(gt=0)
    indicated: float = Field(gt=0)
    inferred: float = Field(gt=0)

    @model_validator(mode="after")
    def _ascending(self) -> "ClassificationRadii":
        if not self.measured < self.indicated < self.inferred:
            raise ValueError(
                f"radius harus menaik: measured({self.measured}) < "
                f"indicated({self.indicated}) < inferred({self.inferred})"
            )
        return self


class RadiiTable(_Strict):
    sederhana: ClassificationRadii
    moderat: ClassificationRadii
    kompleks: ClassificationRadii


class Cutoffs(_Strict):
    min_seam_thickness_m: float = Field(gt=0)
    max_parting_thickness_m: float = Field(gt=0)
    min_core_recovery_pct: float = Field(ge=0, le=100)
    min_quality_coverage_frac: float = Field(ge=0, le=1)


class RpeeeConstraints(_Strict):
    """Batasan prospek ekonomi.

    max_depth_m adalah kedalaman roof seam di bawah topografi, bukan pit shell.
    Ketiadaannya tidak menghentikan run, tapi mengubah label seluruh keluaran
    menjadi Inventori Batubara (aturan 8.4).
    """

    max_depth_m: float | None = Field(default=None, gt=0)
    max_depth_basis: str = ""
    min_cv_ar_kcal_kg: float | None = Field(default=None, gt=0)
    max_ash_adb_pct: float | None = Field(default=None, ge=0, le=100)
    excluded_area_wkt: str = ""
    excluded_area_basis: str = ""

    @model_validator(mode="after")
    def _depth_needs_basis(self) -> "RpeeeConstraints":
        # Batas kedalaman tanpa dasar yang dinyatakan lebih buruk daripada tidak
        # ada batas sama sekali, karena ia tampak dapat dipertanggungjawabkan.
        if self.max_depth_m is not None and not self.max_depth_basis.strip():
            raise ValueError(
                "max_depth_m diisi tetapi max_depth_basis kosong. Batas kedalaman "
                "tanpa dasar yang dinyatakan tidak dapat diterima."
            )
        if self.excluded_area_wkt.strip() and not self.excluded_area_basis.strip():
            raise ValueError("excluded_area_wkt diisi tetapi excluded_area_basis kosong.")
        return self

    @property
    def has_economic_constraint(self) -> bool:
        return self.max_depth_m is not None

    @property
    def resource_label(self) -> str:
        """Label yang menempel pada SETIAP keluaran (aturan 8.4)."""
        return "Sumberdaya" if self.has_economic_constraint else "Inventori Batubara"

    @property
    def class_prefix(self) -> str:
        return "Sumberdaya" if self.has_economic_constraint else "Inventori"


class MapSettings(_Strict):
    """Parameter penyajian peta kontur struktur."""

    contour_interval_m: float = Field(gt=0)
    index_contour_every: int = Field(ge=1)
    label_contours: bool = True


class Paths(_Strict):
    workbook_dir: Path
    las_dir: Path | None = None
    quality_table: Path | None = None
    topography_dxf: Path | None = None
    output_dir: Path


class Config(_Strict):
    paths: Paths

    authoritative_coordinate_source: CoordinateSource
    coal_thickness_source: ThicknessSource
    core_loss_treatment: CoreLossTreatment

    geological_condition: GeologicalCondition
    geological_condition_justification: str

    classification_radii_m: RadiiTable
    cutoffs: Cutoffs
    rpeee_constraints: RpeeeConstraints

    maps: MapSettings
    estimation_method: EstimationMethod = "voronoi"
    block_boundary_wkt: str = ""
    assumed_rd_t_per_m3: float | None = Field(default=None, gt=0)

    @field_validator("geological_condition_justification")
    @classmethod
    def _justification_substantive(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError(
                "geological_condition_justification wajib diisi. Kondisi geologi "
                "menggerakkan setiap radius klasifikasi (measured 500 m pada "
                "sederhana melawan 100 m pada kompleks); dasar yang lemah pada "
                "satu setelan ini membatalkan seluruh estimasi."
            )
        if len(text) < JUSTIFICATION_MIN_CHARS:
            raise ValueError(
                f"geological_condition_justification hanya {len(text)} karakter, "
                f"minimum {JUSTIFICATION_MIN_CHARS}. Uraikan kontinuitas seam, "
                "sesar, variasi ketebalan, dan dip."
            )
        return text

    @property
    def radii(self) -> ClassificationRadii:
        return getattr(self.classification_radii_m, self.geological_condition)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"file konfigurasi tidak ditemukan: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        try:
            cfg = cls.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(f"konfigurasi tidak valid ({path}):\n{exc}") from exc
        return cfg


def file_digest(path: Path) -> str:
    """SHA-256 file masukan, untuk log run dan ketertelusuran."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
