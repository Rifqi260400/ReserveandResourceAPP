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
InputFormat = Literal["bgg_workbook", "minex_flat"]

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


class MinexSpec(_Strict):
    """Berkas flat Minex dan arti kolomnya.

    Berkas ini tidak berheader, jadi arti kolom WAJIB dinyatakan di sini. Salah
    urut kolom tidak akan memunculkan kesalahan apa pun - tebal dan kualitas
    tetap terbaca sebagai angka yang wajar.
    """

    survey_file: Path
    lithology_file: Path
    quality_file: Path | None = None
    topography_file: Path | None = None
    faults_file: Path | None = None

    survey_columns: list[str]
    lithology_columns: list[str]
    quality_columns: list[str] = Field(default_factory=list)
    fault_columns: list[str] = Field(default_factory=list)

    # Baris berketebalan nol yang menandai horizon, bukan seam (mis. 'W').
    marker_seams: list[str] = Field(default_factory=list)

    # Basis RD dinyatakan di sini karena berkasnya tidak berheader. Ia tidak
    # boleh disimpulkan dari nilainya.
    quality_rd_basis: RDBasis = "unknown"

    # Basis kolom moisture. Sumber data kerap hanya menulis "Moisture" tanpa
    # menyatakan IM atau TM. Ketika quality_rd_basis = 'in_situ', ambiguitas ini
    # TIDAK menyentuh tonase - konversi densitas tidak diperlukan - dan hanya
    # menentukan label basis pada kualitas yang dilaporkan.
    quality_moisture_basis: Literal["adb", "ar", "unknown"] = "unknown"

    # Satuan nilai kalori. kcal/kg dan cal/g bernilai sama secara numerik;
    # dinyatakan agar tidak ada yang mengalikan 1000 di kemudian hari.
    quality_cv_unit: Literal["kcal/kg", "cal/g", "MJ/kg"] = "kcal/kg"

    # Interval kualitas terbalik (to <= from) adalah cacat data. Perbaikannya
    # bukan urusan kode - menukar from dan to akan menebak niat penulisnya.
    #   stop    : hentikan run (bawaan)
    #   exclude : keluarkan baris itu dan catat, sisanya tetap diproses
    on_invalid_quality_interval: Literal["stop", "exclude"] = "stop"

    @model_validator(mode="after")
    def _quality_needs_columns(self) -> "MinexSpec":
        if self.quality_file is not None and not self.quality_columns:
            raise ValueError("quality_file diisi tetapi quality_columns kosong.")
        if self.faults_file is not None and not self.fault_columns:
            raise ValueError("faults_file diisi tetapi fault_columns kosong.")
        return self


class ValidationSettings(_Strict):
    collar_vs_topo_tolerance_m: float = Field(gt=0)
    mass_balance_tolerance_pct: float = Field(gt=0)


class DxfExportSettings(_Strict):
    """Ekspor kontur struktur seam ke DXF."""

    enabled: bool = True
    contour_interval_m: float = Field(default=2.0, gt=0)
    index_every: int = Field(default=5, ge=1)
    # Nama layer. {seam} dan {surface} diisi, mis. 'Seam A Roof'.
    layer_template: str = "Seam {seam} {surface}"
    index_suffix: str = " Index"
    # Satu berkas per seam (roof dan floor menyatu), selain berkas gabungan.
    per_seam_files: bool = True
    # Satu berkas per PERMUKAAN: 'Seam A Roof.dxf' dan 'Seam A Floor.dxf'
    # terpisah. Ini yang biasanya diharapkan saat memuat satu permukaan ke CAD.
    per_surface_files: bool = True
    include_boreholes: bool = True
    include_subcrop: bool = True
    # Kontur dipotong oleh subcrop: di luar itu seam berada di atas topografi
    # dan batubaranya sudah tererosi.
    clip_to_subcrop: bool = True


class MapSettings(_Strict):
    """Parameter penyajian peta kontur struktur."""

    contour_interval_m: float = Field(gt=0)
    index_contour_every: int = Field(ge=1)
    label_contours: bool = True
    dxf_export: DxfExportSettings = Field(default_factory=DxfExportSettings)


class Paths(_Strict):
    workbook_dir: Path
    las_dir: Path | None = None
    quality_table: Path | None = None
    topography_dxf: Path | None = None
    output_dir: Path


class Config(_Strict):
    paths: Paths
    input_format: InputFormat = "bgg_workbook"
    minex: MinexSpec | None = None

    authoritative_coordinate_source: CoordinateSource
    coal_thickness_source: ThicknessSource
    core_loss_treatment: CoreLossTreatment

    geological_condition: GeologicalCondition
    geological_condition_justification: str

    # Urutan stratigrafi dari MUDA (atas) ke TUA (bawah).
    stratigraphy: list[str] = Field(default_factory=list)
    # Seam induk yang terpecah menjadi beberapa anak, mis. {"A": ["A1", "A2"]}.
    # Anak mewarisi kedudukan stratigrafi induknya dan diurutkan sesuai daftar.
    seam_splits: dict[str, list[str]] = Field(default_factory=dict)

    classification_radii_m: RadiiTable
    cutoffs: Cutoffs
    validation: ValidationSettings
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

    @model_validator(mode="after")
    def _minex_spec_required(self) -> "Config":
        if self.input_format == "minex_flat" and self.minex is None:
            raise ValueError("input_format='minex_flat' menuntut blok 'minex'.")
        return self

    def stratigraphic_rank(self) -> dict[str, tuple[int, int]]:
        """Peringkat (induk, anak) tiap seam. Peringkat kecil = lebih muda."""
        rank: dict[str, tuple[int, int]] = {}
        for index, parent in enumerate(self.stratigraphy):
            rank[parent] = (index, 0)
            for child_index, child in enumerate(self.seam_splits.get(parent, [])):
                rank[child] = (index, child_index)
        return rank

    def parent_seam(self, seam: str) -> str:
        for parent, children in self.seam_splits.items():
            if seam in children:
                return parent
        return seam

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
