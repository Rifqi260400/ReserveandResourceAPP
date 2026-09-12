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

# Basis pelaporan kualitas. 'unknown' bukan nilai netral: ia menghentikan run.
QualityBasis = Literal["adb", "ar", "db", "daf", "dmmf", "unknown"]

# Penyelesaian tabrakan induk-anak seam. 'undeclared' menghentikan run.
#   merge_children    : anak digabung kembali menjadi induk (A1+A2 -> A)
#   split_parent      : induk dipecah mengikuti pola anak di lubang tetangga
#   treat_as_distinct : induk dan anak adalah seam berbeda yang tidak bertumpuk
SeamCollisionResolution = Literal[
    "undeclared", "merge_children", "split_parent", "treat_as_distinct"
]

# Asal angka zona pelapukan.
WeatheringProvenance = Literal["measured", "assumed", "unknown"]

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

    # Catatan: label sebenarnya dibaca lewat Config.resource_label, yang juga
    # menghormati posisi "tanpa batas kedalaman" yang dinyatakan di limits.depth.

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

    # Penggantian nama seam saat pembacaan, {nama_di_berkas: nama_dipakai}.
    # Dipakai ketika penamaan di berkas tidak mencerminkan korelasi yang
    # disepakati - misalnya seam yang menyatu dilog 'A' sementara di tempat
    # lain ia memecah menjadi 'A1' dan 'A2'; menamai yang menyatu sebagai 'A1'
    # membuat keduanya satu seam yang sama, bukan dua seam yang bertabrakan.
    # Penggantian dicatat di provenance dan ikut ke keluaran.
    seam_aliases: dict[str, str] = Field(default_factory=dict)

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

    # Basis per kolom kualitas. Kunci = nama kolom di quality_columns.
    # Kolom yang tidak terdaftar dianggap 'unknown' dan menghentikan run.
    # Basis TIDAK boleh disimpulkan dari nilainya oleh kode; tetapi bila basis
    # yang dinyatakan bertentangan dengan bukti korelasi abu-kalori, run
    # dihentikan - lihat audit G2.
    quality_column_basis: dict[str, QualityBasis] = Field(default_factory=dict)

    # Ketika basis yang dinyatakan bertentangan dengan bukti korelasi atau
    # kelayakan fisik, run dihentikan. Diisi, string ini MENIMPA penghentian itu
    # - tetapi pertentangannya tetap dicetak pada audit dan ikut ke keluaran,
    # sehingga keputusannya terlihat, bukan hilang.
    quality_basis_override_basis: str = ""

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


class SeamPolicy(_Strict):
    """Kebijakan seam yang tidak boleh ditebak oleh kode.

    Ketika seam induk 'A' muncul di sebagian lubang dan anaknya 'A1'/'A2' di
    lubang lain - tanpa satu pun lubang memuat keduanya - tidak ada gejala per
    lubang. Yang terjadi justru di bidang datar: satu tesselasi untuk A dan satu
    lagi untuk A1/A2 sama-sama menutup SELURUH area, dan tonasenya terhitung dua
    kali. Ini overestimasi diam terbesar yang tersedia pada data semacam ini,
    jadi penyelesaiannya wajib dinyatakan manusia.
    """

    collision_resolution: SeamCollisionResolution = "undeclared"
    collision_basis: str = ""

    @model_validator(mode="after")
    def _resolution_needs_basis(self) -> "SeamPolicy":
        if self.collision_resolution != "undeclared" and not self.collision_basis.strip():
            raise ValueError(
                "seam_policy.collision_resolution dinyatakan tetapi "
                "collision_basis kosong. Pilihan ini menggerakkan tonase "
                "puluhan persen; dasarnya wajib tercatat."
            )
        return self


class LegalStatus(_Strict):
    """KCMI 4.6.1: legalitas. Tanpa ini Sumber daya tidak dapat dilaporkan."""

    permit_type: Literal["IUP", "IUPK", "PKP2B", "KK", "lainnya"] | None = None
    permit_valid_until: str = ""
    permit_covers_mine_life: bool | None = None
    cnc_certified: bool | None = None
    # KCMI 4.6.1.3: data bor dari kawasan hutan wajib didukung IPPKH eksplorasi.
    exploration_in_forest_area: bool | None = None
    ippkh_exploration_held: bool | None = None


class LandStatus(_Strict):
    """KCMI 4.6.2: lahan dan tata ruang.

    Pedoman menyatakan CPI TIDAK DAPAT melaporkan Sumber daya tambang terbuka
    di Hutan Lindung, area konservasi, atau area lain yang terlarang untuk
    penambangan. Ini larangan, bukan pertimbangan.
    """

    forest_category: Literal["APL", "hutan_produksi", "hutan_lindung",
                             "konservasi", "campuran", "tidak_diketahui"] | None = None
    # Poligon area terlarang (WKT). Dikeluarkan dari pelaporan.
    prohibited_area_wkt: str = ""
    prohibited_area_basis: str = ""
    rtrw_allows_mining: bool | None = None
    rtrw_basis: str = ""


class MineabilityCutoffs(_Strict):
    """KCMI 4.6.3.1: cut off parameters Sumber daya."""

    min_mineable_thickness_m: float | None = Field(default=None, gt=0)
    max_ash_pct: float | None = Field(default=None, ge=0, le=100)
    max_total_sulphur_pct: float | None = Field(default=None, ge=0)
    max_total_moisture_pct: float | None = Field(default=None, ge=0, le=100)
    min_calorific_value: float | None = Field(default=None, gt=0)
    apply_weathering_depth: bool = True


class DepthLimit(_Strict):
    """KCMI 4.6.3.2: batas maksimum kedalaman.

    Pedoman menyebut tiga acuan yang sah: kedalaman pit pada BESR yang
    diperdalam, surface hasil pit optimisasi pada harga tertinggi yang pernah
    tercapai, atau rekomendasi studi geoteknik. Angka tanpa salah satu acuan itu
    bukan batas ekonomi - ia tebakan yang tampak berwibawa.
    """

    max_depth_m: float | None = Field(default=None, gt=0)
    basis_kind: Literal["besr_diperdalam", "pit_optimisasi_harga_tertinggi",
                        "geoteknik", "lainnya"] | None = None
    basis_note: str = ""

    # Tidak memakai batas kedalaman adalah posisi yang SAH - KCMI 4.6.3.2
    # menulis "dapat menggunakan", bukan "wajib". Klasifikasi memang tidak
    # bergantung kedalaman; ia bergantung jarak dari titik pengamatan di bidang
    # X-Y. Tetapi keprospekan beralasan (4.6) tetap harus ditunjukkan, jadi
    # ketiadaan batas wajib DINYATAKAN beserta alasannya - bukan sekadar kosong.
    # Diisi, keluaran tetap berlabel Sumber daya; dikosongkan, ia Inventori.
    no_depth_limit_basis: str = ""

    @model_validator(mode="after")
    def _depth_needs_a_recognised_basis(self) -> "DepthLimit":
        if self.max_depth_m is not None and self.no_depth_limit_basis.strip():
            raise ValueError(
                "limits.depth: max_depth_m dan no_depth_limit_basis tidak boleh "
                "diisi bersamaan - pilih satu posisi.")
        if self.no_depth_limit_basis.strip() and len(
                self.no_depth_limit_basis.strip()) < 60:
            raise ValueError(
                "limits.depth.no_depth_limit_basis terlalu pendek (minimum 60 "
                "karakter). Ketiadaan batas kedalaman adalah klaim keprospekan "
                "beralasan dan harus beralasan.")
        if self.max_depth_m is not None:
            if self.basis_kind is None:
                raise ValueError(
                    "limits.depth.max_depth_m diisi tetapi basis_kind kosong. "
                    "KCMI 4.6.3.2 menyebut acuan yang sah: BESR diperdalam, pit "
                    "optimisasi pada harga tertinggi, atau studi geoteknik.")
            if len(self.basis_note.strip()) < 40:
                raise ValueError(
                    "limits.depth.basis_note terlalu pendek (minimum 40 karakter).")
        return self


class Limits(_Strict):
    """Tahap 9: seluruh batas KCMI 4.6 dalam satu blok."""

    legal: LegalStatus = Field(default_factory=LegalStatus)
    land: LandStatus = Field(default_factory=LandStatus)
    cutoffs: MineabilityCutoffs = Field(default_factory=MineabilityCutoffs)
    depth: DepthLimit = Field(default_factory=DepthLimit)
    # Keputusan proyek yang tercatat: null berarti batas IUP TIDAK diterapkan.
    iup_boundary_wkt: str | None = None
    iup_boundary_basis: str = ""
    apply_subcrop: bool = True


class PoOSpec(_Strict):
    """Kriteria Titik Pengamatan menurut Pedoman Praktis KCMI 2017 pasal 4.5.2.

    Ketiga butirnya wajib. Butir (a) metoda survey dan butir (b) jenis lubang
    serta logging geofisika TIDAK ada di berkas flat, jadi keduanya dinyatakan
    di sini. Selama belum dinyatakan, tidak ada lubang yang lolos: menganggap
    terpenuhi berarti mengklaim mutu data yang belum pernah diperiksa.
    """

    # 4.5.2 (a) - total station atau GPS geodetik. Selain itu tidak memenuhi.
    survey_method: Literal["total_station", "gps_geodetik", "handheld_gps",
                           "tidak_diketahui"] | None = None

    # 4.5.2 (b) - full coring, atau logging geofisika WAJIB bila bukan.
    all_holes_full_cored: bool | None = None
    hole_types: dict[str, Literal["full_coring", "open_hole", "touch_coring"]] | None = None
    all_holes_geophysically_logged: bool | None = None
    geophysically_logged_holes: list[str] = Field(default_factory=list)

    # 4.5.2 (c) - keterwakilan sampel. None = tidak diperiksa (data tidak ada).
    min_coal_recovery_pct: float | None = Field(default=None, ge=0, le=100)

    # 4.5.5 - "kemenerusan pada dua arah" tidak diberi angka oleh pedoman.
    #   radius_provides_dip  : titik boleh berjajar searah strike; jangkauan
    #                          arah dip datang dari RADIUS kelasnya sendiri,
    #                          yang memang menyapu ke segala arah. Ini posisi
    #                          pemilik data.
    #   require_offset_point : menuntut titik fisik yang bergeser searah dip,
    #                          mengikuti bunyi harfiah KCMI 4.5.4.
    two_direction_policy: Literal["radius_provides_dip",
                                  "require_offset_point"] = "radius_provides_dip"
    two_direction_basis: str = ""

    # 4.5.3 - jarak antar PoO ditentukan per seam dari variabilitasnya.
    # Geostatistik disarankan bila data >= 30 (Journel & Huijbregts 1978);
    # di bawah itu, pendekatan kompleksitas geologi SNI 5015:2019 jadi cadangan.
    spacing_method: Literal["geostatistik", "sni_kompleksitas", "otomatis"] = "otomatis"
    geostatistics_min_data: int = Field(default=30, ge=3)


class ObservationPointSpec(_Strict):
    """Apa yang membuat sebuah lubang menjadi TITIK OBSERVASI.

    SNI menuntut ketebalan DAN kualitas pada titik observasi. Bila kualitas
    tidak dituntut, populasi titik membengkak dan kelas naik atas dasar geometri
    semata. Karena keduanya dipakai di lapangan, pilihannya wajib dinyatakan -
    dan ikut tercetak pada keluaran.
    """

    requires_quality: bool | None = None
    basis: str = ""

    @model_validator(mode="after")
    def _choice_needs_basis(self) -> "ObservationPointSpec":
        if self.requires_quality is not None and not self.basis.strip():
            raise ValueError(
                "observation_point.requires_quality dinyatakan tetapi basis kosong."
            )
        return self


class WeatheringSpec(_Strict):
    """Zona pelapukan (base of weathering).

    Batubara di atas BOW tidak dapat dijual. Bila angkanya konstanta yang
    diasumsikan, bukan hasil pembacaan log, itu asumsi pemodelan - bukan data -
    dan wajib muncul sebagai asumsi pada keluaran.
    """

    marker_seam: str = "W"
    provenance: WeatheringProvenance = "unknown"
    provenance_basis: str = ""
    # Dipakai HANYA untuk lubang yang tidak punya baris penanda.
    constant_depth_m: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _provenance_needs_basis(self) -> "WeatheringSpec":
        if self.provenance != "unknown" and not self.provenance_basis.strip():
            raise ValueError(
                "weathering.provenance dinyatakan tetapi provenance_basis kosong."
            )
        return self


class ValidationSettings(_Strict):
    collar_vs_topo_tolerance_m: float = Field(gt=0)
    mass_balance_tolerance_pct: float = Field(gt=0)

    # Proksimat yang tidak menutup 100% menghentikan run. Diisi, string ini
    # menimpa penghentian itu - dipakai ketika datanya memang sintetis dan
    # ketidakkonsistenannya sudah diketahui. Temuannya tetap dicetak.
    proximate_closure_waiver_basis: str = ""


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


class GrdExportSettings(_Strict):
    """Ekspor grid Surfer untuk dikonsumsi perangkat lunak tambang.

    Penamaan berkas mengikuti konvensi pengguna. Bawaannya meniru konvensi
    Minex yang terlihat pada data referensi: SG<seam><kode>.grid, mis.
    SG05SR (roof), SG05SF (floor), SG05ST (thickness), SG05CV (calorific value).
    """

    enabled: bool = True
    format: Literal["ascii", "binary"] = "ascii"

    filename_template: str = "SG{seam}{code}"
    extension: str = ".grid"
    # Kode permukaan. Ketebalan adalah ketebalan VERTIKAL (roof RL - floor RL).
    structure_codes: dict[str, str] = Field(
        default_factory=lambda: {"roof": "SR", "floor": "SF", "thickness": "ST",
                                 "depth": "DP"}
    )
    # Kode atribut kualitas. Atribut yang tidak terdaftar memakai namanya sendiri.
    quality_codes: dict[str, str] = Field(
        default_factory=lambda: {
            "rd_t_per_m3": "RD", "ASH": "AS", "ASH_adb": "AS", "CV": "CV",
            "CV_adb": "CV", "VM": "VM", "VM_adb": "VM", "FC": "FC",
            "FC_adb": "FC", "TS": "TS", "TS_adb": "TS", "MOISTURE": "IM",
            "M_adb": "IM", "TM_ar": "TM", "HGI": "HG",
        }
    )
    # Ketebalan uncut: seluruh interseksi, tanpa cutoff dan tanpa aturan
    # penambangan. Ini ketebalan GEOLOGI.
    write_uncut_thickness: bool = True
    # Ketebalan cut: setelah cutoff ketebalan minimum dan pengecualian parting.
    write_cut_thickness: bool = True
    write_structure: bool = True
    write_quality: bool = True
    # Atribut kualitas yang di-grid. Kosong = seluruh atribut yang tersedia.
    quality_attributes: list[str] = Field(default_factory=list)


class MapSettings(_Strict):
    """Parameter penyajian peta kontur struktur."""

    contour_interval_m: float = Field(gt=0)
    index_contour_every: int = Field(ge=1)
    label_contours: bool = True
    dxf_export: DxfExportSettings = Field(default_factory=DxfExportSettings)
    grd_export: GrdExportSettings = Field(default_factory=GrdExportSettings)


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
    seam_policy: SeamPolicy = Field(default_factory=SeamPolicy)
    weathering: WeatheringSpec = Field(default_factory=WeatheringSpec)
    observation_point: ObservationPointSpec = Field(default_factory=ObservationPointSpec)
    poo: PoOSpec = Field(default_factory=PoOSpec)
    limits: Limits = Field(default_factory=Limits)

    classification_radii_m: RadiiTable
    cutoffs: Cutoffs
    validation: ValidationSettings
    rpeee_constraints: RpeeeConstraints

    maps: MapSettings
    # Pedoman Praktis KCMI 2017 mengilustrasikan metode CIRCULAR pada pasal
    # 4.5.4 dan 4.5.5: gabungan cakram di sekeliling tiap titik pengamatan,
    # berpita Terukur - Tertunjuk - Tereka. Itu bawaannya.
    estimation_method: EstimationMethod = "circular"
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
    def _one_depth_limit_only(self) -> "Config":
        """Batas kedalaman hanya boleh punya SATU sumber kebenaran.

        `limits.depth` (KCMI 4.6.3.2) adalah yang otoritatif: hanya ia yang
        menuntut acuan yang diakui pedoman. Blok `rpeee_constraints` yang lama
        masih dibaca modul hilir, jadi ia DISINKRONKAN dari limits.depth.

        Bila keduanya diisi dengan angka yang berbeda, run dihentikan. Dua batas
        kedalaman yang berbeda di satu konfigurasi berarti dua angka sumberdaya
        yang berbeda, dan yang mana yang keluar hanya bergantung pada modul mana
        yang kebetulan dipanggil - persis kekeliruan tanpa gejala.
        """
        authoritative = self.limits.depth.max_depth_m
        legacy = self.rpeee_constraints.max_depth_m

        if authoritative is not None and legacy is not None \
                and abs(authoritative - legacy) > 1e-9:
            raise ValueError(
                f"dua batas kedalaman yang berbeda: limits.depth.max_depth_m = "
                f"{authoritative} m tetapi rpeee_constraints.max_depth_m = "
                f"{legacy} m. Isi limits.depth saja - ia yang menuntut acuan "
                "sesuai KCMI 4.6.3.2 - dan kosongkan rpeee_constraints.max_depth_m."
            )

        if authoritative is not None and legacy is None:
            depth = self.limits.depth
            object.__setattr__(self, "rpeee_constraints",
                               self.rpeee_constraints.model_copy(update={
                                   "max_depth_m": authoritative,
                                   "max_depth_basis":
                                       f"[{depth.basis_kind}] {depth.basis_note}"}))
        return self

    @property
    def max_depth_m(self) -> float | None:
        """Batas kedalaman yang berlaku, satu-satunya."""
        return self.limits.depth.max_depth_m or self.rpeee_constraints.max_depth_m

    @property
    def rpeee_demonstrated(self) -> bool:
        """Apakah keprospekan beralasan ditunjukkan - lewat batas, atau alasan."""
        return (self.max_depth_m is not None
                or bool(self.limits.depth.no_depth_limit_basis.strip()))

    @property
    def resource_label(self) -> str:
        """Label yang menempel pada SETIAP keluaran."""
        return "Sumberdaya" if self.rpeee_demonstrated else "Inventori Batubara"

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
