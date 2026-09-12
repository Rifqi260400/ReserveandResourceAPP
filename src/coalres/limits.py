"""Tahap 9: batas pelaporan Sumber daya menurut Pedoman Praktis KCMI 2017 4.6.

Pasal 4.6 menyusun batas dalam empat kelompok, dan ketiganya yang pertama
bersifat MENGGUGURKAN - bukan pertimbangan yang boleh dicatat lalu dilewati:

  4.6.1 Legalitas      izin masih berlaku dan menutup umur tambang; CnC bila
                       regulasi menuntut; data bor dari kawasan hutan wajib
                       didukung IPPKH eksplorasi.
  4.6.2 Lahan          "CPI tidak bisa melaporkan Sumber daya untuk tambang
                       terbuka di Hutan Lindung, area konservasi atau area lain
                       yang terlarang untuk kegiatan penambangan." Ini larangan.
                       RTRW harus memungkinkan usaha pertambangan.
  4.6.3 Cut off        kedalaman pelapukan, tebal minimum yang dapat ditambang,
                       maksimum Abu/Sulfur/TM, minimum CV, dan RD: "Untuk
                       batubara peringkat rendah, WAJIB menggunakan RD insitu
                       hasil konversi RD laboratorium dengan mengaplikasikan
                       formula Preston Sanders."
  4.6.4 Lainnya        teknologi, infrastruktur, prospek pemasaran - diulas,
                       tidak dihitung.

Modul ini MENERAPKAN yang dapat dihitung dan MENGGUGAT yang belum dinyatakan.
Ia tidak pernah mengisi sendiri: batas yang ditebak lebih buruk daripada tidak
ada batas, karena ia tampak dapat dipertanggungjawabkan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logging_setup import get_logger
from .model import SUPPORT_EXTRAPOLATED, SeamModel

log = get_logger("limits")

# KCMI 4.4.2 / 4.6.3.1: di bawah ini batubara diperlakukan peringkat rendah,
# dan konversi Preston & Sanders menjadi WAJIB.
LOW_RANK_CV_ADB = 6100.0

PROHIBITED_FOREST = {"hutan_lindung", "konservasi"}


@dataclass
class LimitResult:
    """Hasil penerapan batas pada satu seam."""

    key: str
    seam: str
    cells_before: int
    cells_after: int
    removed: dict[str, int] = field(default_factory=dict)
    mask: np.ndarray | None = None

    @property
    def fraction_removed(self) -> float:
        return 1.0 - self.cells_after / max(self.cells_before, 1)


@dataclass
class LimitsReport:
    results: list[LimitResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def reportable(self) -> bool:
        """Sumber daya boleh dilaporkan hanya bila tidak ada penggugur."""
        return not self.blockers

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"seam/domain": r.key, "sel_awal": r.cells_before,
                   "sel_akhir": r.cells_after,
                   "frac_dibuang": round(r.fraction_removed, 3)}
            row.update({f"dibuang: {k}": v for k, v in r.removed.items()})
            rows.append(row)
        return pd.DataFrame(rows).fillna(0)


def check_legal(cfg, report: LimitsReport) -> None:
    """KCMI 4.6.1."""
    legal = cfg.limits.legal
    if legal.permit_type is None:
        report.blockers.append(
            "KCMI 4.6.1.1: jenis izin tambang belum dinyatakan. Sumber daya "
            "tidak dapat dilaporkan tanpa dasar perizinan.")
    if legal.permit_covers_mine_life is None:
        report.blockers.append(
            "KCMI 4.6.1.1: belum dinyatakan apakah sisa masa berlaku izin "
            "(termasuk peluang perpanjangan) menutup perkiraan umur tambang.")
    elif not legal.permit_covers_mine_life:
        report.blockers.append(
            "KCMI 4.6.1.1: masa berlaku izin TIDAK menutup perkiraan umur "
            "tambang. Sumber daya di luar masa itu tidak memenuhi keprospekan "
            "beralasan.")
    if legal.cnc_certified is False:
        report.warnings.append(
            "KCMI 4.6.1.2: perusahaan belum tersertifikasi CnC. Periksa apakah "
            "regulasi yang berlaku saat estimasi menuntutnya.")
    if legal.exploration_in_forest_area:
        if not legal.ippkh_exploration_held:
            report.blockers.append(
                "KCMI 4.6.1.3: ada data bor dari kawasan hutan tetapi IPPKH "
                "eksplorasi tidak dinyatakan dimiliki. Data itu tidak dapat "
                "menopang pelaporan Sumber daya.")


def check_land(cfg, report: LimitsReport) -> None:
    """KCMI 4.6.2 - larangan, bukan pertimbangan."""
    land = cfg.limits.land
    if land.forest_category is None or land.forest_category == "tidak_diketahui":
        report.blockers.append(
            "KCMI 4.6.2.1: status kawasan hutan belum dinyatakan. Sumber daya "
            "tambang terbuka TIDAK DAPAT dilaporkan di Hutan Lindung maupun "
            "area konservasi, jadi statusnya harus diketahui lebih dulu.")
    elif land.forest_category in PROHIBITED_FOREST:
        report.blockers.append(
            f"KCMI 4.6.2.1: area berstatus '{land.forest_category}'. Pedoman "
            "menyatakan CPI tidak dapat melaporkan Sumber daya tambang terbuka "
            "di sana.")
    elif land.forest_category == "hutan_produksi":
        if not cfg.limits.legal.ippkh_exploration_held:
            report.blockers.append(
                "KCMI 4.6.2.1: deposit di Hutan Produksi menuntut bukti bahwa "
                "pemboran eksplorasi didukung IPPKH eksplorasi.")
    if land.rtrw_allows_mining is None:
        report.warnings.append(
            "KCMI 4.6.2.2: kesesuaian RTRW belum dinyatakan.")
    elif not land.rtrw_allows_mining:
        report.blockers.append(
            "KCMI 4.6.2.2: RTRW tidak memungkinkan usaha pertambangan di area "
            "ini.")


def check_density_rule(quality: pd.DataFrame | None, cfg,
                       report: LimitsReport) -> None:
    """KCMI 4.6.3.1 butir RD - kewajiban, bukan saran."""
    if quality is None:
        return
    cv = next((c for c in ("CV", "CV_adb") if c in quality.columns), None)
    if cv is None:
        return
    median_cv = float(quality[cv].dropna().median())
    if median_cv >= LOW_RANK_CV_ADB:
        return
    basis = cfg.minex.quality_rd_basis if cfg.minex else "unknown"
    if basis == "in_situ":
        report.notes.append(
            f"KCMI 4.6.3.1: CV median {median_cv:.0f} kcal/kg - batubara "
            "peringkat rendah, sehingga RD in-situ WAJIB. Konfigurasi menyatakan "
            "RD yang dipasok SUDAH in-situ, jadi Preston & Sanders tidak "
            "diterapkan ulang. Kewajiban ini bertumpu sepenuhnya pada pernyataan "
            "itu - bila sebenarnya RD laboratorium, tonase kelebihan sekitar 10%.")
    else:
        report.blockers.append(
            f"KCMI 4.6.3.1: CV median {median_cv:.0f} kcal/kg menandakan "
            f"batubara peringkat rendah, tetapi basis RD '{basis}'. Pedoman "
            "MEWAJIBKAN RD in-situ hasil konversi Preston & Sanders untuk "
            "peringkat rendah.")


def apply_to_model(model: SeamModel, cfg, weathering_depth_m: float | None,
                   topo=None, key: str | None = None) -> LimitResult:
    """Terapkan batas yang dapat dihitung pada satu seam.

    Urutannya sengaja: pelapukan, tebal minimum, lalu kedalaman. Tiap tahap
    melaporkan berapa sel yang ia buang, supaya tidak ada penyusutan yang
    tercampur menjadi satu angka.
    """
    limits = cfg.limits
    alive = np.isfinite(model.isopach.z)
    before = int(alive.sum())
    removed: dict[str, int] = {}

    # 4.6.3.1 - tebal minimum yang dapat ditambang.
    minimum = limits.cutoffs.min_mineable_thickness_m
    if minimum is not None:
        thin = alive & (model.isopach.z < minimum)
        removed["tebal < minimum"] = int(thin.sum())
        alive &= ~thin

    # 4.6.3.1 - zona pelapukan: batubara di ATAS BOW tidak dapat dijual.
    if limits.cutoffs.apply_weathering_depth and weathering_depth_m is not None \
            and topo is not None:
        gx, gy = np.meshgrid(model.roof.x, model.roof.y)
        surface = topo.sample(gx.ravel(), gy.ravel()).reshape(gx.shape)
        bow_rl = surface - weathering_depth_m
        weathered = alive & np.isfinite(bow_rl) & (model.roof.z > bow_rl)
        removed["roof di atas BOW"] = int(weathered.sum())
        alive &= ~weathered

    # 4.6.3.2 - batas maksimum kedalaman, diukur dari topografi ke roof.
    max_depth = limits.depth.max_depth_m
    if max_depth is not None and topo is not None:
        gx, gy = np.meshgrid(model.roof.x, model.roof.y)
        surface = topo.sample(gx.ravel(), gy.ravel()).reshape(gx.shape)
        depth = surface - model.roof.z
        too_deep = alive & np.isfinite(depth) & (depth > max_depth)
        removed["lebih dalam dari batas"] = int(too_deep.sum())
        alive &= ~too_deep

    # 4.6.2.3 - area terlarang di permukaan.
    if limits.land.prohibited_area_wkt.strip():
        from shapely import wkt
        from shapely.geometry import Point
        polygon = wkt.loads(limits.land.prohibited_area_wkt)
        gx, gy = np.meshgrid(model.roof.x, model.roof.y)
        inside = np.array([polygon.contains(Point(x, y))
                           for x, y in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
        blocked = alive & inside
        removed["area terlarang"] = int(blocked.sum())
        alive &= ~blocked

    return LimitResult(key=key or model.seam, seam=model.seam,
                       cells_before=before, cells_after=int(alive.sum()),
                       removed=removed, mask=alive)


def run(models: dict[str, SeamModel], cfg, topo=None,
        weathering_depth_m: float | None = None,
        quality: pd.DataFrame | None = None) -> LimitsReport:
    """Jalankan tahap 9 penuh: gugatan legal/lahan, lalu penerapan cut off."""
    report = LimitsReport()
    check_legal(cfg, report)
    check_land(cfg, report)
    check_density_rule(quality, cfg, report)

    if cfg.limits.iup_boundary_wkt is None:
        report.notes.append(
            "Batas IUP TIDAK diterapkan - keputusan proyek yang tercatat. "
            "Seluruh angka Sumber daya karena itu tidak dipotong batas izin, "
            "dan catatan ini wajib muncul pada dokumen asumsi, laporan QA, dan "
            "header ringkasan Sumber daya.")

    if cfg.limits.depth.max_depth_m is None:
        report.warnings.append(
            "KCMI 4.6.3.2: batas maksimum kedalaman belum dinyatakan. Tanpa "
            "batas ekonomi, keluaran adalah INVENTORI BATUBARA, bukan Sumber "
            "daya.")

    for key, model in models.items():
        report.results.append(
            apply_to_model(model, cfg, weathering_depth_m, topo, key=key))

    for line in report.blockers:
        log.error(line)
    for line in report.warnings:
        log.warning(line)
    return report
