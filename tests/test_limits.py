"""Tahap 9: batas pelaporan Sumber daya (KCMI 4.6)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import limits, model
from coalres.config import Config
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame
from coalres.topo import build_surface

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"


@pytest.fixture(scope="module")
def scene():
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    dataset = load_minex(cfg)
    collars = dataset.collars.set_index("hole_id")
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    models = model.build(intersections, collars, cfg, spacing=25.0)
    t = dataset.topo_points
    topo = build_surface(t[:, 0], t[:, 1], t[:, 2], spacing=25.0, name="topo")
    return models, cfg, topo, dataset


def _stopping(cfg):
    """Konfigurasi tanpa fallback: RD yang tidak dapat dikonversi menghentikan."""
    return cfg.model_copy(update={"minex": cfg.minex.model_copy(
        update={"rd_fallback_when_unconvertible": "stop", "rd_fallback_basis": ""})})


def _with_tm(dataset, tm: float = 25.0):
    """Kualitas dengan kolom TM, supaya gerbang densitas tidak ikut menyala.

    Uji legal dan lahan harus menguji legal dan lahan saja; membiarkan gerbang
    densitas menyala di dalamnya membuat kegagalannya tidak dapat dibaca.
    """
    return dataset.quality.assign(TM=tm)


def _declared(cfg, **land):
    legal = cfg.limits.legal.model_copy(update={
        "permit_type": "IUP", "permit_covers_mine_life": True})
    land_block = cfg.limits.land.model_copy(update={
        "forest_category": "APL", "rtrw_allows_mining": True, **land})
    return cfg.model_copy(update={"limits": cfg.limits.model_copy(
        update={"legal": legal, "land": land_block})})


# --- 4.6.1 legalitas / 4.6.2 lahan: menggugurkan ---------------------------

def test_undeclared_legal_status_does_not_block_the_estimate(scene):
    """KCMI 4.6 menulis RPEEE "dapat mengacu" pada faktor pengubah.

    Estimasi sumber daya adalah pekerjaan geologi; tidak tahu jenis izin bukan
    alasan menolak menghitung batubara yang ada di tanah. Butir yang belum
    dinyatakan dicatat sebagai kesiapan pelaporan, bukan penggugur.
    """
    models, cfg, topo, dataset = scene
    report = limits.run(models, cfg, topo=topo, quality=_with_tm(dataset))
    assert report.reportable
    assert report.blockers == []
    assert len(report.readiness) >= 3
    assert any("4.6.1.1" in r for r in report.readiness)
    assert any("4.6.2.1" in r for r in report.readiness)
    assert not report.reporting_ready


def test_protected_forest_cannot_be_reported_at_all(scene):
    """KCMI 4.6.2.1 adalah larangan, bukan pertimbangan."""
    models, cfg, topo, dataset = scene
    for category in ("hutan_lindung", "konservasi"):
        variant = _declared(cfg, forest_category=category)
        report = limits.run(models, variant, topo=topo, quality=dataset.quality)
        assert not report.reportable
        assert any(category in b for b in report.blockers)


def test_production_forest_records_the_ippkh_requirement(scene):
    models, cfg, topo, dataset = scene
    variant = _declared(cfg, forest_category="hutan_produksi")
    report = limits.run(models, variant, topo=topo, quality=_with_tm(dataset))
    assert report.reportable
    assert any("IPPKH" in r for r in report.readiness)

    with_permit = variant.model_copy(update={"limits": variant.limits.model_copy(
        update={"legal": variant.limits.legal.model_copy(
            update={"ippkh_exploration_held": True})})})
    report2 = limits.run(models, with_permit, topo=topo, quality=_with_tm(dataset))
    assert not any("IPPKH" in r for r in report2.readiness)


def test_a_permit_not_covering_mine_life_warns_but_still_counts(scene):
    models, cfg, topo, dataset = scene
    variant = _declared(cfg)
    variant = variant.model_copy(update={"limits": variant.limits.model_copy(
        update={"legal": variant.limits.legal.model_copy(
            update={"permit_covers_mine_life": False})})})
    report = limits.run(models, variant, topo=topo, quality=_with_tm(dataset))
    assert report.reportable
    assert any("umur tambang" in w for w in report.warnings)


def test_rtrw_refusal_still_blocks(scene):
    """RTRW yang DINYATAKAN melarang adalah fakta, bukan ketidaktahuan."""
    models, cfg, topo, dataset = scene
    variant = _declared(cfg, rtrw_allows_mining=False)
    report = limits.run(models, variant, topo=topo, quality=dataset.quality)
    assert any("RTRW" in b for b in report.blockers)


def test_fully_declared_status_clears_everything(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, _declared(cfg), topo=topo, quality=_with_tm(dataset))
    assert report.reportable, report.blockers
    assert report.reporting_ready, report.readiness


def test_only_a_declared_prohibition_stops_reporting(scene):
    """Garisnya: fakta terlarang yang DINYATAKAN menghentikan; data kosong tidak."""
    models, cfg, topo, dataset = scene
    unknown = limits.run(models, cfg, topo=topo, quality=_with_tm(dataset))
    declared_bad = limits.run(models, _declared(cfg, forest_category="hutan_lindung"),
                              topo=topo, quality=_with_tm(dataset))
    assert unknown.reportable and not unknown.reporting_ready
    assert not declared_bad.reportable


# --- 4.6.3.1 aturan RD ------------------------------------------------------

def test_low_rank_coal_with_unconvertible_lab_density_is_blocked(scene):
    """KCMI 4.6.3.1: RD laboratorium wajib dikonversi sebelum jadi tonase."""
    models, cfg, topo, dataset = scene
    report = limits.LimitsReport()
    limits.check_density_rule(pd.DataFrame({"CV": [4500.0] * 20, "RD": [1.35] * 20}),
                              _stopping(cfg), report)
    assert any("Preston & Sanders" in b for b in report.blockers)


def test_declared_in_situ_density_is_noted_not_blocked(scene):
    """Bila penyedia data memang sudah mengonversi, tidak ada yang dihitung ulang."""
    models, cfg, topo, dataset = scene
    already = cfg.model_copy(update={
        "minex": cfg.minex.model_copy(update={"quality_rd_basis": "in_situ"})})
    report = limits.LimitsReport()
    limits.check_density_rule(pd.DataFrame({"CV": [4500.0] * 20}), already, report)
    assert not report.blockers
    assert any("bertumpu sepenuhnya pada pernyataan" in n for n in report.notes)


# --- 4.6.3 penerapan cut off ------------------------------------------------

def test_weathered_coal_above_bow_is_removed(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, _declared(cfg), topo=topo,
                        weathering_depth_m=3.0, quality=dataset.quality)
    frame = report.summary()
    assert "dibuang: roof di atas BOW" in frame.columns
    assert frame["dibuang: roof di atas BOW"].sum() > 0
    assert (frame["sel_akhir"] < frame["sel_awal"]).all()


def test_no_weathering_depth_removes_nothing_for_it(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, _declared(cfg), topo=topo,
                        weathering_depth_m=None, quality=dataset.quality)
    assert "dibuang: roof di atas BOW" not in report.summary().columns


def test_minimum_mineable_thickness_is_applied(scene):
    models, cfg, topo, dataset = scene
    variant = _declared(cfg)
    variant = variant.model_copy(update={"limits": variant.limits.model_copy(
        update={"cutoffs": variant.limits.cutoffs.model_copy(
            update={"min_mineable_thickness_m": 2.0})})})
    report = limits.run(models, variant, topo=topo, quality=dataset.quality)
    frame = report.summary()
    assert frame["dibuang: tebal < minimum"].sum() > 0


def test_each_cut_off_reports_its_own_removal_separately(scene):
    """Penyusutan tidak boleh tercampur menjadi satu angka."""
    models, cfg, topo, dataset = scene
    variant = _declared(cfg)
    variant = variant.model_copy(update={"limits": variant.limits.model_copy(
        update={"cutoffs": variant.limits.cutoffs.model_copy(
            update={"min_mineable_thickness_m": 1.5})})})
    report = limits.run(models, variant, topo=topo, weathering_depth_m=3.0,
                        quality=dataset.quality)
    columns = set(report.summary().columns)
    assert "dibuang: tebal < minimum" in columns
    assert "dibuang: roof di atas BOW" in columns


def test_a_depth_limit_without_a_recognised_basis_is_refused():
    """KCMI 4.6.3.2 menyebut acuan yang sah; angka telanjang ditolak."""
    from coalres.config import DepthLimit
    with pytest.raises(ValueError, match="basis_kind"):
        DepthLimit(max_depth_m=100.0)
    with pytest.raises(ValueError, match="terlalu pendek"):
        DepthLimit(max_depth_m=100.0, basis_kind="geoteknik", basis_note="ok")
    good = DepthLimit(max_depth_m=100.0, basis_kind="besr_diperdalam",
                      basis_note="Kedalaman pit pada BESR diperdalam 5,6 menurut "
                                 "kajian internal 2026.")
    assert good.max_depth_m == 100.0


def test_the_missing_iup_boundary_is_recorded_as_a_project_decision(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, _declared(cfg), topo=topo, quality=dataset.quality)
    assert any("Batas IUP TIDAK diterapkan" in n for n in report.notes)


def test_two_disagreeing_depth_limits_are_refused(tmp_path):
    """Dua batas kedalaman berbeda = dua angka sumberdaya berbeda.

    Yang mana yang keluar hanya bergantung modul mana yang dipanggil - persis
    kekeliruan tanpa gejala yang program ini ada untuk mencegah.
    """
    import yaml
    from coalres.errors import ConfigError
    raw = yaml.safe_load(CONFIG.read_text())
    raw["rpeee_constraints"] = {**raw["rpeee_constraints"], "max_depth_m": 100.0,
                                "max_depth_basis": "Angka latihan blok lama."}
    raw["limits"] = {"depth": {
        "max_depth_m": 80.0, "basis_kind": "geoteknik",
        "basis_note": "Rekomendasi studi geoteknik lereng 2026 untuk batas bawah."}}
    path = tmp_path / "dua.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="dua batas kedalaman"):
        Config.load(path)


def test_the_new_block_syncs_into_the_legacy_one(tmp_path):
    import yaml
    raw = yaml.safe_load(CONFIG.read_text())
    raw["rpeee_constraints"] = {**raw["rpeee_constraints"],
                                "max_depth_m": None, "max_depth_basis": ""}
    raw["limits"] = {"depth": {
        "max_depth_m": 80.0, "basis_kind": "besr_diperdalam",
        "basis_note": "Kedalaman pit pada kondisi BESR diperdalam, kajian internal 2026."}}
    path = tmp_path / "satu.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = Config.load(path)
    assert cfg.max_depth_m == 80.0
    assert cfg.rpeee_constraints.max_depth_m == 80.0
    assert "besr_diperdalam" in cfg.rpeee_constraints.max_depth_basis
    assert cfg.rpeee_constraints.resource_label == "Sumberdaya"


def test_no_depth_limit_declared_keeps_the_sumberdaya_label(scene):
    """Posisi pemilik data: kedalaman bukan kriteria klasifikasi.

    KCMI 4.6.3.2 menulis CPI "dapat menggunakan" acuan kedalaman, bukan wajib,
    dan klasifikasi bersandar pada jarak di bidang X-Y. Ketiadaan batas karena
    itu sah - tetapi harus DINYATAKAN, bukan sekadar kosong.
    """
    models, cfg, topo, dataset = scene
    assert cfg.max_depth_m is None
    assert cfg.rpeee_demonstrated
    assert cfg.resource_label == "Sumberdaya"
    report = limits.run(models, _declared(cfg), topo=topo, quality=dataset.quality)
    assert any("SENGAJA TIDAK diterapkan" in n for n in report.notes)
    assert not any("INVENTORI BATUBARA" in w for w in report.warnings)


def test_silence_is_not_a_declaration(scene):
    """Kolom yang terlewat berbeda dari posisi yang dinyatakan."""
    models, cfg, topo, dataset = scene
    silent = _declared(cfg).model_copy(update={
        "limits": cfg.limits.model_copy(update={
            "depth": cfg.limits.depth.model_copy(
                update={"no_depth_limit_basis": ""})})})
    assert not silent.rpeee_demonstrated
    assert silent.resource_label == "Inventori Batubara"
    report = limits.run(models, silent, topo=topo, quality=dataset.quality)
    assert any("INVENTORI BATUBARA" in w for w in report.warnings)


def test_a_depth_limit_and_its_absence_cannot_both_be_declared():
    from coalres.config import DepthLimit
    with pytest.raises(ValueError, match="tidak boleh diisi bersamaan"):
        DepthLimit(max_depth_m=100.0, basis_kind="geoteknik",
                   basis_note="Rekomendasi studi geoteknik lereng 2026 untuk batas.",
                   no_depth_limit_basis="x" * 80)


def test_declining_a_depth_limit_still_demands_a_reason():
    from coalres.config import DepthLimit
    with pytest.raises(ValueError, match="terlalu pendek"):
        DepthLimit(no_depth_limit_basis="tidak perlu")


# --- 4.6.3.1 konversi RD in-situ -------------------------------------------

def test_lab_density_without_tm_cannot_be_converted(scene):
    """RD laboratorium WAJIB dikonversi; tanpa TM ia tidak dapat dikonversi.

    Memakai RD lab apa adanya melebihkan tonase karena air yang menguap di lab
    tetap ada di dalam tanah.
    """
    models, cfg, topo, dataset = scene
    report = limits.LimitsReport()
    limits.check_density_rule(dataset.quality, _stopping(cfg), report)
    assert cfg.minex.quality_rd_basis == "air_dried"
    assert any("TM (as-received)" in b for b in report.blockers)
    assert not report.reportable


def test_the_declared_fallback_treats_lab_density_as_in_situ(scene):
    """Keputusan pemilik data ketika TM dan IM memang tidak ada.

    Dihormati, tetapi tidak pernah diam: ia dicatat sebagai asumsi yang
    diketahui MELEBIHKAN tonase, beserta besarannya.
    """
    models, cfg, topo, dataset = scene
    assert cfg.minex.rd_fallback_when_unconvertible == "treat_as_in_situ"
    report = limits.LimitsReport()
    limits.check_density_rule(dataset.quality, cfg, report)
    assert report.blockers == []
    note = next(n for n in report.notes if "4.6.3.1" in n)
    assert "ASUMSI" in note and "MELEBIHKAN" in note


def test_the_fallback_cannot_be_chosen_without_a_reason():
    """Melebihkan tonase atas keputusan sendiri menuntut alasan tercatat."""
    from coalres.config import MinexSpec
    with pytest.raises(ValueError, match="rd_fallback_basis"):
        MinexSpec(survey_file="a", lithology_file="b",
                  survey_columns=["hole_id", "east", "north", "rl"],
                  lithology_columns=["hole_id", "seam", "depth_from", "depth_to"],
                  rd_fallback_when_unconvertible="treat_as_in_situ")


def test_the_fallback_flags_the_basis_not_the_value(scene):
    """RD lab yang diperlakukan in-situ: NILAInya terukur, BASISnya diasumsikan."""
    from coalres import estimate_grid
    models, cfg, topo, dataset = scene
    collars = dataset.collars.set_index("hole_id")
    from coalres.seams import build_intersections_from_dataset, to_frame
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    _, no_lab, basis_assumed, _ = estimate_grid._rd_grids(
        models["B"], intersections, collars, dataset.quality, 1.30, cfg=cfg)
    # Nilainya terukur di sel yang punya hasil lab; yang diasumsikan basisnya.
    assert float(basis_assumed.mean()) > 0
    assert float((no_lab + basis_assumed).mean()) == pytest.approx(1.0)


def test_the_conversion_gate_does_not_depend_on_coal_rank(scene):
    """Peringkat hanya mengatur seberapa keras pedoman menuntut.

    Memakai RD laboratorium untuk tonase salah pada peringkat mana pun; CV
    median data ini 6.328 kcal/kg, di ATAS ambang peringkat rendah, dan
    gerbangnya tetap menyala.
    """
    models, cfg, topo, dataset = scene
    assert float(dataset.quality["CV"].median()) > limits.LOW_RANK_CV_ADB
    report = limits.LimitsReport()
    limits.check_density_rule(dataset.quality, _stopping(cfg), report)
    assert report.blockers


def test_supplying_tm_converts_and_lowers_the_density(scene):
    """Arah koreksinya satu arah: RD in-situ SELALU lebih rendah dari air-dried."""
    from coalres.density import preston_sanders_insitu_ard
    models, cfg, topo, dataset = scene
    quality = dataset.quality.assign(TM=25.0)
    report = limits.LimitsReport()
    limits.check_density_rule(quality, cfg, report)
    assert not report.blockers
    lab = float(dataset.quality["RD"].median())
    converted = preston_sanders_insitu_ard(lab, 25.0,
                                           float(dataset.quality["MOISTURE"].median()))
    assert converted < lab
    assert 0.94 < converted / lab < 0.96      # sekitar 5-6% lebih rendah


def test_tm_below_im_is_refused():
    """Air-dried berarti sebagian air sudah hilang, jadi TM selalu >= IM."""
    from coalres.density import resolve_in_situ_rd
    from coalres.errors import MissingDataError
    with pytest.raises(MissingDataError, match="tidak lebih besar dari IM"):
        resolve_in_situ_rd(1.31, "air_dried", 4.0, 6.5, None)
