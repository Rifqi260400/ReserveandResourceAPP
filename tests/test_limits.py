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


def _declared(cfg, **land):
    legal = cfg.limits.legal.model_copy(update={
        "permit_type": "IUP", "permit_covers_mine_life": True})
    land_block = cfg.limits.land.model_copy(update={
        "forest_category": "APL", "rtrw_allows_mining": True, **land})
    return cfg.model_copy(update={"limits": cfg.limits.model_copy(
        update={"legal": legal, "land": land_block})})


# --- 4.6.1 legalitas / 4.6.2 lahan: menggugurkan ---------------------------

def test_undeclared_legal_status_blocks_reporting(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, cfg, topo=topo, quality=dataset.quality)
    assert not report.reportable
    assert any("4.6.1.1" in b for b in report.blockers)
    assert any("4.6.2.1" in b for b in report.blockers)


def test_protected_forest_cannot_be_reported_at_all(scene):
    """KCMI 4.6.2.1 adalah larangan, bukan pertimbangan."""
    models, cfg, topo, dataset = scene
    for category in ("hutan_lindung", "konservasi"):
        variant = _declared(cfg, forest_category=category)
        report = limits.run(models, variant, topo=topo, quality=dataset.quality)
        assert not report.reportable
        assert any(category in b for b in report.blockers)


def test_production_forest_needs_ippkh(scene):
    models, cfg, topo, dataset = scene
    variant = _declared(cfg, forest_category="hutan_produksi")
    report = limits.run(models, variant, topo=topo, quality=dataset.quality)
    assert any("IPPKH" in b for b in report.blockers)

    with_permit = variant.model_copy(update={"limits": variant.limits.model_copy(
        update={"legal": variant.limits.legal.model_copy(
            update={"ippkh_exploration_held": True})})})
    report2 = limits.run(models, with_permit, topo=topo, quality=dataset.quality)
    assert not any("IPPKH" in b for b in report2.blockers)


def test_a_permit_not_covering_mine_life_blocks(scene):
    models, cfg, topo, dataset = scene
    variant = _declared(cfg)
    variant = variant.model_copy(update={"limits": variant.limits.model_copy(
        update={"legal": variant.limits.legal.model_copy(
            update={"permit_covers_mine_life": False})})})
    report = limits.run(models, variant, topo=topo, quality=dataset.quality)
    assert any("umur tambang" in b for b in report.blockers)


def test_rtrw_refusal_blocks(scene):
    models, cfg, topo, dataset = scene
    variant = _declared(cfg, rtrw_allows_mining=False)
    report = limits.run(models, variant, topo=topo, quality=dataset.quality)
    assert any("RTRW" in b for b in report.blockers)


def test_fully_declared_status_clears_the_blockers(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, _declared(cfg), topo=topo, quality=dataset.quality)
    assert report.reportable, report.blockers


# --- 4.6.3.1 aturan RD ------------------------------------------------------

def test_low_rank_coal_with_lab_density_is_blocked(scene):
    """KCMI 4.6.3.1: peringkat rendah WAJIB RD in-situ Preston & Sanders."""
    models, cfg, topo, dataset = scene
    variant = _declared(cfg).model_copy(update={
        "minex": cfg.minex.model_copy(update={"quality_rd_basis": "air_dried"})})
    quality = pd.DataFrame({"CV": [4500.0] * 20})
    report = limits.LimitsReport()
    limits.check_density_rule(quality, variant, report)
    assert any("MEWAJIBKAN RD in-situ" in b for b in report.blockers)


def test_declared_in_situ_density_is_noted_not_blocked(scene):
    models, cfg, topo, dataset = scene
    report = limits.LimitsReport()
    limits.check_density_rule(pd.DataFrame({"CV": [4500.0] * 20}), cfg, report)
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


def test_a_legacy_only_depth_limit_is_flagged_as_lacking_a_kcmi_basis(scene):
    models, cfg, topo, dataset = scene
    report = limits.run(models, _declared(cfg), topo=topo, quality=dataset.quality)
    assert cfg.max_depth_m == 100.0
    assert any("blok lama rpeee_constraints" in w for w in report.warnings)


def test_without_a_depth_limit_the_output_is_inventori(scene):
    models, cfg, topo, dataset = scene
    stripped = _declared(cfg).model_copy(update={
        "rpeee_constraints": cfg.rpeee_constraints.model_copy(
            update={"max_depth_m": None, "max_depth_basis": ""})})
    report = limits.run(models, stripped, topo=topo, quality=dataset.quality)
    assert stripped.max_depth_m is None
    assert any("INVENTORI BATUBARA" in w for w in report.warnings)
