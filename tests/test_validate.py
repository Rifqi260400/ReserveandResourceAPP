"""Tahap 5: validasi model (KCMI 4.4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import model, validate
from coalres.config import Config
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"
CELL = 25.0


@pytest.fixture(scope="module")
def built():
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    dataset = load_minex(cfg)
    collars = dataset.collars.set_index("hole_id")
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    models = model.build(intersections, collars, cfg, spacing=CELL)
    return models, intersections, collars, cfg, dataset


def test_a_sound_model_honours_its_holes(built):
    models, intersections, collars, cfg, dataset = built
    report = validate.run(models, intersections, collars, cfg,
                          quality=dataset.quality, spacing=CELL,
                          cross_validation=False)
    assert report.failures == []


def test_the_honour_check_still_catches_swapped_coordinates(built):
    """Uji terpenting: pemeriksaan yang selalu lulus tidak berguna.

    Easting dan northing ditukar pada collar. Modelnya tetap terbangun dan
    permukaannya tetap tampak wajar - itulah bahayanya - tetapi tidak lagi
    menghormati lubangnya.
    """
    models, intersections, collars, cfg, _ = built
    swapped = collars.copy()
    swapped[["east", "north"]] = swapped[["north", "east"]].to_numpy()
    report = validate.run(models, intersections, swapped, cfg, quality=None,
                          spacing=CELL, cross_validation=False)
    assert report.failures, "koordinat tertukar harus tertangkap"
    assert not report.passed


def test_the_honour_check_catches_a_shifted_surface(built):
    """Permukaan digeser 5 m: cacat mekanis yang tidak terlihat di kontur."""
    models, intersections, collars, cfg, _ = built
    broken = {}
    for key, m in models.items():
        copy = model.SeamModel(
            seam=m.seam, domain=m.domain, roof=m.roof, isopach=m.isopach,
            floor=m.floor, support=m.support, n_support=m.n_support,
            n_holes=m.n_holes)
        copy.roof = type(m.roof)(**{**m.roof.__dict__, "z": m.roof.z + 5.0})
        broken[key] = copy
    report = validate.run(broken, intersections, collars, cfg, quality=None,
                          spacing=CELL, cross_validation=False)
    assert report.failures


def test_tolerance_is_per_hole_and_follows_local_relief(built):
    """Lubang di lereng curam tidak dihukum sekeras lubang di daerah landai."""
    models, intersections, collars, cfg, _ = built
    m = models["B"]
    frame = intersections[intersections["seam"] == "B"]
    deviations = validate.honour_check(m, frame, collars, 1.0, 0.30)
    for deviation in deviations:
        table = deviation.frame
        assert "toleransi" in table and "relief_lokal" in table
        assert (table["toleransi"] >= 0).all()
        # Toleransi naik bersama relief lokal, tidak pernah di bawah lantainya.
        steep = table.nlargest(3, "relief_lokal")["toleransi"].mean()
        flat = table.nsmallest(3, "relief_lokal")["toleransi"].mean()
        assert steep >= flat


def test_holes_on_the_support_edge_are_separated_from_real_failures(built):
    """H057 terbaca meleset karena node tetangganya NaN, bukan karena cacat."""
    models, intersections, collars, cfg, _ = built
    m = models["A@menyatu"]
    frame = intersections[intersections["seam"] == "A"]
    deviations = validate.honour_check(m, frame, collars, 1.0, 0.30)
    assert any(d.frame["di_tepi_dukungan"].any() for d in deviations)


def test_honour_deviation_shrinks_with_cell_size(built):
    """Buktinya diskretisasi: perkecil sel, galatnya ikut mengecil."""
    _, intersections, collars, cfg, _ = built
    worst = {}
    for spacing in (50.0, 10.0):
        models = model.build(intersections, collars, cfg, spacing=spacing)
        deviations = validate.honour_check(
            models["B"], intersections[intersections["seam"] == "B"],
            collars, 1.0, 0.30)
        worst[spacing] = max(
            float(d.frame["deviasi"].abs().max()) for d in deviations)
    assert worst[10.0] < worst[50.0]


def test_cross_validation_is_the_honest_measure(built):
    """Tinggalkan-satu: galat ramalan yang sebenarnya, jauh lebih besar."""
    models, intersections, collars, cfg, _ = built
    frame = intersections[intersections["seam"] == "B"]
    honour = validate.honour_check(models["B"], frame, collars, 1.0, 0.30)
    cross = validate.cross_validate("B", frame, collars, CELL, 119.0, 1.0, 0.30)
    assert cross, "validasi silang harus menghasilkan sesuatu"
    honour_rms = max(d.stats()["rms"] for d in honour)
    cross_rms = max(d.stats()["rms"] for d in cross)
    assert cross_rms > honour_rms * 2


def test_implausible_quality_values_fail_kcmi_442(built):
    *_, cfg, _ = built
    quality = pd.DataFrame({"RD": [1.30, 1.35, 9.99], "ASH": [5.0, 6.0, 7.0]})
    report = validate.ValidationReport()
    validate.check_quality_model(quality, cfg, report)
    assert any("di luar rentang wajar" in f for f in report.failures)


def test_low_rank_coal_triggers_the_mhc_recommendation(built):
    *_, cfg, _ = built
    quality = pd.DataFrame({"CV": [4200.0] * 10, "RD": [1.35] * 10})
    report = validate.ValidationReport()
    validate.check_quality_model(quality, cfg, report)
    assert any("MHC" in w and "EQM" in w for w in report.warnings)


def test_an_undeclared_rd_basis_fails(built):
    *_, cfg, _ = built
    variant = cfg.model_copy(update={
        "minex": cfg.minex.model_copy(update={"quality_rd_basis": "unknown"})})
    report = validate.ValidationReport()
    validate.check_quality_model(pd.DataFrame({"RD": [1.3] * 5}), variant, report)
    assert any("basis Rd belum dinyatakan" in f for f in report.failures)
