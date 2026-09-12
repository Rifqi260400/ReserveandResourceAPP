"""Tahap 10: estimasi circular di atas sel model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import (complexity, estimate_grid, limits, model, observation, poo,
                     radius)
from coalres.config import Config
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame
from coalres.topo import build_surface

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"
INTRUSI = ("intrusi", "sederhana",
           "Tidak ada indikasi batuan beku pada seluruh 60 lubang maupun peta regional.")


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
    declared = cfg.model_copy(update={"limits": cfg.limits.model_copy(update={
        "legal": cfg.limits.legal.model_copy(
            update={"permit_type": "IUP", "permit_covers_mine_life": True}),
        "land": cfg.limits.land.model_copy(
            update={"forest_category": "APL", "rtrw_allows_mining": True})})})
    report = limits.run(models, declared, topo=topo, weathering_depth_m=3.0,
                        quality=dataset.quality)
    masks = {r.key: r.mask for r in report.results}
    points = observation.build(intersections, dataset, declared)
    assessment = complexity.assess(dataset, declared, {INTRUSI[0]: INTRUSI[1:]})
    radii = radius.from_assessment(assessment)
    return (models, masks, points, intersections, collars, radii, declared,
            dataset)


def _run(scene):
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    return estimate_grid.run(models, masks, points.frame, intersections, collars,
                             radii, cfg, quality=dataset.quality)


def test_only_qualifying_observation_points_may_classify(scene):
    """Lubang tanpa kualitas tidak boleh menaikkan kelas.

    Melewatkan seluruh lubang ke tahap klasifikasi menggelembungkan Terukur
    tanpa gejala - inilah bug yang ditemukan dan ditutup di sini.
    """
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    strict = _run(scene)
    loose = estimate_grid.run(
        models, masks, points.frame.assign(qualifies=True), intersections,
        collars, radii, cfg, quality=dataset.quality)
    strict_total = sum(e.tonnes.get("terukur", 0.0) for e in strict.estimates)
    loose_total = sum(e.tonnes.get("terukur", 0.0) for e in loose.estimates)
    assert loose_total > strict_total


def test_coal_beyond_the_inferred_radius_is_excluded_not_folded(scene):
    """Aturan keras: tidak pernah dilipat menjadi Tereka."""
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    strict = cfg.model_copy(update={"poo": cfg.poo.model_copy(
        update={"two_direction_policy": "require_offset_point"})})
    report = estimate_grid.run(models, masks, points.frame, intersections,
                               collars, radii, strict, quality=dataset.quality)
    outside = sum(e.tonnes.get(estimate_grid.OUTSIDE, 0.0) for e in report.estimates)
    inferred = sum(e.tonnes.get("tereka", 0.0) for e in report.estimates)
    assert outside > 0
    assert inferred == 0.0
    assert any("tidak dilipat menjadi Tereka" in w for w in report.warnings)


def test_extrapolated_cells_are_never_measured(scene):
    models, masks, *_ = scene
    report = _run(scene)
    for estimate in report.estimates:
        m = models[estimate.key]
        measured = estimate.klass == "terukur"
        assert not (measured & (m.support == model.SUPPORT_EXTRAPOLATED)).any()
    assert any("ekstrapolasi tidak jadi Terukur" in n for n in report.notes)


def test_the_map_self_check_passes_and_separates_its_three_benign_states(scene):
    report = _run(scene)
    assert report.passed, report.failures
    assert any("GAGAL KCMI" in n for n in report.notes)
    assert any("DIBUANG cut-off" in n for n in report.notes)
    assert any("sel KOSONG pada model" in w for w in report.warnings)


def test_the_self_check_still_catches_a_real_desync(scene):
    """Sel hidup tanpa kelas padahal titiknya sah = peta dan tabel berbeda."""
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    report = _run(scene)
    estimate = next(e for e in report.estimates if (e.klass == "terukur").any())
    broken = estimate_grid.EstimateReport()
    estimate.klass[estimate.klass == "terukur"] = estimate_grid.OUTSIDE
    estimate_grid.self_check_map([estimate], models, masks, points.frame,
                                 intersections, collars, radii, broken)
    assert broken.failures


def test_tonnage_uses_plan_area_times_vertical_thickness(scene):
    """Tanpa koreksi cos(dip): prisma tegak sudah eksak."""
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    report = _run(scene)
    estimate = next(e for e in report.estimates if e.seam == "A")
    m = models[estimate.key]
    mask = estimate.klass == "terukur"
    rd_grid, _ = estimate_grid._rd_grids(m, intersections, collars,
                                         dataset.quality, 1.30)
    expected = float(np.nansum(m.isopach.z[mask] * rd_grid[mask])) * estimate.cell_area_m2
    assert estimate.tonnes["terukur"] == pytest.approx(expected)


def _total(report):
    return sum(e.tonnes.get(k, 0.0) for e in report.estimates
               for k in estimate_grid.CLASSES)


def test_the_two_direction_reading_moves_the_total_three_fold(scene):
    """KCMI 4.5.5 menuntut "kemenerusan dua arah" tanpa memberi angka.

    Kedua pembacaan sah, dan selisihnya besar - jadi pilihannya wajib terlihat
    di konfigurasi, bukan terkubur di konstanta modul.
    """
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    totals = {}
    for policy in ("radius_provides_dip", "require_offset_point"):
        variant = cfg.model_copy(update={"poo": cfg.poo.model_copy(
            update={"two_direction_policy": policy})})
        totals[policy] = _total(estimate_grid.run(
            models, masks, points.frame, intersections, collars, radii, variant,
            quality=dataset.quality))
    assert totals["radius_provides_dip"] > totals["require_offset_point"] * 2.5


def test_strike_aligned_points_pass_under_the_owners_reading(scene):
    """Tiga titik berjajar searah strike sah: radius yang memberi arah dip."""
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    report = _run(scene)
    # Seam B: 8 titik dalam satu lintasan searah strike (simpangan 0,2 derajat).
    b = next(e for e in report.estimates if e.seam == "B")
    assert b.tonnes["terukur"] > 0
    assert b.tonnes[estimate_grid.OUTSIDE] == 0.0


def test_the_minimum_of_three_points_still_bites_under_both_readings(scene):
    """4.5.4 tetap berlaku: seam A1 hanya punya 1 titik pengamatan."""
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    for policy in ("radius_provides_dip", "require_offset_point"):
        variant = cfg.model_copy(update={"poo": cfg.poo.model_copy(
            update={"two_direction_policy": policy})})
        report = estimate_grid.run(models, masks, points.frame, intersections,
                                   collars, radii, variant, quality=dataset.quality)
        a1 = next(e for e in report.estimates if e.seam == "A1")
        assert sum(a1.tonnes.get(k, 0.0) for k in estimate_grid.CLASSES) == 0.0


def test_collinearity_is_still_measured_even_when_it_no_longer_disqualifies(scene):
    """Angkanya tetap dilaporkan supaya CPI dapat menimbang sendiri."""
    models, masks, points, intersections, collars, radii, cfg, dataset = scene
    frame = poo.evaluate_areas(
        points.frame[points.frame["qualifies"]], intersections, collars, "B",
        250.0, policy="radius_provides_dip")
    assert "rasio" in frame and "rentang_arah_2_m" in frame
    assert (frame["kebijakan"] == "radius_provides_dip").all()


def test_the_rd_assumed_fraction_is_carried_per_seam(scene):
    report = _run(scene)
    for estimate in report.estimates:
        assert 0.0 <= estimate.rd_assumed_fraction <= 1.0
    # Sebagian besar sel masih memakai RD asumsi, dan itu harus terlihat.
    assert max(e.rd_assumed_fraction for e in report.estimates) > 0.5
