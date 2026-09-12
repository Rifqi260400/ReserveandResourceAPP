"""Tahap 6: titik observasi."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coalres import observation
from coalres.config import Config
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"


@pytest.fixture(scope="module")
def data():
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    dataset = load_minex(cfg)
    return to_frame(build_intersections_from_dataset(dataset, cfg)), dataset, cfg


def test_the_module_applies_the_decision_it_does_not_make_it(data):
    intersections, dataset, cfg = data
    undecided = cfg.model_copy(update={
        "observation_point": cfg.observation_point.model_copy(
            update={"requires_quality": None, "basis": ""})})
    with pytest.raises(ValueError, match="belum diputuskan"):
        observation.build(intersections, dataset, undecided)


def test_quality_is_measured_per_seam_not_per_hole(data):
    """Kualitas pada seam B tidak menjadikan lubang itu titik observasi seam A1.

    Diukur per lubang, 18 dari 60 lubang punya kualitas. Diukur per seam -
    dan itu yang berlaku - hanya 27 dari 135 interseksi yang punya.
    """
    intersections, dataset, cfg = data
    with_quality = sum(
        1 for _, row in intersections.iterrows()
        if observation._quality_coverage(dataset.quality, row["hole_id"],
                                         row["seam"], row["roof_m"], row["floor_m"]) > 0)
    assert len(set(dataset.quality["hole_id"])) == 18
    assert with_quality == 27
    assert len(intersections) == 135


def test_coverage_is_a_fraction_of_seam_thickness_not_a_sample_count(data):
    """Sepuluh sampel pendek di satu ujung seam tebal bukan cakupan."""
    quality = pd.DataFrame([
        {"hole_id": "X", "seam": "A1", "depth_from": 10.0 + i * 0.1,
         "depth_to": 10.1 + i * 0.1} for i in range(10)
    ])
    coverage = observation._quality_coverage(quality, "X", "A1", 10.0, 20.0)
    assert coverage == pytest.approx(0.10)


def test_coverage_outside_the_seam_envelope_does_not_count(data):
    quality = pd.DataFrame([{"hole_id": "X", "seam": "A1",
                             "depth_from": 0.0, "depth_to": 100.0}])
    # Interval membentang jauh melewati seam; yang dihitung hanya irisannya.
    assert observation._quality_coverage(quality, "X", "A1", 10.0, 20.0) == pytest.approx(1.0)
    assert observation._quality_coverage(quality, "X", "B", 10.0, 20.0) == 0.0


def test_every_disqualified_hole_carries_a_reason(data):
    intersections, dataset, cfg = data
    frame = observation.build(intersections, dataset, cfg).frame
    rejected = frame[~frame["qualifies"]]
    assert len(rejected) > 0
    assert (rejected["reason"].str.len() > 0).all()
    assert set(rejected["reason"]) <= {
        observation.NO_INTERSECTION, observation.BELOW_CUTOFF,
        observation.NO_QUALITY, observation.THIN_QUALITY}


def test_requiring_quality_never_adds_observation_points(data):
    intersections, dataset, cfg = data
    comparison = observation.compare_populations(intersections, dataset, cfg)
    assert (comparison["selisih"] <= 0).all()
    # Pada dataset ini penyusutannya besar: 135 interseksi menjadi 19.
    assert comparison["cukup ketebalan"].sum() == 135
    assert comparison["menuntut kualitas"].sum() == 19


def test_the_binding_constraint_is_availability_not_the_threshold(data):
    """Ambang cakupan hanya memangkas 27 menjadi 19.

    Yang memangkas 135 menjadi 27 adalah ketiadaan kualitas sama sekali. Jadi
    menurunkan ambang TIDAK akan menyelamatkan klasifikasi - hanya menambah
    data kualitas yang bisa.
    """
    intersections, dataset, cfg = data
    def qualifying(frac: float) -> int:
        variant = cfg.model_copy(update={
            "cutoffs": cfg.cutoffs.model_copy(
                update={"min_quality_coverage_frac": frac})})
        return int(observation.build(intersections, dataset, variant).frame["qualifies"].sum())
    assert qualifying(0.90) == 19
    assert qualifying(0.50) == 27      # seluruh yang punya kualitas
    assert qualifying(0.0001) == 27    # menurunkan ambang tidak menambah apa pun
    assert qualifying(1.00) == 15


def test_for_seam_returns_only_qualifying_points(data):
    intersections, dataset, cfg = data
    points = observation.build(intersections, dataset, cfg)
    for seam in ("A1", "A2", "B"):
        frame = points.for_seam(seam)
        assert list(frame.columns) == ["hole_id", "east", "north"]
        assert frame["hole_id"].is_unique
        assert frame["east"].notna().all()
