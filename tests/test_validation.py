"""Validasi harus menangkap cacat yang benar-benar terjadi di data lapangan."""
import numpy as np
import pandas as pd
import pytest

from coalres.config import Config
from coalres.io.excel import normalise_hole_id
from coalres.validate import (
    ValidationReport, check_collar_vs_topo, check_intervals, check_quality,
    check_stratigraphic_order,
)


@pytest.fixture
def cfg():
    return Config.load()


def _codes(report):
    return {f.code for f in report.findings}


def test_hole_id_normalisation_joins_variants():
    """LAS menulis DH-09-05C1, spreadsheet DH09_05C1, BHC DH09 05C1."""
    variants = ["DH-09-05C1", "DH09_05C1", "DH09 05C1", "dh09_05c1"]
    assert len({normalise_hole_id(v) for v in variants}) == 1


def test_collar_vs_topo_detects_gps_elevation_error(cfg):
    """Kasus nyata DH11_01: RL GPS 84 m vs Total Station 69,816 m."""
    collar = pd.DataFrame({"hole_id": [f"H{i}" for i in range(6)],
                           "rl": [70.0, 70.5, 71.0, 84.0, 70.2, 70.8]})
    topo = np.array([70.1, 70.4, 71.1, 69.8, 70.3, 70.7])
    report = ValidationReport()
    check_collar_vs_topo(collar, topo, report, cfg)
    hits = [f for f in report.findings if f.code == "COLLAR_VS_TOPO"]
    assert len(hits) == 1 and hits[0].hole_id == "H3"
    assert hits[0].value == pytest.approx(14.2, abs=0.1)


def test_systematic_datum_bias_is_an_error_not_a_warning(cfg):
    """Offset seragam menggeser SELURUH seam dan tak tertangkap QC di hilir."""
    collar = pd.DataFrame({"hole_id": [f"H{i}" for i in range(8)], "rl": [100.0] * 8})
    report = ValidationReport()
    check_collar_vs_topo(collar, np.full(8, 72.0), report, cfg)
    assert "TOPO_DATUM_BIAS" in _codes(report)


def test_mass_balance_failure_detected(cfg):
    seam = pd.DataFrame({
        "hole_id": ["A", "B"], "seam": ["S1", "S1"],
        "im_adb": [17.55, 17.55], "ash_adb": [10.84, 14.84],
        "vm_adb": [39.96, 39.96], "fc_adb": [31.65, 31.65],
    })
    report = ValidationReport()
    check_quality(seam, report, cfg)
    hits = [f for f in report.findings if f.code == "MASS_BALANCE"]
    assert len(hits) == 1 and hits[0].hole_id == "B"


def test_cv_conversion_failure_detected(cfg):
    seam = pd.DataFrame({
        "hole_id": ["A", "B"], "seam": ["S1", "S1"],
        "tm_ar": [42.25, 42.25], "im_adb": [17.55, 17.55],
        "cv_adb": [4815, 4815], "cv_ar": [3373, 3793],
    })
    report = ValidationReport()
    check_quality(seam, report, cfg)
    hits = [f for f in report.findings if f.code == "CV_CONVERSION"]
    assert len(hits) == 1 and hits[0].hole_id == "B"


def test_true_density_reported_as_ard_is_flagged(cfg):
    seam = pd.DataFrame({"hole_id": ["A"], "seam": ["S1"],
                         "rd_adb": [1.52], "im_adb": [17.55]})
    report = ValidationReport()
    check_quality(seam, report, cfg)
    assert "ARD_NOT_APPARENT" in _codes(report)


def test_seam_out_of_stratigraphic_order(cfg):
    seam = pd.DataFrame({
        "hole_id": ["A", "A", "B", "B"],
        "seam": ["S10A", "S10B", "S10B", "S10A"],
        "depth_from": [70.0, 84.0, 70.0, 84.0],
        "depth_to": [80.0, 95.0, 80.0, 95.0],
    })
    report = ValidationReport()
    check_stratigraphic_order(seam, ["S10A", "S10B"], report)
    hits = [f for f in report.findings if f.code == "STRAT_OUT_OF_ORDER"]
    assert len(hits) == 1 and hits[0].hole_id == "B"


def test_interval_overlap_and_beyond_td(cfg):
    collar = pd.DataFrame({"hole_id": ["A"], "td": [100.0]})
    seam = pd.DataFrame({
        "hole_id": ["A", "A", "A"], "seam": ["S1", "S2", "S3"],
        "depth_from": [10.0, 14.0, 95.0], "depth_to": [15.0, 20.0, 105.0],
    })
    report = ValidationReport()
    check_intervals(collar, seam, report, cfg)
    assert {"INTERVAL_OVERLAP", "INTERVAL_BEYOND_TD"} <= _codes(report)


def test_missing_quality_for_a_seam_is_an_error(cfg):
    seam = pd.DataFrame({
        "hole_id": ["A", "B"], "seam": ["S1", "S1"],
        "ash_adb": [np.nan, np.nan], "cv_adb": [np.nan, np.nan],
        "rd_adb": [np.nan, np.nan],
    })
    report = ValidationReport()
    check_quality(seam, report, cfg)
    hits = [f for f in report.findings if f.code == "QUALITY_COVERAGE"]
    assert hits and hits[0].severity == "ERROR"
