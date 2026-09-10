"""Gerbang audit Phase 0 terhadap data BGG yang sebenarnya."""
import pandas as pd
import pytest

from coalres.audit.checks import (
    Severity, coal_codes, run_audit, seam_envelopes, seam_intervals, unrecovered_codes,
)
from coalres.config import Config
from coalres.io.excel import load_workbook
from coalres.io.las import load_las
from coalres.io.quality_table import load_quality_table


@pytest.fixture(scope="module")
def workbooks():
    from conftest import WORKBOOKS
    if not WORKBOOKS.exists():
        pytest.skip("workbook referensi tidak tersedia")
    return [load_workbook(p) for p in sorted(WORKBOOKS.glob("*.xlsx"))]


@pytest.fixture(scope="module")
def las_files():
    from conftest import LAS_DIR
    from coalres.io.excel import normalise_hole_id
    if not LAS_DIR.exists():
        return {}
    out = {}
    for p in sorted(LAS_DIR.glob("*.LAS")):
        las = load_las(p)
        out[normalise_hole_id(las.well_name or p.stem)] = las
    return out


@pytest.fixture
def audit(workbooks, las_files, base_config_dict, write_config):
    cfg = Config.load(write_config(base_config_dict))
    return run_audit(workbooks, las_files, None, None, cfg), cfg


def _checks(report, prefix):
    return [f for f in report.findings if f.check.startswith(prefix)]


def test_audit_does_not_pass_on_incomplete_inputs(audit):
    report, _ = audit
    assert not report.passed
    assert report.stops


def test_missing_quality_table_stops(audit):
    report, _ = audit
    stops = [f for f in report.stops if f.check.startswith("05_")]
    assert stops and "PDF" in stops[0].message


def test_rd_basis_cannot_be_checked_without_quality(audit):
    report, _ = audit
    assert [f for f in report.stops if f.check.startswith("06_")]


def test_missing_topography_stops(audit):
    report, _ = audit
    assert any("topografi" in f.message.lower() for f in report.stops)


def test_seams_with_fewer_than_three_holes_stop(audit):
    """Voronoi menuntut tiga titik tidak segaris; satu lubang bukan poligon."""
    report, _ = audit
    stops = [f for f in report.stops if f.check.startswith("09_")]
    assert len(stops) == 9          # S11, S10A, S10B, S10C, S7, S7A..S7D
    assert all("1 lubang" in f.message for f in stops)


def test_coordinate_conflict_reported_per_hole(audit):
    """Nilai referensi: DH09_05C1 Collar vs BHC GPS beda 1,778 m northing,
    0,220 m RL. DH11_01 beda 14,184 m RL - kasus GPS handheld."""
    report, _ = audit
    frame = report.tables["coordinate_conflict"].set_index("hole_id")
    assert frame.loc["DH09_05C1", "dnorth_vs_gps"] == pytest.approx(1.778, abs=1e-3)
    assert frame.loc["DH09_05C1", "drl_vs_gps"] == pytest.approx(-0.220, abs=1e-3)
    assert frame.loc["DH11_01", "drl_vs_gps"] == pytest.approx(-14.184, abs=1e-3)
    # DH11_01 punya BHC TS terisi dan identik dengan Collar.
    assert frame.loc["DH11_01", "drl_vs_ts"] == pytest.approx(0.0, abs=1e-9)


def test_lithology_sampling_conflict_separates_core_loss_from_parting(audit):
    """Core loss di dalam interval sampel bukan parting: ia panjang yang tidak
    terambil, dan harus dilaporkan terpisah."""
    report, _ = audit
    frame = report.tables["lithology_vs_sampling"]
    assert not frame.empty
    kinds = frame["conflict_type"].value_counts().to_dict()
    assert kinds["parting"] == 9
    assert kinds["core_loss"] == 3
    assert frame.loc[frame["conflict_type"] == "core_loss", "lith_code"].eq("KL").all()


def test_depth_basis_offset_escalates_when_comparable_to_seam_thickness(audit):
    """DH09_05C1/S11: rekonsiliasi mengubah tebal 1,6 m pada seam 1,4 m."""
    report, _ = audit
    warns = [f for f in _checks(report, "03_") if f.severity == Severity.WARN]
    assert any(f.hole_id == "DH0905C1" and f.seam == "S11" for f in warns)


def test_open_hole_seams_flagged(audit):
    report, _ = audit
    frame = report.tables["core_coverage"].set_index(["hole_id", "seam"])
    assert bool(frame.loc[("DH0905C1", "S11"), "open_hole"])
    assert not bool(frame.loc[("DH0905C1", "S10A"), "open_hole"])
    assert all(frame.loc[("DH1101", s), "open_hole"] for s in ("S7", "S7A", "S7B"))


def test_real_interval_overlap_detected(audit):
    """Cacat asli di DH09_05C1: claystone 96,515-99,015 menimpa coal 99,000."""
    report, _ = audit
    assert any("tumpang tindih" in f.message for f in report.stops)


def test_coal_codes_exclude_coaly_rocks(workbooks):
    lib = workbooks[0].lithology_library
    codes = coal_codes(lib)
    assert {"CO", "C1", "C3", "C5", "LG"} <= codes
    assert "ZC" not in codes and "ZM" not in codes and "XC" not in codes


def test_unrecovered_codes_found_from_library(workbooks):
    assert "KL" in unrecovered_codes(workbooks[0].lithology_library)


def test_seam_envelopes_match_hand_calculation(workbooks):
    """DH09_05C1 S10B: roof 84,065 floor 95,835 dari SLL_Reconciled."""
    wb = next(w for w in workbooks if w.hole_id == "DH09_05C1")
    env = seam_envelopes(seam_intervals(wb), coal_codes(wb.lithology_library))
    row = env.set_index("seam").loc["S10B"]
    assert row["roof_m"] == pytest.approx(84.065, abs=1e-3)
    assert row["floor_m"] == pytest.approx(95.835, abs=1e-3)
    assert row["gross_thickness_m"] == pytest.approx(11.770, abs=1e-3)
    assert row["parting_thickness_m"] > 0
