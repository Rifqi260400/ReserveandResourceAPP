"""Base of Weathering: posisi tiap seam terhadap horizon pelapukan."""
import numpy as np
import pandas as pd
import pytest

from coalres.bow import ABOVE, BELOW, NO_BOW, STRADDLE, seam_vs_bow, summarise_by_seam


@pytest.fixture
def bow():
    return pd.DataFrame({"hole_id": ["H1", "H2"], "bow_depth_m": [10.0, 10.0],
                         "bow_rl_m": [40.0, 40.0], "collar_rl_m": [50.0, 50.0]})


def intercept(hole, seam, roof, floor, collar_rl=50.0):
    return {"hole_id": hole, "seam": seam, "roof_m": roof, "floor_m": floor,
            "collar_rl_m": collar_rl, "roof_rl_m": collar_rl - roof,
            "floor_rl_m": collar_rl - floor, "coal_thickness_m": floor - roof}


def test_seam_entirely_above_bow_is_fully_weathered(bow):
    detail = seam_vs_bow(pd.DataFrame([intercept("H1", "A", 4.0, 8.0)]), bow)
    row = detail.iloc[0]
    assert row["status"] == ABOVE
    assert row["weathered_thickness_m"] == pytest.approx(4.0)
    assert row["fresh_thickness_m"] == pytest.approx(0.0)


def test_seam_entirely_below_bow_is_fresh(bow):
    detail = seam_vs_bow(pd.DataFrame([intercept("H1", "A", 20.0, 24.0)]), bow)
    row = detail.iloc[0]
    assert row["status"] == BELOW
    assert row["weathered_thickness_m"] == pytest.approx(0.0)
    assert row["cover_below_bow_m"] == pytest.approx(10.0)


def test_seam_straddling_bow_reports_partial_weathering(bow):
    detail = seam_vs_bow(pd.DataFrame([intercept("H1", "A", 8.0, 14.0)]), bow)
    row = detail.iloc[0]
    assert row["status"] == STRADDLE
    assert row["weathered_thickness_m"] == pytest.approx(2.0)   # 8 -> 10
    assert row["fresh_thickness_m"] == pytest.approx(4.0)       # 10 -> 14


def test_roof_exactly_at_bow_counts_as_fresh(bow):
    """Kasus batas nyata: H001/A punya roof 3,00 m dan BOW 3,00 m."""
    detail = seam_vs_bow(pd.DataFrame([intercept("H1", "A", 10.0, 14.0)]), bow)
    assert detail.iloc[0]["status"] == BELOW
    assert detail.iloc[0]["cover_below_bow_m"] == pytest.approx(0.0)


def test_hole_without_bow_is_flagged_not_assumed(bow):
    """Ketiadaan BOW tidak boleh diam-diam dianggap 'segar'."""
    detail = seam_vs_bow(pd.DataFrame([intercept("H9", "A", 4.0, 8.0)]), bow)
    row = detail.iloc[0]
    assert row["status"] == NO_BOW
    assert np.isnan(row["weathered_thickness_m"])


def test_summary_counts_each_status(bow):
    frame = pd.DataFrame([
        intercept("H1", "A", 4.0, 8.0),      # lapuk
        intercept("H2", "A", 20.0, 24.0),    # segar
        intercept("H1", "B", 8.0, 14.0),     # terpotong
    ])
    summary = summarise_by_seam(seam_vs_bow(frame, bow)).set_index("seam")
    assert summary.loc["A", "n_seluruhnya_lapuk"] == 1
    assert summary.loc["A", "n_seluruhnya_segar"] == 1
    assert summary.loc["B", "n_terpotong_bow"] == 1


def test_bow_extracted_from_marker_rows():
    from coalres.bow import extract_bow
    from coalres.config import Config
    from coalres.io.minex import load_minex
    from conftest import ROOT

    path = ROOT / "config" / "minex_dummy_resolved.yaml"
    if not path.exists():
        pytest.skip("konfigurasi Minex tidak tersedia")
    cfg = Config.load(path)
    bow = extract_bow(load_minex(cfg), cfg)
    assert len(bow) == 60
    assert bow["bow_depth_m"].between(2.0, 4.0).all()
    assert np.allclose(bow["bow_rl_m"], bow["collar_rl_m"] - bow["bow_depth_m"])
