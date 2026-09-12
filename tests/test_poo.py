"""Titik Pengamatan menurut Pedoman Praktis KCMI 2017 pasal 4.5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import observation, poo, radius
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
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    points = observation.build(intersections, dataset, cfg)
    return intersections, dataset, cfg, points


# --- 4.5.2 kriteria -------------------------------------------------------

def test_undeclared_survey_method_fails_every_hole(data):
    """Butir (a) belum dinyatakan, jadi tidak ada yang lolos.

    Menganggapnya terpenuhi berarti mengklaim mutu data yang belum diperiksa.
    """
    _, _, cfg, points = data
    result = poo.check_criteria(points.frame, cfg)
    assert "poo.survey_method" in result.undeclared
    assert int(result.frame["lolos"].sum()) == 0


def test_handheld_gps_does_not_satisfy_criterion_a(data):
    """KCMI 4.5.2 (a) menyebut total station atau GPS geodetik saja."""
    _, _, cfg, points = data
    for method, expect_any in (("handheld_gps", False), ("gps_geodetik", True),
                               ("total_station", True)):
        variant = cfg.model_copy(update={"poo": cfg.poo.model_copy(update={
            "survey_method": method, "all_holes_full_cored": True})})
        passing = int(poo.check_criteria(points.frame, variant).frame["lolos"].sum())
        assert (passing > 0) is expect_any


def test_a_non_cored_hole_without_geophysical_logging_fails(data):
    """KCMI 4.5.2 (b): selain full coring, logging geofisika WAJIB."""
    _, _, cfg, points = data
    base = {"survey_method": "total_station"}
    cored = cfg.model_copy(update={"poo": cfg.poo.model_copy(
        update={**base, "all_holes_full_cored": True})})
    open_unlogged = cfg.model_copy(update={"poo": cfg.poo.model_copy(
        update={**base, "all_holes_full_cored": False,
                "all_holes_geophysically_logged": False})})
    open_logged = cfg.model_copy(update={"poo": cfg.poo.model_copy(
        update={**base, "all_holes_full_cored": False,
                "all_holes_geophysically_logged": True})})
    assert int(poo.check_criteria(points.frame, cored).frame["lolos"].sum()) == 19
    assert int(poo.check_criteria(points.frame, open_unlogged).frame["lolos"].sum()) == 0
    assert int(poo.check_criteria(points.frame, open_logged).frame["lolos"].sum()) == 19
    reasons = set(poo.check_criteria(points.frame, open_unlogged).frame["alasan_kcmi"])
    assert poo.FAIL_LOGGING in reasons


# --- 4.5.4 jumlah minimum -------------------------------------------------

def test_fewer_than_three_points_cannot_form_a_resource_area():
    points = np.array([[0.0, 0.0], [100.0, 100.0]])
    primary, secondary, ratio = poo.two_direction_spread(points, None)
    frame = poo.evaluate_areas(
        pd.DataFrame({"seam": ["X", "X"], "hole_id": ["a", "b"],
                      "east": points[:, 0], "north": points[:, 1]}),
        pd.DataFrame(columns=["seam", "hole_id", "floor_m"]),
        pd.DataFrame(columns=["east", "north", "rl"]).set_index(
            pd.Index([], name="hole_id")),
        "X", 250.0)
    assert len(frame) == 1 and not frame.iloc[0]["memenuhi_kcmi"]
    assert "minimum 3" in frame.iloc[0]["temuan"]


def test_three_well_spread_points_pass():
    points = np.array([[0.0, 0.0], [400.0, 0.0], [200.0, 350.0]])
    frame = poo.evaluate_areas(
        pd.DataFrame({"seam": ["X"] * 3, "hole_id": list("abc"),
                      "east": points[:, 0], "north": points[:, 1]}),
        pd.DataFrame(columns=["seam", "hole_id", "floor_m"]),
        pd.DataFrame(columns=["east", "north", "rl"]).set_index(
            pd.Index([], name="hole_id")),
        "X", 250.0)
    assert frame.iloc[0]["memenuhi_kcmi"]
    assert frame.iloc[0]["n_titik"] == 3


# --- 4.5.5 spotted dog ----------------------------------------------------

def _collinear_frame():
    points = np.array([[0.0, 0.0], [300.0, 10.0], [600.0, 20.0]])
    return (pd.DataFrame({"seam": ["X"] * 3, "hole_id": list("abc"),
                          "east": points[:, 0], "north": points[:, 1]}),
            pd.DataFrame(columns=["seam", "hole_id", "floor_m"]),
            pd.DataFrame(columns=["east", "north", "rl"]).set_index(
                pd.Index([], name="hole_id")))


def test_collinear_points_fail_under_the_literal_reading():
    """Bunyi harfiah 4.5.4 menuntut satu titik FISIK ke arah down dip."""
    frame = poo.evaluate_areas(*_collinear_frame(), "X", 250.0,
                               policy="require_offset_point")
    assert not frame.iloc[0]["memenuhi_kcmi"]
    assert "spotted dog" in frame.iloc[0]["temuan"]
    assert frame.iloc[0]["rasio"] < poo.COLLINEARITY_RATIO


def test_collinear_points_pass_when_the_radius_supplies_the_dip_direction():
    """Posisi pemilik data: cakram menyapu ke segala arah, termasuk dip."""
    frame = poo.evaluate_areas(*_collinear_frame(), "X", 250.0,
                               policy="radius_provides_dip")
    assert frame.iloc[0]["memenuhi_kcmi"]
    # Rasionya tetap diukur dan dilaporkan, hanya tidak lagi menggugurkan.
    assert frame.iloc[0]["rasio"] < poo.COLLINEARITY_RATIO


def test_two_points_fail_under_both_readings():
    """Minimum 3 titik (4.5.4) tidak bergantung pembacaan mana pun."""
    points, intersections, collars = _collinear_frame()
    two = points.iloc[:2]
    for policy in ("radius_provides_dip", "require_offset_point"):
        frame = poo.evaluate_areas(two, intersections, collars, "X", 250.0,
                                   policy=policy)
        assert not frame.iloc[0]["memenuhi_kcmi"]
        assert "minimum 3" in frame.iloc[0]["temuan"]


def test_components_split_at_the_smaller_radius():
    """Spotted dog muncul di BATAS kelas.

    Titik yang menyatu pada radius Tereka dapat pecah jadi pulau terisolasi
    pada radius Terukur yang lebih kecil - persis ilustrasi KCMI 4.5.5.
    """
    points = np.array([[0.0, 0.0], [400.0, 0.0], [1400.0, 0.0]])
    assert len(poo.connected_components(points, 250.0)) == 2
    assert len(poo.connected_components(points, 1000.0)) == 1


def test_the_quality_holes_in_this_dataset_form_a_traverse(data):
    """Temuan nyata: lubang berkualitas tersebar praktis dalam satu garis.

    Seam B pada radius Tertunjuk: 8 titik membentang 3.250 m pada arah pertama
    tetapi hanya 423 m pada arah kedua. Itu lintasan eksplorasi, bukan jaring,
    dan KCMI 4.5.5 menolaknya sebagai dasar area Sumber daya.
    """
    intersections, dataset, cfg, points = data
    report = poo.spotted_dog_report(
        points.frame[points.frame["qualifies"]], intersections,
        dataset.collars.set_index("hole_id"), radius.radii_for("moderat"),
        policy="require_offset_point")
    row = report[(report["seam"] == "B") & (report["kelas"] == "tertunjuk")].iloc[0]
    assert row["n_titik"] == 8
    assert row["rentang_arah_1_m"] > 3000
    assert row["rasio"] < poo.COLLINEARITY_RATIO
    assert not row["memenuhi_kcmi"]
    # Sebagian besar area gagal salah satu dari kedua pasal.
    assert int((~report["memenuhi_kcmi"]).sum()) >= 15


def test_the_thresholds_are_ours_not_the_guidelines(data):
    """KCMI tidak memberi angka untuk "dua arah"; ambang ini operasionalisasi."""
    assert poo.COLLINEARITY_RATIO == 0.15
    assert poo.MIN_SPREAD_FRAC == 0.25
    assert "OPERASIONALISASI KAMI" in Path(
        ROOT / "src" / "coalres" / "poo.py").read_text()


def test_the_default_estimation_method_is_circular():
    """Pedoman KCMI 4.5.4 dan 4.5.5 mengilustrasikan gabungan cakram."""
    cfg = Config.load(CONFIG)
    assert cfg.estimation_method == "circular"
