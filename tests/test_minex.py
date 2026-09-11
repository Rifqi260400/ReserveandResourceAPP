"""Jalur flat file Minex, dan perbaikan hitung ganda seam terpecah."""
import numpy as np
import pandas as pd
import pytest
from shapely.ops import unary_union

from coalres.config import Config
from coalres.errors import SchemaError
from coalres.estimate import HolePoint, build_polygons, build_polygons_for_unit, plan_overlap_report
from coalres.io.minex import read_lithology, read_survey
from coalres.seams import build_intersections_from_dataset

MINEX_DIR = None


@pytest.fixture(scope="module")
def minex_dir():
    from conftest import ROOT
    path = ROOT / "data" / "minex"
    if not (path / "surv.txt").exists():
        pytest.skip("dataset Minex tidak tersedia")
    return path


@pytest.fixture
def cfg(base_config_dict, write_config, minex_dir):
    base_config_dict["input_format"] = "minex_flat"
    base_config_dict["minex"] = {
        "survey_file": str(minex_dir / "surv.txt"),
        "lithology_file": str(minex_dir / "lit.txt"),
        "quality_file": str(minex_dir / "qual.txt"),
        "topography_file": str(minex_dir / "topo.dat"),
        "faults_file": str(minex_dir / "faults01.dat"),
        "survey_columns": ["hole_id", "east", "north", "rl", "total_depth"],
        "lithology_columns": ["hole_id", "seam", "depth_from", "depth_to", "code"],
        # Nama polos, sesuai berkas yang sebenarnya: sumber flat tidak
        # menyatakan basis di nama kolom.
        "quality_columns": ["hole_id", "seam", "depth_from", "depth_to", "RD",
                            "MOISTURE", "ASH", "VM", "FC", "TS", "CV"],
        "fault_columns": ["name", "east", "north", "z", "f5", "f6", "f7", "f8", "f9", "f10"],
        "marker_seams": ["W"],
        "quality_rd_basis": "in_situ",
        "quality_moisture_basis": "unknown",
        "quality_cv_unit": "kcal/kg",
        "on_invalid_quality_interval": "exclude",
    }
    base_config_dict["stratigraphy"] = ["A", "B"]
    base_config_dict["seam_splits"] = {"A": ["A1", "A2"]}
    # 42 dari 60 lubang tidak punya hasil lab. Tanpa nilai asumsi mereka
    # dikeluarkan seluruhnya - perilaku yang benar, tapi ia menyembunyikan
    # jalur "RD asumsi" dari pengujian.
    base_config_dict["assumed_rd_t_per_m3"] = 1.30
    return Config.load(write_config(base_config_dict))


# --------------------------------------------------------------------------- #
# Pembacaan
# --------------------------------------------------------------------------- #
def test_duplicate_rows_are_removed_and_reported(minex_dir):
    """Berkas ini berisi isi yang sama dua kali; dibiarkan, tonase berlipat."""
    collars, dup_surv = read_survey(minex_dir / "surv.txt",
                                    ["hole_id", "east", "north", "rl", "total_depth"])
    assert dup_surv == 60 and len(collars) == 60

    intervals, dup_lit = read_lithology(
        minex_dir / "lit.txt", ["hole_id", "seam", "depth_from", "depth_to", "code"], ["W"]
    )
    assert dup_lit == 197 and len(intervals) == 197


def test_marker_rows_are_separated_not_dropped(minex_dir):
    intervals, _ = read_lithology(
        minex_dir / "lit.txt", ["hole_id", "seam", "depth_from", "depth_to", "code"], ["W"]
    )
    markers = intervals[intervals["is_marker"]]
    assert len(markers) == 60
    assert (markers["depth_from"] == markers["depth_to"]).all()


def test_wrong_column_count_is_rejected(minex_dir):
    with pytest.raises(SchemaError, match="kolom"):
        read_survey(minex_dir / "surv.txt", ["hole_id", "east", "north", "rl",
                                             "total_depth", "extra", "more"])


def test_missing_required_column_name_is_rejected(minex_dir):
    with pytest.raises(SchemaError, match="wajib"):
        read_survey(minex_dir / "surv.txt", ["hole", "east", "north", "rl", "td"])


# --------------------------------------------------------------------------- #
# Seam
# --------------------------------------------------------------------------- #
def test_minex_thickness_is_interval_length_not_envelope(cfg):
    """Berkas lit Minex hanya memuat interval seam - tidak ada parting yang
    dicatat, jadi tebal batubara adalah to - from."""
    from coalres.io.minex import load_minex

    dataset = load_minex(cfg)
    items = build_intersections_from_dataset(dataset, cfg)
    by_key = {(i.hole_id, i.seam): i for i in items}
    item = by_key[("H017", "A2")]      # lit: H017 A2 71.00 86.80
    assert item.coal_thickness_m == pytest.approx(15.80, abs=1e-6)
    assert item.parting_thickness_m == pytest.approx(0.0, abs=1e-9)


def test_marker_seam_excluded_from_intersections(cfg):
    from coalres.io.minex import load_minex

    items = build_intersections_from_dataset(load_minex(cfg), cfg)
    assert "W" not in {i.seam for i in items}


# --------------------------------------------------------------------------- #
# Hitung ganda seam terpecah - inti perbaikan
# --------------------------------------------------------------------------- #
def _hole(hid, x, y, thickness=2.0):
    return HolePoint(hid, x, y, thickness, 1.30, "in_situ", False)


def test_separate_tessellations_overlap_parent_and_child(cfg):
    """Perilaku yang SALAH, dijaga agar tidak kembali diam-diam.

    Membentuk tesselasi terpisah untuk seam induk dan anaknya membuat domainnya
    bertindih - batubara yang sama terhitung dua kali.
    """
    parent = [_hole("P1", 0, 0), _hole("P2", 2000, 0), _hole("P3", 1000, 2000)]
    child = [_hole("C1", 900, 500), _hole("C2", 1100, 500), _hole("C3", 1000, 700)]

    a = unary_union([p.geometry for p in build_polygons("A", parent, cfg)])
    b = unary_union([p.geometry for p in build_polygons("A2", child, cfg)])
    assert a.intersection(b).area > 1000.0


def test_shared_tessellation_removes_parent_child_overlap(cfg):
    parent = [_hole("P1", 0, 0), _hole("P2", 2000, 0), _hole("P3", 1000, 2000)]
    child = [_hole("C1", 900, 500), _hole("C2", 1100, 500), _hole("C3", 1000, 700)]

    polygons = build_polygons_for_unit("A", {"A": parent, "A2": child}, cfg)
    a = unary_union([p.geometry for p in polygons if p.seam == "A"])
    b = unary_union([p.geometry for p in polygons if p.seam == "A2"])
    assert a.intersection(b).area < 1.0


def test_stacked_children_may_still_overlap_in_plan(cfg):
    """A1 di atas A2 pada lubang yang sama MEMANG bertindih dalam peta -
    keduanya seam berbeda pada kedudukan stratigrafi berbeda."""
    holes = [_hole("C1", 0, 0), _hole("C2", 800, 0), _hole("C3", 400, 700)]
    polygons = build_polygons_for_unit("A", {"A1": holes, "A2": holes}, cfg)
    a1 = unary_union([p.geometry for p in polygons if p.seam == "A1"])
    a2 = unary_union([p.geometry for p in polygons if p.seam == "A2"])
    assert a1.intersection(a2).area > 1000.0


def test_overlap_report_ignores_floating_point_noise(cfg):
    parent = [_hole("P1", 0, 0), _hole("P2", 2000, 0), _hole("P3", 1000, 2000)]
    child = [_hole("C1", 900, 500), _hole("C2", 1100, 500), _hole("C3", 1000, 700)]
    polygons = build_polygons_for_unit("A", {"A": parent, "A2": child}, cfg)
    assert plan_overlap_report(polygons, cfg).empty


def test_overlap_report_catches_real_double_counting(cfg):
    parent = [_hole("P1", 0, 0), _hole("P2", 2000, 0), _hole("P3", 1000, 2000)]
    child = [_hole("C1", 900, 500), _hole("C2", 1100, 500), _hole("C3", 1000, 700)]
    polygons = build_polygons("A", parent, cfg) + build_polygons("A2", child, cfg)
    report = plan_overlap_report(polygons, cfg)
    assert not report.empty
    assert report.iloc[0]["tonase_terhitung_ganda"] > 0


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_full_pipeline_on_minex_dataset(cfg, write_config, base_config_dict, tmp_path):
    from coalres.pipeline import run

    path = write_config(cfg.model_dump(mode="json"))
    results = run(path, verbose=False)
    assert results.polygons
    assert results.frames["plan_overlap"].empty, "hitung ganda induk-anak muncul kembali"

    frame = results.frames["polygons"]
    assert set(frame["seam"]) <= {"A", "A1", "A2", "B"}
    # Tanpa batasan ekonomi, seluruh label adalah Inventori.
    assert frame["class"].str.startswith("Inventori").all()


def test_unsuffixed_quality_columns_reach_the_report(cfg, write_config):
    """Berkas flat tidak menyatakan basis di nama kolom (ASH, bukan ASH_adb).

    Tanpa nama polos di daftar atribut, kualitas terbaca dari berkas tetapi
    tidak pernah sampai ke tabel keluaran, dan laporan tampak seolah tidak ada
    data kualitas sama sekali.
    """
    from coalres.pipeline import run
    from coalres.quality import AVERAGEABLE

    assert {"ASH", "CV", "TS", "VM", "FC", "MOISTURE"} <= set(AVERAGEABLE)

    results = run(write_config(cfg.model_dump(mode="json")), verbose=False)
    by_seam = results.frames["by_seam"]
    assert {"ASH", "CV", "TS"} <= set(by_seam.columns)
    assert by_seam["ASH"].notna().any()
    # Jumlah sampel ikut di setiap angka rata-rata.
    assert (by_seam["ASH_n"] > 0).any()


def test_assumed_rd_is_flagged_per_polygon(cfg, write_config):
    """Poligon tanpa hasil lab memakai RD asumsi dan HARUS tertandai."""
    from coalres.pipeline import run

    results = run(write_config(cfg.model_dump(mode="json")), verbose=False)
    frame = results.frames["polygons"]
    assert frame["rd_is_assumed"].any()
    assert not frame["rd_is_assumed"].all()
