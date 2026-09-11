"""Permukaan, batasan RPEEE, dan aturan kualitas."""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from coalres.config import Config
from coalres.errors import ConfigError, MissingDataError
from coalres.estimate import HolePoint, build_polygons
from coalres.quality import summarise, weighted_average
from coalres.rpeee import apply_constraints, label_warning, parse_wkt, reconciliation_table
from coalres.topo import build_surface, depth_limit_extent, difference, subcrop_extent


@pytest.fixture
def cfg(base_config_dict, write_config):
    return Config.load(write_config(base_config_dict))


def plane(x, y, z0=100.0, slope=0.0):
    return z0 - slope * x


def make_surface(name, z0=100.0, slope=0.0, spacing=25.0, extent=1000.0):
    xs = np.linspace(0, extent, 12)
    ys = np.linspace(0, extent, 12)
    gx, gy = np.meshgrid(xs, ys)
    return build_surface(gx.ravel(), gy.ravel(), plane(gx.ravel(), gy.ravel(), z0, slope),
                         spacing=spacing, name=name)


# --------------------------------------------------------------------------- #
# Permukaan
# --------------------------------------------------------------------------- #
def test_surface_needs_three_points():
    with pytest.raises(MissingDataError, match="minimum 3"):
        build_surface([0, 1], [0, 1], [10, 20], spacing=10.0, name="uji")


def test_surface_rejects_collinear_points():
    with pytest.raises(MissingDataError, match="triangulasi gagal"):
        build_surface([0, 1, 2, 3], [0, 1, 2, 3], [1, 2, 3, 4], spacing=10.0, name="uji")


def test_surface_does_not_extrapolate_beyond_hull():
    """Permukaan seam yang menjalar jauh dari lubang adalah cara termudah
    memperoleh tonase yang tidak pernah dibor."""
    surface = build_surface([0, 100, 0], [0, 0, 100], [10, 20, 30],
                            spacing=10.0, margin=200.0, name="uji")
    assert np.isnan(surface.sample(np.array([-150.0]), np.array([-150.0]))[0])
    assert np.isfinite(surface.sample(np.array([20.0]), np.array([20.0]))[0])


def test_surface_sampling_reproduces_a_plane():
    surface = make_surface("plane", z0=100.0, slope=0.05)
    for x, y in ((100.0, 200.0), (450.0, 700.0)):
        assert surface.sample(np.array([x]), np.array([y]))[0] == pytest.approx(
            plane(x, y, 100.0, 0.05), abs=0.05
        )


def test_subcrop_excludes_area_where_seam_is_above_topography():
    topo = make_surface("topo", z0=100.0, slope=0.0)
    # Roof menanjak melewati topografi di paruh timur.
    roof = make_surface("roof", z0=60.0, slope=-0.08)
    extent = subcrop_extent(topo, roof)
    assert extent.area > 0
    assert extent.contains(Point(100, 500))        # roof jauh di bawah topo
    assert not extent.contains(Point(900, 500))    # roof sudah di atas topo


def test_depth_limit_extent_cuts_where_seam_is_too_deep():
    topo = make_surface("topo", z0=100.0, slope=0.0)
    roof = make_surface("roof", z0=60.0, slope=0.20)   # makin dalam ke timur
    depth = difference(topo, roof, "depth")
    extent = depth_limit_extent(depth, 100.0)
    assert extent.contains(Point(100, 500))
    assert not extent.contains(Point(900, 500))


def test_difference_rejects_mismatched_grids():
    a = make_surface("a", spacing=25.0)
    b = make_surface("b", spacing=50.0)
    with pytest.raises(MissingDataError, match="grid yang sama"):
        difference(a, b, "beda")


# --------------------------------------------------------------------------- #
# RPEEE
# --------------------------------------------------------------------------- #
def _points():
    return [HolePoint("A", 200, 200, 2.0, 1.30, "in_situ", False),
            HolePoint("B", 800, 200, 2.0, 1.30, "in_situ", False),
            HolePoint("C", 500, 800, 2.0, 1.30, "in_situ", False)]


def test_depth_limit_applied_to_polygons_not_just_holes(base_config_dict, write_config):
    """Poligon membentang jauh melewati lubangnya: seam di 80 m pada lubang
    bisa berada di 300 m di seberang poligon."""
    base_config_dict["rpeee_constraints"]["max_depth_m"] = 100.0
    base_config_dict["rpeee_constraints"]["max_depth_basis"] = (
        "Batas uji: kedalaman tambang terbuka blok sekitar dengan geometri sebanding."
    )
    cfg = Config.load(write_config(base_config_dict))

    topo = make_surface("topo", z0=100.0, slope=0.0)
    roof = make_surface("roof", z0=60.0, slope=0.20)
    depth = difference(topo, roof, "depth")
    surfaces = {"S1": {"subcrop": subcrop_extent(topo, roof),
                       "depth_limit": depth_limit_extent(depth, 100.0)}}

    polygons = build_polygons("S1", _points(), cfg)
    before = sum(p.tonnes for p in polygons)
    kept, steps = apply_constraints(polygons, surfaces, cfg)
    after = sum(p.tonnes for p in kept)

    assert after < before
    depth_step = next(s for s in steps if s.name == "batas kedalaman")
    assert depth_step.applied and depth_step.tonnes_removed > 0


def test_every_constraint_reports_what_it_removed(cfg):
    topo = make_surface("topo", z0=100.0, slope=0.0)
    roof = make_surface("roof", z0=60.0, slope=0.0)
    surfaces = {"S1": {"subcrop": subcrop_extent(topo, roof)}}
    polygons = build_polygons("S1", _points(), cfg)
    _, steps = apply_constraints(polygons, surfaces, cfg)

    table = reconciliation_table(steps, cfg)
    assert set(table["batasan"]) == {"subcrop", "batas kedalaman", "batas blok",
                                     "cutoff kualitas", "area terlarang"}
    for _, row in table.iterrows():
        assert row["tonnes_sesudah"] == pytest.approx(
            row["tonnes_sebelum"] - row["tonnes_dibuang"], rel=1e-9
        )


def test_excluded_area_removes_tonnage(base_config_dict, write_config):
    base_config_dict["rpeee_constraints"]["excluded_area_wkt"] = box(0, 0, 600, 1000).wkt
    base_config_dict["rpeee_constraints"]["excluded_area_basis"] = "Sungai dan sempadan."
    cfg = Config.load(write_config(base_config_dict))

    topo = make_surface("topo", z0=100.0, slope=0.0)
    roof = make_surface("roof", z0=60.0, slope=0.0)
    surfaces = {"S1": {"subcrop": subcrop_extent(topo, roof)}}
    polygons = build_polygons("S1", _points(), cfg)
    _, steps = apply_constraints(polygons, surfaces, cfg)
    step = next(s for s in steps if s.name == "area terlarang")
    assert step.applied and step.tonnes_removed > 0


def test_excluded_area_without_basis_is_rejected(base_config_dict, write_config):
    base_config_dict["rpeee_constraints"]["excluded_area_wkt"] = box(0, 0, 10, 10).wkt
    with pytest.raises(ConfigError, match="excluded_area_basis kosong"):
        Config.load(write_config(base_config_dict))


def test_bad_wkt_is_rejected():
    with pytest.raises(ConfigError, match="WKT"):
        parse_wkt("POLYGON((bukan wkt))", "uji")


def test_label_warning_present_only_without_economic_constraint(base_config_dict, write_config):
    cfg = Config.load(write_config(base_config_dict))
    assert "INVENTORI BATUBARA" in label_warning(cfg)

    base_config_dict["rpeee_constraints"]["max_depth_m"] = 120.0
    base_config_dict["rpeee_constraints"]["max_depth_basis"] = (
        "Batas uji: kedalaman tambang terbuka blok sekitar dengan geometri sebanding."
    )
    assert label_warning(Config.load(write_config(base_config_dict))) is None


# --------------------------------------------------------------------------- #
# Kualitas
# --------------------------------------------------------------------------- #
def test_averages_are_weighted_by_tonnage_not_hole_count():
    frame = pd.DataFrame({
        "seam": ["S1", "S1"], "class": ["Terukur", "Terukur"],
        "hole_id": ["A", "B"], "area_ha": [1.0, 9.0], "tonnes": [100.0, 900.0],
        "coal_thickness_m": [1.0, 9.0], "rd_t_per_m3": [1.3, 1.3],
        "rd_is_assumed": [False, False], "quality_coverage_frac": [1.0, 1.0],
        "ASH_adb": [5.0, 15.0],
    })
    summary = summarise(frame, ["seam", "class"])
    # Rata-rata sederhana 10,0; terbobot tonase 14,0.
    assert summary["ASH_adb"].iloc[0] == pytest.approx(14.0)
    assert summary["ASH_adb_n"].iloc[0] == 2


def test_sample_count_travels_with_every_average():
    frame = pd.DataFrame({
        "seam": ["S1", "S1"], "hole_id": ["A", "B"], "area_ha": [1.0, 1.0],
        "tonnes": [100.0, 100.0], "coal_thickness_m": [1.0, 1.0],
        "rd_t_per_m3": [1.3, 1.3], "rd_is_assumed": [False, False],
        "quality_coverage_frac": [1.0, 1.0], "CV_ar": [3400.0, np.nan],
    })
    summary = summarise(frame, ["seam"])
    assert summary["CV_ar_n"].iloc[0] == 1     # bukan 2


def test_weighted_average_ignores_zero_and_missing_weights():
    frame = pd.DataFrame({"v": [1.0, 100.0], "tonnes": [10.0, 0.0]})
    assert weighted_average(frame, "v") == pytest.approx(1.0)
