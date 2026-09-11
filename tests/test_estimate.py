"""Estimasi poligonal: rumus tonase, Voronoi, dan anti-hitung-ganda."""
import numpy as np
import pytest
from shapely.geometry import Point, box

from coalres.classify import ResourceClass, bands, class_label
from coalres.config import Config
from coalres.errors import MissingDataError
from coalres.estimate import (
    HolePoint, build_polygons, clip_polygons, rd_sensitivity, to_frame, voronoi_cells,
)


def hole(hole_id, east, north, thickness=2.0, rd=1.30):
    return HolePoint(hole_id=hole_id, east=east, north=north,
                     coal_thickness_m=thickness, rd_t_per_m3=rd,
                     rd_basis="in_situ", rd_is_assumed=False)


@pytest.fixture
def cfg(base_config_dict, write_config):
    return Config.load(write_config(base_config_dict))


# --------------------------------------------------------------------------- #
# Tonase
# --------------------------------------------------------------------------- #
def test_single_hole_tonnage_matches_hand_calculation(cfg):
    """Satu lubang, luas dan tebal dan RD diketahui -> tonase hitung tangan.

    Radius measured 250 m: luas cakram = pi r^2 = 196.349,54 m2.
    Tonase = 196.349,54 x 2,0 m x 1,30 t/m3 = 510.508,8 t (galat segmentasi
    lingkaran -0,04% pada 128 segmen).
    """
    points = [hole("A", 0, 0), hole("B", 5000, 0), hole("C", 0, 5000)]
    polygons = build_polygons("S1", points, cfg, method="voronoi")
    measured = [p for p in polygons
                if p.hole_id == "A" and p.resource_class is ResourceClass.MEASURED]
    assert len(measured) == 1

    expected_area = np.pi * cfg.radii.measured ** 2
    assert measured[0].area_m2 == pytest.approx(expected_area, rel=1e-3)
    assert measured[0].tonnes == pytest.approx(expected_area * 2.0 * 1.30, rel=1e-3)


def test_tonnage_is_linear_in_each_factor(cfg):
    points = [hole("A", 0, 0), hole("B", 5000, 0), hole("C", 0, 5000)]
    base = build_polygons("S1", points, cfg)
    thick = build_polygons("S1", [hole("A", 0, 0, thickness=4.0),
                                  hole("B", 5000, 0), hole("C", 0, 5000)], cfg)
    a_base = sum(p.tonnes for p in base if p.hole_id == "A")
    a_thick = sum(p.tonnes for p in thick if p.hole_id == "A")
    assert a_thick == pytest.approx(2 * a_base, rel=1e-9)


def test_no_cosine_dip_correction_is_applied(cfg):
    """Tebal VERTIKAL dikalikan luas DALAM PETA sudah memberi volume prisma.

    Kalau suatu saat cos(dip) disisipkan ke jalur tonase, tes ini gagal.
    """
    points = [hole("A", 0, 0, thickness=2.309), hole("B", 5000, 0), hole("C", 0, 5000)]
    polygons = [p for p in build_polygons("S1", points, cfg) if p.hole_id == "A"]
    for polygon in polygons:
        assert polygon.volume_m3 == pytest.approx(polygon.area_m2 * 2.309, rel=1e-12)


def test_thickness_comes_from_the_polygons_own_hole(cfg):
    """Di bawah Voronoi, tebal tidak boleh diinterpolasi antar poligon."""
    points = [hole("A", 0, 0, thickness=1.0), hole("B", 600, 0, thickness=9.0),
              hole("C", 0, 600, thickness=5.0)]
    polygons = build_polygons("S1", points, cfg)
    by_hole = {p.hole_id: p.coal_thickness_m for p in polygons}
    assert by_hole["A"] == 1.0 and by_hole["B"] == 9.0 and by_hole["C"] == 5.0


# --------------------------------------------------------------------------- #
# Voronoi
# --------------------------------------------------------------------------- #
def test_voronoi_requires_three_points(cfg):
    with pytest.raises(MissingDataError, match="minimal 3"):
        voronoi_cells([hole("A", 0, 0), hole("B", 100, 0)])


def test_voronoi_rejects_coincident_points(cfg):
    with pytest.raises(MissingDataError, match="berimpit"):
        voronoi_cells([hole("A", 0, 0), hole("B", 0, 0), hole("C", 0, 0)])


def test_voronoi_cells_are_matched_by_containment_not_order(cfg):
    """shapely tidak menjamin urutan keluaran; kesalahan pencocokan akan
    menukar tebal antar lubang tanpa gejala yang terlihat."""
    points = [hole("A", 0, 0), hole("B", 800, 0), hole("C", 400, 700)]
    cells = voronoi_cells(points)
    for point in points:
        assert cells[point.hole_id].contains(Point(point.east, point.north))


def test_voronoi_cells_do_not_overlap(cfg):
    points = [hole("A", 0, 0), hole("B", 800, 0), hole("C", 400, 700)]
    cells = list(voronoi_cells(points).values())
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            assert cells[i].intersection(cells[j]).area == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Sirkular
# --------------------------------------------------------------------------- #
def test_circular_method_does_not_double_count_overlap(cfg):
    """Cakram yang bertumpang tindih diberi kelas TERTINGGI, sekali saja."""
    points = [hole("A", 0, 0), hole("B", 200, 0), hole("C", 100, 180)]
    polygons = build_polygons("S1", points, cfg, method="circular")
    total = sum(p.area_m2 for p in polygons)

    from shapely.ops import unary_union

    from coalres.classify import CIRCLE_SEGMENTS
    union = unary_union([
        Point(p.east, p.north).buffer(cfg.radii.inferred, quad_segs=CIRCLE_SEGMENTS // 4)
        for p in points
    ])
    assert total <= union.area * 1.001

    for i, a in enumerate(polygons):
        for b in polygons[i + 1:]:
            assert a.geometry.intersection(b.geometry).area == pytest.approx(0.0, abs=1e-3)


def test_circular_overlap_takes_highest_class(cfg):
    points = [hole("A", 0, 0), hole("B", 300, 0), hole("C", 150, 260)]
    polygons = build_polygons("S1", points, cfg, method="circular")
    midpoint = Point(150, 0)
    covering = [p for p in polygons if p.geometry.covers(midpoint)]
    assert covering
    assert max(p.resource_class for p in covering) is ResourceClass.MEASURED


# --------------------------------------------------------------------------- #
# Pemotongan
# --------------------------------------------------------------------------- #
def test_clipping_reduces_area_and_reports_removal(cfg):
    points = [hole("A", 0, 0), hole("B", 5000, 0), hole("C", 0, 5000)]
    polygons = build_polygons("S1", points, cfg)
    before = sum(p.area_m2 for p in polygons)

    kept, removed = clip_polygons(polygons, box(-100, -100, 100, 100), "uji")
    after = sum(p.area_m2 for p in kept)
    assert after < before
    assert not removed.empty
    assert removed["area_removed_m2"].sum() == pytest.approx(before - after, rel=1e-6)


def test_clip_with_empty_mask_is_a_noop(cfg):
    points = [hole("A", 0, 0), hole("B", 5000, 0), hole("C", 0, 5000)]
    polygons = build_polygons("S1", points, cfg)
    kept, removed = clip_polygons(polygons, None, "uji")
    assert len(kept) == len(polygons) and removed.empty


# --------------------------------------------------------------------------- #
def test_class_bands_are_annuli_not_overlapping_discs(cfg):
    band_list = bands(cfg.radii)
    geometries = [b.geometry(0, 0) for b in band_list]
    for i in range(len(geometries)):
        for j in range(i + 1, len(geometries)):
            assert geometries[i].intersection(geometries[j]).area == pytest.approx(0.0, abs=1e-3)
    total = sum(g.area for g in geometries)
    assert total == pytest.approx(np.pi * cfg.radii.inferred ** 2, rel=1e-3)


def test_rd_sensitivity_is_linear(cfg):
    points = [hole("A", 0, 0), hole("B", 5000, 0), hole("C", 0, 5000)]
    polygons = build_polygons("S1", points, cfg)
    table = rd_sensitivity(polygons).set_index("rd_delta_pct")
    base = table.loc[0, "tonnes"]
    assert table.loc[10, "tonnes"] == pytest.approx(base * 1.10, rel=1e-9)
    assert table.loc[-10, "tonnes"] == pytest.approx(base * 0.90, rel=1e-9)


def test_class_labels_follow_rule_84(base_config_dict, write_config):
    cfg = Config.load(write_config(base_config_dict))     # max_depth_m kosong
    assert class_label(ResourceClass.MEASURED, cfg) == "Inventori Terukur"

    base_config_dict["rpeee_constraints"]["max_depth_m"] = 100.0
    base_config_dict["rpeee_constraints"]["max_depth_basis"] = (
        "Kedalaman tambang terbuka pada blok sekitar dengan geometri sebanding."
    )
    cfg = Config.load(write_config(base_config_dict))
    assert class_label(ResourceClass.MEASURED, cfg) == "Sumberdaya Terukur"
