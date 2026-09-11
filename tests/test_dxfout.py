"""Ekspor kontur struktur seam ke DXF."""
import ezdxf
import numpy as np
import pytest

from coalres.dxfout import (
    contour_segments, export_seam_contours, layer_names, sanitize_layer, subcrop_mask,
)
from coalres.topo import build_surface


def plane_surface(name, z0=0.0, slope=0.05, spacing=25.0, extent=1000.0):
    xs = np.linspace(0, extent, 12)
    ys = np.linspace(0, extent, 12)
    gx, gy = np.meshgrid(xs, ys)
    return build_surface(gx.ravel(), gy.ravel(), z0 - slope * gx.ravel(),
                         spacing=spacing, name=name)


@pytest.fixture
def seams():
    return {"A": {"roof": plane_surface("A_roof", z0=0.0),
                  "floor": plane_surface("A_floor", z0=-10.0)}}


def test_layer_names_match_requested_form():
    spec = layer_names("A", "roof", "Seam {seam} {surface}", " Index")
    assert spec.main == "Seam A Roof"
    assert spec.index == "Seam A Roof Index"
    assert layer_names("A1", "floor", "Seam {seam} {surface}", " Index").main == "Seam A1 Floor"


def test_sanitize_keeps_spaces_but_removes_illegal_characters():
    """Spasi sah di AutoCAD dan diminta pengguna; karakter terlarang tidak."""
    assert sanitize_layer("Seam A Roof") == "Seam A Roof"
    assert sanitize_layer('Seam A/B "Roof"') == "Seam A_B _Roof_"


def test_contour_interval_is_respected(seams):
    segments = contour_segments(seams["A"]["roof"], interval_m=2.0)
    levels = sorted(segments)
    assert len(levels) > 3
    assert all(abs(level % 2.0) < 1e-6 for level in levels)


def test_contours_are_clipped_by_mask(seams):
    surface = seams["A"]["roof"]
    full = contour_segments(surface, 2.0)
    half = np.zeros(surface.z.shape, dtype=bool)
    half[:, : surface.z.shape[1] // 2] = True
    clipped = contour_segments(surface, 2.0, mask=half)
    length = lambda segs: sum(len(p) for paths in segs.values() for p in paths)
    assert length(clipped) < length(full)


def test_export_writes_expected_layers(tmp_path, seams):
    path, summary = export_seam_contours(
        tmp_path / "out.dxf", seams, interval_m=2.0, index_every=5,
        layer_template="Seam {seam} {surface}", index_suffix=" Index",
    )
    document = ezdxf.readfile(path)
    layers = {layer.dxf.name for layer in document.layers}
    assert {"Seam A Roof", "Seam A Roof Index",
            "Seam A Floor", "Seam A Floor Index"} <= layers
    assert summary["Seam A Roof"]["polylines"] > 0


def test_elevation_is_carried_by_each_polyline(tmp_path, seams):
    """Elevasi harus terbaca CAD dan GIS, bukan hanya tersirat dari nama layer."""
    path, _ = export_seam_contours(
        tmp_path / "out.dxf", seams, interval_m=2.0, index_every=5,
        layer_template="Seam {seam} {surface}", index_suffix=" Index",
    )
    msp = ezdxf.readfile(path).modelspace()
    elevations = [float(e.dxf.elevation) for e in msp
                  if e.dxftype() == "LWPOLYLINE" and e.dxf.layer.startswith("Seam A Roof")]
    assert elevations
    assert all(abs(z % 2.0) < 1e-6 for z in elevations)
    assert len(set(elevations)) > 1


def test_index_contours_land_on_their_own_layer(tmp_path, seams):
    path, _ = export_seam_contours(
        tmp_path / "out.dxf", seams, interval_m=2.0, index_every=5,
        layer_template="Seam {seam} {surface}", index_suffix=" Index",
    )
    msp = ezdxf.readfile(path).modelspace()
    index = [float(e.dxf.elevation) for e in msp
             if e.dxftype() == "LWPOLYLINE" and e.dxf.layer == "Seam A Roof Index"]
    assert index
    assert all(abs(z % 10.0) < 1e-6 for z in index)   # 2 m x 5


def test_units_are_metres(tmp_path, seams):
    path, _ = export_seam_contours(
        tmp_path / "out.dxf", seams, interval_m=2.0, index_every=5,
        layer_template="Seam {seam} {surface}", index_suffix=" Index",
    )
    assert ezdxf.readfile(path).header["$INSUNITS"] == 6


def test_subcrop_is_draped_onto_topography(tmp_path, seams):
    """Subcrop adalah garis tempat seam memotong permukaan tanah.

    Menulisnya pada elevasi 0 membuatnya melayang di tempat yang salah pada
    model 3D - di dataset nyata RL berkisar -100 sampai +60 m.
    """
    from shapely.geometry import box

    topo = plane_surface("topo", z0=60.0, slope=0.0)
    path, summary = export_seam_contours(
        tmp_path / "out.dxf", seams, interval_m=2.0, index_every=5,
        layer_template="Seam {seam} {surface}", index_suffix=" Index",
        subcrop_lines={"A": box(100, 100, 800, 800)}, topography=topo,
    )
    msp = ezdxf.readfile(path).modelspace()
    traces = [e for e in msp if e.dxftype() == "POLYLINE" and "Subcrop" in e.dxf.layer]
    assert traces
    z = [v.dxf.location.z for v in traces[0].vertices]
    assert all(abs(value - 60.0) < 1.0 for value in z)
    assert summary["Seam A Subcrop"]["draped_to"] == "topografi"


def test_seam_filter_writes_only_that_seam(tmp_path):
    seams = {
        "A": {"roof": plane_surface("A_roof"), "floor": plane_surface("A_floor", z0=-10)},
        "B": {"roof": plane_surface("B_roof", z0=-40), "floor": plane_surface("B_floor", z0=-45)},
    }
    path, _ = export_seam_contours(
        tmp_path / "a.dxf", seams, interval_m=2.0, index_every=5,
        layer_template="Seam {seam} {surface}", index_suffix=" Index",
        seam_filter=["A"],
    )
    msp = ezdxf.readfile(path).modelspace()
    assert not [e for e in msp if e.dxf.layer.startswith("Seam B")]
    assert [e for e in msp if e.dxf.layer.startswith("Seam A")]


def test_subcrop_mask_excludes_eroded_area():
    topo = plane_surface("topo", z0=0.0, slope=0.0)
    roof = plane_surface("roof", z0=20.0, slope=0.05)   # di atas topo di barat
    mask = subcrop_mask(topo, roof)
    assert mask.any() and not mask.all()
