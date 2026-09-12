"""Tahap 4: pemodelan seam."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import model
from coalres.config import Config
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"
CELL = 25.0


@pytest.fixture(scope="module")
def built():
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    dataset = load_minex(cfg)
    collars = dataset.collars.set_index("hole_id")
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    models = model.build(intersections, collars, cfg, spacing=CELL)
    return models, intersections, collars, cfg


def test_thickness_is_never_negative_anywhere(built):
    """Akibat stacking: floor = roof - isopach, bukan dua interpolasi bebas."""
    models, *_ = built
    for key, m in models.items():
        thickness = m.thickness()
        finite = thickness[np.isfinite(thickness)]
        assert (finite >= -1e-9).all(), key


def test_floor_is_derived_not_interpolated(built):
    models, *_ = built
    for m in models.values():
        assert "isopach" in m.floor.method
        expected = m.roof.z - m.isopach.z
        finite = np.isfinite(expected)
        assert np.allclose(m.floor.z[finite], expected[finite])


def test_roof_and_isopach_share_one_grid(built):
    """Syarat agar floor = roof - isopach sah sel demi sel."""
    models, *_ = built
    for m in models.values():
        assert np.array_equal(m.roof.x, m.isopach.x)
        assert np.array_equal(m.roof.y, m.isopach.y)


def test_extrapolated_cells_can_never_be_measured(built):
    """Aturan keras, bukan setelan."""
    models, *_ = built
    for m in models.values():
        assert not m.measurable[m.support == model.SUPPORT_EXTRAPOLATED].any()
        assert m.measurable[m.support == model.SUPPORT_INTERPOLATED].all()


def test_the_split_lens_domains_are_modelled_separately(built):
    models, *_ = built
    assert "A@menyatu" in models
    assert "A1@terpecah" in models and "A2@terpecah" in models
    assert models["A@menyatu"].domain == "menyatu"
    assert models["A2@terpecah"].domain == "terpecah"


def test_domain_clipping_removes_false_coal_in_the_lens(built):
    """Hull seam A membentang barat-timur dan menelan zona tengah.

    Di zona itu seam A justru TIDAK ADA - yang ada A1 dan A2. Tanpa pemotongan
    domain, permukaan seam A mengklaim batubara di sana. Pemotongan tetangga
    terdekat membuang 2.025 sel, sekitar 21 juta ton semu.
    """
    models, intersections, collars, cfg = built
    domains = model.split_domains(intersections, collars, cfg)
    merged = domains["A:menyatu"]
    merged = merged[merged["seam"] == "A"]
    rival = collars.reindex(sorted(set(domains["A:terpecah"]["hole_id"]))).dropna(
        subset=["east", "north"])[["east", "north"]].to_numpy(float)

    unclipped = model.build_seam("A", merged, collars, CELL, 119.0, domain="menyatu")
    clipped = model.build_seam("A", merged, collars, CELL, 119.0, domain="menyatu",
                               rival_points=rival)

    def tonnes(m):
        z = m.isopach.z
        return float(np.nansum(z[np.isfinite(z)])) * CELL * CELL * 1.30

    assert np.isfinite(unclipped.isopach.z).sum() > np.isfinite(clipped.isopach.z).sum()
    removed = tonnes(unclipped) - tonnes(clipped)
    assert removed > 15e6
    assert removed / tonnes(unclipped) > 0.5
    assert any("domain saingan" in note for note in clipped.notes)


def test_seam_a_barely_reaches_into_the_split_zone(built):
    """Sisa sel di zona tengah hanya di tepi lensa, bukan menembusnya."""
    models, *_ = built
    m = models["A@menyatu"]
    gx, _ = np.meshgrid(m.roof.x, m.roof.y)
    inside_lens = (gx > 4140) & (gx < 6087) & np.isfinite(m.roof.z)
    assert int(inside_lens.sum()) < 100


def test_a_domain_without_a_rival_is_not_clipped():
    surface_points = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 100.0]])
    from coalres.topo import build_surface
    surface = build_surface(surface_points[:, 0], surface_points[:, 1],
                            np.ones(3), spacing=10.0, name="x")
    mask = model.domain_mask(surface, surface_points, np.empty((0, 2)))
    assert mask.all()


def test_negative_isopach_cells_are_clamped_and_reported():
    """Interpolasi tebal dapat menembus nol antar lubang tipis."""
    collars = pd.DataFrame({
        "hole_id": list("abcd"),
        "east": [0.0, 1000.0, 0.0, 1000.0],
        "north": [0.0, 0.0, 1000.0, 1000.0],
        "rl": [100.0, 100.0, 100.0, 100.0],
    }).set_index("hole_id")
    frame = pd.DataFrame({
        "hole_id": list("abcd"), "seam": ["X"] * 4,
        "roof_m": [10.0, 10.0, 10.0, 10.0],
        "coal_thickness_m": [5.0, 0.01, 0.01, 5.0],
    })
    m = model.build_seam("X", frame, collars, spacing=50.0, influence_m=500.0)
    assert m is not None
    thickness = m.thickness()
    assert (thickness[np.isfinite(thickness)] >= -1e-9).all()
