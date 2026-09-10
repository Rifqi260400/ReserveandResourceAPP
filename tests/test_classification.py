"""Klasifikasi: pengaman yang membedakan kriteria SPASI dari buffer."""
import numpy as np
import pandas as pd
import pytest

from coalres.classify import INDICATED, INFERRED, MEASURED, UNCLASSIFIED, classify_seam, spacing_statistics
from coalres.config import Config
from coalres.grid import Grid


@pytest.fixture
def cfg():
    return Config.load()


@pytest.fixture
def grid():
    return Grid.from_extent(0, 2000, 0, 2000, cell_size=50.0)


def _points(coords, has_quality=True):
    return pd.DataFrame(
        {"east": [c[0] for c in coords], "north": [c[1] for c in coords],
         "has_quality": [has_quality] * len(coords)}
    )


def test_isolated_hole_never_reaches_measured(grid, cfg):
    """Satu lubang terisolasi tidak membuktikan kontinuitas apa pun.

    Ini kesalahan konseptual paling umum pada implementasi berbasis buffer:
    memberi radius Terukur di sekeliling satu lubang. Kriteria SNI adalah
    spasi antar titik, bukan jarak ke satu titik.
    """
    result = classify_seam(grid, _points([(1000, 1000)]), cfg)
    assert MEASURED not in np.unique(result)
    assert INDICATED not in np.unique(result)
    assert INFERRED in np.unique(result)  # satu titik cukup untuk Tereka


def test_cluster_of_three_reaches_measured(grid, cfg):
    coords = [(1000, 1000), (1100, 1000), (1050, 1100)]
    result = classify_seam(grid, _points(coords), cfg)
    assert (result == MEASURED).any()


def test_measured_requires_quality_data(grid, cfg):
    """Spasi rapat tanpa data kualitas tidak menghasilkan Terukur.

    Klasifikasi mencerminkan keyakinan pada tonase DAN kualitas, bukan hanya
    geometri. Lubang tanpa lab tidak boleh menaikkan kelas.
    """
    coords = [(1000, 1000), (1100, 1000), (1050, 1100)]
    without = classify_seam(grid, _points(coords, has_quality=False), cfg)
    assert not (without == MEASURED).any()
    with_quality = classify_seam(grid, _points(coords, has_quality=True), cfg)
    assert (with_quality == MEASURED).any()


def test_classification_respects_present_mask(grid, cfg):
    """Sel tanpa batubara tidak boleh punya kelas."""
    coords = [(1000, 1000), (1100, 1000), (1050, 1100)]
    mask = np.zeros(grid.shape, dtype=bool)
    result = classify_seam(grid, _points(coords), cfg, present_mask=mask)
    assert (result == UNCLASSIFIED).all()


def test_higher_class_overrides_lower(grid, cfg):
    coords = [(1000, 1000), (1050, 1000), (1000, 1050), (1050, 1050)]
    result = classify_seam(grid, _points(coords), cfg)
    centre = result[grid.nrows // 2, grid.ncols // 2]
    assert centre == MEASURED


def test_geological_condition_changes_reach(grid, cfg):
    """Menilai kondisi geologi lebih optimis melipatgandakan luas Terukur.

    Inilah titik manipulasi paling sulit dideteksi dalam estimasi sumberdaya:
    tidak ada satu pun angka di laporan yang terlihat salah.
    """
    coords = [(1000, 1000), (1100, 1000), (1050, 1100)]
    areas = {}
    for condition in ("complex", "moderate", "simple"):
        cfg.data["classification"]["geological_condition"] = condition
        areas[condition] = int((classify_seam(grid, _points(coords), cfg) == MEASURED).sum())
    assert areas["complex"] < areas["moderate"] < areas["simple"]


def test_spacing_statistics_suggests_cell_size():
    coords = [(x, y) for x in range(0, 1000, 200) for y in range(0, 1000, 200)]
    stats = spacing_statistics(_points(coords))
    assert stats["nearest_median"] == pytest.approx(200.0, abs=1e-6)
    assert stats["suggested_cell_size"] == pytest.approx(50.0, abs=1e-6)
