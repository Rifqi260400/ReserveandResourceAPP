"""KCMI 4.5.3: jarak antar PoO dari variabilitas seam."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coalres import variography
from coalres.config import Config
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"


@pytest.fixture(scope="module")
def variograms():
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    dataset = load_minex(cfg)
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    return variography.analyse(intersections, dataset.collars.set_index("hole_id"))


def test_the_threshold_is_the_one_kcmi_cites():
    """Journel & Huijbregts 1978, dikutip KCMI 4.5.3."""
    assert variography.GEOSTATISTICS_MIN_DATA == 30


def test_seams_below_thirty_take_the_sni_route(variograms):
    for seam in ("A", "A1"):
        v = variograms[seam]
        assert v.n_data < 30
        assert v.route == "sni_kompleksitas"
        assert "Journel" in v.note


def test_seams_at_or_above_thirty_take_the_geostatistics_route(variograms):
    for seam in ("A2", "B"):
        assert variograms[seam].n_data >= 30
        assert variograms[seam].route == "geostatistik"


def test_a_range_beyond_half_the_data_extent_is_refused(variograms):
    """Range 3.867 m dan 4.083 m melewati separuh bentang datanya sendiri.

    Variogram hanya dapat dipercaya sampai sekitar separuh bentang. Range
    sebesar itu menandakan sill belum tercapai karena ada tren, bukan korelasi
    sejauh itu - memakainya akan membenarkan spasi bor yang jauh lebih longgar
    daripada yang didukung data.
    """
    for seam in ("A2", "B"):
        v = variograms[seam]
        assert v.range_m > variography.MAX_RELIABLE_RANGE_FRAC * v.data_extent_m
        assert not v.range_usable
        assert "TIDAK DAPAT DIPAKAI" in v.note


def test_no_seam_in_this_dataset_yields_a_usable_range(variograms):
    """Jadi seluruhnya jatuh ke tabel kompleksitas SNI - jalur cadangan KCMI."""
    assert not any(v.range_usable for v in variograms.values())


def test_a_weak_fit_is_refused_even_within_extent():
    """R2 rendah membatalkan range meski panjangnya masuk akal."""
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 4000, size=(60, 2))
    values = rng.normal(5.0, 1.0, size=60)      # murni acak: tanpa struktur
    v = variography.for_seam("X", points, values)
    assert v.route == "geostatistik"
    assert not v.range_usable


def test_a_clean_spherical_field_recovers_its_range():
    """Uji positif: data berstruktur nyata harus memberi range yang terpakai."""
    rng = np.random.default_rng(3)
    n, length_scale = 400, 80.0
    points = rng.uniform(0, 2000, size=(n, 2))
    # Bidang berulang tiap ~2*pi*length_scale (~500 m), jadi korelasinya habis
    # jauh di bawah separuh bentang (~1400 m).
    values = (np.sin(points[:, 0] / length_scale) * np.cos(points[:, 1] / length_scale)
              + rng.normal(0, 0.05, size=n))
    v = variography.for_seam("X", points, values)
    assert v.range_usable
    assert v.fit_quality > 0.5
    assert 0 < v.range_m < variography.MAX_RELIABLE_RANGE_FRAC * v.data_extent_m


def test_the_spherical_model_flattens_at_the_sill():
    h = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    gamma = variography._spherical(h, nugget=1.0, sill=4.0, rng=200.0)
    assert gamma[0] == pytest.approx(1.0)
    assert gamma[-1] == pytest.approx(5.0)
    assert gamma[-2] == pytest.approx(5.0)
    assert np.all(np.diff(gamma) >= -1e-12)


def test_variability_is_measured_per_seam_as_kcmi_requires(variograms):
    """KCMI 4.5.3: "dilakukan pada TIAP SEAM"."""
    assert set(variograms) == {"A", "A1", "A2", "B"}
    assert len({v.n_data for v in variograms.values()}) > 1
