"""Ekspor grid Surfer (.grd): uncut, cut, dan atribut kualitas."""
import numpy as np
import pytest

from coalres.grdout import (
    SURFER_BLANK, read_surfer_ascii, write_surfer_ascii, write_surfer_binary, write_sidecar,
)
from coalres.topo import build_surface


@pytest.fixture
def surface():
    xs = np.linspace(0, 500, 8)
    ys = np.linspace(0, 400, 8)
    gx, gy = np.meshgrid(xs, ys)
    return build_surface(gx.ravel(), gy.ravel(), 10.0 + 0.01 * gx.ravel(),
                         spacing=50.0, name="uji")


def test_ascii_round_trip_preserves_values(surface):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = write_surfer_ascii(surface, Path(tmp) / "a.grd")
        z, header = read_surfer_ascii(path)
    assert z.shape == surface.z.shape
    assert header["nx"] == len(surface.x) and header["ny"] == len(surface.y)
    ok = np.isfinite(surface.z) & np.isfinite(z)
    assert np.allclose(z[ok], surface.z[ok], atol=1e-5)


def test_nan_is_written_as_surfer_blank_not_zero(tmp_path, surface):
    """Sel bernilai nol dan sel tanpa data adalah dua hal berbeda.

    Menyamakannya menciptakan ketebalan nol palsu di area tanpa dukungan data.
    """
    values = np.array(surface.z, dtype=float)
    values[0, 0] = np.nan
    path = write_surfer_ascii(surface, tmp_path / "a.grd", values)
    text = path.read_text()
    assert "1.70141e+38" in text

    z, _ = read_surfer_ascii(path)
    assert np.isnan(z[0, 0])
    assert not np.isclose(z[0, 0], 0.0, equal_nan=False)


def test_rows_run_from_low_y_to_high_y(tmp_path, surface):
    """Spesifikasi DSAA: baris pertama adalah Y terendah."""
    values = np.zeros(surface.z.shape)
    values[0, :] = 1.0          # baris Y terendah
    values[-1, :] = 9.0
    path = write_surfer_ascii(surface, tmp_path / "a.grd", values)
    rows = path.read_text().splitlines()[5:]
    assert rows[0].split()[0].startswith("1.")
    assert rows[-1].split()[0].startswith("9.")


def test_binary_header_is_dsbb(tmp_path, surface):
    path = write_surfer_binary(surface, tmp_path / "a.grd")
    assert path.read_bytes()[:4] == b"DSBB"


def test_shape_mismatch_is_rejected(tmp_path, surface):
    with pytest.raises(ValueError, match="tidak sama"):
        write_surfer_ascii(surface, tmp_path / "a.grd", np.zeros((3, 3)))


def test_sidecar_carries_warnings(tmp_path, surface):
    path = write_sidecar(
        tmp_path / "a.txt", name="A_uncut", description="uji", units="meter",
        surface=surface, support_note="convex hull",
        warnings=["BUKAN ketebalan yang dapat ditambang"],
    )
    text = path.read_text()
    assert "PERINGATAN" in text and "BUKAN ketebalan yang dapat ditambang" in text
    assert "1.70141e+38" in text


# --------------------------------------------------------------------------- #
# Uncut vs cut - inti perbedaannya
# --------------------------------------------------------------------------- #
def test_uncut_keeps_intersections_below_cutoff():
    """Grid uncut WAJIB dibangun dari interseksi PRA-cutoff.

    Memakai interseksi yang sudah lolos cutoff menghasilkan berkas bernama
    uncut yang isinya cut, dan tidak ada cara membedakannya dari yang benar.
    """
    from conftest import ROOT
    from coalres.config import Config
    from coalres.io.minex import load_minex
    from coalres.seams import build_intersections_from_dataset

    path = ROOT / "config" / "minex_dummy_resolved.yaml"
    if not path.exists():
        pytest.skip("konfigurasi Minex tidak tersedia")
    cfg = Config.load(path)
    dataset = load_minex(cfg)

    uncut = build_intersections_from_dataset(dataset, cfg, apply_cutoffs=False)
    cut = build_intersections_from_dataset(dataset, cfg, apply_cutoffs=True)

    assert len(uncut) > len(cut)
    thinnest_uncut = min(i.coal_thickness_m for i in uncut)
    thinnest_cut = min(i.coal_thickness_m for i in cut)
    assert thinnest_uncut < cfg.cutoffs.min_seam_thickness_m <= thinnest_cut


def test_uncut_grid_is_wider_than_cut_grid():
    from conftest import ROOT

    directory = ROOT / "output" / "minex" / "grd"
    if not (directory / "A_uncut.grd").exists():
        pytest.skip("keluaran grd belum dihasilkan")
    uncut, _ = read_surfer_ascii(directory / "A_uncut.grd")
    cut, _ = read_surfer_ascii(directory / "A_cut.grd")
    assert np.isfinite(uncut).sum() > np.isfinite(cut).sum()
    assert np.nanmin(uncut) < np.nanmin(cut)
