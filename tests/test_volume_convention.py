"""Konvensi volume - inti kebenaran seluruh angka tonase.

Volume = luas DALAM PETA x ketebalan VERTIKAL, dan ini benar berapa pun dip-nya.
Menerapkan koreksi cos(dip) pada volume adalah kesalahan yang MENGURANGI tonase
secara keliru. Koreksi itu hanya untuk cutoff ketebalan.
"""
import numpy as np
import pytest

from coalres.grid import Grid, surface_dip_degrees


@pytest.mark.parametrize("dip_deg", [0.0, 5.0, 15.0, 30.0, 45.0])
def test_vertical_prism_volume_is_dip_invariant(dip_deg):
    """Seam dengan true thickness sama menghasilkan volume sama, berapa pun dip.

    Tebal vertikal membesar mengikuti 1/cos(dip), sementara luas bidang seam
    yang sebenarnya juga membesar 1/cos(dip) - keduanya saling meniadakan pada
    prisma vertikal.
    """
    true_thickness = 2.0
    plan_area = 10_000.0  # 100 m x 100 m
    vertical_thickness = true_thickness / np.cos(np.radians(dip_deg))

    volume_prism = plan_area * vertical_thickness
    true_area = plan_area / np.cos(np.radians(dip_deg))
    volume_true = true_area * true_thickness

    assert volume_prism == pytest.approx(volume_true, rel=1e-12)


def test_applying_cos_dip_to_volume_understates_tonnage():
    """Kesalahan yang sengaja dijaga agar tidak masuk kembali ke kode."""
    plan_area, vertical_thickness, dip = 10_000.0, 2.309, 30.0
    correct = plan_area * vertical_thickness
    wrong = plan_area * vertical_thickness * np.cos(np.radians(dip))
    assert wrong < correct
    assert wrong / correct == pytest.approx(np.cos(np.radians(dip)), rel=1e-9)


def test_dip_from_surface_gradient():
    """Dip diturunkan dari gradien grid struktur, bukan dari kemiringan lubang.

    Lubang vertikal tidak membuat dip menjadi nol - ia hanya membuat
    perhitungannya sederhana.
    """
    grid = Grid.from_extent(0, 1000, 0, 1000, cell_size=50.0)
    gx, _ = grid.meshgrid()
    for dip_deg in (5.0, 10.0, 20.0):
        surface = -gx * np.tan(np.radians(dip_deg))
        computed = surface_dip_degrees(surface, grid.cell_size)
        assert float(np.median(computed)) == pytest.approx(dip_deg, abs=0.01)


def test_true_thickness_is_smaller_than_vertical():
    vertical, dip = 0.35, 25.0
    true = vertical * np.cos(np.radians(dip))
    assert true < vertical
    # Cutoff 0,30 m: lolos pada tebal vertikal, gagal pada true thickness.
    assert vertical > 0.30 and true < 0.32
