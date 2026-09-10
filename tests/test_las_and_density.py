"""LAS cacah mentah, dan koreksi terhadap catatan RD di spesifikasi."""
import pytest

from coalres.density import (
    dry_matter_ard, mass_fraction_basis_conversion, preston_sanders_insitu_ard,
)
from coalres.errors import MissingDataError
from coalres.io.las import load_las

# Sertifikat asli 01221.00923, DH09_05C1.
S10A = dict(tm=42.25, im=17.55, rd=1.36, cv_ar=3373)
S10B = dict(tm=43.73, im=18.56, rd=1.35, cv_ar=3470)


@pytest.fixture(scope="module")
def las():
    from conftest import LAS_DIR
    path = LAS_DIR / "DH09_05C1.LAS"
    if not path.exists():
        pytest.skip("LAS referensi tidak tersedia")
    return load_las(path)


def test_las_density_curves_are_raw_counts(las):
    """LD dan SD bersatuan CPS, bukan g/cc - harus terblokir dari tonase."""
    assert las.density_curves_are_raw_counts
    for mnemonic in ("LD", "SD"):
        curve = las.curves[mnemonic]
        assert curve.unit.upper() == "CPS"
        assert not curve.is_calibrated_density
        assert curve.data[curve.data > 0].max() > 1000     # ribuan cacah


def test_las_curve_coverage_differs_between_sensors(las):
    """Offset sensor dalam rangkaian tool: gamma berhenti lebih dangkal."""
    gr = las.curve_coverage("GR")
    ld = las.curve_coverage("LD")
    assert gr is not None and ld is not None
    assert ld[1] > gr[1]


def test_caliper_present_despite_collar_flag(las):
    """Sheet Collar mencatat CAL = 'x' (tidak ada), tetapi LAS memuatnya.

    Metadata workbook tidak boleh dipercaya melebihi file LAS itu sendiri.
    """
    assert "CL" in las.curves
    assert las.curves["CL"].n_valid > 4000


@pytest.mark.parametrize("s,expected", [(S10A, 1.2276), (S10B, 1.2182)])
def test_preston_sanders_against_certificate(s, expected):
    got = preston_sanders_insitu_ard(s["rd"], s["tm"], s["im"])
    assert got == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("s", [S10A, S10B])
def test_insitu_ard_is_lower_than_air_dried(s):
    """Arahnya, bukan hanya besarannya: memakai RD lab MELEBIHKAN tonase."""
    insitu = preston_sanders_insitu_ard(s["rd"], s["tm"], s["im"])
    assert insitu < s["rd"]
    assert 0.08 < s["rd"] / insitu - 1.0 < 0.12


def test_spec_note_about_194_comes_from_the_wrong_formula():
    """Spesifikasi menyebut 1,36 air-dried -> ~1,94 t/m3 sebagai bukti bahwa
    basisnya bukan air-dried. Angka itu berasal dari rumus konversi KADAR yang
    diterapkan pada densitas; konversi densitas yang benar memberi 1,2276."""
    naive = mass_fraction_basis_conversion(
        S10A["rd"], S10A["tm"], S10A["im"], direction="ar_to_adb"
    )
    assert naive == pytest.approx(1.9417, abs=1e-3)     # angka di spesifikasi

    correct = preston_sanders_insitu_ard(S10A["rd"], S10A["tm"], S10A["im"])
    assert correct == pytest.approx(1.2276, abs=1e-4)
    assert correct < S10A["rd"] < naive


def test_identity_when_tm_equals_im():
    assert preston_sanders_insitu_ard(1.36, 17.55, 17.55) == pytest.approx(1.36, abs=1e-9)


@pytest.mark.parametrize("s", [S10A, S10B])
def test_dry_matter_ard_supports_apparent_density_reading(s):
    """1,36 dengan M_adb 17,55% menyiratkan matriks kering 1,47 - wajar untuk
    batubara ash ~11%. Itu mendukung pembacaan ARD air-dried."""
    assert 1.40 < dry_matter_ard(s["rd"], s["im"]) < 1.55


def test_true_density_reported_as_ard_falls_outside_plausible_range():
    assert dry_matter_ard(1.52, 17.55) > 1.70


def test_no_silent_default_when_moisture_missing():
    with pytest.raises(MissingDataError):
        preston_sanders_insitu_ard(1.36, float("nan"), 17.55)
