"""Uji terhadap sertifikat lab nyata DH09_05C1 (Report No. 01221.00923)."""
import numpy as np
import pytest

from coalres.quality import (
    adb_to_ar, adb_to_daf, dry_matter_ard, preston_sanders_insitu_ard,
)

# S10A (PH21.00629) dan S10B (PH21.00630), angka apa adanya dari sertifikat.
S10A = dict(tm=42.25, im=17.55, ash=10.84, vm=39.96, fc=31.65, ts=0.20,
            cv_adb=4815, cv_ar=3373, cv_daf=6724, rd=1.36)
S10B = dict(tm=43.73, im=18.56, ash=8.08, vm=40.96, fc=32.40, ts=0.31,
            cv_adb=5022, cv_ar=3470, cv_daf=6846, rd=1.35)


@pytest.mark.parametrize("s", [S10A, S10B])
def test_mass_balance_certificate(s):
    assert s["im"] + s["ash"] + s["vm"] + s["fc"] == pytest.approx(100.0, abs=0.01)


@pytest.mark.parametrize("s", [S10A, S10B])
def test_cv_ar_matches_certificate(s):
    assert adb_to_ar(s["cv_adb"], s["tm"], s["im"]) == pytest.approx(s["cv_ar"], abs=1)


@pytest.mark.parametrize("s", [S10A, S10B])
def test_cv_daf_matches_certificate(s):
    assert adb_to_daf(s["cv_adb"], s["im"], s["ash"]) == pytest.approx(s["cv_daf"], abs=1)


@pytest.mark.parametrize("s,expected", [(S10A, 1.2276), (S10B, 1.2182)])
def test_preston_sanders(s, expected):
    assert preston_sanders_insitu_ard(s["rd"], s["tm"], s["im"]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("s", [S10A, S10B])
def test_insitu_ard_is_lower_than_lab_for_low_rank_coal(s):
    """Arah efeknya, bukan sekadar besarannya.

    Untuk batubara dengan TM jauh di atas IM, air yang ditambahkan menarik
    densitas curah ke arah 1,0. RD in-situ LEBIH RENDAH dari RD lab, sehingga
    memakai RD lab MELEBIHKAN tonase. Salah arah di sini berarti tonase meleset
    ~10% ke arah yang berlawanan.
    """
    insitu = float(preston_sanders_insitu_ard(s["rd"], s["tm"], s["im"]))
    assert insitu < s["rd"]
    overstatement = s["rd"] / insitu - 1.0
    assert 0.08 < overstatement < 0.12


def test_preston_sanders_identity_when_tm_equals_im():
    assert preston_sanders_insitu_ard(1.36, 17.55, 17.55) == pytest.approx(1.36, abs=1e-9)


def test_high_rank_coal_barely_changes():
    """TM hampir sama dengan IM (batubara peringkat tinggi): koreksi kecil."""
    insitu = float(preston_sanders_insitu_ard(1.45, 8.0, 6.0))
    assert abs(insitu - 1.45) < 0.02


@pytest.mark.parametrize("s", [S10A, S10B])
def test_dry_matter_ard_plausible(s):
    """Uji apakah RD yang dilaporkan benar-benar apparent density."""
    assert 1.40 < float(dry_matter_ard(s["rd"], s["im"])) < 1.55


def test_dry_matter_ard_flags_true_density():
    """RD dari piknometer pada sampel digerus akan keluar dari rentang wajar."""
    assert float(dry_matter_ard(1.52, 17.55)) > 1.70
