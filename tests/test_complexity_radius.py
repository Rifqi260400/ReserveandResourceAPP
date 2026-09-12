"""Tahap 7 dan 8: penilaian kompleksitas dan pencarian radius."""
from __future__ import annotations

from pathlib import Path

import pytest

from coalres import complexity, radius
from coalres.config import Config
from coalres.io.minex import load_minex

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"

INTRUSI = ("intrusi", "sederhana",
           "Tidak ada indikasi batuan beku pada seluruh 60 lubang maupun peta regional.")


@pytest.fixture(scope="module")
def data():
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    return load_minex(cfg), cfg


# --- tahap 8: fungsi murni -------------------------------------------------

def test_radius_lookup_takes_no_config():
    """Tandatangannya hanya kondisi dan kelas. Tidak ada jalan masuk config."""
    import inspect
    assert list(inspect.signature(radius.radius_m).parameters) == ["condition", "klass"]


def test_radii_ascend_within_every_condition():
    for condition in ("sederhana", "moderat", "kompleks"):
        r = radius.radii_for(condition)
        assert r["terukur"] < r["tertunjuk"] < r["tereka"]


def test_simpler_geology_never_gets_a_smaller_radius():
    for klass in ("terukur", "tertunjuk", "tereka"):
        assert (radius.radius_m("sederhana", klass)
                > radius.radius_m("moderat", klass)
                > radius.radius_m("kompleks", klass))


def test_unknown_condition_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="tidak dikenal"):
        radius.radius_m("sangat_sederhana", "terukur")
    with pytest.raises(ValueError, match="tidak dikenal"):
        radius.radius_m("sederhana", "terbukti")


def test_config_radii_that_differ_from_the_table_are_reported():
    differences = radius.compare_to_table(
        "sederhana", {"terukur": 250.0, "tertunjuk": 1000.0, "tereka": 1500.0})
    assert len(differences) == 1 and "terukur" in differences[0]
    assert radius.compare_to_table("sederhana", radius.radii_for("sederhana")) == []


# --- tahap 7: penilaian ----------------------------------------------------

def test_seven_of_eight_parameters_are_assessable_from_this_data(data):
    dataset, cfg = data
    suggestions = complexity.suggest(dataset, cfg)
    assert len(suggestions) == 8
    unassessable = [s.parameter for s in suggestions if not s.assessable]
    # Intrusi tidak dapat dinilai: berkas lit hanya memuat interval seam.
    assert unassessable == ["intrusi"]


def test_every_suggestion_carries_its_evidence_and_threshold(data):
    dataset, cfg = data
    for suggestion in complexity.suggest(dataset, cfg):
        assert suggestion.evidence.strip()
        assert suggestion.threshold.strip()


def test_an_unassessable_parameter_blocks_rather_than_defaulting(data):
    dataset, cfg = data
    with pytest.raises(ValueError, match="belum dinyatakan manusia"):
        complexity.assess(dataset, cfg)


def test_a_tie_is_never_broken_by_the_code(data):
    """Skor pada dataset ini seri 3-3.

    Pengurutan stabil akan memenangkan 'sederhana' - kelas dengan radius
    TERBESAR - tanpa gejala apa pun. Seri harus menghentikan run.
    """
    dataset, cfg = data
    with pytest.raises(complexity.TiedAssessment, match="seri"):
        complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]})


def test_breaking_a_tie_requires_a_substantive_reason(data):
    dataset, cfg = data
    with pytest.raises(ValueError, match="terlalu pendek"):
        complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]},
                          condition_override=("moderat", "konservatif"))


def test_a_broken_tie_is_recorded_as_a_mandatory_warning(data):
    dataset, cfg = data
    reason = ("Seri 3-3 diputus ke moderat sebagai arah konservatif karena "
              "bentang dan dip belum diverifikasi penampang.")
    assessment = complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]},
                                   condition_override=("moderat", reason))
    assert assessment.condition == "moderat"
    assert any("diputus manusia" in w for w in assessment.warnings)
    assert any(reason in w for w in assessment.warnings)


def test_thin_justification_on_a_parameter_override_is_refused(data):
    dataset, cfg = data
    with pytest.raises(ValueError, match="terlalu pendek"):
        complexity.assess(dataset, cfg, {"intrusi": ("sederhana", "tidak ada")})


def test_both_mandatory_warnings_are_always_emitted():
    """Keduanya wajib, apa pun hasilnya - tidak dapat dimatikan."""
    entries = [
        complexity.Entry(p, g, "sederhana", "x" * 50, "manual")
        for g, params in complexity.FORM.items() for p in params
    ]
    assessment = complexity.tally_and_warn(entries)
    assert assessment.condition == "sederhana" and assessment.margin == 8
    assert len(assessment.warnings) == 2
    assert any("bias pembobotan kelompok" in w for w in assessment.warnings)
    assert any("margin tepi jurang" in w for w in assessment.warnings)


def test_the_group_bias_warning_names_the_heaviest_group():
    entries = [
        complexity.Entry(p, g, "moderat", "x" * 50, "manual")
        for g, params in complexity.FORM.items() for p in params
    ]
    warning = next(w for w in complexity.tally_and_warn(entries).warnings
                   if "bias pembobotan" in w)
    # Tektonik memuat 4 dari 8 parameter - kelompok terberat.
    assert "tektonik" in warning and "4 dari 8" in warning


def test_the_form_matches_the_reference_report_shape():
    """Tabel 5-32: 8 parameter dalam tiga kelompok."""
    sizes = {g: len(p) for g, p in complexity.FORM.items()}
    assert sizes == {"sedimentasi": 3, "tektonik": 4, "kualitas": 1}
    assert sum(sizes.values()) == 8
    for params in complexity.FORM.values():
        for options in params.values():
            assert set(options) == set(complexity.SCORES)
