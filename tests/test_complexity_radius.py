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


def test_the_table_matches_tabel_5_33_of_the_reference_report():
    """Pita kumulatif Tabel 5-33 (mengutip SNI 5015:2019).

    Pencocokan ini memperbaiki dua kekeliruan pada baris kompleks: tertunjuk
    tertulis 200 m (seharusnya 250) dan tereka 400 m (seharusnya 500).
    """
    assert radius.SNI_5015_2019_RADII_M == {
        "sederhana": {"terukur": 500.0, "tertunjuk": 1000.0, "tereka": 1500.0},
        "moderat": {"terukur": 250.0, "tertunjuk": 500.0, "tereka": 1000.0},
        "kompleks": {"terukur": 100.0, "tertunjuk": 250.0, "tereka": 500.0},
    }


def test_the_assessment_is_the_only_road_to_a_radius(data):
    """Urutan yang ditetapkan pemilik data: bobot dulu, baru radius.

    Tidak ada masukan lain - variogram, jarak bor, setelan config - yang boleh
    menggeser radius. Satu-satunya jalan lewat hasil pembobotan tahap 7.
    """
    dataset, cfg = data
    reason = ("Diputus ke moderat sebagai arah konservatif karena bentang dan "
              "dip belum diverifikasi penampang.")
    assessment = complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]})
    assert radius.from_assessment(assessment) == radius.radii_for(assessment.condition)


def test_without_a_weighting_there_is_no_radius():
    class Unassessed:
        condition = None
    with pytest.raises(ValueError, match="belum menghasilkan kelas"):
        radius.from_assessment(Unassessed())


def test_a_variogram_range_never_becomes_a_radius():
    """KCMI 4.5.3 dihitung, tetapi tempatnya di HULU formulir kompleksitas.

    Ia mengisi skor 'kesinambungan' dengan angka, lalu pembobotan yang memilih
    radius. Range tidak pernah jadi radius secara langsung.
    """
    import inspect
    from coalres import variography
    source = inspect.getsource(radius)
    assert "variogram" not in source.lower().replace(
        "variogram - bukti", "").split("def from_assessment")[0]
    # Jalur satu-satunya dari variogram adalah usulan skor formulir.
    assert hasattr(variography, "continuity_evidence")
    assert "usul_kesinambungan" in inspect.getsource(variography.continuity_evidence)


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


def _entries(**scores) -> list[complexity.Entry]:
    groups = {p: g for g, params in complexity.FORM.items() for p in params}
    return [complexity.Entry(p, groups[p], s, "x" * 50, "manual")
            for p, s in scores.items()]


SGM_REPORT_CHECKLIST = dict(
    variasi="moderat", kesinambungan="sederhana", percabangan="moderat",
    sesar="sederhana", lipatan="sederhana", intrusi="sederhana",
    kemiringan="sederhana", variasi_kualitas="sederhana",
)


def test_the_method_reproduces_the_reference_report_conclusion():
    """Ceklis Tabel 5-32 PT SGM harus menghasilkan 'sederhana'.

    Ini uji paling kuat yang tersedia: satu-satunya penilaian lengkap yang
    hasilnya sudah diketahui. Sedimentasi (2+1+2)/3 = 1,667; tektonik
    (1+1+1+1)/4 = 1,000; kualitas 1,000. Rata-rata ketiganya 1,222 -> sederhana.
    """
    assessment = complexity.tally_and_warn(_entries(**SGM_REPORT_CHECKLIST))
    assert assessment.aspect_values["sedimentasi"] == pytest.approx(5 / 3)
    assert assessment.aspect_values["tektonik"] == pytest.approx(1.0)
    assert assessment.aspect_values["kualitas"] == pytest.approx(1.0)
    assert assessment.final_value == pytest.approx(1.2222, abs=1e-4)
    assert assessment.condition == "sederhana"


def test_aspects_carry_equal_weight_regardless_of_subaspect_count():
    """Kualitas hanya satu subaspek, tetapi tetap bernilai sepertiga.

    Dengan penjumlahan ceklis mentah, tujuh 'sederhana' melawan satu
    'kompleks' akan menang telak. Dengan rata-rata dua tingkat, satu subaspek
    kompleks pada aspek Kualitas menaikkan nilai akhir sepenuh sepertiga.
    """
    all_simple = complexity.tally_and_warn(
        _entries(**{p: "sederhana" for p in
                    (q for params in complexity.FORM.values() for q in params)}))
    quality_complex = complexity.tally_and_warn(
        _entries(**{**{p: "sederhana" for p in
                       (q for params in complexity.FORM.values() for q in params)},
                    "variasi_kualitas": "kompleks"}))
    assert all_simple.final_value == pytest.approx(1.0)
    # (1 + 1 + 3)/3 = 1,667 - naik dua pertiga dari satu ceklis saja.
    assert quality_complex.final_value == pytest.approx(5 / 3)
    assert quality_complex.condition == "moderat"


def test_a_value_exactly_on_a_band_boundary_stops(data):
    """1,5 dan 2,5 mendua. Membulatkan ke bawah selalu memilih radius terbesar."""
    tied = _entries(variasi="moderat", kesinambungan="moderat", percabangan="moderat",
                    sesar="sederhana", lipatan="sederhana", intrusi="sederhana",
                    kemiringan="sederhana", variasi_kualitas="moderat")
    # (2 + 1 + 2)/3 = 1,667... bukan tepat batas; bangun yang tepat 1,5:
    exact = _entries(variasi="moderat", kesinambungan="moderat", percabangan="moderat",
                     sesar="sederhana", lipatan="sederhana", intrusi="sederhana",
                     kemiringan="sederhana", variasi_kualitas="sederhana")
    assert complexity.tally_and_warn(tied).final_value == pytest.approx(5 / 3)
    # (2 + 1 + 1)/3 = 1,333 -> sederhana, masih bukan batas.
    assert complexity.tally_and_warn(exact).condition == "sederhana"
    # Tepat 1,5: sedimentasi 2, tektonik 1,5, kualitas 1 -> (2+1.5+1)/3 = 1,5.
    boundary = _entries(variasi="moderat", kesinambungan="moderat",
                        percabangan="moderat", sesar="moderat", lipatan="moderat",
                        intrusi="sederhana", kemiringan="sederhana",
                        variasi_kualitas="sederhana")
    assert complexity.aspect_value(boundary, "tektonik") == pytest.approx(1.5)
    with pytest.raises(complexity.TiedAssessment, match="batas pita"):
        complexity.tally_and_warn(boundary)


def test_breaking_a_boundary_requires_a_substantive_reason(data):
    dataset, cfg = data
    with pytest.raises(ValueError, match="terlalu pendek"):
        complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]},
                          condition_override=("kompleks", "konservatif"))


def test_overriding_the_class_is_recorded_as_a_mandatory_warning(data):
    dataset, cfg = data
    reason = ("Ditimpa ke kompleks karena variasi kualitas dan ketebalan belum "
              "diverifikasi penampang di area barat.")
    assessment = complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]},
                                   condition_override=("kompleks", reason))
    assert assessment.condition == "kompleks"
    assert any("ditimpa manusia" in w for w in assessment.warnings)
    assert any(reason in w for w in assessment.warnings)


def test_thin_justification_on_a_parameter_override_is_refused(data):
    dataset, cfg = data
    with pytest.raises(ValueError, match="terlalu pendek"):
        complexity.assess(dataset, cfg, {"intrusi": ("sederhana", "tidak ada")})


def test_both_mandatory_warnings_are_always_emitted():
    """Keduanya wajib, apa pun hasilnya - tidak dapat dimatikan."""
    all_params = [p for params in complexity.FORM.values() for p in params]
    assessment = complexity.tally_and_warn(
        _entries(**{p: "sederhana" for p in all_params}))
    assert assessment.condition == "sederhana"
    assert assessment.margin == pytest.approx(0.5)
    assert len(assessment.warnings) == 2
    assert any("cara pembobotan" in w for w in assessment.warnings)
    assert any("margin tepi jurang" in w for w in assessment.warnings)


def test_the_weighting_warning_flags_the_reference_reports_different_method():
    """Laporan rujukan MENJUMLAH ceklis (6 lawan 2); di sini dirata-ratakan."""
    all_params = [p for params in complexity.FORM.values() for p in params]
    warning = next(w for w in complexity.tally_and_warn(
        _entries(**{p: "moderat" for p in all_params})).warnings
        if "cara pembobotan" in w)
    assert "6 lawan 2" in warning and "bobot SAMA" in warning


def test_a_thin_margin_is_named_as_thin():
    warning = next(w for w in complexity.tally_and_warn(_entries(
        variasi="moderat", kesinambungan="moderat", percabangan="sederhana",
        sesar="sederhana", lipatan="sederhana", intrusi="sederhana",
        kemiringan="sederhana", variasi_kualitas="moderat")).warnings
        if "margin tepi jurang" in w)
    assert "TIPIS" in warning


def test_subaspect_assessment_is_a_checklist_not_a_number(data):
    """Ceklis: satu kolom ditandai per subaspek, tidak ada angka dimasukkan."""
    dataset, cfg = data
    frame = complexity.assess(dataset, cfg, {INTRUSI[0]: INTRUSI[1:]}).to_checklist_frame()
    rows = frame[frame["subaspek"].isin(
        [p for params in complexity.FORM.values() for p in params])]
    assert len(rows) == 8
    for _, row in rows.iterrows():
        assert sum(1 for s in complexity.SCORES if row[s] == "v") == 1


def test_the_form_matches_the_reference_report_shape():
    """Tabel 5-32: 8 parameter dalam tiga kelompok."""
    sizes = {g: len(p) for g, p in complexity.FORM.items()}
    assert sizes == {"sedimentasi": 3, "tektonik": 4, "kualitas": 1}
    assert sum(sizes.values()) == 8
    for params in complexity.FORM.values():
        for options in params.values():
            assert set(options) == set(complexity.SCORES)
