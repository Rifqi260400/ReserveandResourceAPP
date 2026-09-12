"""Gerbang Tahap 2 diuji terhadap data nyata di data/minex.

Nilai yang dipatok di sini diverifikasi langsung dari berkas mentah, bukan dari
keluaran kode: bila kode berubah sehingga angkanya bergeser, ujinya gagal.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from coalres.audit.checks import AuditReport, Severity
from coalres.audit.gate2 import run_gate2
from coalres.config import Config
from coalres.io.minex import load_minex

ROOT = Path(__file__).resolve().parents[1]
GATED = ROOT / "config" / "minex_dummy.yaml"
RESOLVED = ROOT / "config" / "minex_dummy_resolved.yaml"


def _run(path: Path) -> AuditReport:
    cfg = Config.load(path)
    return run_gate2(load_minex(cfg), cfg, AuditReport())


def _undeclared_config() -> Config:
    """config/minex_dummy.yaml dengan seluruh deklarasi dicabut kembali.

    Dibangun di sini, bukan dibaca dari berkas, supaya uji "gerbang terbuka
    ketika belum dinyatakan" tidak ikut berubah setiap kali satu keputusan
    dimasukkan ke konfigurasi kerja.
    """
    cfg = Config.load(GATED)
    cfg.seam_policy = cfg.seam_policy.model_copy(
        update={"collision_resolution": "undeclared", "collision_basis": ""})
    cfg.weathering = cfg.weathering.model_copy(
        update={"provenance": "unknown", "provenance_basis": ""})
    cfg.observation_point = cfg.observation_point.model_copy(
        update={"requires_quality": None, "basis": ""})
    cfg.minex = cfg.minex.model_copy(
        update={"quality_column_basis": {}, "quality_basis_override_basis": ""})
    cfg.validation = cfg.validation.model_copy(
        update={"proximate_closure_waiver_basis": ""})
    return cfg


@pytest.fixture(scope="module")
def gated() -> AuditReport:
    if not GATED.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = _undeclared_config()
    return run_gate2(load_minex(cfg), cfg, AuditReport())


@pytest.fixture(scope="module")
def resolved() -> AuditReport:
    if not RESOLVED.exists():
        pytest.skip("dataset minex tidak tersedia")
    return _run(RESOLVED)


def _checks(report: AuditReport, severity: Severity) -> set[str]:
    return {f.check for f in report.of(severity)}


def test_undeclared_config_opens_every_gate(gated):
    # G1 tidak lagi muncul: penggantian nama A -> A1 menghapus tabrakannya di
    # akar, bukan menyelesaikannya dengan aturan hitung.
    assert _checks(gated, Severity.STOP) == {
        "G2_quality_basis", "G4_weathering", "G6_populations", "G7_proximate",
    }


def test_declaring_the_decisions_closes_every_gate(resolved):
    assert resolved.of(Severity.STOP) == []


def test_renaming_removed_the_collision_at_the_root(gated):
    """Tabrakan induk-anak sudah tidak ada pada konfigurasi kerja.

    Sebelumnya seam 'A' di 23 lubang bertabrakan dengan 'A1'/'A2' di 30 lubang,
    hull bertumpang 94,8 ha. Pemilik data menyatakan 'A' adalah seam yang MASIH
    MENYATU, jadi ia dinamai ulang 'A1'. Penamaan ulang menghapus tabrakannya
    di akar - tidak ada lagi induk dan anak yang menempati ruang yang sama -
    bukan menutupinya dengan aturan hitung.
    """
    assert "seam_collision" not in gated.tables
    assert not [f for f in gated.of(Severity.STOP) if f.check == "G1_seam_collision"]
    assert any("tidak ada seam induk" in f.message
               for f in gated.of(Severity.INFO) if f.check == "G1_seam_collision")


def test_the_alias_is_applied_once_at_read_time(gated):
    cfg = Config.load(GATED)
    dataset = load_minex(cfg)
    applied = dataset.provenance["seam_aliases"]
    assert applied["lithology"] == {"A -> A1": 23}
    assert applied["quality"] == {"A -> A1": 44}
    body = dataset.intervals[~dataset.intervals["is_marker"]]
    assert "A" not in set(body["seam"])
    assert body[body["seam"] == "A1"]["hole_id"].nunique() == 48
    assert body[body["seam"] == "A2"]["hole_id"].nunique() == 30


def test_the_collision_detector_still_fires_when_a_collision_exists():
    """Detektornya tidak dilemahkan - hanya datanya yang berubah."""
    cfg = Config.load(GATED)
    cfg.minex = cfg.minex.model_copy(update={"seam_aliases": {}})
    cfg.stratigraphy = ["A", "B"]
    cfg.seam_splits = {"A": ["A1", "A2"]}
    cfg.seam_policy = cfg.seam_policy.model_copy(
        update={"collision_resolution": "undeclared", "collision_basis": ""})
    report = run_gate2(load_minex(cfg), cfg, AuditReport())
    row = report.tables["seam_collision"].iloc[0]
    assert row["induk"] == "A" and row["anak"] == "A1, A2"
    assert row["lubang_keduanya"] == 0
    assert row["lubang_induk"] == 23 and row["lubang_anak"] == 30
    assert row["luas_hull_bertumpang_ha"] > 50
    assert "G1_seam_collision" in {f.check for f in report.of(Severity.STOP)}


def test_ash_cv_correlation_exposes_a_daf_column(gated):
    row = gated.tables["ash_cv_test"].iloc[0]
    assert row["n_sampel"] == 116
    # Apa adanya korelasinya lemah; sebagai daf ia kuat dan negatif.
    assert abs(row["r(ASH, CV apa adanya)"]) < 0.50
    assert row["r(ASH, CV jika daf -> adb)"] < -0.70
    assert any("daf" in f.message for f in gated.of(Severity.STOP)
               if f.check == "G2_quality_basis")


def test_the_ash_cv_test_leans_on_a_factor_the_data_does_not_support(gated):
    """Batas uji abu-kalori, dicatat agar tidak dilupakan.

    Konversi daf memakai faktor (100 - M - ASH)/100. Faktor itu hanya sah bila
    proksimat menutup 100%; di sini ia menutup 86%. Jadi bukti korelasi bersifat
    menunjuk, bukan membuktikan - dan G7 wajib diselesaikan lebih dulu.
    """
    proximate = gated.tables["proximate_closure"].iloc[0]
    assert proximate["jumlah_median_pct"] < 99.5
    assert "G7_proximate" in _checks(gated, Severity.STOP)


def test_declaring_adb_against_the_evidence_still_stops():
    cfg = Config.load(RESOLVED)
    cfg.minex = cfg.minex.model_copy(update={
        "quality_column_basis": {**cfg.minex.quality_column_basis, "CV": "adb"}})
    report = run_gate2(load_minex(cfg), cfg, AuditReport())
    stops = [f for f in report.of(Severity.STOP) if f.check == "G2_quality_basis"]
    assert stops and "BERTENTANGAN" in stops[0].message


def test_an_override_downgrades_the_contradiction_but_never_hides_it():
    """Menimpa gerbang boleh; menghilangkan temuannya tidak."""
    cfg = Config.load(RESOLVED)
    cfg.minex = cfg.minex.model_copy(update={
        "quality_column_basis": {**cfg.minex.quality_column_basis, "CV": "adb"},
        "quality_basis_override_basis": "Dinyatakan pemilik data setelah melihat bukti."})
    report = run_gate2(load_minex(cfg), cfg, AuditReport())
    assert not [f for f in report.of(Severity.STOP) if f.check == "G2_quality_basis"]
    warns = [f for f in report.of(Severity.WARN) if f.check == "G2_quality_basis"]
    assert warns and "BERTENTANGAN" in warns[0].message
    assert "DITIMPA" in warns[0].message


def test_proximate_does_not_close_on_this_dataset(gated):
    """Uji yang tidak bergantung basis: M + ASH + VM + FC harus 100%.

    Pada data ini jumlahnya 86%. Selama kekurangan 14% itu belum dijelaskan,
    setiap uji basis - termasuk uji abu-kalori di atas - bertumpu pada kolom
    yang belum tentu benar.
    """
    row = gated.tables["proximate_closure"].iloc[0]
    assert row["n_sampel"] == 116
    assert row["jumlah_median_pct"] == 86.0
    assert row["kekurangan_median_pct"] == 14.0
    assert "G7_proximate" in _checks(gated, Severity.STOP)


def test_weathering_constant_is_detected_as_a_typed_number(gated):
    table = gated.tables["weathering_depths"].set_index("kedalaman_m")["n_lubang"]
    assert table.loc[3.0] == 54 and table.sum() == 60
    assert any("54 dari 60" in f.message for f in gated.of(Severity.STOP)
               if f.check == "G4_weathering")


def test_the_working_config_records_weathering_as_an_assumption():
    cfg = Config.load(GATED)
    assert cfg.weathering.provenance == "assumed"
    assert cfg.weathering.constant_depth_m == 3.0
    assert len(cfg.weathering.provenance_basis.strip()) > 100


def test_the_working_config_requires_quality_at_observation_points():
    cfg = Config.load(GATED)
    assert cfg.observation_point.requires_quality is True
    assert len(cfg.observation_point.basis.strip()) > 100


def test_the_ten_centimetre_rule_makes_a1_and_a2_separate_seams():
    """Aturan pemilik data: interburden > 10 cm adalah seam berbeda.

    IB A1<->A2 paling tipis pun 40 cm, jadi keduanya seam berbeda di SELURUH
    25 lubang - menggabungkannya akan menelan parting median 3,40 m.
    """
    cfg = Config.load(GATED)
    assert cfg.cutoffs.max_parting_thickness_m == 0.10


def test_barren_hole_is_carried_through_as_evidence(gated):
    frame = gated.tables["barren_holes"]
    assert list(frame["hole_id"]) == ["H029"]
    assert frame.iloc[0]["total_depth_m"] == 60.0


def test_conflicting_duplicates_stop_but_identical_ones_only_warn(gated):
    summary = gated.tables["duplicate_summary"].set_index("berkas")
    assert summary.loc["survey", "duplikat_identik_dibuang"] == 60
    assert summary.loc["survey", "baris_akhir"] == 60
    assert int(summary["kunci_bertentangan"].sum()) == 0
    assert not [f for f in gated.of(Severity.STOP) if f.check == "G3_duplicates"]


def test_requiring_quality_shrinks_the_measured_class(gated):
    """Kualitas dihitung PER SEAM, dan itu jauh lebih menggigit.

    Diukur per lubang, seam B tampak punya 18 titik - seluruh lubang
    berkualitas. Diukur per seam, hanya 8 lubang yang punya kualitas pada B.
    Pengukuran per lubang melebih-lebihkan populasi lebih dari dua kali lipat.
    """
    frame = gated.tables["hole_populations"]
    totals = frame.groupby("populasi")["terukur_ha"].sum()
    assert totals["lubang berkualitas"] < totals["seluruh lubang"]
    per_seam = frame.set_index(["seam", "populasi"])["n_lubang"]
    assert per_seam[("B", "seluruh lubang")] == 59
    assert per_seam[("B", "lubang berkualitas")] == 8
    assert per_seam[("A1", "lubang berkualitas")] == 9
    assert per_seam[("A2", "lubang berkualitas")] == 10
    # Susutnya sekitar separuh, bukan seperempat seperti pengukuran per lubang.
    shrink = 1 - totals["lubang berkualitas"] / totals["seluruh lubang"]
    assert 0.45 < shrink < 0.60
