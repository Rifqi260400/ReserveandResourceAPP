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


@pytest.fixture(scope="module")
def gated() -> AuditReport:
    if not GATED.exists():
        pytest.skip("dataset minex tidak tersedia")
    return _run(GATED)


@pytest.fixture(scope="module")
def resolved() -> AuditReport:
    if not RESOLVED.exists():
        pytest.skip("dataset minex tidak tersedia")
    return _run(RESOLVED)


def _checks(report: AuditReport, severity: Severity) -> set[str]:
    return {f.check for f in report.of(severity)}


def test_undeclared_config_opens_every_gate(gated):
    assert _checks(gated, Severity.STOP) == {
        "G1_seam_collision", "G2_quality_basis", "G4_weathering", "G6_populations"
    }


def test_declaring_the_decisions_closes_every_gate(resolved):
    assert resolved.of(Severity.STOP) == []


def test_parent_child_collision_is_areal_not_per_hole(gated):
    """Tidak satu pun lubang memuat A dan A1/A2 sekaligus.

    Justru itu sebabnya pemeriksaan per lubang tidak menemukan apa pun, dan
    tesselasi terpisah menghitung area yang sama dua kali.
    """
    row = gated.tables["seam_collision"].iloc[0]
    assert row["induk"] == "A" and row["anak"] == "A1, A2"
    assert row["lubang_keduanya"] == 0
    assert row["lubang_induk"] == 23 and row["lubang_anak"] == 30
    assert row["luas_hull_bertumpang_ha"] > 50


def test_ash_cv_correlation_exposes_a_daf_column(gated):
    row = gated.tables["ash_cv_test"].iloc[0]
    assert row["n_sampel"] == 116
    # Apa adanya korelasinya lemah; sebagai daf ia kuat dan negatif.
    assert abs(row["r(ASH, CV apa adanya)"]) < 0.50
    assert row["r(ASH, CV jika daf -> adb)"] < -0.70
    assert any("daf" in f.message for f in gated.of(Severity.STOP)
               if f.check == "G2_quality_basis")


def test_declaring_adb_against_the_evidence_still_stops():
    cfg = Config.load(RESOLVED)
    cfg.minex.quality_column_basis = {**cfg.minex.quality_column_basis, "CV": "adb"}
    report = run_gate2(load_minex(cfg), cfg, AuditReport())
    stops = [f for f in report.of(Severity.STOP) if f.check == "G2_quality_basis"]
    assert stops and "BERTENTANGAN" in stops[0].message


def test_weathering_constant_is_detected_as_a_typed_number(gated):
    table = gated.tables["weathering_depths"].set_index("kedalaman_m")["n_lubang"]
    assert table.loc[3.0] == 54 and table.sum() == 60
    assert any("54 dari 60" in f.message for f in gated.of(Severity.STOP)
               if f.check == "G4_weathering")


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
    frame = gated.tables["hole_populations"]
    totals = frame.groupby("populasi")["terukur_ha"].sum()
    assert totals["lubang berkualitas"] < totals["seluruh lubang"]
    # 18 dari 60 lubang punya kualitas; plafon kelas ikut turun.
    assert frame[frame["seam"] == "B"].set_index("populasi").loc[
        "lubang berkualitas", "n_lubang"] == 18
