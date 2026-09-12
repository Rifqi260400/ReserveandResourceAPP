"""Orkestrator 11 tahap dan perilaku gerbangnya."""
from __future__ import annotations

from pathlib import Path

import pytest

from coalres.config import Config
from coalres.ui import pipeline

ROOT = Path(__file__).resolve().parents[1]
GATED = ROOT / "config" / "minex_dummy.yaml"
FULL = ROOT / "config" / "minex_demo_lengkap.yaml"
INTRUSI = {"intrusi": ("sederhana",
                       "Tidak ada indikasi batuan beku pada 60 lubang maupun peta regional.")}


def _skip_unless(path: Path):
    if not path.exists():
        pytest.skip(f"konfigurasi tidak tersedia: {path.name}")


def test_an_undeclared_complexity_parameter_is_a_gate_not_a_crash():
    """Tahap 7 menunggu keputusan manusia; ia berhenti, tidak meledak."""
    _skip_unless(GATED)
    stages = pipeline.run(Config.load(GATED), spacing=50.0, cross_validation=False)
    assert stages.stopped_at == "7_kompleksitas"
    assert not stages.complete
    assert any("intrusi" in m for m in stages.messages)
    # Usulan otomatis tetap disiapkan supaya formulir UI dapat mengisinya.
    assert stages.suggestions and len(stages.suggestions) == 8


def test_declaring_it_moves_the_stop_to_the_next_gate():
    """Gerbang berikutnya KCMI 4.6: izin dan status lahan belum dinyatakan."""
    _skip_unless(GATED)
    stages = pipeline.run(Config.load(GATED), spacing=50.0,
                          complexity_overrides=INTRUSI, cross_validation=False)
    assert stages.stopped_at == "9_batas"
    assert stages.limits is not None and not stages.limits.reportable
    assert len(stages.limits.blockers) == 3


def test_a_fully_declared_config_runs_to_the_end():
    _skip_unless(FULL)
    stages = pipeline.run(Config.load(FULL), spacing=50.0, cross_validation=False)
    assert stages.stopped_at is None
    assert stages.complete
    assert stages.estimate is not None and stages.estimate.passed


def test_the_stages_run_in_order_and_stop_where_they_stop():
    """Tahap setelah gerbang yang gagal TIDAK dijalankan."""
    _skip_unless(GATED)
    stages = pipeline.run(Config.load(GATED), spacing=50.0, cross_validation=False)
    # Berhenti di 7, jadi 8, 9, 10 tidak pernah berjalan.
    assert stages.models is not None            # tahap 4 sudah
    assert stages.validation is not None        # tahap 5 sudah
    assert stages.radii is None                 # tahap 8 belum
    assert stages.limits is None                # tahap 9 belum
    assert stages.estimate is None              # tahap 10 belum


def test_export_refuses_an_unfinished_run():
    """Tidak ada jalan pintas: gerbang terbuka berarti tidak ada keluaran."""
    _skip_unless(GATED)
    stages = pipeline.run(Config.load(GATED), spacing=50.0, cross_validation=False)
    with pytest.raises(ValueError, match="belum selesai"):
        pipeline.export_all(stages, "output/tidak_boleh", "UJI")


def test_export_writes_both_packages_from_one_run(tmp_path):
    _skip_unless(FULL)
    stages = pipeline.run(Config.load(FULL), spacing=50.0, cross_validation=False)
    run_obj, problems = pipeline.export_all(stages, tmp_path, "UJI")
    assert problems == []
    kinds = {entry["kind"] for entry in run_obj.files}
    # Paket serah-terima dan paket Bab V keluar dari run yang sama.
    assert {"grid_uncut_roof", "grid_limited_roof", "kontur_roof"} <= kinds
    assert {"tabel_csv", "gambar_pdf", "slot_manual"} <= kinds
    assert (run_obj.root / "02_reserve_handover" / "manifest.json").exists()
    # Keempat asumsi ikut ke dokumen.
    assert len(run_obj.assumptions) == 4


def test_the_config_carries_the_complexity_declaration():
    """Deklarasi dapat hidup di config, bukan hanya di formulir UI."""
    _skip_unless(FULL)
    cfg = Config.load(FULL)
    overrides = cfg.complexity_input.as_overrides()
    assert "intrusi" in overrides
    score, justification = overrides["intrusi"]
    assert score == "sederhana"
    assert len(justification) >= 40


def test_the_streamlit_app_imports_as_a_top_level_script():
    """`streamlit run app.py` memuatnya sebagai skrip, jadi impornya absolut."""
    source = (ROOT / "src" / "coalres" / "ui" / "app.py").read_text()
    assert "from coalres.config import Config" in source
    assert "from ..config" not in source
    assert "COALRES_CONFIG" in source


def test_the_orchestrator_has_no_switch_to_disable_the_gates():
    """Diuji dari TANDA TANGANNYA, bukan dari kata-kata di berkasnya.

    Saklar untuk mematikan gerbang akan menjadi jalan termudah menuju angka
    yang tidak boleh dipakai - dan jalan termudah adalah jalan yang ditempuh.
    """
    import inspect
    names = set(inspect.signature(pipeline.run).parameters)
    assert names == {"cfg", "spacing", "complexity_overrides",
                     "condition_override", "cross_validation"}
    for forbidden in ("stop_at_gates", "skip_gates", "ignore_gates", "force"):
        assert forbidden not in names


def test_a_stopped_run_yields_no_estimate_to_show_or_export():
    """Satu-satunya jalan melewati gerbang adalah menyelesaikannya."""
    _skip_unless(GATED)
    stages = pipeline.run(Config.load(GATED), spacing=50.0, cross_validation=False)
    assert stages.estimate is None
    assert not stages.complete
    with pytest.raises(ValueError):
        pipeline.export_all(stages, "output/tidak_boleh", "UJI")
