"""Uji end-to-end: Excel + DXF masuk, laporan sumberdaya keluar."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"),
         "--holes", "60", "--out", str(out)],
        check=True, capture_output=True,
    )
    cfg_path = out / "config.yml"
    cfg_path.write_text(yaml.safe_dump({
        "input": {"collar_sheet": "Collar", "seam_sheet": "Seam"},
        "stratigraphy": ["S11", "S10A", "S10B", "S10C"],
        "grid": {"cell_size": 75.0, "max_extrapolation_m": 300.0},
        "output": {"dir": str(out / "output"), "write_maps": False, "write_grids": False},
    }))
    return out / "drillholes.xlsx", out / "topo.dxf", cfg_path, out / "output"


def test_pipeline_runs_end_to_end(dataset):
    holes, topo, cfg, out_dir = dataset
    from coalres.pipeline import run

    results = run(holes, topo, cfg, out_dir, verbose=False)

    assert not results.summary.empty
    assert not results.totals.empty
    assert (out_dir / "resource_report.xlsx").exists()
    assert set(results.totals["class"]) <= {"Terukur", "Tertunjuk", "Tereka"}


def test_pipeline_catches_injected_defects(dataset):
    """Kelima cacat yang disisipkan generator harus muncul di laporan validasi."""
    holes, topo, cfg, out_dir = dataset
    from coalres.pipeline import run

    results = run(holes, topo, cfg, out_dir, verbose=False)
    codes = set(results.findings["code"])

    assert "COLLAR_VS_TOPO" in codes          # RL collar salah 14,18 m
    assert "MASS_BALANCE" in codes            # IM+Ash+VM+FC != 100
    assert "CV_CONVERSION" in codes           # CV(ar) tidak konsisten
    assert "STRAT_OUT_OF_ORDER" in codes      # seam tertukar
    assert "ARD_NOT_APPARENT" in codes        # true density dilaporkan sebagai ARD
    assert "NO_QAQC" in codes                 # tidak ada duplikat/CRM/umpire


def test_tonnage_uses_insitu_density_not_lab(dataset):
    """ARD in-situ harus lebih rendah dari ARD lab (~1,28-1,40 pada data ini)."""
    holes, topo, cfg, out_dir = dataset
    from coalres.pipeline import run

    results = run(holes, topo, cfg, out_dir, verbose=False)
    ard = results.summary["ard_insitu"].dropna()
    assert (ard < 1.30).all(), "ARD in-situ terlalu tinggi - kemungkinan ARD lab terpakai"
    assert (ard > 1.15).all()


def test_reported_area_matches_classification_map(dataset):
    """Luas di tabel harus sama dengan luas kelas di grid keluaran.

    Kalau cutoff diterapkan pada tabel tapi tidak pada grid kelas, peta akan
    melaporkan luas lebih besar daripada yang benar-benar terhitung.
    """
    holes, topo, cfg, out_dir = dataset
    from coalres.classify import CLASS_NAMES
    from coalres.pipeline import run

    results = run(holes, topo, cfg, out_dir, verbose=False)
    cell_ha = results.grid.cell_area / 10_000.0
    for seam_name, class_grid in results.class_grids.items():
        rows = results.summary[results.summary["seam"] == seam_name]
        for _, row in rows.iterrows():
            level = next(k for k, v in CLASS_NAMES.items() if v == row["class"])
            assert row["area_ha"] == pytest.approx(
                int((class_grid == level).sum()) * cell_ha, rel=1e-9
            )


def test_cli_returns_nonzero_when_validation_errors(dataset):
    holes, topo, cfg, out_dir = dataset
    result = subprocess.run(
        [sys.executable, "-m", "coalres.cli", "--holes", str(holes),
         "--topo", str(topo), "--config", str(cfg), "--out", str(out_dir), "--quiet"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert result.returncode == 1
    assert "ERROR validasi" in result.stdout
