import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORKBOOKS = ROOT / "data" / "workbooks"
LAS_DIR = ROOT / "data" / "las"

JUSTIFICATION = (
    "Justifikasi uji: seam menerus antar lubang pada jarak bor saat ini, tidak "
    "ditemukan indikasi sesar pada log, variasi ketebalan lateral moderat, dan "
    "dip landai seragam berdasarkan interpretasi penampang."
)


@pytest.fixture(scope="session")
def workbook_dir() -> Path:
    if not WORKBOOKS.exists():
        pytest.skip("workbook referensi tidak tersedia")
    return WORKBOOKS


@pytest.fixture
def base_config_dict(tmp_path, workbook_dir) -> dict:
    return {
        "paths": {
            "workbook_dir": str(workbook_dir), "las_dir": str(LAS_DIR),
            "quality_table": None, "topography_dxf": None,
            "output_dir": str(tmp_path / "out"),
        },
        "authoritative_coordinate_source": "collar_ts",
        "coal_thickness_source": "lithology",
        "core_loss_treatment": "as_coal",
        "geological_condition": "moderat",
        "geological_condition_justification": JUSTIFICATION,
        "classification_radii_m": {
            "sederhana": {"measured": 500, "indicated": 1000, "inferred": 1500},
            "moderat": {"measured": 250, "indicated": 500, "inferred": 1000},
            "kompleks": {"measured": 100, "indicated": 200, "inferred": 400},
        },
        "cutoffs": {
            "min_seam_thickness_m": 0.4, "max_parting_thickness_m": 0.3,
            "min_core_recovery_pct": 90, "min_quality_coverage_frac": 0.9,
        },
        "validation": {"collar_vs_topo_tolerance_m": 2.0,
                       "mass_balance_tolerance_pct": 0.5},
        "rpeee_constraints": {
            "max_depth_m": None, "max_depth_basis": "", "min_cv_ar_kcal_kg": None,
            "max_ash_adb_pct": None, "excluded_area_wkt": "", "excluded_area_basis": "",
        },
        "maps": {"contour_interval_m": 5.0, "index_contour_every": 5,
                 "label_contours": True},
        "estimation_method": "voronoi",
        "block_boundary_wkt": "",
        "assumed_rd_t_per_m3": None,
    }


@pytest.fixture
def write_config(tmp_path):
    def _write(data: dict) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return path
    return _write
