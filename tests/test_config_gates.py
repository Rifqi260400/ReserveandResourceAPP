"""Gerbang konfigurasi: harus menolak, bukan mundur ke nilai cadangan."""
import pytest

from coalres.config import Config
from coalres.errors import ConfigError


def test_empty_justification_stops(base_config_dict, write_config):
    base_config_dict["geological_condition_justification"] = ""
    with pytest.raises(ConfigError, match="wajib diisi"):
        Config.load(write_config(base_config_dict))


def test_short_justification_stops(base_config_dict, write_config):
    base_config_dict["geological_condition_justification"] = "Seam menerus, dip landai."
    with pytest.raises(ConfigError, match="minimum 100"):
        Config.load(write_config(base_config_dict))


def test_depth_limit_without_basis_stops(base_config_dict, write_config):
    base_config_dict["rpeee_constraints"]["max_depth_m"] = 100.0
    base_config_dict["rpeee_constraints"]["max_depth_basis"] = ""
    with pytest.raises(ConfigError, match="max_depth_basis kosong"):
        Config.load(write_config(base_config_dict))


def test_depth_limit_with_basis_loads(base_config_dict, write_config):
    base_config_dict["rpeee_constraints"]["max_depth_m"] = 100.0
    base_config_dict["rpeee_constraints"]["max_depth_basis"] = (
        "Batas kedalaman tambang terbuka pada blok sekitar dengan geometri serupa."
    )
    cfg = Config.load(write_config(base_config_dict))
    assert cfg.rpeee_constraints.resource_label == "Sumberdaya"
    assert cfg.rpeee_constraints.class_prefix == "Sumberdaya"


def test_null_depth_limit_labels_everything_inventori(base_config_dict, write_config):
    """Aturan 8.4: tanpa batasan ekonomi, tidak ada keluaran berlabel Sumberdaya."""
    cfg = Config.load(write_config(base_config_dict))
    assert cfg.rpeee_constraints.max_depth_m is None
    assert cfg.rpeee_constraints.resource_label == "Inventori Batubara"
    assert cfg.rpeee_constraints.class_prefix == "Inventori"
    assert not cfg.rpeee_constraints.has_economic_constraint


def test_radii_must_ascend(base_config_dict, write_config):
    base_config_dict["classification_radii_m"]["moderat"] = {
        "measured": 500, "indicated": 250, "inferred": 1000
    }
    with pytest.raises(ConfigError, match="menaik"):
        Config.load(write_config(base_config_dict))


def test_geological_condition_selects_radii(base_config_dict, write_config):
    for condition, expected in (("sederhana", 500), ("moderat", 250), ("kompleks", 100)):
        base_config_dict["geological_condition"] = condition
        cfg = Config.load(write_config(base_config_dict))
        assert cfg.radii.measured == expected


def test_unknown_key_is_rejected(base_config_dict, write_config):
    """Salah ketik di YAML tidak boleh lolos diam-diam."""
    base_config_dict["max_dept_m"] = 100
    with pytest.raises(ConfigError):
        Config.load(write_config(base_config_dict))
