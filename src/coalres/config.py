"""Pemuatan dan validasi konfigurasi."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    """Konfigurasi proyek, dengan akses bertitik: cfg.get('grid.cell_size')."""

    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        with open(DEFAULT_CONFIG_PATH) as fh:
            data = yaml.safe_load(fh)
        if path is not None:
            with open(path) as fh:
                user = yaml.safe_load(fh) or {}
            data = _deep_merge(data, user)
        cfg = cls(data)
        cfg.validate()
        return cfg

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"kunci konfigurasi tidak ada: {dotted}")
        return value

    def validate(self) -> None:
        """Tangkap konfigurasi yang mustahil sebelum pipeline berjalan."""
        errors: list[str] = []

        if self["grid.cell_size"] <= 0:
            errors.append("grid.cell_size harus > 0")

        cond = self["classification.geological_condition"]
        distances = self.get(f"classification.distances.{cond}")
        if distances is None:
            errors.append(
                f"classification.geological_condition='{cond}' tidak punya entri jarak"
            )
        else:
            m, i, inf = distances["measured"], distances["indicated"], distances["inferred"]
            if not (m < i < inf):
                errors.append(
                    "jarak klasifikasi harus menaik: measured < indicated < inferred "
                    f"(dapat {m}, {i}, {inf})"
                )

        method = self["density.insitu_method"]
        if method not in {"preston_sanders", "fixed", "none"}:
            errors.append(f"density.insitu_method tidak dikenal: {method}")

        gmethod = self["grid.method"]
        if gmethod not in {"idw", "nearest", "linear"}:
            errors.append(f"grid.method tidak dikenal: {gmethod}")

        lo, hi = self["density.plausible_range"]
        if lo >= hi:
            errors.append("density.plausible_range harus [min, max] dengan min < max")

        if errors:
            raise ValueError("Konfigurasi tidak valid:\n  - " + "\n  - ".join(errors))
