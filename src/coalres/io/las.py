"""Pembaca LAS geofisika.

PENTING - kurva LD dan SD pada dataset BGG bersatuan CPS (cacah per detik),
bukan bulk density terkalibrasi dalam g/cc. Nilainya berkisar ribuan sampai
puluhan ribu. Kurva ini SAH untuk verifikasi pick seam dan rekonsiliasi
kedalaman, dan TIDAK BOLEH dipakai untuk tonase.

Bila suatu LAS memuat kurva RHOB bersatuan g/cc, itu kasus berbeda dan
ditangani secara eksplisit lewat pengecekan satuan - bukan lewat tebakan dari
besaran nilainya.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lasio
import numpy as np

from ..errors import SchemaError
from ..logging_setup import get_logger

log = get_logger("io.las")

# Satuan yang menandai densitas terkalibrasi. Apa pun di luar ini diperlakukan
# sebagai cacah mentah dan diblokir dari jalur tonase.
CALIBRATED_DENSITY_UNITS = {"G/CC", "G/CM3", "GCC", "KG/M3", "K/M3"}
CALIBRATED_DENSITY_MNEMONICS = {"RHOB", "DENS", "ZDEN", "RHOZ"}


@dataclass
class LasCurve:
    mnemonic: str
    unit: str
    description: str
    data: np.ndarray

    @property
    def is_calibrated_density(self) -> bool:
        return (
            self.mnemonic.upper() in CALIBRATED_DENSITY_MNEMONICS
            and self.unit.upper().replace(" ", "") in CALIBRATED_DENSITY_UNITS
        )

    @property
    def n_valid(self) -> int:
        return int(np.isfinite(self.data).sum())


@dataclass
class LasFile:
    path: Path
    well_name: str
    field_name: str
    company: str
    service_company: str
    log_date: str
    start_m: float
    stop_m: float
    step_m: float
    depth: np.ndarray
    curves: dict[str, LasCurve]

    @property
    def logged_interval(self) -> tuple[float, float]:
        finite = self.depth[np.isfinite(self.depth)]
        return (float(finite.min()), float(finite.max())) if finite.size else (np.nan, np.nan)

    def curve_coverage(self, mnemonic: str) -> tuple[float, float] | None:
        """Rentang kedalaman dengan data valid untuk satu kurva.

        Sensor pada satu rangkaian tool punya offset, sehingga gamma dapat
        berhenti lebih dangkal daripada density di lubang yang sama.
        """
        curve = self.curves.get(mnemonic.upper())
        if curve is None:
            return None
        ok = np.isfinite(curve.data)
        if not ok.any():
            return None
        return (float(self.depth[ok].min()), float(self.depth[ok].max()))

    @property
    def density_curves_are_raw_counts(self) -> bool:
        """True bila tidak ada kurva densitas terkalibrasi sama sekali."""
        return not any(c.is_calibrated_density for c in self.curves.values())


def load_las(path: str | Path) -> LasFile:
    path = Path(path)
    try:
        las = lasio.read(str(path))
    except Exception as exc:
        raise SchemaError(f"gagal membaca LAS {path.name}: {exc}") from exc

    def header(key: str, default: str = "") -> str:
        try:
            item = las.well[key]
        except (KeyError, AttributeError):
            return default
        value = getattr(item, "value", None)
        return "" if value is None else str(value).strip()

    depth = np.asarray(las.index, dtype=float)
    curves: dict[str, LasCurve] = {}
    for curve in las.curves:
        mnemonic = str(curve.mnemonic).upper()
        if mnemonic in {"DEPT", "DEPTH", "MD"}:
            continue
        curves[mnemonic] = LasCurve(
            mnemonic=mnemonic,
            unit=str(curve.unit or "").strip(),
            description=str(curve.descr or "").strip(),
            data=np.asarray(curve.data, dtype=float),
        )

    def numeric(key: str) -> float:
        text = header(key)
        try:
            return float(text)
        except (TypeError, ValueError):
            return float("nan")

    return LasFile(
        path=path,
        well_name=header("WELL"),
        field_name=header("FLD"),
        company=header("COMP"),
        service_company=header("SRVC"),
        log_date=header("DATE"),
        start_m=numeric("STRT"),
        stop_m=numeric("STOP"),
        step_m=numeric("STEP"),
        depth=depth,
        curves=curves,
    )
