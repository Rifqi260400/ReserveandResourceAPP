"""Tabel kualitas yang dipasok pengguna.

Kualitas TIDAK ada di dalam workbook - ia hanya ada di PDF laporan lab, dan
laporan itu memuat komposit, bukan tiap ply. Tool ini sengaja tidak mem-parse
PDF. Pengguna memasok CSV/Excel dengan skema di bawah.

`composite_of` memuat daftar sample_id yang digabung menjadi satu hasil,
sehingga interval kedalaman yang benar-benar diwakili nilai kualitas itu dapat
direkonstruksi dengan join balik ke sheet `Sampling`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..errors import SchemaError

QUALITY_COLUMNS = [
    "sample_id", "lab_sample_id", "hole_id", "seam", "composite_of",
    "TM_ar", "M_adb", "ASH_adb", "VM_adb", "FC_adb", "TS_adb",
    "CV_adb", "CV_ar", "CV_daf", "RD", "RD_basis", "report_no",
]
REQUIRED_COLUMNS = ["sample_id", "hole_id", "seam", "RD_basis"]
VALID_RD_BASIS = {"in_situ", "air_dried", "as_received", "unknown"}


@dataclass
class QualityTable:
    path: Path
    frame: pd.DataFrame

    def composite_members(self, sample_id: str) -> list[str]:
        row = self.frame.loc[self.frame["sample_id"] == sample_id]
        if row.empty:
            return []
        raw = row.iloc[0]["composite_of"]
        if pd.isna(raw) or not str(raw).strip():
            return [sample_id]
        return [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]


def load_quality_table(path: str | Path) -> QualityTable:
    path = Path(path)
    if not path.exists():
        raise SchemaError(f"tabel kualitas tidak ditemukan: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)

    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"{path.name}: kolom wajib tidak ada: {missing}\n"
            f"  skema yang diharapkan: {QUALITY_COLUMNS}"
        )

    bad = sorted(set(frame["RD_basis"].dropna().astype(str)) - VALID_RD_BASIS)
    if bad:
        raise SchemaError(
            f"{path.name}: RD_basis memuat nilai tidak sah {bad}. "
            f"Harus salah satu dari {sorted(VALID_RD_BASIS)}."
        )
    return QualityTable(path=path, frame=frame)
