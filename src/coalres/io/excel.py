"""Import data lubang bor dari Excel dengan pemetaan kolom yang fleksibel.

Format internal kanonik:
    collar : hole_id, east, north, rl, td, block
    seam   : hole_id, seam, depth_from, depth_to  (+ kolom kualitas bila menyatu)
    quality: hole_id, seam, tm_ar, im_adb, ash_adb, vm_adb, fc_adb, ts_adb,
             cv_adb, cv_ar, rd_adb

Kolom kualitas boleh berada di sheet seam itu sendiri, atau di sheet terpisah
yang di-join pada (hole_id, seam).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED = {
    "collar": ["hole_id", "east", "north", "rl"],
    "seam": ["hole_id", "seam", "depth_from", "depth_to"],
    "quality": ["hole_id", "seam"],
}


def _norm(name: object) -> str:
    """Normalisasi header agar cocok tanpa peduli spasi/kapital/tanda baca."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _resolve_columns(
    df: pd.DataFrame, mapping: dict[str, list[str]], table: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Ganti nama kolom sumber menjadi nama kanonik.

    Kolom sumber yang tidak dikenali tetap dipertahankan apa adanya, sehingga
    kolom tambahan (HGI, AFT, keterangan) tidak hilang.
    """
    normalised = {_norm(col): col for col in df.columns}
    rename: dict[str, str] = {}
    found: dict[str, str] = {}

    for canonical, aliases in mapping.items():
        for alias in [canonical, *aliases]:
            source = normalised.get(_norm(alias))
            if source is not None and source not in rename:
                rename[source] = canonical
                found[canonical] = source
                break

    out = df.rename(columns=rename)
    missing = [c for c in REQUIRED.get(table, []) if c not in out.columns]
    if missing:
        raise ValueError(
            f"sheet '{table}': kolom wajib tidak ditemukan: {missing}. "
            f"Header yang terbaca: {list(df.columns)}"
        )
    return out, found


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError as exc:
        available = pd.ExcelFile(path).sheet_names
        raise ValueError(
            f"sheet '{sheet}' tidak ada di {path.name}. Sheet tersedia: {available}"
        ) from exc


@dataclass
class DrillholeData:
    """Data lubang bor yang sudah dinormalisasi dan digabung."""

    collar: pd.DataFrame
    seam: pd.DataFrame
    source_columns: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def n_holes(self) -> int:
        return len(self.collar)

    @property
    def seams(self) -> list[str]:
        return sorted(self.seam["seam"].dropna().unique().tolist())

    def with_quality(self) -> pd.DataFrame:
        """Baris seam yang punya minimal satu nilai kualitas."""
        cols = [c for c in ("ash_adb", "cv_adb", "rd_adb", "tm_ar") if c in self.seam]
        if not cols:
            return self.seam.iloc[0:0]
        return self.seam[self.seam[cols].notna().any(axis=1)]


def load_drillholes(path: str | Path, cfg) -> DrillholeData:
    """Baca satu workbook Excel berisi collar + interval seam (+ kualitas)."""
    path = Path(path)
    col_map = cfg["columns"]

    collar_raw = _read_sheet(path, cfg["input.collar_sheet"])
    collar, collar_src = _resolve_columns(collar_raw, col_map["collar"], "collar")

    seam_raw = _read_sheet(path, cfg["input.seam_sheet"])
    seam, seam_src = _resolve_columns(seam_raw, col_map["seam"], "seam")

    # Kolom kualitas: menyatu di sheet seam, atau di sheet terpisah.
    quality_sheet = cfg.get("input.quality_sheet")
    if quality_sheet:
        qual_raw = _read_sheet(path, quality_sheet)
        qual, qual_src = _resolve_columns(qual_raw, col_map["quality"], "quality")
        qual["hole_id"] = qual["hole_id"].map(normalise_hole_id)
        qual["seam"] = qual["seam"].astype(str).str.strip()
        seam = seam.merge(qual, on=["hole_id", "seam"], how="left", suffixes=("", "_q"))
    else:
        seam, qual_src = _resolve_columns(seam, col_map["quality"], "quality")

    for frame, keys in ((collar, ["hole_id"]), (seam, ["hole_id"])):
        frame["hole_id"] = frame["hole_id"].map(normalise_hole_id)

    seam["seam"] = seam["seam"].astype(str).str.strip()
    for col in ("east", "north", "rl", "td"):
        if col in collar:
            collar[col] = pd.to_numeric(collar[col], errors="coerce")
    for col in ("depth_from", "depth_to", *col_map["quality"]):
        if col in seam and col not in ("hole_id", "seam"):
            seam[col] = pd.to_numeric(seam[col], errors="coerce")

    seam = seam.sort_values(["hole_id", "depth_from"]).reset_index(drop=True)
    collar = collar.sort_values("hole_id").reset_index(drop=True)

    return DrillholeData(
        collar=collar,
        seam=seam,
        source_columns={"collar": collar_src, "seam": seam_src, "quality": qual_src},
    )


def normalise_hole_id(value: object) -> str:
    """Samakan varian penulisan ID lubang.

    Data lapangan kerap menulis lubang yang sama dalam beberapa bentuk
    (DH-09-05C1 / DH09_05C1 / DH09 05C1). Tanpa normalisasi, join antar-sumber
    akan diam-diam gagal dan lubang hilang dari model.
    """
    text = str(value).strip().upper()
    return re.sub(r"[^A-Z0-9]", "", text)
