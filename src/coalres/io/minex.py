"""Pembaca flat file bergaya Minex: surv, lit, qual, faults, topo.

Berkas-berkas ini TIDAK BERHEADER. Arti tiap kolom bersifat posisional dan
harus DINYATAKAN DI KONFIGURASI, bukan ditebak oleh kode. Sekali salah urut,
kesalahannya diam - tebal dan kualitas tetap terbaca sebagai angka yang wajar.

Pembacanya menghasilkan `HoleDataset`, representasi bersama yang juga
dihasilkan pembaca workbook BGG, sehingga modul di hilir tidak perlu tahu
format asalnya.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import SchemaError
from ..logging_setup import get_logger

log = get_logger("io.minex")

COLLAR_REQUIRED = ["hole_id", "east", "north", "rl"]
INTERVAL_REQUIRED = ["hole_id", "seam", "depth_from", "depth_to"]


@dataclass
class Fault:
    """Satu jejak sesar: rangkaian titik beserta atributnya."""

    name: str
    points: np.ndarray                 # (N, 3) x, y, z
    attributes: pd.DataFrame           # kolom mentah, arti dinyatakan konfigurasi

    @property
    def trace_xy(self) -> np.ndarray:
        return self.points[:, :2]


@dataclass
class HoleDataset:
    """Representasi bersama, apa pun format asalnya."""

    collars: pd.DataFrame
    intervals: pd.DataFrame
    quality: pd.DataFrame | None = None
    topo_points: np.ndarray | None = None
    faults: list[Fault] = field(default_factory=list)
    source_format: str = "?"
    provenance: dict = field(default_factory=dict)

    @property
    def hole_ids(self) -> list[str]:
        return sorted(self.collars["hole_id"].unique().tolist())

    @property
    def seams(self) -> list[str]:
        return sorted(self.intervals["seam"].dropna().unique().tolist())


def _read_whitespace(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise SchemaError(f"{label}: berkas tidak ditemukan: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python",
                        comment="#", skip_blank_lines=True)
    if frame.shape[1] < len(columns):
        raise SchemaError(
            f"{label} ({path.name}): berisi {frame.shape[1]} kolom, konfigurasi "
            f"menyebut {len(columns)}: {columns}"
        )
    if frame.shape[1] > len(columns):
        log.warning(
            f"{label} ({path.name}): {frame.shape[1]} kolom terbaca, "
            f"{len(columns)} dinamai; sisanya diabaikan"
        )
        frame = frame.iloc[:, :len(columns)]
    frame.columns = columns
    return frame


def _drop_exact_duplicates(frame: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    """Buang baris yang identik seluruhnya, laporkan jumlahnya.

    Berkas Minex kerap berisi isi yang sama dua kali akibat penggabungan
    ekspor. Membiarkannya akan MELIPATGANDAKAN tonase tanpa gejala apa pun.
    """
    before = len(frame)
    cleaned = frame.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    if removed:
        log.warning(f"{label}: {removed} baris duplikat identik dibuang "
                    f"({before} -> {len(cleaned)})")
    return cleaned, removed


def read_survey(path: Path, columns: list[str]) -> tuple[pd.DataFrame, int]:
    frame = _read_whitespace(Path(path), columns, "survey")
    missing = [c for c in COLLAR_REQUIRED if c not in frame.columns]
    if missing:
        raise SchemaError(f"survey: kolom wajib belum dinamai: {missing}")
    frame["hole_id"] = frame["hole_id"].astype(str).str.strip()
    for column in ("east", "north", "rl", "total_depth"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _drop_exact_duplicates(frame, "survey")


def read_lithology(
    path: Path, columns: list[str], marker_seams: list[str]
) -> tuple[pd.DataFrame, int]:
    """Baca berkas lit.

    Berkas lit Minex mencantumkan interval SEAM, bukan seluruh kolom litologi.
    Karena itu tidak ada parting maupun core loss untuk dikurangkan: tebal
    batubara adalah to - from. Ini perbedaan nyata dari workbook BGG, dan tidak
    boleh diselesaikan dengan memakai logika parting BGG di sini.

    Baris `marker_seams` (mis. 'W') berketebalan nol dan menandai suatu horizon,
    bukan seam. Baris itu dipisahkan, tidak dibuang.
    """
    frame = _read_whitespace(Path(path), columns, "lithology")
    missing = [c for c in INTERVAL_REQUIRED if c not in frame.columns]
    if missing:
        raise SchemaError(f"lithology: kolom wajib belum dinamai: {missing}")

    frame["hole_id"] = frame["hole_id"].astype(str).str.strip()
    frame["seam"] = frame["seam"].astype(str).str.strip()
    for column in ("depth_from", "depth_to"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "code" in frame:
        frame["code"] = frame["code"].astype(str).str.strip().str.strip("'")

    frame, removed = _drop_exact_duplicates(frame, "lithology")
    markers = {m.strip().upper() for m in marker_seams}
    frame["is_marker"] = frame["seam"].str.upper().isin(markers)
    return frame, removed


def read_quality(path: Path, columns: list[str]) -> tuple[pd.DataFrame, int]:
    frame = _read_whitespace(Path(path), columns, "quality")
    missing = [c for c in INTERVAL_REQUIRED if c not in frame.columns]
    if missing:
        raise SchemaError(f"quality: kolom wajib belum dinamai: {missing}")
    frame["hole_id"] = frame["hole_id"].astype(str).str.strip()
    frame["seam"] = frame["seam"].astype(str).str.strip()
    for column in frame.columns:
        if column not in ("hole_id", "seam"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _drop_exact_duplicates(frame, "quality")


def read_faults(path: Path, columns: list[str]) -> list[Fault]:
    """Baca berkas sesar. Titik dikelompokkan menurut nama jejak."""
    frame = _read_whitespace(Path(path), columns, "faults")
    for column in frame.columns:
        if column != "name":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["name"] = frame["name"].astype(str).str.strip()

    faults = []
    for name, group in frame.groupby("name", sort=False):
        points = group[["east", "north", "z"]].to_numpy(float)
        faults.append(Fault(name=name, points=points,
                            attributes=group.drop(columns=["name"]).reset_index(drop=True)))
    return faults


def read_topo_dat(path: Path) -> np.ndarray:
    """Baca topo XYZ tiga kolom (biasanya kontur yang sudah didigitasi)."""
    frame = pd.read_csv(Path(path), sep=r"\s+", header=None, engine="python",
                        names=["x", "y", "z"], comment="#", skip_blank_lines=True)
    points = frame[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna()
    return np.unique(points.to_numpy(float), axis=0)


def _apply_seam_aliases(
    frame: pd.DataFrame, aliases: dict[str, str], label: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Ganti nama seam sesuai peta alias, laporkan berapa baris terkena.

    Penggantian dilakukan SEKALI di titik pembacaan, bukan tersebar di hilir,
    supaya tidak ada modul yang melihat dua penamaan untuk seam yang sama.
    """
    if not aliases or "seam" not in frame.columns:
        return frame, {}
    applied: dict[str, int] = {}
    for source, target in aliases.items():
        mask = frame["seam"].str.upper() == source.strip().upper()
        count = int(mask.sum())
        if count:
            frame.loc[mask, "seam"] = target
            applied[f"{source} -> {target}"] = count
    if applied:
        log.info(f"{label}: alias seam diterapkan {applied}")
    return frame, applied


def load_minex(cfg) -> HoleDataset:
    """Muat seluruh berkas Minex sesuai konfigurasi."""
    spec = cfg.minex
    provenance: dict = {}

    collars, dup_surv = read_survey(spec.survey_file, spec.survey_columns)
    intervals, dup_lit = read_lithology(spec.lithology_file, spec.lithology_columns,
                                        spec.marker_seams)
    provenance["duplicate_rows_removed"] = {"survey": dup_surv, "lithology": dup_lit}

    intervals, alias_lit = _apply_seam_aliases(intervals, spec.seam_aliases, "lithology")

    quality = None
    if spec.quality_file is not None:
        quality, dup_qual = read_quality(spec.quality_file, spec.quality_columns)
        provenance["duplicate_rows_removed"]["quality"] = dup_qual
        quality, alias_qual = _apply_seam_aliases(quality, spec.seam_aliases, "quality")
        provenance["seam_aliases"] = {"lithology": alias_lit, "quality": alias_qual}
    else:
        provenance["seam_aliases"] = {"lithology": alias_lit}

    topo_points = None
    if spec.topography_file is not None:
        path = Path(spec.topography_file)
        if path.suffix.lower() == ".dxf":
            from .dxf import load_topography
            topo = load_topography(path)
            topo_points = topo.points
            provenance["topography"] = {"source": path.name, **topo.entity_counts}
        else:
            topo_points = read_topo_dat(path)
            provenance["topography"] = {"source": path.name, "points": len(topo_points)}

    faults = []
    if spec.faults_file is not None:
        faults = read_faults(spec.faults_file, spec.fault_columns)
        provenance["faults"] = {f.name: len(f.points) for f in faults}

    return HoleDataset(
        collars=collars, intervals=intervals, quality=quality,
        topo_points=topo_points, faults=faults,
        source_format="minex_flat", provenance=provenance,
    )
