"""Penulis grid format Surfer (.grd) — ASCII DSAA dan biner DSBB.

Dipakai untuk grid yang akan dikonsumsi ulang oleh perangkat lunak tambang:
ketebalan uncut, ketebalan cut, dan atribut kualitas.

Format Minex sendiri bersifat proprietary dan tidak ditulis di sini. Surfer
ASCII (DSAA) adalah format grid yang paling luas dapat dibaca - Minex, Surpac,
Micromine, dan Surfer semuanya menerimanya - jadi ia yang dipakai sebagai bawaan.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .logging_setup import get_logger
from .topo import Surface

log = get_logger("grdout")

# Nilai kosong Surfer. Sel NaN ditulis dengan angka ini, bukan 0 - sel bernilai
# nol dan sel tanpa data adalah dua hal berbeda, dan menyamakannya akan
# menciptakan ketebalan nol palsu di area tanpa dukungan data.
SURFER_BLANK = 1.70141e38
# Nilai kosong ditulis dalam notasi ilmiah. Dengan format desimal tetap, ia
# menjadi angka 39 digit yang membengkakkan berkas - grid uncut pada dataset ini
# 75% selnya kosong - dan sebagian pembaca .grd menolaknya.
SURFER_BLANK_TEXT = "1.70141e+38"


def _bounds(surface: Surface, values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 0.0)
    return (float(finite.min()), float(finite.max()))


def write_surfer_ascii(surface: Surface, path: Path, values: np.ndarray | None = None) -> Path:
    """Tulis grid Surfer ASCII (DSAA).

    Baris ditulis dari Y terendah ke tertinggi, sesuai spesifikasi format.
    """
    z = surface.z if values is None else values
    if z.shape != surface.z.shape:
        raise ValueError(f"bentuk nilai {z.shape} tidak sama dengan grid {surface.z.shape}")

    zlo, zhi = _bounds(surface, z)
    filled = np.where(np.isfinite(z), z, SURFER_BLANK)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as fh:
        fh.write("DSAA\n")
        fh.write(f"{len(surface.x)} {len(surface.y)}\n")
        fh.write(f"{surface.x.min():.6f} {surface.x.max():.6f}\n")
        fh.write(f"{surface.y.min():.6f} {surface.y.max():.6f}\n")
        fh.write(f"{zlo:.6f} {zhi:.6f}\n")
        blank = np.isfinite(z)
        for row, row_ok in zip(filled, blank):   # baris 0 = Y terendah
            fh.write(" ".join(
                f"{v:.6f}" if ok else SURFER_BLANK_TEXT
                for v, ok in zip(row, row_ok)
            ) + "\n")
    return path


def write_surfer_binary(surface: Surface, path: Path, values: np.ndarray | None = None) -> Path:
    """Tulis grid Surfer biner (DSBB)."""
    z = surface.z if values is None else values
    zlo, zhi = _bounds(surface, z)
    filled = np.where(np.isfinite(z), z, SURFER_BLANK).astype("<f4")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as fh:
        fh.write(b"DSBB")
        fh.write(struct.pack("<hh", len(surface.x), len(surface.y)))
        fh.write(struct.pack("<6d", float(surface.x.min()), float(surface.x.max()),
                             float(surface.y.min()), float(surface.y.max()), zlo, zhi))
        fh.write(filled.tobytes())
    return path


def write_grd(surface: Surface, path: Path, values: np.ndarray | None = None,
              fmt: str = "ascii") -> Path:
    if fmt == "ascii":
        return write_surfer_ascii(surface, path, values)
    if fmt == "binary":
        return write_surfer_binary(surface, path, values)
    raise ValueError(f"format grd tidak dikenal: {fmt}")


def read_surfer_ascii(path: Path) -> tuple[np.ndarray, dict]:
    """Baca kembali DSAA. Dipakai untuk verifikasi bolak-balik."""
    with open(path) as fh:
        if fh.readline().strip() != "DSAA":
            raise ValueError(f"{path.name}: bukan grid Surfer ASCII")
        nx, ny = (int(v) for v in fh.readline().split())
        xlo, xhi = (float(v) for v in fh.readline().split())
        ylo, yhi = (float(v) for v in fh.readline().split())
        zlo, zhi = (float(v) for v in fh.readline().split())
        values = np.fromstring(fh.read().replace("\n", " "), sep=" ")
    z = values[: nx * ny].reshape(ny, nx)
    z = np.where(z >= SURFER_BLANK * 0.999, np.nan, z)
    return z, {"nx": nx, "ny": ny, "xlo": xlo, "xhi": xhi,
               "ylo": ylo, "yhi": yhi, "zlo": zlo, "zhi": zhi}


ATTRIBUTE_UNITS = {
    "rd_t_per_m3": "t/m3", "ASH": "%", "VM": "%", "FC": "%", "TS": "%",
    "MOISTURE": "%", "CV": "kcal/kg", "ASH_adb": "%", "CV_adb": "kcal/kg",
    "CV_ar": "kcal/kg", "TS_adb": "%", "TM_ar": "%", "M_adb": "%",
}


def write_sidecar(path: Path, *, name: str, description: str, units: str,
                  surface: Surface, support_note: str, warnings: list[str]) -> Path:
    """Berkas keterangan pendamping. Grid tanpa keterangan mudah disalahpakai."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Grid          : {name}",
        f"Isi           : {description}",
        f"Satuan        : {units}",
        f"Interpolasi   : TIN Delaunay, {surface.method}",
        f"Spasi grid    : {surface.spacing} m",
        f"Titik pendukung: {surface.n_points}",
        f"Extent        : {surface.extent}",
        f"Nilai kosong  : {SURFER_BLANK:g} (Surfer blank)",
        f"Dukungan data : {support_note}",
        "",
    ]
    if warnings:
        lines.append("PERINGATAN")
        lines.extend(f"  - {w}" for w in warnings)
        lines.append("")
    path.write_text("\n".join(lines))
    return path
