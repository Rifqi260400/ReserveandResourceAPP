"""Keluaran: tabel, peta, dan laporan exception.

Palet mengikuti aturan: magnitudo kontinu memakai ramp SATU HUE terang->gelap
(tidak pernah pelangi/jet, yang menciptakan batas palsu di tempat yang bukan
batas geologi), dan kelas sumberdaya memakai ramp ordinal satu hue karena
kelas itu berjenjang, bukan kategori sembarang.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .classify import CLASS_NAMES, INDICATED, INFERRED, MEASURED
from .grid import Grid

# --- Token warna (light surface) ------------------------------------------- #
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Ramp biru sekuensial 100 -> 700, untuk magnitudo kontinu.
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Ramp ordinal 3 langkah untuk kelas sumberdaya (tervalidasi: satu hue,
# lightness monoton, ujung terang 2,06:1 terhadap surface).
CLASS_COLORS = {INFERRED: "#86b6ef", INDICATED: "#2a78d6", MEASURED: "#104281"}


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def _style_axes(ax, title: str, subtitle: str | None = None) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes,
                color=INK_SECONDARY, fontsize=8.5, va="bottom")
    ax.set_aspect("equal")
    # Koordinat UTM ditulis penuh: notasi 1e6 pada sumbu northing tidak
    # terbaca sebagai koordinat oleh siapa pun yang memakai peta ini.
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
        spine.set_linewidth(0.6)
    ax.grid(True, color=GRIDLINE, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Easting (m)", color=INK_MUTED, fontsize=8)
    ax.set_ylabel("Northing (m)", color=INK_MUTED, fontsize=8)


def _extent(grid: Grid):
    half = grid.cell_size / 2.0
    return (
        grid.xmin - half, grid.xmin + (grid.ncols - 1) * grid.cell_size + half,
        grid.ymin - half, grid.ymin + (grid.nrows - 1) * grid.cell_size + half,
    )


def map_continuous(
    grid: Grid, values: np.ndarray, title: str, label: str, path: Path,
    holes: pd.DataFrame | None = None, subtitle: str | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    image = ax.imshow(
        values, origin="lower", extent=_extent(grid), cmap=_cmap(),
        interpolation="nearest",
    )
    if holes is not None and len(holes):
        # Lubang bor memakai tinta, bukan warna seri: identitas dibawa oleh
        # bentuk marker, sementara warna sudah dipakai untuk magnitudo.
        ax.scatter(holes["east"], holes["north"], s=9, marker="o",
                   facecolors="none", edgecolors=INK_PRIMARY, linewidths=0.7,
                   label=f"Lubang bor (n={len(holes)})", zorder=5)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True,
                  facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK_SECONDARY)

    bar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    bar.set_label(label, color=INK_SECONDARY, fontsize=8.5)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=3, width=0.6)
    bar.outline.set_edgecolor(GRIDLINE)
    bar.outline.set_linewidth(0.6)

    _style_axes(ax, title, subtitle)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def map_classification(
    grid: Grid, class_grid: np.ndarray, title: str, path: Path,
    holes: pd.DataFrame | None = None, subtitle: str | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.2, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    shown = np.where(class_grid > 0, class_grid, np.nan)
    cmap = ListedColormap([CLASS_COLORS[INFERRED], CLASS_COLORS[INDICATED],
                           CLASS_COLORS[MEASURED]])
    ax.imshow(shown, origin="lower", extent=_extent(grid), cmap=cmap,
              norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N),
              interpolation="nearest")

    handles = [
        Patch(facecolor=CLASS_COLORS[level], edgecolor=SURFACE, linewidth=1.5,
              label=f"{CLASS_NAMES[level]}  ({int((class_grid == level).sum()) * grid.cell_area / 1e4:,.0f} ha)")
        for level in (MEASURED, INDICATED, INFERRED)
        if (class_grid == level).any()
    ]
    if holes is not None and len(holes):
        ax.scatter(holes["east"], holes["north"], s=9, marker="o",
                   facecolors="none", edgecolors=INK_PRIMARY, linewidths=0.7, zorder=5)
        handles.append(Patch(facecolor="none", edgecolor=INK_PRIMARY,
                             label=f"Lubang bor (n={len(holes)})"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=True,
                  facecolor=SURFACE, edgecolor=GRIDLINE, labelcolor=INK_SECONDARY)

    _style_axes(ax, title, subtitle)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_grid_ascii(grid: Grid, values: np.ndarray, path: Path, nodata: float = -9999.0) -> Path:
    """Tulis ESRI ASCII grid agar hasil dapat dibuka di GIS / software tambang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.where(np.isfinite(values), values, nodata)
    header = (
        f"ncols {grid.ncols}\nnrows {grid.nrows}\n"
        f"xllcorner {grid.xmin - grid.cell_size / 2:.4f}\n"
        f"yllcorner {grid.ymin - grid.cell_size / 2:.4f}\n"
        f"cellsize {grid.cell_size}\nNODATA_value {nodata}\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        for row in data[::-1]:  # ASCII grid ditulis dari baris paling utara
            fh.write(" ".join(f"{v:.4f}" for v in row) + "\n")
    return path


def write_excel_report(
    path: Path,
    summary: pd.DataFrame,
    totals: pd.DataFrame,
    findings: pd.DataFrame,
    spacing: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
    notes: pd.DataFrame | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if notes is not None:
            notes.to_excel(writer, sheet_name="Catatan Penting", index=False)
        totals.to_excel(writer, sheet_name="Total per Kelas", index=False)
        summary.to_excel(writer, sheet_name="Sumberdaya per Seam", index=False)
        findings.to_excel(writer, sheet_name="Validasi", index=False)
        if spacing is not None:
            spacing.to_excel(writer, sheet_name="Spasi Bor", index=False)
        if residuals is not None:
            residuals.to_excel(writer, sheet_name="Residual Struktur", index=False)
    return path


def reporting_notes() -> pd.DataFrame:
    """Catatan yang WAJIB ikut ke mana pun angka ini dibawa."""
    return pd.DataFrame(
        {
            "No": range(1, 8),
            "Catatan": [
                "Angka ini adalah hasil perhitungan geometri dan BUKAN pernyataan "
                "sumberdaya. Pernyataan sumberdaya menuntut penilaian dan tanda "
                "tangan Competent Person.",
                "KCMI tidak memuat tabel radius klasifikasi. Jarak yang dipakai di "
                "sini berasal dari konfigurasi (lazimnya mengacu SNI 5015) dan HARUS "
                "diverifikasi ke dokumen standar versi terkini.",
                "Klasifikasi di sini hanya menilai spasi titik data. Ia TIDAK menilai "
                "kualitas korelasi seam, kerapatan struktur, maupun kecukupan QAQC - "
                "dan ketiganya adalah bagian dari klasifikasi.",
                "Tonase dihitung dengan ARD IN-SITU (Preston-Sanders), bukan ARD lab. "
                "Memakai ARD lab pada batubara peringkat rendah melebihkan tonase "
                "sekitar 10%.",
                "Tonase adalah tonase in-situ pada total moisture (basis ar). "
                "Kualitas yang dilaporkan harus dibaca pada basis yang sama.",
                "Volume = luas peta x ketebalan VERTIKAL. Tidak ada koreksi cos(dip) "
                "pada volume; koreksi itu hanya berlaku untuk cutoff ketebalan.",
                "Batas IUP, sungai, jalan, permukiman, dan kawasan lindung BELUM "
                "dikurangkan kecuali dimasukkan sebagai poligon batas.",
            ],
        }
    )
