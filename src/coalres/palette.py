"""Token warna dan gaya sumbu bersama untuk seluruh keluaran grafis.

Aturan yang dipatuhi:
  - Magnitudo kontinu (isopach, struktur, kedalaman) memakai ramp SATU HUE
    terang->gelap. Tidak pernah pelangi/jet, yang menciptakan batas palsu di
    tempat yang bukan batas geologi.
  - Kelas sumberdaya memakai ramp ORDINAL satu hue, karena kelas itu berjenjang
    (Tereka -> Tertunjuk -> Terukur), bukan kategori sembarang. Tiga langkah di
    bawah lolos validasi: satu hue, lightness monoton, ujung terang 2,06:1
    terhadap surface.
  - Teks memakai token tinta, tidak pernah warna seri.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"

# Ramp biru sekuensial 100 -> 700 untuk magnitudo kontinu.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Ramp ordinal tiga langkah untuk kelas sumberdaya (tervalidasi).
CLASS_COLOURS = {"Tereka": "#86b6ef", "Tertunjuk": "#2a78d6", "Terukur": "#104281"}

# Kontur: satu hue, terang untuk garis biasa dan gelap untuk garis indeks,
# sehingga hierarkinya terbaca tanpa bergantung pada perbedaan warna.
CONTOUR_MINOR = "#86b6ef"
CONTOUR_INDEX = "#184f95"
# Subcrop adalah jenis fitur yang berbeda, bukan tingkatan kontur - ia memakai
# hue kedua dan selalu punya entri legenda.
SUBCROP_COLOUR = "#eb6834"

LITHOLOGY_FILL = {
    "coal": "#2a2a28",
    "waste": "#d8d7cf",
    "core_loss": "#eb6834",
}


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)


def style_axes(ax, title: str, subtitle: str | None = None, grid_axis: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter

    ax.set_title(title, color=INK_PRIMARY, fontsize=10.5, loc="left",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes,
                color=INK_SECONDARY, fontsize=8.5, va="bottom")
    ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
        spine.set_linewidth(0.6)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    if grid_axis in {"both", "x"}:
        ax.xaxis.set_major_formatter(plain)


def style_map_axes(ax, title: str, subtitle: str | None = None) -> None:
    """Sumbu peta: koordinat UTM ditulis penuh, tanpa notasi 1e6."""
    from matplotlib.ticker import FuncFormatter

    style_axes(ax, title, subtitle, grid_axis="both")
    ax.set_aspect("equal")
    plain = FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel("Easting (m)", color=INK_MUTED, fontsize=8)
    ax.set_ylabel("Northing (m)", color=INK_MUTED, fontsize=8)
