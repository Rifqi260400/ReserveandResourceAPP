"""Paket laporan Bab V: tabel, gambar, dan draf DOCX.

Susunannya mengikuti laporan rujukan (Bab V Pemodelan dan Estimasi Sumber daya),
dibagi tiga kelas menurut apa yang benar-benar dapat dihasilkan dari data:

  A  dihasilkan penuh   Tabel 5-1, 5-2, 5-3..5-30, 5-31..5-33, 5-35, 5-36;
                        Gambar 5.1, 5.4..5.19, 5.25..5.33
  B  butuh masukan      Tabel 5-34 (BESR), Gambar 5.24 (design pit) - keduanya
     ekonomi            menuntut kajian ekonomi yang tidak ada di data geologi
  C  slot bernarasi     Gambar 5.2, 5.3, 5.20..5.22 - penampang, interpretasi
                        log, dan diagram kerangka yang digambar manusia

Kelas B dan C DICETAK sebagai slot bertanda "BUTUH ISI MANUAL", bukan
dikosongkan diam-diam. Laporan yang kehilangan satu gambar tanpa jejak jauh
lebih berbahaya daripada laporan yang menyatakan gambar itu belum ada.

Tiap gambar ditulis tiga kali: PDF vektor untuk dicetak, PNG 300 dpi untuk
ditempel, dan CSV berisi angka yang membentuknya - supaya pembaca dapat
memeriksa gambarnya tanpa menjalankan ulang programnya.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger("export.bab5")

DPI = 300

# Ramp ORDINAL satu warna untuk kelas sumberdaya. Kelas itu peringkat keyakinan
# (Terukur > Tertunjuk > Tereka), bukan identitas, jadi encoding-nya berurutan -
# satu hue menggelap - bukan kategorikal. Lolos uji ordinal: monoton, jarak
# lightness cukup, ujung terang 2,06:1 terhadap permukaan.
CLASS_COLORS = {"terukur": "#1c5cab", "tertunjuk": "#3987e5", "tereka": "#86b6ef"}
CLASS_LABELS = {"terukur": "Terukur", "tertunjuk": "Tertunjuk", "tereka": "Tereka"}
OUTSIDE_COLOR = "#d9d8d4"

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8984"
SURFACE = "#fcfcfb"

MANUAL = "BUTUH ISI MANUAL"

CLASS_C_SLOTS = {
    "Gambar 5.2": "Model monoklin area (penampang interpretatif)",
    "Gambar 5.3": "Interpretasi log geofisika",
    "Gambar 5.20": "Hubungan Inventori, Sumber daya dan Cadangan",
    "Gambar 5.21": "Diagram kerangka estimasi Sumber daya",
    "Gambar 5.22": "Gambaran keseragaman kemiringan batubara",
}
CLASS_B_SLOTS = {
    "Tabel 5-34": "Evaluasi perhitungan BESR - menuntut harga, biaya, dan SR",
    "Gambar 5.24": "Peta design pit - menuntut hasil pit optimisasi",
}


def _style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Sumbu dan grid yang resesif; teks memakai token tinta, bukan warna seri."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=12, weight="bold", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9.5)


def save_figure(run, figure, number: str, caption: str,
                data: pd.DataFrame | None = None) -> list[Path]:
    """Tulis PDF vektor, PNG 300 dpi, dan CSV angkanya - ketiganya."""
    stem = number.replace(" ", "_").replace(".", "-")
    written = []
    for suffix, kind in ((".pdf", "gambar_pdf"), (".png", "gambar_png")):
        target = run.path("01_bab5_laporan", "gambar", f"{stem}{suffix}")
        figure.savefig(target, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
        written.append(run.record(target, kind, note=caption))
    if data is not None:
        target = run.path("01_bab5_laporan", "gambar", f"{stem}.csv")
        data.to_csv(target, index=False)
        written.append(run.record(target, "gambar_data", note=caption))
    import matplotlib.pyplot as plt
    plt.close(figure)
    return written


def save_table(run, frame: pd.DataFrame, number: str, caption: str) -> list[Path]:
    stem = number.replace(" ", "_").replace(".", "-")
    written = []
    csv_path = run.path("01_bab5_laporan", "tabel", f"{stem}.csv")
    frame.to_csv(csv_path, index=False)
    written.append(run.record(csv_path, "tabel_csv", note=caption))
    try:
        xlsx = run.path("01_bab5_laporan", "tabel", f"{stem}.xlsx")
        frame.to_excel(xlsx, index=False)
        written.append(run.record(xlsx, "tabel_xlsx", note=caption))
    except Exception as exc:
        log.warning(f"{number}: Excel dilewati ({exc})")
    return written


# --- Tabel ----------------------------------------------------------------

def tabel_5_1(run, weathering_depths: pd.DataFrame) -> list[Path]:
    """Tebal zona pelapukan."""
    return save_table(run, weathering_depths, "Tabel 5-1",
                      "Tebal zona pelapukan berdasarkan analisa log bor")


def tabel_5_2(run, intersections: pd.DataFrame) -> tuple[list[Path], pd.DataFrame]:
    """Stratigrafi seam dan statistik ketebalan."""
    frame = intersections.assign(t=intersections["coal_thickness_m"])
    stats = (frame.groupby("seam")["t"]
             .agg(n_lubang="count", min="min", rerata="mean", median="median",
                  maks="max", std="std")
             .round(2).reset_index())
    return save_table(run, stats, "Tabel 5-2",
                      "Stratigrafi dan statistik ketebalan seam"), stats


def tabel_deviasi(run, validation_report, start: int = 3) -> list[Path]:
    """Tabel 5.3..5.30: deviasi model terhadap data bor, per seam per atribut."""
    written, number = [], start
    for deviation in validation_report.deviations:
        if deviation.kind != "validasi_silang":
            continue
        label = "Ketebalan" if deviation.attribute == "tebal" else "Roof"
        written += save_table(
            run, deviation.frame.round(3), f"Tabel 5-{number}",
            f"Deviasi {label} Model-Drill hole Seam {deviation.seam}")
        number += 1
    return written


def tabel_5_32(run, assessment) -> list[Path]:
    return save_table(run, assessment.to_checklist_frame().fillna(""),
                      "Tabel 5-32", "Pembobotan penentuan kelas geologi")


def tabel_5_33(run, radii: dict[str, float], condition: str) -> list[Path]:
    frame = pd.DataFrame([{
        "Kondisi Geologi": condition.capitalize(),
        "Kriteria": "Jarak Titik Pengamatan (m)",
        "Tereka": f"{radii['tertunjuk']:.0f} < x <= {radii['tereka']:.0f}",
        "Tertunjuk": f"{radii['terukur']:.0f} < x <= {radii['tertunjuk']:.0f}",
        "Terukur": f"x <= {radii['terukur']:.0f}",
    }])
    return save_table(run, frame, "Tabel 5-33",
                      "Jarak titik informasi berdasarkan kondisi geologi (SNI 5015:2019)")


def tabel_5_35_36(run, estimate_report, label: str) -> list[Path]:
    frame = estimate_report.table(label)
    return (save_table(run, frame, "Tabel 5-35",
                       f"Ringkasan {label} batubara setiap seam")
            + save_table(run, frame, "Tabel 5-36",
                         f"Tabulasi estimasi {label} batubara"))


# --- Gambar ---------------------------------------------------------------

def gambar_5_1(run, stats: pd.DataFrame) -> list[Path]:
    """Rerata ketebalan seam - satu seri, jadi tanpa legenda; judul menamainya."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = stats.sort_values("rerata")
    figure, ax = plt.subplots(figsize=(7.2, 0.62 * len(data) + 1.9))
    figure.patch.set_facecolor(SURFACE)
    y = np.arange(len(data))
    ax.barh(y, data["rerata"], height=0.6, color=SERIES, zorder=3)
    ax.set_yticks(y, [f"Seam {s}" for s in data["seam"]])
    for index, (value, n) in enumerate(zip(data["rerata"], data["n_lubang"])):
        ax.text(value + max(data["rerata"]) * 0.015, index,
                f"{value:.2f} m   (n={n})", va="center", fontsize=9, color=INK_SOFT)
    ax.set_xlim(0, max(data["rerata"]) * 1.30)
    ax.xaxis.grid(True, color=INK_MUTED, alpha=0.25, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    _style(ax, "Rerata ketebalan seam batubara", "Ketebalan rerata (m)")
    figure.tight_layout()
    return save_figure(run, figure, "Gambar 5.1",
                       "Rerata ketebalan seam batubara", data)


def gambar_kontur_floor(run, models: dict, number_start: int = 4,
                        interval_m: float = 5.0) -> list[Path]:
    """Peta kontur struktur floor per seam - elevasi sebagai ramp sekuensial."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written, index = [], 0
    for key, model in sorted(models.items()):
        z = model.floor.z
        if not np.isfinite(z).any():
            continue
        gx, gy = np.meshgrid(model.floor.x, model.floor.y)
        figure, ax = plt.subplots(figsize=(8.4, 6.2))
        figure.patch.set_facecolor(SURFACE)
        fill = ax.contourf(gx, gy, z, levels=14, cmap="Blues", alpha=0.9)
        lines = ax.contour(gx, gy, z, levels=14, colors=INK_SOFT,
                           linewidths=0.6, linestyles="solid")
        ax.clabel(lines, inline=True, fontsize=7, fmt="%.0f", colors=INK)
        bar = figure.colorbar(fill, ax=ax, shrink=0.82, pad=0.02)
        bar.set_label("RL floor (m)", color=INK_SOFT, fontsize=9)
        bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
        ax.set_aspect("equal")
        _style(ax, f"Peta kontur struktur floor Seam {model.seam}",
               "Easting (m)", "Northing (m)")
        figure.tight_layout()
        number = f"Gambar 5.{number_start + index}"
        data = pd.DataFrame({"x": gx[np.isfinite(z)], "y": gy[np.isfinite(z)],
                             "floor_rl": z[np.isfinite(z)]})
        written += save_figure(run, figure, number,
                               f"Peta kontur struktur floor Seam {model.seam}", data)
        index += 1
    return written


def gambar_peta_sumberdaya(run, estimates, models: dict, points_frame,
                           number_start: int = 25) -> list[Path]:
    """Peta sumberdaya per seam - kelas sebagai ramp ordinal satu warna.

    Kelas adalah peringkat keyakinan, bukan identitas, jadi warnanya menggelap
    searah keyakinan. Legenda selalu ada, dan titik pengamatan ikut digambar
    supaya pembaca melihat dari mana tiap poligon tumbuh.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    order = ["terukur", "tertunjuk", "tereka", "di luar radius"]
    colors = [CLASS_COLORS["terukur"], CLASS_COLORS["tertunjuk"],
              CLASS_COLORS["tereka"], OUTSIDE_COLOR]
    written = []
    for offset, estimate in enumerate(estimates):
        model = models[estimate.key]
        coded = np.full(estimate.klass.shape, np.nan)
        for value, name in enumerate(order):
            coded[estimate.klass == name] = value
        if not np.isfinite(coded).any():
            continue

        figure, ax = plt.subplots(figsize=(8.4, 6.2))
        figure.patch.set_facecolor(SURFACE)
        ax.pcolormesh(model.roof.x, model.roof.y, coded,
                      cmap=ListedColormap(colors),
                      norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), shading="auto")
        sub = points_frame[(points_frame["seam"] == estimate.seam)
                           & points_frame["qualifies"]]
        if len(sub):
            ax.scatter(sub["east"], sub["north"], s=34, c="white",
                       edgecolors=INK, linewidths=1.0, zorder=5,
                       label="Titik pengamatan")
        handles = [Patch(facecolor=c, label=CLASS_LABELS.get(n, "Di luar radius"))
                   for n, c in zip(order, colors)]
        if len(sub):
            handles.append(plt.Line2D([], [], marker="o", color="none",
                                      markerfacecolor="white",
                                      markeredgecolor=INK, markersize=7,
                                      label="Titik pengamatan"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=9, labelcolor=INK_SOFT)
        ax.set_aspect("equal")
        _style(ax, f"Peta sumberdaya Seam {estimate.seam}",
               "Easting (m)", "Northing (m)")
        figure.tight_layout()
        rows = pd.DataFrame([{"kelas": CLASS_LABELS.get(n, "Di luar radius"),
                              "luas_ha": round(estimate.area_ha.get(n, 0.0), 1),
                              "ton": round(estimate.tonnes.get(n, 0.0))}
                             for n in order])
        written += save_figure(run, figure, f"Gambar 5.{number_start + offset}",
                               f"Peta sumberdaya Seam {estimate.seam}", rows)
    return written


# --- Slot yang belum dapat dihasilkan --------------------------------------

def write_manual_slots(run) -> list[Path]:
    """Cetak slot kelas B dan C sebagai penanda, bukan dihilangkan diam-diam."""
    lines = ["# Slot yang menunggu isi manual", "",
             "Berkas ini SENGAJA ada. Laporan yang kehilangan satu gambar tanpa",
             "jejak jauh lebih berbahaya daripada laporan yang menyatakan gambar",
             "itu belum dibuat.", "",
             "## Kelas B - menunggu masukan ekonomi", ""]
    for number, what in CLASS_B_SLOTS.items():
        lines.append(f"- **{number}** - {what}  -> {MANUAL}")
    lines += ["", "## Kelas C - digambar manusia", ""]
    for number, what in CLASS_C_SLOTS.items():
        lines.append(f"- **{number}** - {what}  -> {MANUAL}")
    target = run.path("01_bab5_laporan", "SLOT_MANUAL.md")
    target.write_text("\n".join(lines) + "\n")
    return [run.record(target, "slot_manual",
                       note=f"{len(CLASS_B_SLOTS) + len(CLASS_C_SLOTS)} slot")]


def write_docx(run, label: str, condition: str, estimate_report,
               assessment) -> list[Path]:
    """Draf DOCX Bab V. Draf, bukan laporan jadi - penilaian CP belum ada."""
    try:
        from docx import Document
    except Exception as exc:
        log.warning(f"DOCX dilewati, python-docx tidak tersedia: {exc}")
        return []

    document = Document()
    document.add_heading("BAB V  PEMODELAN DAN ESTIMASI SUMBER DAYA", 0)
    document.add_paragraph(
        f"DRAF OTOMATIS - {run.project}, run {run.root.name}. Dokumen ini "
        "dihasilkan program dan BELUM memuat penilaian Competent Person.")

    document.add_heading("5.1  Pemodelan Geologi", level=1)
    document.add_paragraph(
        f"Kondisi geologi dinilai '{condition}' melalui pembobotan dua tingkat "
        f"pada delapan subaspek (nilai akhir {assessment.final_value:.3f}). "
        "Rinciannya pada Tabel 5-32.")
    for warning in assessment.warnings:
        document.add_paragraph(warning, style="Intense Quote")

    document.add_heading("5.2  Estimasi Sumber Daya", level=1)
    frame = estimate_report.table(label)
    table = document.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light Grid Accent 1"
    for cell, column in zip(table.rows[0].cells, frame.columns):
        cell.text = str(column)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = "" if pd.isna(value) else str(value)

    document.add_heading("5.3  Asumsi dan Batasan", level=1)
    for item in run.assumptions:
        document.add_paragraph(f"{item['topic']}: {item['statement']}", style="List Bullet")

    document.add_heading("5.4  Slot yang menunggu isi manual", level=1)
    for number, what in {**CLASS_B_SLOTS, **CLASS_C_SLOTS}.items():
        document.add_paragraph(f"{number} - {what} [{MANUAL}]", style="List Bullet")

    target = run.path("01_bab5_laporan", "draf_bab5.docx")
    document.save(target)
    return [run.record(target, "draf_docx")]
