"""Plot log per lubang: kurva LAS di samping litologi dan pick seam.

Tujuannya verifikasi visual pick, bukan pengukuran. Kurva LD dan SD adalah
CACAH MENTAH (CPS) - sumbunya diberi label satuan aslinya dan TIDAK PERNAH
dikonversi menjadi densitas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .audit.checks import coal_codes, seam_intervals, unrecovered_codes
from .io.excel import Workbook
from .io.las import LasFile
from .logging_setup import get_logger
from .palette import (
    GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, LITHOLOGY_FILL, SERIES_BLUE,
    SERIES_ORANGE, SURFACE, style_axes,
)

log = get_logger("logs")


def plot_hole(wb: Workbook, las: LasFile | None, path: Path,
              title_suffix: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    intervals = seam_intervals(wb, "SLL_Reconciled")
    if intervals.empty:
        raise ValueError(f"{wb.hole_id}: tidak ada interval litologi untuk diplot")

    coal = coal_codes(wb.lithology_library)
    lost = unrecovered_codes(wb.lithology_library)
    depth_max = float(intervals["depth_to"].max())

    n_panels = 1 + (2 if las is not None else 0)
    widths = [1.0] + ([1.6, 1.6] if las is not None else [])
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.0 * n_panels + 1.4, 9.0), sharey=True,
        gridspec_kw={"width_ratios": widths}, facecolor=SURFACE,
    )
    axes = np.atleast_1d(axes)

    # --- Panel litologi ----------------------------------------------------- #
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for row in intervals.itertuples():
        if row.lithology in coal:
            kind = "coal"
        elif row.lithology in lost:
            kind = "core_loss"
        else:
            kind = "waste"
        ax.add_patch(plt.Rectangle(
            (0, row.depth_from), 1, row.depth_to - row.depth_from,
            facecolor=LITHOLOGY_FILL[kind], edgecolor=SURFACE, linewidth=0.4,
        ))
    named = intervals[intervals["seam"].notna() & (intervals["seam"].astype(str) != "nan")
                      & (intervals["seam"].astype(str).str.strip() != "")
                      & (intervals["seam"].astype(str).str.upper() != "PA")]
    for seam, group in named.groupby("seam"):
        roof, floor = group["depth_from"].min(), group["depth_to"].max()
        ax.text(1.06, (roof + floor) / 2, str(seam), va="center", ha="left",
                fontsize=8, color=INK_PRIMARY)
        for depth in (roof, floor):
            ax.plot([0, 1], [depth, depth], color=INK_PRIMARY, linewidth=0.9)
    ax.set_xlim(0, 1.35)
    ax.set_xticks([])
    ax.set_ylabel("Kedalaman (m)", color=INK_SECONDARY, fontsize=9)
    style_axes(ax, "Litologi & pick seam", grid_axis="y")
    ax.legend(handles=[
        Patch(facecolor=LITHOLOGY_FILL["coal"], label="Batubara"),
        Patch(facecolor=LITHOLOGY_FILL["waste"], label="Non-batubara"),
        Patch(facecolor=LITHOLOGY_FILL["core_loss"], label="Core loss"),
    ], loc="lower left", fontsize=7.5, frameon=True, facecolor=SURFACE,
        edgecolor=GRIDLINE, labelcolor=INK_SECONDARY)

    # --- Panel kurva LAS ---------------------------------------------------- #
    if las is not None:
        for ax_, mnemonics, colour in (
            (axes[1], ["GR"], SERIES_BLUE),
            (axes[2], ["LD", "SD"], SERIES_ORANGE),
        ):
            ax_.set_facecolor(SURFACE)
            plotted = []
            for index, mnemonic in enumerate(mnemonics):
                curve = las.curves.get(mnemonic)
                if curve is None:
                    continue
                ok = np.isfinite(curve.data)
                ax_.plot(curve.data[ok], las.depth[ok], linewidth=1.0,
                         color=colour if index == 0 else INK_MUTED,
                         label=f"{mnemonic} ({curve.unit or '?'})")
                plotted.append(mnemonic)
            unit = las.curves[plotted[0]].unit if plotted else "?"
            # Satuan asli dipertahankan: cacah mentah bukan densitas.
            style_axes(ax_, f"{' / '.join(plotted)} [{unit}]", grid_axis="both")
            if len(plotted) >= 1:
                ax_.legend(loc="lower right", fontsize=7.5, frameon=True,
                           facecolor=SURFACE, edgecolor=GRIDLINE,
                           labelcolor=INK_SECONDARY)
            for row in named.itertuples():
                ax_.axhspan(row.depth_from, row.depth_to,
                            color=LITHOLOGY_FILL["coal"], alpha=0.18, zorder=0)

    axes[0].set_ylim(depth_max * 1.02, 0)
    fig.suptitle(f"{wb.hole_id}{title_suffix}", x=0.02, ha="left",
                 color=INK_PRIMARY, fontsize=12)
    if las is not None and las.density_curves_are_raw_counts:
        fig.text(0.02, 0.965,
                 "LD/SD adalah cacah mentah (CPS), bukan bulk density - hanya untuk "
                 "verifikasi pick, tidak dipakai untuk tonase.",
                 color=INK_SECONDARY, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path
