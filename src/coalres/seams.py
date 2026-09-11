"""Rekonstruksi interseksi seam dari log litologi terekonsiliasi.

Keputusan yang dikunci konfigurasi dan dicap pada setiap keluaran:

  coal_thickness_source = lithology
      Tebal batubara berasal dari log litologi, bukan dari interval sampling.
      Ini konvensi Minex, dipilih agar angka sebanding dengan model terdahulu.
      Konsekuensinya, interval kualitas TIDAK sama dengan interval tebal; selisih
      itu tidak disembunyikan melainkan dilaporkan lewat thickness_quality_
      reconciliation().

  core_loss_treatment = as_coal
      Core loss YANG BERATRIBUT SEAM dihitung sebagai batubara, mengikuti Minex.
      Ini asumsi, bukan pengamatan: material itu tidak terambil, sehingga isinya
      tidak diketahui. Panjangnya selalu dilaporkan terpisah.

Core loss DI LUAR amplop seam selalu waste, apa pun setelannya. Pada DH09_05C1
ada 0,445 m core loss di batuan penutup dan interburden (70,070-70,270 dan
83,470-83,715) yang akan salah terhitung sebagai batubara bila aturan ini tidak
dibatasi pada atribusi seam yang dibuat geolog.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from .audit.checks import coal_codes, seam_intervals, unrecovered_codes
from .config import Config
from .errors import MissingDataError
from .io.excel import Workbook, normalise_hole_id
from .logging_setup import get_logger

log = get_logger("seams")


class Material(str, Enum):
    COAL = "coal"
    CORE_LOSS = "core_loss"
    PARTING = "parting"
    WASTE = "waste"


@dataclass(frozen=True)
class SeamIntersection:
    """Satu interseksi seam pada satu lubang, dalam meter kedalaman."""

    hole_id: str
    seam: str
    roof_m: float
    floor_m: float
    gross_thickness_m: float
    coal_thickness_m: float
    parting_thickness_m: float
    core_loss_thickness_m: float
    core_loss_in_coal_m: float
    n_intervals: int
    max_parting_m: float
    thickness_source: str
    core_loss_treatment: str
    split_from: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.hole_id, self.seam)


def classify_materials(
    intervals: pd.DataFrame, library: dict[str, str]
) -> pd.DataFrame:
    """Beri label material pada tiap interval berdasarkan kode dari Library SLL."""
    coal = coal_codes(library)
    lost = unrecovered_codes(library)
    named = (intervals["seam"].notna() & (intervals["seam"].astype(str) != "nan")
             & (intervals["seam"].astype(str).str.strip() != ""))

    out = intervals.copy()
    out["seam_attributed"] = named
    out["material"] = np.where(
        out["lithology"].isin(coal), Material.COAL.value,
        np.where(out["lithology"].isin(lost), Material.CORE_LOSS.value,
                 Material.PARTING.value),
    )
    return out


def _envelope_rows(intervals: pd.DataFrame, roof: float, floor: float) -> pd.DataFrame:
    return intervals[(intervals["depth_from"] >= roof - 1e-9)
                     & (intervals["depth_to"] <= floor + 1e-9)]


def build_intersections(wb: Workbook, cfg: Config,
                        apply_cutoffs: bool = True) -> list[SeamIntersection]:
    """Bangun interseksi seam untuk satu lubang."""
    if cfg.coal_thickness_source != "lithology":
        raise MissingDataError(
            f"coal_thickness_source='{cfg.coal_thickness_source}' belum "
            "diimplementasikan; hanya 'lithology' yang tersedia."
        )

    raw = seam_intervals(wb, "SLL_Reconciled")
    if raw.empty:
        return []
    labelled = classify_materials(raw, wb.lithology_library)
    hole = normalise_hole_id(wb.hole_id)

    treatment = cfg.core_loss_treatment
    max_parting = cfg.cutoffs.max_parting_thickness_m
    min_seam = cfg.cutoffs.min_seam_thickness_m

    # Amplop seam: baris yang diberi nama seam oleh geolog. 'PA' adalah penanda
    # parting di kolom Seam, bukan nama seam.
    named = labelled[labelled["seam_attributed"]
                     & (labelled["seam"].str.upper() != "PA")]

    results: list[SeamIntersection] = []
    for seam, group in named.groupby("seam", sort=False):
        roof, floor = float(group["depth_from"].min()), float(group["depth_to"].max())
        envelope = _envelope_rows(labelled, roof, floor).sort_values("depth_from")

        # Parting di atas cutoff MEMISAHKAN seam; di bawah cutoff ia masuk gross
        # tetapi keluar dari tebal batubara.
        splitters = envelope[
            (envelope["material"] == Material.PARTING.value)
            & ((envelope["depth_to"] - envelope["depth_from"]) > max_parting)
        ]
        segments: list[tuple[float, float]] = []
        cursor = roof
        for sp in splitters.itertuples():
            if sp.depth_from - cursor > 1e-9:
                segments.append((cursor, sp.depth_from))
            cursor = sp.depth_to
        if floor - cursor > 1e-9:
            segments.append((cursor, floor))
        if not segments:
            segments = [(roof, floor)]

        for index, (seg_roof, seg_floor) in enumerate(segments):
            rows = _envelope_rows(labelled, seg_roof, seg_floor)
            length = (rows["depth_to"] - rows["depth_from"]).to_numpy(float)
            material = rows["material"].to_numpy()
            attributed = rows["seam_attributed"].to_numpy(bool)

            coal_m = float(length[material == Material.COAL.value].sum())
            parting_m = float(length[material == Material.PARTING.value].sum())
            loss_all = material == Material.CORE_LOSS.value
            loss_m = float(length[loss_all].sum())
            # Hanya core loss yang diatribusikan geolog ke seam yang boleh ikut.
            loss_in_seam = float(length[loss_all & attributed].sum())

            if treatment == "as_coal":
                coal_total = coal_m + loss_in_seam
                parting_total = parting_m + (loss_m - loss_in_seam)
                gross = seg_floor - seg_roof
            elif treatment == "as_waste":
                coal_total = coal_m
                parting_total = parting_m + loss_m
                gross = seg_floor - seg_roof
            else:  # exclude
                coal_total = coal_m
                parting_total = parting_m
                gross = (seg_floor - seg_roof) - loss_m

            partings = length[material == Material.PARTING.value]
            results.append(SeamIntersection(
                hole_id=hole,
                seam=str(seam) if len(segments) == 1 else f"{seam}_{index + 1}",
                roof_m=seg_roof, floor_m=seg_floor,
                gross_thickness_m=gross,
                coal_thickness_m=coal_total,
                parting_thickness_m=parting_total,
                core_loss_thickness_m=loss_m,
                core_loss_in_coal_m=loss_in_seam if treatment == "as_coal" else 0.0,
                n_intervals=len(rows),
                max_parting_m=float(partings.max()) if partings.size else 0.0,
                thickness_source=cfg.coal_thickness_source,
                core_loss_treatment=treatment,
                split_from=str(seam) if len(segments) > 1 else None,
            ))

    if not apply_cutoffs:
        # Ketebalan UNCUT: seluruh interseksi apa adanya, tanpa aturan
        # penambangan. Grid yang dibangun darinya adalah ketebalan geologi,
        # bukan ketebalan yang dapat ditambang.
        return results

    kept = []
    for item in results:
        if item.coal_thickness_m < min_seam:
            log.info(
                f"{item.hole_id}/{item.seam}: tebal batubara "
                f"{item.coal_thickness_m:.3f} m di bawah cutoff {min_seam} m - dikeluarkan"
            )
            continue
        kept.append(item)
    return kept


def to_frame(intersections: list[SeamIntersection]) -> pd.DataFrame:
    if not intersections:
        return pd.DataFrame()
    return pd.DataFrame([vars(i) for i in intersections])


def thickness_quality_reconciliation(
    intersections: list[SeamIntersection],
    workbooks: dict[str, Workbook],
    quality_spans: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Selisih antara interval yang MENENTUKAN TONASE dan yang DIANALISA LAB.

    Minex menerima tebal dari tabel litologi dan kualitas dari tabel komposit
    tanpa memeriksa bahwa keduanya menutupi interval yang sama. Ketidakcocokan
    itu tidak memunculkan peringatan di sana. Fungsi ini membuatnya terlihat dan
    terukur, sehingga angka tetap sebanding dengan Minex tanpa ikut menyembunyikan
    selisihnya.
    """
    rows = []
    for item in intersections:
        wb = workbooks.get(item.hole_id)
        span = (quality_spans or {}).get(item.key)
        record = {
            "hole_id": item.hole_id, "seam": item.seam,
            "roof_m": item.roof_m, "floor_m": item.floor_m,
            "coal_thickness_m": item.coal_thickness_m,
            "parting_in_envelope_m": item.parting_thickness_m,
            "core_loss_counted_as_coal_m": item.core_loss_in_coal_m,
        }
        if span is None:
            record.update({
                "quality_from_m": np.nan, "quality_to_m": np.nan,
                "quality_span_m": np.nan, "uncovered_m": np.nan,
                "uncovered_frac": np.nan,
            })
        else:
            q_from, q_to = span
            q_span = q_to - q_from
            uncovered = max(item.coal_thickness_m - q_span, 0.0)
            record.update({
                "quality_from_m": q_from, "quality_to_m": q_to,
                "quality_span_m": q_span, "uncovered_m": uncovered,
                "uncovered_frac": uncovered / item.coal_thickness_m
                if item.coal_thickness_m > 0 else np.nan,
            })
        rows.append(record)
    return pd.DataFrame(rows)


def assumptions(cfg: Config) -> list[str]:
    """Asumsi yang WAJIB ikut ke setiap tabel keluaran."""
    notes = [
        f"Tebal batubara berasal dari log litologi terekonsiliasi "
        f"(coal_thickness_source='{cfg.coal_thickness_source}'), mengikuti konvensi "
        f"Minex agar sebanding dengan model terdahulu.",
        "Interval kualitas tidak sama dengan interval tebal; selisihnya "
        "dilaporkan pada sheet rekonsiliasi tebal-kualitas, bukan diserap diam-diam.",
    ]
    if cfg.core_loss_treatment == "as_coal":
        notes.append(
            "Core loss yang diatribusikan geolog ke suatu seam dihitung sebagai "
            "batubara (core_loss_treatment='as_coal', konvensi Minex). Ini ASUMSI: "
            "material tidak terambil sehingga isinya tidak diketahui. Panjangnya "
            "dilaporkan terpisah pada kolom core_loss_counted_as_coal_m."
        )
    elif cfg.core_loss_treatment == "as_waste":
        notes.append("Core loss dihitung bersama parting (core_loss_treatment='as_waste').")
    else:
        notes.append(
            "Core loss dikeluarkan dari gross maupun tebal batubara "
            "(core_loss_treatment='exclude') dan dilaporkan sebagai panjang tak diketahui."
        )
    notes.append(
        "Core loss di luar amplop seam selalu dihitung waste, apa pun setelan di atas."
    )
    return notes


def build_intersections_from_dataset(dataset, cfg: Config,
                                     apply_cutoffs: bool = True) -> list[SeamIntersection]:
    """Interseksi seam dari HoleDataset (jalur flat file Minex).

    Berbeda dari jalur workbook BGG: berkas `lit` Minex mencantumkan interval
    SEAM, bukan seluruh kolom litologi. Jadi tidak ada parting maupun core loss
    untuk dikurangkan - tebal batubara adalah to - from. Menerapkan logika
    parting BGG di sini akan mengurangi tebal berdasarkan interval yang memang
    tidak pernah dicatat.

    Baris penanda (mis. 'W') berketebalan nol; ia menandai horizon, bukan seam,
    dan dikeluarkan di sini.
    """
    intervals = dataset.intervals
    if intervals.empty:
        return []

    marker = intervals["is_marker"] if "is_marker" in intervals else False
    body = intervals[~np.asarray(marker, dtype=bool)].copy()
    body = body[np.isfinite(body["depth_from"]) & np.isfinite(body["depth_to"])]

    min_seam = cfg.cutoffs.min_seam_thickness_m
    results: list[SeamIntersection] = []
    for (hole, seam), group in body.groupby(["hole_id", "seam"], sort=False):
        roof = float(group["depth_from"].min())
        floor = float(group["depth_to"].max())
        # Beberapa baris untuk satu seam diperlakukan sebagai ply: tebal
        # batubara adalah jumlah panjangnya, bukan amplop roof-floor.
        coal = float((group["depth_to"] - group["depth_from"]).clip(lower=0).sum())
        gross = floor - roof
        results.append(SeamIntersection(
            hole_id=str(hole), seam=str(seam), roof_m=roof, floor_m=floor,
            gross_thickness_m=gross, coal_thickness_m=coal,
            parting_thickness_m=max(gross - coal, 0.0),
            core_loss_thickness_m=0.0, core_loss_in_coal_m=0.0,
            n_intervals=len(group), max_parting_m=max(gross - coal, 0.0),
            thickness_source="lithology_interval",
            core_loss_treatment="n/a (tidak dicatat di berkas lit)",
        ))

    if not apply_cutoffs:
        return results

    kept = []
    for item in results:
        if item.coal_thickness_m < min_seam:
            log.info(f"{item.hole_id}/{item.seam}: tebal {item.coal_thickness_m:.3f} m "
                     f"di bawah cutoff {min_seam} m - dikeluarkan")
            continue
        kept.append(item)
    return kept
