"""Tahap 10: estimasi Sumber daya dengan metode circular di atas sel model.

Metode circular mengikuti Pedoman Praktis KCMI 2017, yang mengilustrasikannya
pada pasal 4.5.4 dan 4.5.5: gabungan cakram di sekeliling tiap Titik Pengamatan,
berpita Terukur - Tertunjuk - Tereka.

Empat aturan keras yang ditegakkan di sini, semuanya tanpa saklar konfigurasi:

  1. Sel EKSTRAPOLASI tidak pernah Terukur. Di luar hull data, model menebak;
     kelas tertinggi menuntut pengamatan, bukan tebakan.
  2. Batubara di luar radius Tereka DIKELUARKAN, tidak pernah dilipat menjadi
     Tereka. Melipatnya berarti memberi kelas kepada tanah yang tidak memenuhi
     syarat kelas mana pun.
  3. Bagian yang gagal KCMI 4.5.4 (minimum 3 titik) atau 4.5.5 (spotted dog)
     tidak memperoleh kelas itu. Ia masih dapat memenuhi syarat pada radius yang
     lebih besar, tempat bagiannya menyatu dan sebarannya melebar.
  4. Swauji peta: tiap titik pengamatan yang memenuhi syarat WAJIB jatuh di
     dalam poligon kelas tertinggi seam-nya. Bila tidak, peta dan tabel tidak
     sinkron - itu bug, dan run berhenti.

Tonase = luas datar sel x ketebalan VERTIKAL x RD. Tanpa koreksi cos(dip):
luas datar dikali tebal vertikal sudah merupakan volume prisma yang eksak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logging_setup import get_logger
from .model import SUPPORT_EXTRAPOLATED, SeamModel
from .poo import evaluate_areas

log = get_logger("estimate_grid")

CLASSES = ("terukur", "tertunjuk", "tereka")
OUTSIDE = "di luar radius"
LABELS = {"terukur": "Terukur", "tertunjuk": "Tertunjuk", "tereka": "Tereka"}


@dataclass
class SeamEstimate:
    key: str
    seam: str
    cell_area_m2: float
    klass: np.ndarray                 # (ny, nx) string kelas atau "" bila mati
    tonnes: dict[str, float] = field(default_factory=dict)
    area_ha: dict[str, float] = field(default_factory=dict)
    rd_assumed_fraction: float = float("nan")
    demotions: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class EstimateReport:
    estimates: list[SeamEstimate] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def table(self, label: str = "Sumberdaya") -> pd.DataFrame:
        rows = []
        for e in self.estimates:
            row = {"seam": e.seam, "domain": e.key}
            for klass in CLASSES:
                row[f"{label} {LABELS[klass]} (ton)"] = round(e.tonnes.get(klass, 0.0))
                row[f"{LABELS[klass]} (ha)"] = round(e.area_ha.get(klass, 0.0), 1)
            row[f"{OUTSIDE} (ton)"] = round(e.tonnes.get(OUTSIDE, 0.0))
            row[f"{OUTSIDE} (ha)"] = round(e.area_ha.get(OUTSIDE, 0.0), 1)
            row["RD asumsi (frac)"] = round(e.rd_assumed_fraction, 3)
            rows.append(row)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            total = frame.drop(columns=["seam", "domain"]).sum(numeric_only=True)
            total["RD asumsi (frac)"] = float("nan")
            frame.loc[len(frame)] = {"seam": "TOTAL", "domain": "", **total}
        return frame


def _disc_mask(surface, points: np.ndarray, radius: float) -> np.ndarray:
    """Sel yang berada dalam `radius` dari salah satu titik."""
    gx, gy = np.meshgrid(surface.x, surface.y)
    if len(points) == 0:
        return np.zeros(gx.shape, dtype=bool)
    from scipy.spatial import cKDTree
    distance, _ = cKDTree(points).query(
        np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    return (distance.reshape(gx.shape) <= radius)


def _qualifying_points(points_frame: pd.DataFrame, intersections: pd.DataFrame,
                       collars: pd.DataFrame, seam: str, radius: float
                       ) -> tuple[np.ndarray, list[str]]:
    """Titik pengamatan pada bagian yang LOLOS KCMI 4.5.4 dan 4.5.5.

    HANYA titik yang memenuhi syarat titik pengamatan yang ikut. Melewatkan
    lubang yang gagal - tanpa kualitas, di bawah cutoff - berarti mengklasifikasi
    dari lubang yang menurut SNI/KCMI tidak berhak menaikkan kelas, dan itu
    menggelembungkan Terukur secara diam-diam.
    """
    if "qualifies" in points_frame.columns:
        points_frame = points_frame[points_frame["qualifies"]]
    areas = evaluate_areas(points_frame, intersections, collars, seam, radius)
    if areas.empty:
        return np.empty((0, 2)), []
    good = areas[areas["memenuhi_kcmi"]]
    holes: list[str] = []
    for value in good["lubang"]:
        holes.extend(h.strip() for h in str(value).split(",") if h.strip())
    sub = points_frame[points_frame["hole_id"].isin(holes)
                       & (points_frame["seam"] == seam)]
    return sub[["east", "north"]].to_numpy(float), holes


def estimate_seam(key: str, model: SeamModel, alive: np.ndarray,
                  points_frame: pd.DataFrame, intersections: pd.DataFrame,
                  collars: pd.DataFrame, radii: dict[str, float],
                  rd_grid: np.ndarray, rd_assumed: np.ndarray) -> SeamEstimate:
    """Klasifikasikan sel seam ini dan hitung tonasenya."""
    cell = model.roof.spacing ** 2
    klass = np.full(alive.shape, "", dtype=object)
    demotions: dict[str, int] = {}
    notes: list[str] = []

    # Dari kelas TERTINGGI ke terendah: sel mengambil kelas pertama yang cocok.
    for name in CLASSES:
        radius = radii[name]
        points, _ = _qualifying_points(points_frame, intersections, collars,
                                       model.seam, radius)
        if len(points) == 0:
            notes.append(f"kelas {LABELS[name]}: tidak ada bagian yang lolos "
                         "KCMI 4.5.4/4.5.5 pada radius ini.")
            continue
        covered = _disc_mask(model.roof, points, radius) & alive & (klass == "")

        if name == "terukur":
            # Aturan keras 1: ekstrapolasi tidak pernah Terukur.
            extrapolated = covered & (model.support == SUPPORT_EXTRAPOLATED)
            if extrapolated.any():
                demotions["ekstrapolasi tidak jadi Terukur"] = int(extrapolated.sum())
                covered &= ~extrapolated
        klass[covered] = name

    # Aturan keras 2: sisanya DIKELUARKAN, tidak dilipat menjadi Tereka.
    klass[alive & (klass == "")] = OUTSIDE

    tonnes, area = {}, {}
    thickness = model.isopach.z
    for name in list(CLASSES) + [OUTSIDE]:
        mask = klass == name
        n = int(mask.sum())
        area[name] = n * cell / 1e4
        tonnes[name] = float(np.nansum(thickness[mask] * rd_grid[mask])) * cell

    live = alive & np.isfinite(thickness)
    fraction = float(rd_assumed[live].mean()) if live.any() else float("nan")
    return SeamEstimate(key=key, seam=model.seam, cell_area_m2=cell, klass=klass,
                        tonnes=tonnes, area_ha=area, rd_assumed_fraction=fraction,
                        demotions=demotions, notes=notes)


def _rd_grids(model: SeamModel, intersections: pd.DataFrame,
              collars: pd.DataFrame, quality: pd.DataFrame | None,
              assumed_rd: float) -> tuple[np.ndarray, np.ndarray]:
    """Grid RD dan grid penanda "RD ini asumsi", keduanya dari lubang terdekat."""
    rows = []
    for _, row in intersections[intersections["seam"] == model.seam].iterrows():
        hole = row["hole_id"]
        if hole not in collars.index:
            continue
        value, is_assumed = assumed_rd, True
        if quality is not None:
            match = quality[(quality["hole_id"] == hole)
                            & (quality["seam"] == model.seam)]
            if len(match) and "RD" in match and match["RD"].notna().any():
                value, is_assumed = float(match["RD"].mean()), False
        collar = collars.loc[hole]
        rows.append((float(collar["east"]), float(collar["north"]), value,
                     1.0 if is_assumed else 0.0))

    gx, gy = np.meshgrid(model.roof.x, model.roof.y)
    if not rows:
        return (np.full(gx.shape, assumed_rd), np.ones(gx.shape))
    arr = np.asarray(rows, float)
    from scipy.spatial import cKDTree
    _, index = cKDTree(arr[:, :2]).query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    return (arr[index, 2].reshape(gx.shape), arr[index, 3].reshape(gx.shape))


def _kcmi_compliant_holes(points_frame: pd.DataFrame, intersections: pd.DataFrame,
                          collars: pd.DataFrame, seam: str,
                          radii: dict[str, float]) -> set[str]:
    """Lubang yang berada di bagian yang lolos KCMI 4.5.4/4.5.5 pada radius mana pun."""
    holes: set[str] = set()
    for radius in radii.values():
        _, names = _qualifying_points(points_frame, intersections, collars,
                                      seam, radius)
        holes.update(names)
    return holes


def self_check_map(estimates: list[SeamEstimate], models: dict[str, SeamModel],
                   limit_masks: dict[str, np.ndarray],
                   points_frame: pd.DataFrame, intersections: pd.DataFrame,
                   collars: pd.DataFrame, radii: dict[str, float],
                   report: EstimateReport) -> None:
    """Aturan keras 4: tiap titik pengamatan jatuh di kelas tertinggi seam-nya.

    Tiga keadaan yang harus DIBEDAKAN, karena hanya satu yang bug:

      sel masih hidup tetapi tak berkelas - peta dan tabel bercerita berbeda,
      dan pembaca mempercayai petanya. Itu BUG, dan run berhenti.

      sel dibuang cut-off - sah. Batubara di titik itu tersaring pelapukan,
      tebal minimum, atau batas lain, sehingga memang tidak boleh dilaporkan.
      Titik pengamatannya tetap valid untuk seam itu di tempat lain.

      sel kosong di model - lubang berada di tepi dukungan sehingga node grid
      di sekitarnya NaN. Sifat grid, bukan cacat model, tetapi perlu diketahui
      karena titik itu tidak menopang sel mana pun.

      titik sah tetapi bagiannya GAGAL KCMI 4.5.4/4.5.5 - juga sah, dan justru
      inilah temuan spotted dog itu sendiri. Titiknya memenuhi syarat titik
      pengamatan, tetapi sebarannya tidak membentuk area Sumber daya, sehingga
      tidak ada cakram yang ditarik darinya.
    """
    for estimate in estimates:
        model = models[estimate.key]
        alive = limit_masks.get(estimate.key)
        compliant = _kcmi_compliant_holes(points_frame, intersections, collars,
                                          estimate.seam, radii)
        sub = points_frame[(points_frame["seam"] == estimate.seam)
                           & points_frame["qualifies"]]
        for _, row in sub.iterrows():
            ix = int(np.argmin(np.abs(model.roof.x - float(row["east"]))))
            iy = int(np.argmin(np.abs(model.roof.y - float(row["north"]))))
            at = estimate.klass[iy, ix]
            if at not in ("", OUTSIDE):
                continue

            modelled = np.isfinite(model.isopach.z[iy, ix])
            kept = bool(alive[iy, ix]) if alive is not None else modelled

            if kept and str(row["hole_id"]) not in compliant:
                report.notes.append(
                    f"titik pengamatan {row['hole_id']} pada seam "
                    f"{estimate.seam} tidak berkelas karena bagiannya GAGAL KCMI "
                    "4.5.4/4.5.5 - jumlah titik kurang dari 3, atau tidak menerus "
                    "dua arah. Ini temuan spotted dog, bukan ketidaksinkronan "
                    "peta: titiknya sah, sebarannya yang tidak membentuk area.")
            elif kept:
                report.failures.append(
                    f"swauji peta: titik pengamatan {row['hole_id']} pada seam "
                    f"{estimate.seam} berdiri di sel yang MASIH HIDUP tetapi "
                    f"berkelas '{at or 'kosong'}'. Peta dan tabel tidak sinkron, "
                    "dan pembaca mempercayai petanya.")
            elif modelled:
                report.notes.append(
                    f"titik pengamatan {row['hole_id']} pada seam "
                    f"{estimate.seam} berada di sel yang DIBUANG cut-off "
                    "(pelapukan atau tebal minimum). Sah: batubara di titik itu "
                    "memang tidak dapat dilaporkan.")
            else:
                report.warnings.append(
                    f"titik pengamatan {row['hole_id']} pada seam "
                    f"{estimate.seam} berada di sel KOSONG pada model - lubang "
                    "di tepi dukungan, node grid di sekitarnya NaN. Titik itu "
                    "tidak menopang sel mana pun; perkecil sel atau beri margin "
                    "bila daerah tepi ikut dilaporkan.")


def run(models: dict[str, SeamModel], limit_masks: dict[str, np.ndarray],
        points_frame: pd.DataFrame, intersections: pd.DataFrame,
        collars: pd.DataFrame, radii: dict[str, float], cfg,
        quality: pd.DataFrame | None = None) -> EstimateReport:
    report = EstimateReport()
    assumed = cfg.assumed_rd_t_per_m3 or 1.30

    for key, model in models.items():
        alive = limit_masks.get(key)
        if alive is None:
            alive = np.isfinite(model.isopach.z)
        rd_grid, rd_assumed = _rd_grids(model, intersections, collars, quality,
                                        assumed)
        report.estimates.append(estimate_seam(
            key, model, alive, points_frame, intersections, collars, radii,
            rd_grid, rd_assumed))

    self_check_map(report.estimates, models, limit_masks, points_frame,
                   intersections, collars, radii, report)

    for estimate in report.estimates:
        outside = estimate.tonnes.get(OUTSIDE, 0.0)
        total = sum(estimate.tonnes.values())
        if total > 0 and outside / total > 0.5:
            report.warnings.append(
                f"seam {estimate.seam}: {100 * outside / total:.0f}% tonase berada "
                "DI LUAR radius kelas mana pun dan dikeluarkan dari pelaporan. "
                "Sesuai aturan, ia tidak dilipat menjadi Tereka.")
        for reason, count in estimate.demotions.items():
            report.notes.append(f"seam {estimate.seam}: {count} sel - {reason}.")
        report.notes.extend(f"seam {estimate.seam}: {n}" for n in estimate.notes)

    for line in report.failures:
        log.error(line)
    return report
