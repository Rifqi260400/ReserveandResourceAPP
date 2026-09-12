"""Tahap 4: pemodelan seam - roof, isopach, floor, dan dukungannya.

Dua aturan yang menentukan bentuk modul ini.

STACKING STRUKTUR + ISOPACH. Roof dan floor TIDAK diinterpolasi bebas satu
sama lain. Roof diinterpolasi, isopach (ketebalan) diinterpolasi, lalu
floor = roof - isopach. Bila keduanya diinterpolasi terpisah, di daerah
berdata jarang keduanya dapat berpotongan dan menghasilkan ketebalan NEGATIF -
cacat yang lolos dari mata karena tiap permukaannya sendiri tampak wajar.

FLAG DUKUNGAN PER SEL. Tiap sel membawa apakah ia di dalam atau di luar hull
titik data, dan berapa lubang yang menopangnya dalam radius pengaruh. Sel
EKSTRAPOLASI TIDAK PERNAH BOLEH TERUKUR - itu aturan keras, bukan setelan.

Batas dari topo.py tetap berlaku: permukaan di sini TIDAK menyumbang ketebalan
ke jalur tonase. Tonase memakai ketebalan lubangnya sendiri; permukaan hanya
memotong luas lewat subcrop dan batas kedalaman, serta memberi grid keluaran.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon

from .logging_setup import get_logger
from .topo import Surface, build_surface

log = get_logger("model")

SUPPORT_INTERPOLATED = "interpolated"
SUPPORT_EXTRAPOLATED = "extrapolated"


@dataclass
class SeamModel:
    """Model satu seam pada satu domain."""

    seam: str
    domain: str
    roof: Surface
    isopach: Surface
    floor: Surface
    support: np.ndarray          # (ny, nx) string: interpolated / extrapolated
    n_support: np.ndarray        # (ny, nx) int: lubang dalam radius pengaruh
    n_holes: int
    negative_thickness_cells: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def measurable(self) -> np.ndarray:
        """Sel yang BOLEH berkelas Terukur. Ekstrapolasi tidak pernah boleh."""
        return self.support == SUPPORT_INTERPOLATED

    def thickness(self) -> np.ndarray:
        return self.roof.z - self.floor.z


def domain_mask(surface: Surface, own: np.ndarray, rival: np.ndarray) -> np.ndarray:
    """Sel yang lubang TERDEKATNYA milik domain ini.

    Convex hull tidak cukup ketika sebuah seam hadir di dua kelompok yang
    terpisah. Pada pola split lens, seam induk hadir di barat DAN di timur,
    sehingga hull-nya menelan zona tengah tempat induk itu justru TIDAK ADA -
    dan permukaannya akan mengklaim batubara di sana.

    Penugasan tetangga terdekat memotongnya di tempat yang diceritakan data:
    batas domain jatuh di tengah antara lubang terluar kedua kelompok, tanpa
    seorang pun menggambar garis split dengan tangan.
    """
    if len(rival) == 0:
        return np.ones((len(surface.y), len(surface.x)), dtype=bool)
    from scipy.spatial import cKDTree
    gx, gy = np.meshgrid(surface.x, surface.y)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    own_d, _ = cKDTree(own).query(cells, k=1)
    rival_d, _ = cKDTree(rival).query(cells, k=1)
    return (own_d <= rival_d).reshape(gx.shape)


def _support_flags(surface: Surface, points: np.ndarray, influence_m: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Tandai tiap sel sebagai interpolasi atau ekstrapolasi, dan hitung penopang."""
    gx, gy = np.meshgrid(surface.x, surface.y)
    inside = surface.contains(gx.ravel(), gy.ravel()).reshape(gx.shape)
    flags = np.where(inside, SUPPORT_INTERPOLATED, SUPPORT_EXTRAPOLATED)

    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    counts = np.array(
        [len(hits) for hits in tree.query_ball_point(
            np.column_stack([gx.ravel(), gy.ravel()]), r=influence_m)]
    ).reshape(gx.shape)
    return flags, counts


def build_seam(seam: str, frame: pd.DataFrame, collars: pd.DataFrame,
               spacing: float, influence_m: float, domain: str = "penuh",
               method: str = "linear",
               rival_points: np.ndarray | None = None) -> SeamModel | None:
    """Bangun satu seam: roof diinterpolasi, isopach diinterpolasi, floor turunan.

    `frame` adalah interseksi seam pada domain ini; `collars` memberi koordinat.
    """
    rows = []
    for _, row in frame.iterrows():
        hole = row["hole_id"]
        if hole not in collars.index:
            continue
        collar = collars.loc[hole]
        roof_rl = float(collar["rl"]) - float(row["roof_m"])
        thickness = float(row["coal_thickness_m"])
        if not (np.isfinite(roof_rl) and np.isfinite(thickness) and thickness > 0):
            continue
        rows.append((float(collar["east"]), float(collar["north"]), roof_rl, thickness))

    if len(rows) < 3:
        log.warning(f"seam {seam} domain {domain}: {len(rows)} titik, "
                    "tidak cukup untuk TIN")
        return None

    arr = np.asarray(rows, float)
    points = arr[:, :2]
    # Keduanya dibangun dari titik yang SAMA, sehingga sumbu gridnya identik dan
    # floor = roof - isopach sah dilakukan sel demi sel.
    roof = build_surface(points[:, 0], points[:, 1], arr[:, 2], spacing=spacing,
                         name=f"{seam}_{domain}_roof", method=method)
    isopach = build_surface(points[:, 0], points[:, 1], arr[:, 3], spacing=spacing,
                            name=f"{seam}_{domain}_isopach", method=method)

    # Isopach tidak boleh negatif: ketebalan negatif tidak punya arti fisik, dan
    # membiarkannya akan mengurangi tonase secara diam-diam.
    negative = int(np.nansum(isopach.z < 0))
    if negative:
        isopach.z = np.where(isopach.z < 0, 0.0, isopach.z)

    floor = Surface(name=f"{seam}_{domain}_floor", x=roof.x, y=roof.y,
                    z=roof.z - isopach.z, spacing=roof.spacing,
                    method=f"{roof.method}+isopach", n_points=roof.n_points,
                    hull=roof.hull)

    flags, counts = _support_flags(roof, points, influence_m)

    # Potong ke domainnya sendiri bila ada domain saingan (pola split lens).
    clipped = 0
    if rival_points is not None and len(rival_points):
        keep = domain_mask(roof, points, rival_points)
        clipped = int(np.isfinite(roof.z)[~keep].sum())
        roof.z = np.where(keep, roof.z, np.nan)
        isopach.z = np.where(keep, isopach.z, np.nan)
        floor.z = np.where(keep, floor.z, np.nan)
        flags = np.where(keep, flags, SUPPORT_EXTRAPOLATED)
        counts = np.where(keep, counts, 0)

    notes = []
    if clipped:
        notes.append(
            f"{clipped} sel dipotong karena lubang terdekatnya milik domain "
            "saingan. Tanpa ini, hull seam yang hadir di dua kelompok terpisah "
            "akan menelan zona di antaranya - tempat seam itu justru tidak ada.")
    if negative:
        notes.append(
            f"{negative} sel isopach negatif dijepit ke nol. Interpolasi "
            "ketebalan dapat menembus nol di antara lubang tipis; dibiarkan, "
            "floor akan naik di atas roof.")
    extrapolated = int((flags == SUPPORT_EXTRAPOLATED).sum())
    if extrapolated:
        notes.append(
            f"{extrapolated} sel di luar hull data ditandai ekstrapolasi dan "
            "TIDAK PERNAH boleh berkelas Terukur.")

    return SeamModel(seam=seam, domain=domain, roof=roof, isopach=isopach,
                     floor=floor, support=flags, n_support=counts,
                     n_holes=len(rows), negative_thickness_cells=negative,
                     notes=notes)


def split_domains(frame: pd.DataFrame, collars: pd.DataFrame, cfg
                  ) -> dict[str, pd.DataFrame]:
    """Pisahkan interseksi menjadi domain menyatu dan domain terpecah.

    Seam yang memecah tidak boleh diinterpolasi melintasi garis split: di satu
    sisi ada satu lapisan, di sisi lain ada dua yang dipisahkan parting. Satu
    permukaan yang dipaksa mulus melewatinya akan membuat ramp palsu tepat di
    tempat geologinya justru berubah mendadak.

    Domain ditentukan dari data, bukan dari garis yang digambar tangan: tiap
    lubang sudah memberi tahu pola mana yang ia temui.
    """
    splits = cfg.seam_splits
    if not splits:
        return {}
    domains: dict[str, pd.DataFrame] = {}
    for parent, children in splits.items():
        present = set(frame["seam"])
        if parent not in present or not (set(children) & present):
            continue
        merged_holes = set(frame[frame["seam"] == parent]["hole_id"])
        split_holes = set(frame[frame["seam"].isin(children)]["hole_id"])
        domains[f"{parent}:menyatu"] = frame[
            (frame["seam"] == parent) & frame["hole_id"].isin(merged_holes)]
        domains[f"{parent}:terpecah"] = frame[
            frame["seam"].isin(children) & frame["hole_id"].isin(split_holes)]
    return domains


def domain_boundary(collars: pd.DataFrame, inside: set[str], outside: set[str],
                    buffer_m: float = 0.0) -> Polygon | None:
    """Batas domain sebagai hull lubangnya, tanpa menebak garis split.

    Garis split sejati berada di suatu tempat di antara lubang terluar kedua
    kelompok. Menaruhnya di tengah adalah tebakan; yang dilakukan di sini hanya
    membatasi tiap domain pada hull lubangnya sendiri, sehingga tidak ada
    permukaan yang menjulur ke wilayah yang datanya bercerita lain.
    """
    frame = collars.reindex(sorted(inside)).dropna(subset=["east", "north"])
    if len(frame) < 3:
        return None
    hull = MultiPoint(frame[["east", "north"]].to_numpy(float)).convex_hull
    return hull.buffer(buffer_m) if buffer_m else hull


def build(intersections: pd.DataFrame, collars: pd.DataFrame, cfg,
          spacing: float = 25.0, influence_m: float | None = None,
          method: str = "linear") -> dict[str, SeamModel]:
    """Bangun model tiap seam, memisahkan domain split bila ada."""
    if influence_m is None:
        # Radius pengaruh bawaan: jarak tetangga terdekat median antar lubang.
        from scipy.spatial import cKDTree
        pts = collars[["east", "north"]].dropna().to_numpy(float)
        nn, _ = cKDTree(pts).query(pts, k=2)
        influence_m = float(np.median(nn[:, 1]))

    domains = split_domains(intersections, collars, cfg)
    handled = set()
    models: dict[str, SeamModel] = {}

    def coords_of(frame: pd.DataFrame) -> np.ndarray:
        holes = sorted(set(frame["hole_id"]))
        sub = collars.reindex(holes).dropna(subset=["east", "north"])
        return sub[["east", "north"]].to_numpy(float)

    for label, frame in domains.items():
        parent, side = label.split(":")
        other = f"{parent}:{'terpecah' if side == 'menyatu' else 'menyatu'}"
        rival = coords_of(domains[other]) if other in domains else None
        for seam, group in frame.groupby("seam"):
            model = build_seam(str(seam), group, collars, spacing, influence_m,
                               domain=side, method=method, rival_points=rival)
            if model is not None:
                models[f"{seam}@{side}"] = model
            handled.add(str(seam))

    for seam, group in intersections.groupby("seam"):
        if str(seam) in handled:
            continue
        model = build_seam(str(seam), group, collars, spacing, influence_m,
                           method=method)
        if model is not None:
            models[str(seam)] = model

    log.info(f"model dibangun untuk {len(models)} seam/domain "
             f"(radius pengaruh {influence_m:.0f} m)")
    return models


def summary(models: dict[str, SeamModel]) -> pd.DataFrame:
    rows = []
    for key, m in models.items():
        thickness = m.isopach.z
        valid = np.isfinite(thickness)
        rows.append({
            "seam/domain": key, "n_lubang": m.n_holes,
            "sel_valid": int(valid.sum()),
            "sel_interpolasi": int((m.support == SUPPORT_INTERPOLATED).sum()),
            "sel_ekstrapolasi": int((m.support == SUPPORT_EXTRAPOLATED).sum()),
            "isopach_negatif_dijepit": m.negative_thickness_cells,
            "penopang_median_di_hull": int(np.median(
                m.n_support[m.support == SUPPORT_INTERPOLATED]))
            if (m.support == SUPPORT_INTERPOLATED).any() else 0,
            "tebal_min": round(float(np.nanmin(thickness)), 2) if valid.any() else None,
            "tebal_maks": round(float(np.nanmax(thickness)), 2) if valid.any() else None,
        })
    return pd.DataFrame(rows)
