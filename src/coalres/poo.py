"""Titik Pengamatan (Point of Observation) menurut Pedoman Praktis KCMI 2017.

Rujukan pasal, dikutip agar dapat diperiksa balik:

  4.5    PoO adalah titik pada lapisan pembawa batubara yang memberikan
         informasi mengenai KEBERADAAN DAN KUALITAS batubara melalui
         pengamatan, pengukuran dan/atau analisis.

  4.5.2  Kriteria PoO - KETIGANYA wajib:
         a. koordinat collar diambil dengan metoda survey valid (total station
            atau GPS geodetik);
         b. posisi roof dan floor dapat diidentifikasi sehingga tebalnya
            meyakinkan; untuk lubang SELAIN full coring, logging geofisika
            WAJIB;
         c. dilakukan pengujian kualitas laboratorium dari sampel perlapisan
            yang akan dilaporkan Sumber daya-nya, dengan coal recovery yang
            memenuhi aspek keterwakilan sampel.

  4.5.4  Jumlah minimum PoO yang dapat MEMBENTUK area Sumber daya adalah
         3 titik: 2 searah crop line dan 1 ke arah down dip.

  4.5.5  Spotted dog: klasifikasi tidak tepat karena ada titik TERISOLASI, atau
         titik-titik yang terhubung namun TIDAK menunjukkan kemenerusan pada
         DUA ARAH.

Pasal 4.5.2 butir (a) dan (b) tidak dapat dijawab dari berkas flat: tidak ada
kolom metoda survey maupun jenis lubang. Keduanya WAJIB dinyatakan di
konfigurasi, dan selama belum dinyatakan tidak ada lubang yang boleh lolos -
menganggapnya terpenuhi berarti mengklaim kualitas data yang belum diperiksa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logging_setup import get_logger

log = get_logger("poo")

MINIMUM_POO_PER_AREA = 3          # 4.5.4

# KCMI 4.5.5 menyebut "kemenerusan pada dua arah" tanpa memberi angka. Kedua
# ambang di bawah adalah OPERASIONALISASI KAMI, bukan isi pedoman, dan dicetak
# bersama tiap temuan supaya dapat dibantah CPI.
#
#   COLLINEARITY_RATIO : rasio rentang arah kedua terhadap arah pertama. Di
#                        bawah ini, titik dianggap praktis segaris - satu arah.
#   MIN_SPREAD_FRAC    : rentang arah kedua minimum, sebagai pecahan radius
#                        kelasnya. Menangkap sebaran yang tidak segaris tetapi
#                        terlalu sempit untuk menopang lebar poligonnya.
COLLINEARITY_RATIO = 0.15
MIN_SPREAD_FRAC = 0.25

FAIL_SURVEY = "metoda survey collar tidak valid atau belum dinyatakan"
FAIL_LOGGING = "bukan full coring dan tanpa logging geofisika"
FAIL_RECOVERY = "coal recovery di bawah ambang keterwakilan"


@dataclass
class CriteriaResult:
    frame: pd.DataFrame                       # hole_id, seam, lolos, alasan
    undeclared: list[str] = field(default_factory=list)

    @property
    def passing(self) -> pd.DataFrame:
        return self.frame[self.frame["lolos"]]


def check_criteria(candidates: pd.DataFrame, cfg) -> CriteriaResult:
    """Terapkan KCMI 4.5.2 butir a, b, c di atas calon titik pengamatan.

    `candidates` adalah keluaran observation.build(): butir (c) sebagian sudah
    diperiksa di sana lewat cakupan kualitas. Di sini ditambahkan butir (a) dan
    (b), plus coal recovery.
    """
    spec = cfg.poo
    frame = candidates.copy()
    undeclared: list[str] = []

    if spec.survey_method is None:
        undeclared.append("poo.survey_method")
    if spec.hole_types is None and spec.all_holes_full_cored is None:
        undeclared.append("poo.all_holes_full_cored atau poo.hole_types")

    reasons: list[str] = []
    passes: list[bool] = []
    for _, row in frame.iterrows():
        hole = str(row["hole_id"])
        ok, why = bool(row.get("qualifies", False)), str(row.get("reason", ""))

        if ok and spec.survey_method not in ("total_station", "gps_geodetik"):
            ok, why = False, FAIL_SURVEY

        if ok:
            full_cored = (spec.all_holes_full_cored
                          if spec.hole_types is None
                          else spec.hole_types.get(hole) == "full_coring")
            logged = (hole in (spec.geophysically_logged_holes or [])
                      if spec.all_holes_geophysically_logged is None
                      else spec.all_holes_geophysically_logged)
            if not full_cored and not logged:
                ok, why = False, FAIL_LOGGING

        if ok and spec.min_coal_recovery_pct is not None:
            recovery = float(row.get("coal_recovery_pct", float("nan")))
            if np.isfinite(recovery) and recovery < spec.min_coal_recovery_pct:
                ok, why = False, FAIL_RECOVERY

        passes.append(ok)
        reasons.append("" if ok else why)

    frame["lolos"] = passes
    frame["alasan_kcmi"] = reasons
    return CriteriaResult(frame=frame, undeclared=undeclared)


def _plan_dip_direction(intersections: pd.DataFrame, collars: pd.DataFrame,
                        seam: str) -> np.ndarray | None:
    """Arah down dip pada bidang datar, sebagai vektor satuan.

    Dicocokkan dari bidang yang melalui RL floor. Gradien bidang itu menunjuk
    ke arah naik; arah down dip adalah kebalikannya.
    """
    group = intersections[intersections["seam"] == seam]
    rows = []
    for _, row in group.iterrows():
        hole = row["hole_id"]
        if hole not in collars.index:
            continue
        collar = collars.loc[hole]
        rows.append((float(collar["east"]), float(collar["north"]),
                     float(collar["rl"]) - float(row["floor_m"])))
    if len(rows) < 3:
        return None
    arr = np.asarray(rows, float)
    design = np.column_stack([arr[:, 0], arr[:, 1], np.ones(len(arr))])
    coeff, *_ = np.linalg.lstsq(design, arr[:, 2], rcond=None)
    gradient = np.array([coeff[0], coeff[1]])
    norm = np.linalg.norm(gradient)
    return -gradient / norm if norm > 1e-12 else None


def connected_components(points: np.ndarray, radius: float) -> list[list[int]]:
    """Kelompokkan titik yang cakramnya saling bersinggungan.

    Dua cakram berjari-jari r bersinggungan bila jarak pusatnya <= 2r. Inilah
    yang membuat poligon gabungan menjadi satu bagian yang menyatu.
    """
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(points[i] - points[j])) <= 2.0 * radius:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def two_direction_spread(points: np.ndarray, dip_direction: np.ndarray | None
                         ) -> tuple[float, float, float]:
    """Rentang sebaran pada dua arah, dan rasio arah kedua terhadap pertama.

    Bila arah dip diketahui, keduanya diukur searah dip dan searah jurus -
    persis dua arah yang disebut KCMI 4.5.4. Bila tidak, dipakai sumbu utama
    sebarannya sendiri.
    """
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    centred = points - points.mean(axis=0)
    if dip_direction is not None:
        strike = np.array([-dip_direction[1], dip_direction[0]])
        axes = np.vstack([dip_direction, strike])
    else:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axes = vt[:2] if len(vt) >= 2 else np.vstack([vt[0], [-vt[0][1], vt[0][0]]])
    projected = centred @ axes.T
    spans = projected.max(axis=0) - projected.min(axis=0)
    primary, secondary = float(max(spans)), float(min(spans))
    return primary, secondary, (secondary / primary if primary > 0 else 0.0)


def evaluate_areas(points_frame: pd.DataFrame, intersections: pd.DataFrame,
                   collars: pd.DataFrame, seam: str, radius: float,
                   policy: str = "radius_provides_dip") -> pd.DataFrame:
    """Uji KCMI 4.5.4 dan 4.5.5 pada tiap bagian poligon yang menyatu.

    `policy` menentukan bagaimana "kemenerusan pada dua arah" dibaca:

      radius_provides_dip - titik boleh berjajar searah strike. Jangkauan arah
        dip datang dari radius kelasnya sendiri: cakram menyapu ke segala arah,
        jadi mengklaim 250 m searah strike berarti mengklaim 250 m searah dip
        juga. Yang tetap ditegakkan adalah minimum 3 titik (4.5.4) dan larangan
        titik terisolasi (4.5.5). Rasio sebaran tetap DIUKUR dan dilaporkan,
        hanya tidak lagi menggugurkan.

      require_offset_point - menuntut titik fisik yang bergeser searah dip,
        mengikuti bunyi harfiah 4.5.4: "2 titik ke searah crop line dan 1 titik
        ke arah down dip".
    """
    sub = points_frame[points_frame["seam"] == seam]
    coords = sub[["east", "north"]].to_numpy(float)
    holes = sub["hole_id"].tolist()
    dip = _plan_dip_direction(intersections, collars, seam)

    rows = []
    for index, component in enumerate(connected_components(coords, radius), start=1):
        member = coords[component]
        primary, secondary, ratio = two_direction_spread(member, dip)
        n = len(component)

        failures = []
        if n < MINIMUM_POO_PER_AREA:
            failures.append(
                f"hanya {n} titik pengamatan; KCMI 4.5.4 menuntut minimum "
                f"{MINIMUM_POO_PER_AREA} (2 searah crop line, 1 ke arah down dip)")
        elif policy == "require_offset_point":
            if ratio < COLLINEARITY_RATIO:
                failures.append(
                    f"titik praktis segaris (rentang arah kedua {secondary:.0f} m "
                    f"lawan {primary:.0f} m, rasio {ratio:.3f}); KCMI 4.5.5 menyebut "
                    "titik yang terhubung tanpa kemenerusan DUA ARAH sebagai "
                    "spotted dog")
            elif secondary < MIN_SPREAD_FRAC * radius:
                failures.append(
                    f"rentang arah kedua {secondary:.0f} m kurang dari "
                    f"{MIN_SPREAD_FRAC:.0%} radius kelas ({radius:.0f} m); "
                    "kemenerusan arah kedua terlalu tipis untuk diandalkan")

        rows.append({
            "seam": seam, "bagian": index, "n_titik": n,
            "rentang_arah_1_m": round(primary, 1),
            "rentang_arah_2_m": round(secondary, 1),
            "rasio": round(ratio, 3),
            "arah_dari": "dip/jurus" if dip is not None else "sumbu sebaran",
            "kebijakan": policy,
            "memenuhi_kcmi": not failures,
            "temuan": "; ".join(failures),
            "lubang": ", ".join(sorted(holes[i] for i in component)),
        })
    return pd.DataFrame(rows)


def spotted_dog_report(points_frame: pd.DataFrame, intersections: pd.DataFrame,
                       collars: pd.DataFrame, radii: dict[str, float],
                       policy: str = "radius_provides_dip") -> pd.DataFrame:
    """Uji tiap seam pada radius tiap kelas.

    Diuji per kelas karena spotted dog muncul pada BATAS kelas: sekumpulan titik
    dapat menyatu pada radius Tereka tetapi pecah menjadi pulau-pulau terisolasi
    pada radius Terukur yang lebih kecil - persis yang digambarkan KCMI 4.5.5.
    """
    frames = []
    for seam in sorted(points_frame["seam"].unique()):
        for klass, radius in radii.items():
            frame = evaluate_areas(points_frame, intersections, collars, seam,
                                   radius, policy=policy)
            if not frame.empty:
                frame.insert(1, "kelas", klass)
                frame.insert(2, "radius_m", radius)
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    failing = int((~out["memenuhi_kcmi"]).sum())
    if failing:
        log.warning(f"spotted dog / jumlah PoO: {failing} bagian tidak memenuhi "
                    f"KCMI 4.5.4-4.5.5 dari {len(out)} bagian diuji")
    return out
