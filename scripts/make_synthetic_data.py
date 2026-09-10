"""Bangkitkan dataset sintetis bergaya BGG untuk menguji pipeline end-to-end.

Sengaja disisipi cacat yang nyata terjadi di data lapangan, agar modul validasi
benar-benar teruji: RL collar yang salah, mass balance yang tidak seimbang,
CV(ar) yang tidak konsisten, dan seam yang urutannya terbalik.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import ezdxf
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20210406)

# Kerangka acuan longgar mengikuti Blok Lawai (UTM 48S).
X0, Y0 = 359_000.0, 9_594_000.0

SEAMS = [
    # nama, kedalaman acuan di titik asal (m), tebal rerata (m), sebaran tebal
    ("S11",  46.0,  1.40, 0.25),
    ("S10A", 72.0,  9.90, 1.20),
    ("S10B", 84.0, 11.80, 1.60),
    ("S10C", 99.0,  1.10, 0.20),
]

# Kualitas acuan per seam, mendekati profil batubara peringkat rendah Sumsel.
QUALITY = {
    "S11":  dict(tm=41.0, im=17.0, ash=13.5, ts=0.25, cv_adb=4700),
    "S10A": dict(tm=42.3, im=17.6, ash=10.8, ts=0.20, cv_adb=4815),
    "S10B": dict(tm=43.7, im=18.6, ash=8.1,  ts=0.31, cv_adb=5022),
    "S10C": dict(tm=41.5, im=17.2, ash=15.0, ts=0.28, cv_adb=4550),
}

DIP_DEG = 5.0
DIP_AZIMUTH_DEG = 115.0


def topo_rl(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Permukaan bergelombang landai, RL sekitar 55-80 m."""
    dx, dy = (x - X0) / 1000.0, (y - Y0) / 1000.0
    return (
        66.0
        + 6.0 * np.sin(1.7 * dx) * np.cos(1.3 * dy)
        + 3.5 * np.sin(3.1 * dy)
        - 2.0 * dx
    )




def _seam_geometry(name: str, x: np.ndarray, y: np.ndarray):
    depth0, mean_t, sd_t = next((d, t, s) for n, d, t, s in SEAMS if n == name)
    azimuth = np.radians(DIP_AZIMUTH_DEG)
    down_dip = (x - X0) * np.sin(azimuth) + (y - Y0) * np.cos(azimuth)
    dx, dy = (x - X0) / 1000.0, (y - Y0) / 1000.0

    surface = topo_rl(x, y)
    roof_rl = surface - depth0 - down_dip * np.tan(np.radians(DIP_DEG))
    roof_rl += 2.5 * np.sin(2.2 * dx + 0.8) * np.cos(1.6 * dy)

    thickness = mean_t + sd_t * np.sin(1.9 * dx) * np.cos(2.4 * dy)
    thickness += RNG.normal(0.0, sd_t * 0.25, size=np.shape(x))
    thickness = np.maximum(thickness, 0.05)
    return roof_rl, thickness


def build(n_holes: int, quality_fraction: float, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- collar: kisi ~250 m dengan jitter, meniru pola bor eksplorasi ------- #
    side = int(np.ceil(np.sqrt(n_holes)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    east = X0 + gx.ravel()[:n_holes] * 250.0 + RNG.normal(0, 30, n_holes)
    north = Y0 + gy.ravel()[:n_holes] * 250.0 + RNG.normal(0, 30, n_holes)
    rl = topo_rl(east, north) + RNG.normal(0, 0.15, n_holes)

    hole_ids = [f"DH{i // 10 + 1:02d}_{i % 10 + 1:02d}" for i in range(n_holes)]
    collar = pd.DataFrame(
        {"HOLE_ID": hole_ids, "EASTING": east.round(3), "NORTHING": north.round(3),
         "RL": rl.round(3), "BLOCK": "Lawai 1"}
    )

    # --- interval seam + kualitas ------------------------------------------- #
    has_quality = RNG.random(n_holes) < quality_fraction

    deepest = np.zeros(n_holes)
    for i in range(n_holes):
        for name, *_ in SEAMS:
            roof_i, thick_i = _seam_geometry(name, np.array(east[i]), np.array(north[i]))
            deepest[i] = max(deepest[i], float(rl[i] - float(roof_i)) + float(thick_i))
    total_depth = np.ceil(np.maximum(deepest + 6.0, 30.0) / 5.0) * 5.0
    collar["TOTAL_DEPTH"] = total_depth
    rows = []
    for i, hid in enumerate(hole_ids):
        x, y, collar_rl = east[i], north[i], rl[i]
        for name, *_ in SEAMS:
            roof_rl_i, thickness_i = _seam_geometry(name, np.array(x), np.array(y))
            roof_rl_i, thickness_i = float(roof_rl_i), float(thickness_i)
            depth_from = collar_rl - roof_rl_i
            depth_to = depth_from + thickness_i
            if depth_from < 0 or depth_to > total_depth[i]:
                continue  # seam tersingkap atau di bawah TD

            row = {"HOLE_ID": hid, "SEAM": name,
                   "FROM": round(depth_from, 3), "TO": round(depth_to, 3)}
            if has_quality[i]:
                q = QUALITY[name]
                ash = max(2.0, q["ash"] + RNG.normal(0, 1.2))
                im = q["im"] + RNG.normal(0, 0.6)
                tm = q["tm"] + RNG.normal(0, 1.0)
                cv_adb = q["cv_adb"] - 55.0 * (ash - q["ash"]) + RNG.normal(0, 60)
                vm = 40.5 + RNG.normal(0, 0.8)
                fc = 100.0 - im - ash - vm           # dipaksa seimbang
                ard = 1.28 + 0.0075 * ash + RNG.normal(0, 0.008)
                row.update({
                    "TM": round(tm, 2), "IM": round(im, 2), "ASH": round(ash, 2),
                    "VM": round(vm, 2), "FC": round(fc, 2),
                    "TS": round(max(0.05, q["ts"] + RNG.normal(0, 0.05)), 2),
                    "CV_ADB": round(cv_adb),
                    "CV_AR": round(cv_adb * (100 - tm) / (100 - im)),
                    "RD": round(ard, 3),
                })
            rows.append(row)
    seam = pd.DataFrame(rows)

    # --- cacat yang disengaja ------------------------------------------------ #
    defects: list[str] = []

    # 1. RL collar salah 14,18 m - persis kasus GPS handheld vs total station.
    collar.loc[3, "RL"] = round(collar.loc[3, "RL"] + 14.18, 3)
    defects.append(f"{collar.loc[3, 'HOLE_ID']}: RL collar +14,18 m (GPS vs TS)")

    # 2. Mass balance tidak seimbang.
    idx = seam[seam["ASH"].notna()].index[5]
    seam.loc[idx, "ASH"] = round(seam.loc[idx, "ASH"] + 4.0, 2)
    defects.append(f"{seam.loc[idx, 'HOLE_ID']}/{seam.loc[idx, 'SEAM']}: mass balance +4%")

    # 3. CV(ar) tidak konsisten dengan CV(adb), TM, IM.
    idx = seam[seam["CV_AR"].notna()].index[9]
    seam.loc[idx, "CV_AR"] = round(seam.loc[idx, "CV_AR"] + 420)
    defects.append(f"{seam.loc[idx, 'HOLE_ID']}/{seam.loc[idx, 'SEAM']}: CV(ar) +420 cal/g")

    # 4. Urutan seam terbalik - salah korelasi.
    mask = (seam["HOLE_ID"] == hole_ids[7]) & (seam["SEAM"].isin(["S10A", "S10B"]))
    if mask.sum() == 2:
        a, b = seam[mask].index
        seam.loc[[a, b], "SEAM"] = seam.loc[[b, a], "SEAM"].values
        defects.append(f"{hole_ids[7]}: S10A/S10B tertukar (salah korelasi)")

    # 5. ARD dilaporkan sebagai true density (piknometer), bukan apparent.
    idx = seam[seam["RD"].notna()].index[14]
    seam.loc[idx, "RD"] = 1.52
    defects.append(f"{seam.loc[idx, 'HOLE_ID']}/{seam.loc[idx, 'SEAM']}: RD 1,52 (true density?)")

    holes_path = out_dir / "drillholes.xlsx"
    with pd.ExcelWriter(holes_path, engine="openpyxl") as writer:
        collar.to_excel(writer, sheet_name="Collar", index=False)
        seam.to_excel(writer, sheet_name="Seam", index=False)

    # --- topografi DXF: kontur + spot height -------------------------------- #
    topo_path = out_dir / "topo.dxf"
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    pad = 400.0
    xs = np.linspace(east.min() - pad, east.max() + pad, 90)
    ys = np.linspace(north.min() - pad, north.max() + pad, 90)
    gx2, gy2 = np.meshgrid(xs, ys)
    gz2 = topo_rl(gx2, gy2)
    for xi, yi, zi in zip(gx2.ravel(), gy2.ravel(), gz2.ravel()):
        msp.add_point((xi, yi, zi), dxfattribs={"layer": "TOPO_SPOT"})
    for level in np.arange(np.floor(gz2.min()), np.ceil(gz2.max()) + 1, 2.0):
        # Kontur disederhanakan: potongan sepanjang baris grid pada level itu.
        pts = [(x_, y_, level) for x_, y_, z_ in
               zip(gx2.ravel(), gy2.ravel(), gz2.ravel()) if abs(z_ - level) < 0.15]
        if len(pts) > 3:
            msp.add_polyline3d(pts[:400], dxfattribs={"layer": "TOPO_CONTOUR"})
    doc.saveas(topo_path)

    print(f"Ditulis: {holes_path}")
    print(f"         {collar.shape[0]} lubang, {seam.shape[0]} interval, "
          f"{int(has_quality.sum())} lubang punya kualitas "
          f"({100 * has_quality.mean():.0f}%)")
    print(f"Ditulis: {topo_path}")
    print("Cacat yang disengaja disisipkan:")
    for d in defects:
        print(f"  - {d}")
    return holes_path, topo_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes", type=int, default=100)
    ap.add_argument("--quality-fraction", type=float, default=0.30)
    ap.add_argument("--out", default="sample_data")
    args = ap.parse_args()
    build(args.holes, args.quality_fraction, Path(args.out))
