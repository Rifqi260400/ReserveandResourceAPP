"""Konversi basis kualitas, densitas in-situ, dan compositing.

Basis yang dipakai:
    ar   as received     (moisture = TM, kondisi in-situ)
    adb  air dried basis (moisture = IM, kondisi lab)
    db   dry basis       (moisture = 0)
    daf  dry ash free    (moisture = 0, ash = 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Atribut yang boleh dirata-rata linear dengan pembobotan massa.
# tm_ar ikut di sini: ia fraksi massa in-situ, dan dibutuhkan di hilir untuk
# menurunkan CV(ar) dan Ash(ar) dari grid basis adb.
ADDITIVE = ("ash_adb", "vm_adb", "fc_adb", "ts_adb", "cv_adb", "im_adb", "tm_ar")

# Atribut yang TIDAK additive: merata-ratakannya secara aritmetik salah secara
# fisik. HGI dan ash fusion temperature termasuk di sini.
NON_ADDITIVE = ("hgi", "aft", "aft_id", "aft_st", "aft_ht", "aft_ft")


def adb_to_ar(value, tm_ar, im_adb):
    """Konversi nilai basis air-dried ke as-received."""
    return np.asarray(value, float) * (100.0 - np.asarray(tm_ar, float)) / (
        100.0 - np.asarray(im_adb, float)
    )


def adb_to_db(value, im_adb):
    """Konversi nilai basis air-dried ke dry basis."""
    return np.asarray(value, float) * 100.0 / (100.0 - np.asarray(im_adb, float))


def adb_to_daf(value, im_adb, ash_adb):
    """Konversi nilai basis air-dried ke dry-ash-free."""
    denom = 100.0 - np.asarray(im_adb, float) - np.asarray(ash_adb, float)
    return np.where(denom > 0, np.asarray(value, float) * 100.0 / denom, np.nan)


def preston_sanders_insitu_ard(ard_adb, tm_ar, im_adb):
    """ARD air-dried -> ARD in-situ (Preston & Sanders, 1993).

        RD_is = K * RD_ad / (1 + RD_ad * (K - 1)),  K = (100-IM)/(100-TM)

    Model fisiknya: volume matriks kering tetap, dan air tambahan antara
    kondisi air-dried dan in-situ menempati volumenya sendiri (RD air = 1).

    ARAH EFEKNYA PENTING. Untuk batubara peringkat rendah dengan TM jauh di
    atas IM (mis. TM 42%, IM 18%), air yang ditambahkan menurunkan densitas
    curah ke arah 1,0. RD in-situ menjadi LEBIH RENDAH dari RD lab, sehingga
    memakai RD lab apa adanya MELEBIHKAN tonase - untuk batubara Sumatera
    Selatan tipikal, sekitar 10%.

    Hanya berlaku untuk *apparent* relative density (ARD, ASTM D167 - diukur
    pada bongkah utuh sehingga pori ikut terhitung). True/real relative
    density dari piknometer pada sampel digerus TIDAK boleh dipakai di sini
    maupun untuk tonase.
    """
    ard_adb = np.asarray(ard_adb, float)
    tm_ar = np.asarray(tm_ar, float)
    im_adb = np.asarray(im_adb, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        k = (100.0 - im_adb) / (100.0 - tm_ar)
        out = k * ard_adb / (1.0 + ard_adb * (k - 1.0))
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)


def dry_matter_ard(ard_adb, im_adb):
    """ARD matriks kering, tersirat dari ARD air-dried dan IM.

    Berguna sebagai uji kewajaran: untuk batubara dengan ash 5-15% angkanya
    biasanya 1,40-1,55. Hasil di luar itu menandakan ARD yang dilaporkan
    kemungkinan bukan apparent density, atau basisnya bukan air-dried.
    """
    ard_adb = np.asarray(ard_adb, float)
    im_adb = np.asarray(im_adb, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (100.0 - im_adb) / (100.0 / ard_adb - im_adb)
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)


def add_derived_quality(seam: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tambahkan kolom turunan: ketebalan, ARD in-situ, kualitas basis ar."""
    df = seam.copy()
    df["thickness_vertical"] = df["depth_to"] - df["depth_from"]

    method = cfg["density.insitu_method"]
    if method == "preston_sanders":
        df["ard_insitu"] = preston_sanders_insitu_ard(
            df.get("rd_adb"), df.get("tm_ar"), df.get("im_adb")
        )
        # Lubang tanpa data moisture tidak bisa dikonversi; jangan diam-diam
        # memakai nilai lab, karena itu justru bias yang mau dihindari.
    elif method == "fixed":
        df["ard_insitu"] = float(cfg["density.fixed_insitu_ard"])
    elif method == "none":
        df["ard_insitu"] = df.get("rd_adb")
    else:  # pragma: no cover - sudah divalidasi di config
        raise ValueError(method)

    if {"rd_adb", "im_adb"} <= set(df.columns):
        df["ard_dry_matter"] = dry_matter_ard(df["rd_adb"], df["im_adb"])

    if {"tm_ar", "im_adb"} <= set(df.columns):
        for col, out in (("ash_adb", "ash_ar"), ("ts_adb", "ts_ar"), ("cv_adb", "cv_adb_to_ar")):
            if col in df.columns:
                df[out] = adb_to_ar(df[col], df["tm_ar"], df["im_adb"])
        if "cv_adb" in df.columns and "ash_adb" in df.columns:
            df["cv_daf_calc"] = adb_to_daf(df["cv_adb"], df["im_adb"], df["ash_adb"])

    return df


def composite_seam(
    plies: pd.DataFrame, attributes: list[str] | None = None
) -> pd.Series:
    """Gabungkan beberapa ply menjadi satu nilai per seam.

    Pembobotan memakai MASSA (tebal x densitas), bukan tebal saja. Pembobotan
    tebal saja bias untuk seam yang ash/densitas antar-ply-nya berbeda jauh.
    """
    if attributes is None:
        attributes = [c for c in ADDITIVE if c in plies.columns]

    thickness = plies["thickness_vertical"].to_numpy(float)
    density = plies["ard_insitu"].to_numpy(float)
    density = np.where(np.isfinite(density), density, 1.0)
    weight = thickness * density

    out: dict[str, float] = {
        "thickness_vertical": float(np.nansum(thickness)),
        "mass_weight": float(np.nansum(weight)),
    }
    for attr in attributes:
        values = plies[attr].to_numpy(float)
        ok = np.isfinite(values) & np.isfinite(weight)
        out[attr] = float(np.sum(values[ok] * weight[ok]) / np.sum(weight[ok])) if ok.any() else np.nan

    ard = plies["ard_insitu"].to_numpy(float)
    ok = np.isfinite(ard) & np.isfinite(thickness)
    out["ard_insitu"] = (
        float(np.sum(ard[ok] * thickness[ok]) / np.sum(thickness[ok])) if ok.any() else np.nan
    )

    for attr in NON_ADDITIVE:
        if attr in plies.columns:
            # Disimpan sebagai rata-rata sederhana DAN ditandai, karena
            # atribut ini tidak berperilaku linear.
            out[f"{attr}_mean_nonadditive"] = float(plies[attr].mean(skipna=True))
    return pd.Series(out)


def recompute_cv_ar_from_grids(cv_adb_grid, tm_grid, im_grid):
    """Hitung ulang CV(ar) dari grid CV(adb), TM, dan IM.

    Menginterpolasi CV(ar) secara langsung akan mencampur variasi kualitas
    dengan variasi moisture. Yang benar: interpolasi pada basis kering/adb,
    lalu turunkan CV(ar) dari grid hasilnya.
    """
    return adb_to_ar(cv_adb_grid, tm_grid, im_grid)
