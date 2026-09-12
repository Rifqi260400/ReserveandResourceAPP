"""Tahap 8a: jarak antar Titik Pengamatan dari variabilitas seam.

Pedoman Praktis KCMI 2017 pasal 4.5.3, dikutip utuh:

    - Penentuan jarak antar PoO dilakukan pada TIAP SEAM mengacu pada
      variabilitas masing-masing seam. Dalam hal ini, CPI sangat disarankan
      untuk menggunakan pendekatan statistik.
    - Bila jumlah datanya memenuhi syarat (minimal 30 data menurut Journel &
      Huijbregts - Mining Geostatistics 1978), maka disarankan juga untuk
      menggunakan geostatistik.
    - Bila jumlah data kurang dari 30, penentuan jarak antar PoO bisa
      menggunakan pendekatan kompleksitas geologi mengacu ke SNI 5015 2019.

Jadi urutannya: variabilitas per seam lebih dulu, tabel kompleksitas SNI
sebagai CADANGAN ketika datanya kurang - bukan sebaliknya.

Yang TIDAK dinyatakan pedoman, dan karena itu tidak diputuskan modul ini:
bagaimana range variogram dipetakan menjadi ketiga radius kelas. Modul ini
melaporkan range-nya; pemetaannya keputusan manusia yang tercatat.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .logging_setup import get_logger

log = get_logger("variography")

# Journel & Huijbregts (1978), dikutip KCMI 4.5.3.
GEOSTATISTICS_MIN_DATA = 30

# Pasangan minimum per lag agar titik variogram bermakna.
MIN_PAIRS_PER_LAG = 10

# Range yang melewati separuh bentang data tidak dapat dipakai. Variogram hanya
# dapat dipercaya sampai sekitar setengah bentang; di luar itu pasangan datanya
# sedikit dan bias. Range yang "besar" di situ hampir selalu berarti variogram
# belum mencapai sill karena ada TREN (mis. seam menebal searah dip), bukan
# karena korelasinya benar-benar sejauh itu. Memakainya sebagai jarak PoO akan
# membenarkan spasi bor yang jauh lebih longgar daripada yang didukung data.
MAX_RELIABLE_RANGE_FRAC = 0.5

Route = Literal["geostatistik", "sni_kompleksitas"]


@dataclass
class Variogram:
    seam: str
    n_data: int
    route: Route
    lags_m: np.ndarray
    gamma: np.ndarray
    pairs: np.ndarray
    nugget: float = float("nan")
    sill: float = float("nan")
    range_m: float = float("nan")
    fit_quality: float = float("nan")      # R^2
    note: str = ""
    data_extent_m: float = float("nan")
    range_usable: bool = False

    @property
    def nugget_ratio(self) -> float:
        """Nugget / sill. Tinggi berarti ragam acak jarak-pendek mendominasi."""
        return self.nugget / self.sill if self.sill > 0 else float("nan")


def _spherical(h: np.ndarray, nugget: float, sill: float, rng: float) -> np.ndarray:
    """Model sferis baku. Datar di `sill` setelah `rng`."""
    h = np.asarray(h, float)
    out = np.full_like(h, nugget + sill)
    inside = h < rng
    ratio = h[inside] / rng
    out[inside] = nugget + sill * (1.5 * ratio - 0.5 * ratio ** 3)
    return out


def experimental(points: np.ndarray, values: np.ndarray,
                 lag_m: float | None = None, n_lags: int = 12
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semivariogram eksperimental omnidirectional.

    gamma(h) = rata-rata setengah selisih kuadrat pasangan berjarak sekitar h.
    """
    n = len(points)
    i, j = np.triu_indices(n, k=1)
    distance = np.hypot(*(points[i] - points[j]).T)
    difference = 0.5 * (values[i] - values[j]) ** 2

    if lag_m is None:
        # Lag sebesar jarak tetangga terdekat median: cukup halus untuk
        # menangkap struktur jarak pendek, cukup lebar untuk berisi pasangan.
        from scipy.spatial import cKDTree
        nn, _ = cKDTree(points).query(points, k=2)
        lag_m = float(np.median(nn[:, 1]))
    edges = np.arange(0, lag_m * (n_lags + 1), lag_m)

    lags, gammas, counts = [], [], []
    for k in range(len(edges) - 1):
        mask = (distance >= edges[k]) & (distance < edges[k + 1])
        if mask.sum() >= MIN_PAIRS_PER_LAG:
            lags.append(float(distance[mask].mean()))
            gammas.append(float(difference[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(lags), np.array(gammas), np.array(counts)


def fit_spherical(lags: np.ndarray, gamma: np.ndarray, pairs: np.ndarray
                  ) -> tuple[float, float, float, float]:
    """Cocokkan model sferis. Kembalikan nugget, sill, range, R^2."""
    from scipy.optimize import curve_fit

    if len(lags) < 3:
        return (float("nan"),) * 4
    sill0 = float(np.max(gamma))
    guess = [max(float(gamma[0]) * 0.5, 1e-9), max(sill0, 1e-9),
             float(np.median(lags))]
    bounds = ([0.0, 1e-12, lags[0] if lags[0] > 0 else 1.0],
              [sill0 * 2 + 1e-9, sill0 * 5 + 1e-9, lags[-1] * 3])
    try:
        popt, _ = curve_fit(_spherical, lags, gamma, p0=guess, bounds=bounds,
                            sigma=1.0 / np.sqrt(pairs), maxfev=20000)
    except Exception as exc:
        log.warning(f"pencocokan variogram gagal: {exc}")
        return (float("nan"),) * 4
    predicted = _spherical(lags, *popt)
    residual = float(np.sum((gamma - predicted) ** 2))
    total = float(np.sum((gamma - gamma.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return float(popt[0]), float(popt[1]), float(popt[2]), r2


def data_extent(points: np.ndarray) -> float:
    """Bentang data terbesar, dipakai sebagai batas keandalan range."""
    if len(points) < 2:
        return float("nan")
    span = points.max(axis=0) - points.min(axis=0)
    return float(np.hypot(*span))


def for_seam(seam: str, points: np.ndarray, values: np.ndarray,
             min_data: int = GEOSTATISTICS_MIN_DATA) -> Variogram:
    """Terapkan KCMI 4.5.3 pada satu seam."""
    n = len(points)
    if n < min_data:
        return Variogram(
            seam=seam, n_data=n, route="sni_kompleksitas",
            lags_m=np.array([]), gamma=np.array([]), pairs=np.array([]),
            note=(f"{n} data, di bawah {min_data} (Journel & Huijbregts 1978 "
                  "sebagaimana dikutip KCMI 4.5.3). Jalur geostatistik tidak "
                  "dipakai; jarak PoO memakai pendekatan kompleksitas geologi "
                  "SNI 5015:2019."))

    lags, gamma, pairs = experimental(points, values)
    nugget, sill, rng, r2 = fit_spherical(lags, gamma, pairs)
    extent = data_extent(points)
    limit = MAX_RELIABLE_RANGE_FRAC * extent
    note = f"{n} data, memenuhi syarat geostatistik (>= {min_data})."
    usable = True

    if not np.isfinite(rng):
        note += " Pencocokan sferis GAGAL; range tidak dapat dipakai."
        usable = False
    else:
        if np.isfinite(limit) and rng > limit:
            note += (f" RANGE TIDAK DAPAT DIPAKAI: {rng:.0f} m melewati separuh "
                     f"bentang data ({limit:.0f} m dari bentang {extent:.0f} m). "
                     "Variogram hanya dapat dipercaya sampai sekitar separuh "
                     "bentang; range sebesar ini hampir selalu berarti sill belum "
                     "tercapai karena ada TREN, bukan korelasi sejauh itu. "
                     "Hilangkan trennya lebih dulu, atau pakai jalur kompleksitas "
                     "SNI.")
            usable = False
        if r2 < 0.5:
            note += (f" Pencocokan lemah (R2 = {r2:.2f}); range tidak dapat "
                     "diandalkan tanpa pemeriksaan manual.")
            usable = False

    return Variogram(seam=seam, n_data=n, route="geostatistik", lags_m=lags,
                     gamma=gamma, pairs=pairs, nugget=nugget, sill=sill,
                     range_m=rng, fit_quality=r2, note=note,
                     data_extent_m=extent, range_usable=usable)


def analyse(intersections: pd.DataFrame, collars: pd.DataFrame,
            attribute: str = "coal_thickness_m",
            min_data: int = GEOSTATISTICS_MIN_DATA) -> dict[str, Variogram]:
    """Variogram tiap seam atas atribut yang menentukan variabilitasnya."""
    out: dict[str, Variogram] = {}
    for seam, group in intersections.groupby("seam"):
        rows = []
        for _, row in group.iterrows():
            hole = row["hole_id"]
            if hole not in collars.index:
                continue
            collar = collars.loc[hole]
            value = float(row.get(attribute, float("nan")))
            if np.isfinite(value):
                rows.append((float(collar["east"]), float(collar["north"]), value))
        if len(rows) < 3:
            continue
        arr = np.asarray(rows, float)
        out[str(seam)] = for_seam(str(seam), arr[:, :2], arr[:, 2], min_data)
    return out


def summary(variograms: dict[str, Variogram]) -> pd.DataFrame:
    return pd.DataFrame([
        {"seam": v.seam, "n_data": v.n_data, "jalur_kcmi_4.5.3": v.route,
         "nugget": round(v.nugget, 3) if np.isfinite(v.nugget) else None,
         "sill": round(v.sill, 3) if np.isfinite(v.sill) else None,
         "nugget/sill": round(v.nugget_ratio, 3) if np.isfinite(v.nugget_ratio) else None,
         "range_m": round(v.range_m, 0) if np.isfinite(v.range_m) else None,
         "R2": round(v.fit_quality, 3) if np.isfinite(v.fit_quality) else None,
         "bentang_data_m": round(v.data_extent_m, 0) if np.isfinite(v.data_extent_m) else None,
         "range_dapat_dipakai": v.range_usable,
         "catatan": v.note}
        for v in variograms.values()
    ])
