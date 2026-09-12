"""Paket serah-terima cadangan: grid, kontur, batas, tabel, manifest.

Dua aturan penamaan yang menentukan bentuk modul ini.

PENANDA _uncut DAN _ltd WAJIB pada setiap nama berkas grid. Grid uncut dan
limited terlihat IDENTIK saat dibuka di perangkat lunak tambang - sumbu sama,
rentang nilai mirip, kontur serupa. Tertukar, hasilnya salah tanpa satu pun
gejala. Penanda di nama berkas adalah satu-satunya pertahanan yang bertahan
setelah berkas keluar dari sini.

ESRI ASCII GRID sebagai format utama, TIDAK PERNAH biner Minescape proprietary.
Format teks dapat dibaca siapa pun sepuluh tahun lagi tanpa lisensi; biner
proprietary hanya dapat dibaca perangkat lunak yang menulisnya.

  uncut   - ketebalan GEOLOGI: seluruh interseksi, tanpa cutoff dan tanpa
            aturan penambangan.
  limited - setelah seluruh batas tahap 9 diterapkan: pelapukan, tebal minimum,
            kedalaman, area terlarang.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from ..model import SeamModel

log = get_logger("export.handover")

NODATA = -9999.0

# Kode permukaan mengikuti konvensi Minex yang terlihat pada data rujukan.
SURFACE_CODES = {"roof": "SR", "floor": "SF", "thickness": "ST"}
MARKERS = {"uncut": "_uncut", "limited": "_ltd"}


def esri_ascii(surface_x: np.ndarray, surface_y: np.ndarray, z: np.ndarray,
               path: Path, nodata: float = NODATA) -> Path:
    """Tulis ESRI ASCII Grid (.asc).

    Barisnya ditulis dari UTARA ke selatan, sesuai ketentuan format. Grid di
    sini disimpan dari selatan ke utara, jadi ia dibalik sekali di sini - bukan
    di pemanggil, supaya tidak ada yang membalik dua kali.
    """
    dx = float(surface_x[1] - surface_x[0])
    dy = float(surface_y[1] - surface_y[0])
    if abs(dx - dy) > 1e-6:
        raise ValueError(f"ESRI ASCII menuntut sel bujur sangkar: {dx} x {dy}")

    body = np.where(np.isfinite(z), z, nodata)
    header = (
        f"ncols {z.shape[1]}\n"
        f"nrows {z.shape[0]}\n"
        f"xllcorner {float(surface_x[0]) - dx / 2:.6f}\n"
        f"yllcorner {float(surface_y[0]) - dy / 2:.6f}\n"
        f"cellsize {dx:.6f}\n"
        f"NODATA_value {nodata:g}\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        for row in body[::-1]:
            fh.write(" ".join(f"{v:.4f}" for v in row))
            fh.write("\n")
    return path


def csv_xyz(surface_x: np.ndarray, surface_y: np.ndarray, z: np.ndarray,
            path: Path) -> Path:
    """Tulis XYZ, melewatkan sel kosong. Sel kosong bukan nol."""
    gx, gy = np.meshgrid(surface_x, surface_y)
    keep = np.isfinite(z)
    pd.DataFrame({"x": gx[keep], "y": gy[keep], "z": z[keep]}).to_csv(
        path, index=False, float_format="%.4f")
    return path


def geotiff(surface_x: np.ndarray, surface_y: np.ndarray, z: np.ndarray,
            path: Path, nodata: float = NODATA) -> Path | None:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception as exc:
        log.warning(f"GeoTIFF dilewati, rasterio tidak tersedia: {exc}")
        return None
    dx = float(surface_x[1] - surface_x[0])
    dy = float(surface_y[1] - surface_y[0])
    transform = from_origin(float(surface_x[0]) - dx / 2,
                            float(surface_y[-1]) + dy / 2, dx, dy)
    body = np.where(np.isfinite(z), z, nodata).astype("float32")[::-1]
    with rasterio.open(path, "w", driver="GTiff", height=body.shape[0],
                       width=body.shape[1], count=1, dtype="float32",
                       transform=transform, nodata=nodata) as dst:
        dst.write(body, 1)
    return path


def _surfaces(model: SeamModel, mask: np.ndarray | None
              ) -> dict[str, np.ndarray]:
    roof, floor = model.roof.z, model.floor.z
    thickness = model.isopach.z
    if mask is not None:
        roof = np.where(mask, roof, np.nan)
        floor = np.where(mask, floor, np.nan)
        thickness = np.where(mask, thickness, np.nan)
    return {"roof": roof, "floor": floor, "thickness": thickness}


def seam_code(seam: str) -> str:
    """Kode seam untuk nama berkas: SG<seam><kode permukaan>."""
    return "".join(ch for ch in str(seam) if ch.isalnum()).upper()


def write_grids(run, models: dict[str, SeamModel],
                limit_masks: dict[str, np.ndarray] | None = None,
                also_geotiff: bool = True, also_csv: bool = True) -> list[Path]:
    """Tulis grid uncut dan limited untuk tiap seam.

    Keduanya ditulis SELALU, bahkan ketika batasnya tidak membuang apa pun.
    Paket yang kadang memuat `limited` dan kadang tidak memaksa penerimanya
    menebak, dan tebakan itulah yang menukar keduanya.
    """
    written: list[Path] = []
    for key, model in models.items():
        mask = (limit_masks or {}).get(key)
        variants = {"uncut": _surfaces(model, None),
                    "limited": _surfaces(model, mask)}
        for variant, surfaces in variants.items():
            folder = "uncut" if variant == "uncut" else "limited"
            for name, z in surfaces.items():
                stem = (f"SG{seam_code(model.seam)}{SURFACE_CODES[name]}"
                        f"{MARKERS[variant]}")
                base = run.path("02_reserve_handover", folder, stem)
                written.append(run.record(
                    esri_ascii(model.roof.x, model.roof.y, z,
                               base.with_suffix(".asc")),
                    f"grid_{variant}_{name}",
                    note=f"seam {model.seam} domain {model.domain}"))
                if also_csv:
                    written.append(run.record(
                        csv_xyz(model.roof.x, model.roof.y, z,
                                base.with_suffix(".csv")),
                        f"grid_{variant}_{name}_xyz"))
                if also_geotiff:
                    tif = geotiff(model.roof.x, model.roof.y, z,
                                  base.with_suffix(".tif"))
                    if tif is not None:
                        written.append(run.record(tif,
                                                  f"grid_{variant}_{name}_tif"))
    log.info(f"grid serah-terima ditulis: {len(written)} berkas")
    return written


def write_contours(run, models: dict[str, SeamModel], interval_m: float = 2.0,
                   limit_masks: dict[str, np.ndarray] | None = None) -> list[Path]:
    """Kontur roof dan floor per seam, sebagai POLYLINE 3D ber-Z.

    Ber-Z, bukan LWPOLYLINE berelevasi tunggal: penerima memuatnya ke model 3D,
    dan kontur datar di elevasi nol tidak berguna di sana.
    """
    try:
        import ezdxf
    except Exception as exc:
        log.warning(f"DXF dilewati, ezdxf tidak tersedia: {exc}")
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    for key, model in models.items():
        mask = (limit_masks or {}).get(key)
        for name, surface in (("Roof", model.roof), ("Floor", model.floor)):
            z = surface.z if mask is None else np.where(mask, surface.z, np.nan)
            finite = z[np.isfinite(z)]
            if finite.size < 4:
                continue
            low = np.floor(finite.min() / interval_m) * interval_m
            high = np.ceil(finite.max() / interval_m) * interval_m
            levels = np.arange(low, high + interval_m, interval_m)
            if len(levels) < 2:
                continue

            figure = plt.figure()
            try:
                gx, gy = np.meshgrid(surface.x, surface.y)
                contours = plt.contour(gx, gy, z, levels=levels)
                document = ezdxf.new(setup=True)
                space = document.modelspace()
                layer = f"Seam {model.seam} {name}"
                document.layers.add(layer)
                count = 0
                for level, path_collection in zip(
                        contours.levels, contours.allsegs):
                    for segment in path_collection:
                        if len(segment) < 2:
                            continue
                        space.add_polyline3d(
                            [(float(x), float(y), float(level))
                             for x, y in segment],
                            dxfattribs={"layer": layer})
                        count += 1
            finally:
                plt.close(figure)

            if count == 0:
                continue
            target = run.path("02_reserve_handover", "kontur",
                              f"Seam_{seam_code(model.seam)}_{name}.dxf")
            document.saveas(target)
            written.append(run.record(
                target, f"kontur_{name.lower()}",
                note=f"{count} POLYLINE 3D, interval {interval_m} m"))
    log.info(f"kontur DXF ditulis: {len(written)} berkas")
    return written


def write_tables(run, estimate_report, label: str) -> list[Path]:
    """Tabel sumberdaya: Excel dan CSV berdampingan."""
    frame = estimate_report.table(label)
    written = []
    csv_path = run.path("02_reserve_handover", "tabel",
                        "ringkasan_sumberdaya.csv")
    frame.to_csv(csv_path, index=False)
    written.append(run.record(csv_path, "tabel_sumberdaya_csv"))
    try:
        xlsx = run.path("02_reserve_handover", "tabel",
                        "ringkasan_sumberdaya.xlsx")
        frame.to_excel(xlsx, index=False)
        written.append(run.record(xlsx, "tabel_sumberdaya_xlsx"))
    except Exception as exc:
        log.warning(f"Excel dilewati: {exc}")
    return written


def verify_markers(run) -> list[str]:
    """Swauji: setiap grid WAJIB membawa penanda _uncut atau _ltd.

    Dijalankan atas manifest, bukan atas niat penulisnya - berkas yang lolos
    tanpa penanda adalah berkas yang akan tertukar di tangan penerima.
    """
    problems = []
    for entry in run.files:
        if not entry["kind"].startswith("grid_"):
            continue
        name = Path(entry["path"]).name
        if not any(marker in name for marker in MARKERS.values()):
            problems.append(
                f"{entry['path']}: grid tanpa penanda _uncut atau _ltd. Grid "
                "uncut dan limited terlihat identik saat dibuka; tanpa penanda, "
                "tertukarnya tidak bergejala.")
    return problems
