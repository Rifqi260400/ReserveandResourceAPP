"""Antarmuka baris perintah."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="coalres",
        description="Estimasi sumberdaya batubara dari Excel lubang bor + DXF topografi.",
    )
    parser.add_argument("--holes", required=True, help="Excel berisi collar + interval seam + kualitas")
    parser.add_argument("--topo", required=True, help="DXF topografi")
    parser.add_argument("--config", default=None, help="YAML konfigurasi (menimpa default)")
    parser.add_argument("--out", default=None, help="Direktori keluaran")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--allow-errors", action="store_true",
        help="Lanjutkan meski validasi menemukan ERROR (tidak dianjurkan)",
    )
    args = parser.parse_args(argv)

    from .pipeline import run

    results = run(args.holes, args.topo, args.config, args.out, verbose=not args.quiet)

    errors = results.findings[results.findings["severity"] == "ERROR"]
    pd.set_option("display.width", 160, "display.max_columns", 40)

    print("\n" + "=" * 78)
    print("RINGKASAN SUMBERDAYA (in-situ, basis ar)")
    print("=" * 78)
    if results.totals.empty:
        print("Tidak ada sumberdaya yang lolos cutoff.")
    else:
        cols = [c for c in ("class", "seams", "area_ha", "tonnes", "thickness_mean_m",
                            "ard_insitu", "ash_ar", "cv_ar", "ts_adb") if c in results.totals]
        print(results.totals[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\nCATATAN: angka di atas adalah hasil perhitungan geometri, BUKAN")
    print("pernyataan sumberdaya. Lihat sheet 'Catatan Penting' di laporan Excel.")

    if not errors.empty:
        print(f"\n{len(errors)} ERROR validasi ditemukan - lihat sheet 'Validasi'.")
        if not args.allow_errors:
            print("Perbaiki dulu, atau jalankan ulang dengan --allow-errors.")
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
