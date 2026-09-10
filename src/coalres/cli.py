"""Antarmuka baris perintah.

    coalres audit --config config.yaml
    coalres run   --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import render_report, run_audit
from .audit.render import write_markdown
from .config import Config, file_digest
from .errors import CoalResError
from .io.excel import load_workbook, normalise_hole_id
from .io.las import load_las
from .io.dxf import load_topography
from .io.quality_table import load_quality_table
from .logging_setup import get_logger, setup

log = get_logger("cli")


def _gather(cfg: Config):
    workbook_dir = cfg.paths.workbook_dir
    if not workbook_dir.exists():
        raise CoalResError(f"paths.workbook_dir tidak ada: {workbook_dir}")
    paths = sorted(p for p in workbook_dir.glob("*.xls*") if not p.name.startswith("~$"))
    if not paths:
        raise CoalResError(f"tidak ada workbook Excel di {workbook_dir}")

    workbooks = []
    digests: dict[str, str] = {}
    for path in paths:
        workbooks.append(load_workbook(path))
        digests[path.name] = file_digest(path)

    las_files = {}
    if cfg.paths.las_dir and cfg.paths.las_dir.exists():
        for path in sorted(cfg.paths.las_dir.glob("*.[Ll][Aa][Ss]")):
            las = load_las(path)
            key = normalise_hole_id(las.well_name or path.stem)
            las_files[key] = las
            digests[path.name] = file_digest(path)

    quality = None
    if cfg.paths.quality_table:
        quality = load_quality_table(cfg.paths.quality_table)
        digests[cfg.paths.quality_table.name] = file_digest(cfg.paths.quality_table)

    topo = None
    if cfg.paths.topography_dxf and cfg.paths.topography_dxf.exists():
        topo = load_topography(cfg.paths.topography_dxf)
        digests[cfg.paths.topography_dxf.name] = file_digest(cfg.paths.topography_dxf)

    return workbooks, las_files, quality, topo, digests


def cmd_audit(args) -> int:
    cfg = Config.load(args.config)
    workbooks, las_files, quality, topo, digests = _gather(cfg)
    report = run_audit(workbooks, las_files, quality, topo, cfg)

    text = render_report(report, cfg, workbooks, las_files)
    print(text)

    out_dir = cfg.paths.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_phase0.txt").write_text(text)
    write_markdown(report, cfg, workbooks, las_files, out_dir / "audit_phase0.md")
    (out_dir / "input_digests.txt").write_text(
        "\n".join(f"{name}  sha256:{d}" for name, d in sorted(digests.items()))
    )
    log.info(f"laporan audit ditulis ke {out_dir}")
    return 0 if report.passed else 2


def cmd_run(args) -> int:
    cfg = Config.load(args.config)
    workbooks, las_files, quality, topo, _ = _gather(cfg)
    report = run_audit(workbooks, las_files, quality, topo, cfg)
    if not report.passed:
        print(render_report(report, cfg, workbooks, las_files))
        log.critical(
            f"Phase 0 tidak lulus: {len(report.stops)} gerbang terbuka. "
            "Estimasi tidak dijalankan."
        )
        return 2
    log.critical(
        "Modul estimasi belum dibangun. Sesuai urutan kerja yang disepakati, "
        "Phase 0 diselesaikan dan dikonfirmasi lebih dulu."
    )
    return 3


def main(argv: list[str] | None = None) -> int:
    setup()
    parser = argparse.ArgumentParser(
        prog="coalres",
        description="Estimasi sumberdaya batubara poligonal/sirkular (SNI 5015:2019).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("audit", cmd_audit, "Jalankan audit data Phase 0 (gerbang keras)."),
        ("run", cmd_run, "Jalankan estimasi (menuntut Phase 0 lulus)."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, type=Path)
        p.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except CoalResError as exc:
        log.critical(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
