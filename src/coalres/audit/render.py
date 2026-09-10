"""Render laporan audit Phase 0 sebagai teks dan Markdown."""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config
from .checks import AuditReport, Severity

WIDTH = 92
RULE = "=" * WIDTH
THIN = "-" * WIDTH


def _table(frame: pd.DataFrame | None, max_rows: int = 25) -> str:
    if frame is None or frame.empty:
        return "    (kosong)"
    shown = frame.head(max_rows)
    text = shown.to_string(index=False, float_format=lambda v: f"{v:,.3f}", max_colwidth=34)
    body = "\n".join("    " + line for line in text.splitlines())
    if len(frame) > max_rows:
        body += f"\n    ... {len(frame) - max_rows} baris lagi"
    return body


def _wrap(text: str, indent: str = "         ") -> str:
    lines = textwrap.wrap(text, width=WIDTH - len(indent))
    return ("\n" + indent).join(lines) if lines else ""


def render_report(report: AuditReport, cfg: Config, workbooks, las_files) -> str:
    out: list[str] = []
    add = out.append

    label = report.context.get("resource_label", "?")
    add(RULE)
    add("  PHASE 0 - AUDIT DATA".ljust(WIDTH - 20) + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"))
    add(f"  Estimasi ini akan menghasilkan: {label.upper()}")
    add(RULE)
    add("")
    add(f"  Workbook   : {len(workbooks)}")
    add(f"  LAS        : {len(las_files)}")
    add(f"  Metode     : {cfg.estimation_method}")
    add(f"  Kondisi geo: {cfg.geological_condition}")
    add("")

    # ---- 1. Peta header ---------------------------------------------------- #
    add(THIN)
    add("  1. PETA HEADER TERATASI  (wajib dikonfirmasi manusia)")
    add(THIN)
    add(_table(report.tables.get("header_maps")))
    add("")
    for wb in workbooks:
        add(f"    {wb.path.name}  ->  hole '{wb.hole_id}'")
        for name, table in wb.sheets.items():
            add(table.header.describe())
        add(f"      pustaka litologi: {len(wb.lithology_library)} kode dibaca dari 'Library SLL'")
        add("")
        recon = wb.sheets.get("SLL_Reconciled")
        if recon is not None:
            add("      sampel SLL_Reconciled:")
            add(_table(recon.frame.head(5), 5))
            add("")

    # ---- 2..11 ------------------------------------------------------------- #
    sections = [
        ("2. KONFLIK KOORDINAT", "coordinate_conflict", "02_"),
        ("3. BASIS KEDALAMAN", "depth_basis_offset", "03_"),
        ("4. LITOLOGI vs SAMPLING", "lithology_vs_sampling", "04_"),
        ("5. CAKUPAN KUALITAS", "quality_coverage", "05_"),
        ("6. BASIS RD", None, "06_"),
        ("7. OPEN HOLE vs CORED", "core_coverage", "07_"),
        ("8. INTEGRITAS DATA, LAS, TOPOGRAFI", None, "08_"),
        ("9. JUMLAH LUBANG PER SEAM", "seam_hole_counts", "09_"),
        ("10. KONDISI GEOLOGI", None, "10_"),
        ("11. BATASAN PROSPEK EKONOMI (RPEEE)", None, "11_"),
    ]
    for title, table_key, prefix in sections:
        add(THIN)
        add(f"  {title}")
        add(THIN)
        if table_key:
            add(_table(report.tables.get(table_key)))
            add("")
        for f in report.findings:
            if not f.check.startswith(prefix):
                continue
            where = ""
            if f.hole_id:
                where = f" [{f.hole_id}" + (f"/{f.seam}]" if f.seam else "]")
            elif f.seam:
                where = f" [{f.seam}]"
            add(f"   {str(f.severity):>4}{where} {_wrap(f.message)}")
            if f.remedy:
                add(f"         -> {_wrap(f.remedy, '            ')}")
        add("")

    justification = report.context.get("geological_condition_justification", "")
    if justification:
        add(THIN)
        add("  JUSTIFIKASI KONDISI GEOLOGI  (verbatim, ikut ke setiap keluaran)")
        add(THIN)
        for line in textwrap.wrap(justification, width=WIDTH - 6):
            add("    " + line)
        add("")

    # ---- Ringkasan gerbang ------------------------------------------------- #
    stops = report.stops
    add(RULE)
    add("  RINGKASAN GERBANG")
    add(RULE)
    counts = {s: len(report.of(s)) for s in (Severity.STOP, Severity.WARN, Severity.INFO)}
    add(f"    STOP {counts[Severity.STOP]}    WARN {counts[Severity.WARN]}    "
        f"INFO {counts[Severity.INFO]}")
    add("")
    if stops:
        add("    Phase 0 TIDAK LULUS. Gerbang berikut harus diselesaikan manusia")
        add("    sebelum logika estimasi dijalankan:")
        add("")
        for i, f in enumerate(stops, 1):
            where = f" [{f.hole_id or ''}{'/' + f.seam if f.seam else ''}]".replace(" []", "")
            add(f"    {i:>2}. {f.check}{where}")
            add(f"        {_wrap(f.message, '        ')}")
            if f.remedy:
                add(f"        -> {_wrap(f.remedy, '           ')}")
            add("")
    else:
        add("    Phase 0 LULUS. Konfirmasi peta header, sumber koordinat, basis")
        add("    kedalaman, sumber ketebalan dan basis RD sebelum melanjutkan.")
    add(RULE)
    return "\n".join(out)


def write_markdown(report: AuditReport, cfg: Config, workbooks, las_files, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = report.context.get("resource_label", "?")
    lines = [
        f"# Laporan Audit Phase 0 — {label}",
        "",
        f"Dihasilkan: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- Workbook: {len(workbooks)}",
        f"- LAS: {len(las_files)}",
        f"- Kondisi geologi: `{cfg.geological_condition}`",
        f"- Label keluaran: **{label}**",
        "",
        "## Temuan",
        "",
        report.to_frame().to_markdown(index=False),
        "",
    ]
    for name, frame in report.tables.items():
        lines += [f"## Tabel: {name}", "", frame.to_markdown(index=False), ""]
    path.write_text("\n".join(lines))
    return path
