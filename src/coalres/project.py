"""Tahap 0: akar run, manifest, dan register asumsi.

Tiap run menulis ke direktorinya sendiri yang bernama waktu, sehingga dua run
tidak pernah saling menimpa dan tiap berkas dapat ditelusuri ke run yang
membuatnya. Manifest mencatat SETIAP berkas beserta jenisnya pada saat
penulisan - bukan ditebak dari namanya kemudian.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import file_digest
from .logging_setup import get_logger

log = get_logger("project")

SUBDIRS = ("01_bab5_laporan", "02_reserve_handover", "03_audit_and_provenance")


@dataclass
class Run:
    """Satu run: akar direktori, manifest, dan register asumsi."""

    root: Path
    project: str
    started: datetime
    files: list[dict] = field(default_factory=list)
    assumptions: list[dict] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def record(self, path: Path, kind: str, note: str = "") -> Path:
        """Catat berkas yang BARU ditulis, beserta jenisnya."""
        self.files.append({
            "path": str(path.relative_to(self.root)), "kind": kind,
            "bytes": path.stat().st_size if path.exists() else 0, "note": note,
        })
        return path

    def assume(self, topic: str, statement: str, basis: str) -> None:
        """Daftarkan asumsi pemodelan. Ia wajib muncul di keluaran, bukan hilang."""
        self.assumptions.append({"topic": topic, "statement": statement,
                                 "basis": basis})

    def digest_input(self, path: Path) -> None:
        path = Path(path)
        if path.exists() and path.is_file():
            self.inputs[str(path)] = file_digest(path)

    def write_manifest(self) -> Path:
        target = self.path("02_reserve_handover", "manifest.json")
        payload = {
            "project": self.project,
            "run_started": self.started.isoformat(timespec="seconds"),
            "run_root": self.root.name,
            "n_files": len(self.files),
            "files": self.files,
            "assumptions": self.assumptions,
            "input_digests": self.inputs,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return target

    def write_assumptions(self) -> Path:
        """Dokumen asumsi - salah satu dari TIGA tempat yang wajib memuatnya."""
        lines = [f"# Asumsi Pemodelan - {self.project}", "",
                 f"Run: {self.root.name}", ""]
        for item in self.assumptions:
            lines += [f"## {item['topic']}", "", item["statement"], "",
                      f"**Dasar:** {item['basis']}", ""]
        target = self.path("03_audit_and_provenance", "asumsi.md")
        target.write_text("\n".join(lines))
        return self.record(target, "asumsi")


def start(output_dir: Path | str, project: str,
          when: datetime | None = None) -> Run:
    when = when or datetime.now()
    root = Path(output_dir) / f"run_{project}_{when:%Y%m%d_%H%M}"
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    log.info(f"akar run: {root}")
    return Run(root=root, project=project, started=when)
