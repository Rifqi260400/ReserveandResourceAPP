"""Antarmuka web lokal untuk estimasi sumberdaya.

Dijalankan di mesin pengguna dan bekerja pada berkas lokal, bukan layanan
daring: data bor dan kualitas tidak pernah meninggalkan mesin. Karena itu
server hanya mengikat 127.0.0.1 secara bawaan.

Endpoint sengaja tipis - seluruh logika tetap di modul pipeline, sehingga UI
dan CLI menghasilkan angka yang sama persis.
"""
from __future__ import annotations

import io
import shutil
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..audit.checks import Severity
from ..config import Config
from ..errors import CoalResError
from ..logging_setup import get_logger

log = get_logger("webui")
STATIC = Path(__file__).parent / "static"

# Lima kategori ekspor. Anggotanya dibaca dari manifest.json yang ditulis
# pipeline, bukan ditebak dari nama berkas.
EXPORT_KINDS = {
    "resource_table": "Tabel sumberdaya (Excel)",
    "contours_dxf": "Kontur roof & floor (DXF)",
    "uncut_grid": "Uncut grid: roof, floor, thickness",
    "quality_grid": "Quality grid",
    "bow_recap": "Rekap BOW (Excel)",
}


@dataclass
class Session:
    """Status satu sesi. Aplikasi ini bersifat satu-pengguna dan lokal."""

    config_path: Path | None = None
    config: Config | None = None
    results: Any = None
    audit: Any = None
    error: str | None = None
    log: list[str] = field(default_factory=list)


SESSION = Session()


class ConfigPayload(BaseModel):
    path: str


class ConfigPatch(BaseModel):
    path: str
    data: dict


def _frame_to_records(frame: pd.DataFrame | None, limit: int | None = None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    body = frame.head(limit) if limit else frame
    return yaml.safe_load(body.replace({np.nan: None}).to_json(orient="records"))


def _audit_payload(report) -> dict:
    counts = {str(s): len(report.of(s)) for s in (Severity.STOP, Severity.WARN, Severity.INFO)}
    return {
        "passed": report.passed,
        "counts": counts,
        "findings": [
            {"severity": str(f.severity), "check": f.check, "hole_id": f.hole_id,
             "seam": f.seam, "message": f.message, "remedy": f.remedy}
            for f in sorted(report.findings, key=lambda f: -int(f.severity))
        ],
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Estimasi Sumberdaya Batubara", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    # ---- Konfigurasi ------------------------------------------------------- #
    @app.post("/api/config/load")
    def load_config(payload: ConfigPayload) -> dict:
        path = Path(payload.path).expanduser()
        try:
            cfg = Config.load(path)
        except CoalResError as exc:
            raise HTTPException(400, str(exc)) from exc
        SESSION.config_path, SESSION.config = path, cfg
        SESSION.results = SESSION.audit = SESSION.error = None
        return {"path": str(path), "config": yaml.safe_load(cfg.model_dump_json()),
                "label": cfg.rpeee_constraints.resource_label}

    @app.post("/api/config/save")
    def save_config(patch: ConfigPatch) -> dict:
        path = Path(patch.path).expanduser()
        try:
            Config.model_validate(patch.data)
        except Exception as exc:
            raise HTTPException(400, f"konfigurasi tidak valid: {exc}") from exc
        path.write_text(yaml.safe_dump(patch.data, sort_keys=False, allow_unicode=True))
        return load_config(ConfigPayload(path=str(path)))

    @app.get("/api/config")
    def get_config() -> dict:
        if SESSION.config is None:
            raise HTTPException(400, "belum ada konfigurasi yang dimuat")
        return {"path": str(SESSION.config_path),
                "config": yaml.safe_load(SESSION.config.model_dump_json()),
                "label": SESSION.config.rpeee_constraints.resource_label}

    # ---- Audit dan run ----------------------------------------------------- #
    @app.post("/api/audit")
    def run_audit_endpoint() -> dict:
        if SESSION.config_path is None:
            raise HTTPException(400, "muat konfigurasi lebih dulu")
        from ..audit.checks import run_audit
        from ..audit.minex_checks import run_minex_audit
        from ..pipeline import gather, gather_minex

        cfg = SESSION.config
        try:
            if cfg.input_format == "minex_flat":
                dataset, _ = gather_minex(cfg)
                report = run_minex_audit(dataset, cfg)
            else:
                inputs = gather(cfg)
                report = run_audit(inputs.workbooks, inputs.las_files, inputs.quality,
                                   inputs.topo_points, cfg)
        except CoalResError as exc:
            raise HTTPException(400, str(exc)) from exc
        SESSION.audit = report
        return _audit_payload(report)

    @app.post("/api/run")
    def run_endpoint() -> dict:
        if SESSION.config_path is None:
            raise HTTPException(400, "muat konfigurasi lebih dulu")
        from ..pipeline import gather, gather_minex, run, write_outputs

        cfg = SESSION.config
        try:
            results = run(SESSION.config_path, verbose=False)
            inputs = (gather_minex(cfg)[1] if cfg.input_format == "minex_flat"
                      else gather(cfg))
            written = write_outputs(results, inputs, SESSION.config_path)
        except CoalResError as exc:
            SESSION.error = str(exc)
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # pragma: no cover - diteruskan ke UI
            SESSION.error = traceback.format_exc(limit=3)
            raise HTTPException(500, str(exc)) from exc

        SESSION.results = results
        SESSION.audit = results.audit
        return summary_endpoint()

    @app.get("/api/summary")
    def summary_endpoint() -> dict:
        results = SESSION.results
        if results is None:
            raise HTTPException(400, "belum ada hasil; jalankan estimasi lebih dulu")
        cfg = results.config
        totals = results.frames["grand_total"]
        return {
            "label": cfg.rpeee_constraints.resource_label,
            "method": cfg.estimation_method,
            "geological_condition": cfg.geological_condition,
            "radii": {"measured": cfg.radii.measured, "indicated": cfg.radii.indicated,
                      "inferred": cfg.radii.inferred},
            "output_dir": str(cfg.paths.output_dir),
            "by_seam_class": _frame_to_records(results.frames["by_seam_class"]),
            "by_seam": _frame_to_records(results.frames["by_seam"]),
            "total": _frame_to_records(totals),
            "rpeee": _frame_to_records(results.frames["rpeee"]),
            "rd_sensitivity": _frame_to_records(results.frames["rd_sensitivity"]),
            "bow_summary": _frame_to_records(results.bow_summary),
            "seams": [s for s in results.surfaces if s != "topo"],
            "audit": _audit_payload(results.audit),
        }

    # ---- Peta -------------------------------------------------------------- #
    @app.get("/api/map/{seam}")
    def map_endpoint(seam: str, interval: float = 5.0) -> dict:
        results = SESSION.results
        if results is None:
            raise HTTPException(400, "belum ada hasil; jalankan estimasi lebih dulu")
        if seam not in results.surfaces:
            raise HTTPException(404, f"seam '{seam}' tidak ada")

        from ..classify import ResourceClass
        from ..dxfout import contour_segments, subcrop_mask

        surfaces = results.surfaces[seam]
        topo = results.surfaces["topo"]["topo"]
        mask = subcrop_mask(topo, surfaces["roof"])

        layers: dict[str, list] = {}
        for key in ("roof", "floor"):
            segments = contour_segments(surfaces[key], interval, mask=mask)
            layers[key] = [
                {"z": level, "points": [[round(float(x), 2), round(float(y), 2)]
                                        for x, y in path]}
                for level, paths in sorted(segments.items()) for path in paths
            ]

        polygons = []
        for polygon in results.polygons:
            if polygon.seam != seam:
                continue
            for geom in getattr(polygon.geometry, "geoms", [polygon.geometry]):
                if geom.geom_type != "Polygon":
                    continue
                polygons.append({
                    "class": polygon.resource_class.sni_name,
                    "hole_id": polygon.hole_id,
                    "tonnes": round(polygon.tonnes, 1),
                    "points": [[round(x, 2), round(y, 2)] for x, y in geom.exterior.coords],
                })

        subcrop = []
        extent_geom = surfaces.get("subcrop")
        if extent_geom is not None and not extent_geom.is_empty:
            for geom in getattr(extent_geom, "geoms", [extent_geom]):
                if geom.geom_type == "Polygon":
                    subcrop.append([[round(x, 2), round(y, 2)]
                                    for x, y in geom.exterior.coords])

        holes = []
        intercepts = results.frames["intercepts"]
        if not intercepts.empty:
            sub = intercepts[intercepts["seam"] == seam]
            holes = [{"hole_id": r["hole_id"], "x": round(float(r["east"]), 2),
                      "y": round(float(r["north"]), 2),
                      "thickness": round(float(r["coal_thickness_m"]), 2)}
                     for _, r in sub.iterrows()]

        return {"seam": seam, "interval": interval, "contours": layers,
                "polygons": polygons, "subcrop": subcrop, "holes": holes,
                "classes": [c.sni_name for c in ResourceClass]}

    # ---- Ekspor ------------------------------------------------------------ #
    @app.get("/api/exports")
    def list_exports() -> dict:
        if SESSION.results is None:
            raise HTTPException(400, "belum ada hasil; jalankan estimasi lebih dulu")
        out_dir = SESSION.results.config.paths.output_dir
        available = {}
        for kind, label in EXPORT_KINDS.items():
            paths = _export_paths(out_dir, kind)
            available[kind] = {"label": label, "count": len(paths),
                               "bytes": sum(p.stat().st_size for p in paths if p.exists())}
        return {"output_dir": str(out_dir), "kinds": available}

    @app.get("/api/export/{kind}")
    def export_endpoint(kind: str):
        if kind not in EXPORT_KINDS:
            raise HTTPException(404, f"kategori ekspor tidak dikenal: {kind}")
        if SESSION.results is None:
            raise HTTPException(400, "belum ada hasil; jalankan estimasi lebih dulu")

        out_dir = SESSION.results.config.paths.output_dir
        paths = _export_paths(out_dir, kind)
        if not paths:
            raise HTTPException(404, "tidak ada berkas untuk kategori ini")
        if len(paths) == 1 and paths[0].is_file():
            return FileResponse(paths[0], filename=paths[0].name)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=path.relative_to(out_dir))
        buffer.seek(0)
        return StreamingResponse(
            buffer, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{kind}.zip"'},
        )

    @app.exception_handler(CoalResError)
    def coalres_error(request, exc):  # pragma: no cover
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def _manifest(out_dir: Path) -> dict[str, list[str]]:
    import json

    path = out_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _export_paths(out_dir: Path, kind: str) -> list[Path]:
    """Anggota kategori dibaca dari manifest yang ditulis saat penulisan berkas.

    Menebak dari akhiran nama berkas rapuh: kode kualitas dan kode struktur
    bisa bertabrakan begitu konvensi penamaan diubah pengguna.
    """
    entries = _manifest(out_dir).get(kind, [])
    return [out_dir / name for name in entries if (out_dir / name).exists()]


def serve(host: str = "127.0.0.1", port: int = 8000, config: str | None = None) -> None:
    import uvicorn

    if config:
        SESSION.config_path = Path(config).expanduser()
        SESSION.config = Config.load(SESSION.config_path)
        log.info(f"konfigurasi awal dimuat: {SESSION.config_path}")
    log.info(f"UI berjalan di http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
