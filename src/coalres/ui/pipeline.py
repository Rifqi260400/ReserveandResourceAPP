"""Orkestrator 11 tahap, dipakai UI maupun CLI.

Fungsi di sini TIDAK memutuskan apa pun. Ia menjalankan tahap berurutan dan
berhenti di gerbang, mengembalikan apa adanya - termasuk kegagalan. UI yang
memutuskan bagaimana menampilkannya; ia tidak boleh melewati gerbang.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..audit.minex_checks import run_minex_audit
from ..config import Config
from ..io.minex import load_minex
from ..logging_setup import get_logger
from ..seams import build_intersections_from_dataset, to_frame
from ..topo import build_surface
from .. import (complexity, estimate_grid, limits, model, observation, poo,
                project, radius, validate, variography)

log = get_logger("ui.pipeline")


@dataclass
class Stages:
    """Hasil tiap tahap. `stopped_at` menyebut gerbang yang menghentikannya."""

    cfg: Config
    dataset: Any = None
    audit: Any = None
    intersections: pd.DataFrame | None = None
    models: dict | None = None
    topo: Any = None
    validation: Any = None
    variograms: dict | None = None
    observation: Any = None
    assessment: Any = None
    radii: dict | None = None
    limits: Any = None
    masks: dict | None = None
    estimate: Any = None
    poo_criteria: Any = None
    spotted_dog: pd.DataFrame | None = None
    suggestions: list | None = None
    stopped_at: str | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.estimate is not None and self.stopped_at is None


def run(cfg: Config, spacing: float = 25.0,
        complexity_overrides: dict | None = None,
        condition_override: tuple | None = None,
        cross_validation: bool = True) -> Stages:
    """Jalankan tahap 1 sampai 10. Berhenti di gerbang yang gagal.

    TIDAK ADA parameter untuk mematikan gerbang. Saklar semacam itu akan menjadi
    jalan termudah menuju angka yang tidak boleh dipakai - dan jalan termudah
    adalah jalan yang akhirnya ditempuh.
    """
    stages = Stages(cfg=cfg)

    # 1 input
    stages.dataset = load_minex(cfg)
    collars = stages.dataset.collars.set_index("hole_id")

    # 2 audit [GERBANG]
    stages.audit = run_minex_audit(stages.dataset, cfg)
    if not stages.audit.passed:
        stages.stopped_at = "2_audit"
        stages.messages.append(
            f"Tahap 2 audit: {len(stages.audit.stops)} gerbang terbuka.")
        return stages

    # 3 basis data seam
    stages.intersections = to_frame(
        build_intersections_from_dataset(stages.dataset, cfg))

    # 4 pemodelan
    stages.models = model.build(stages.intersections, collars, cfg, spacing=spacing)
    if stages.dataset.topo_points is not None:
        t = stages.dataset.topo_points
        stages.topo = build_surface(t[:, 0], t[:, 1], t[:, 2], spacing=spacing,
                                    name="topo")

    # 5 validasi [GERBANG -> kembali ke 4]
    stages.validation = validate.run(
        stages.models, stages.intersections, collars, cfg,
        quality=stages.dataset.quality, spacing=spacing,
        cross_validation=cross_validation)
    if not stages.validation.passed:
        stages.stopped_at = "5_validasi"
        stages.messages.append(
            f"Tahap 5 validasi: {len(stages.validation.failures)} kegagalan. "
            "Jalurnya kembali ke tahap 4, bukan maju.")
        return stages

    # 6 titik observasi  ||  7 kompleksitas
    stages.observation = observation.build(stages.intersections,
                                           stages.dataset, cfg)
    stages.poo_criteria = poo.check_criteria(stages.observation.frame, cfg)
    stages.variograms = variography.analyse(stages.intersections, collars)
    overrides = complexity_overrides or cfg.complexity_input.as_overrides()
    chosen = condition_override or cfg.complexity_input.as_condition_override()
    try:
        stages.assessment = complexity.assess(
            stages.dataset, cfg, overrides, condition_override=chosen)
    except (complexity.TiedAssessment, ValueError) as exc:
        # Ini GERBANG, bukan kecelakaan: pembobotan menunggu keputusan manusia.
        stages.stopped_at = "7_kompleksitas"
        stages.messages.append(f"Tahap 7 kompleksitas: {exc}")
        stages.suggestions = complexity.suggest(stages.dataset, cfg)
        return stages

    # 8 radius - lookup murni dari hasil pembobotan
    stages.radii = radius.from_assessment(stages.assessment)
    stages.spotted_dog = poo.spotted_dog_report(
        stages.observation.frame[stages.observation.frame["qualifies"]],
        stages.intersections, collars, stages.radii,
        policy=cfg.poo.two_direction_policy)

    # 9 batas + RPEEE
    stages.limits = limits.run(
        stages.models, cfg, topo=stages.topo,
        weathering_depth_m=cfg.weathering.constant_depth_m,
        quality=stages.dataset.quality)
    stages.masks = {r.key: r.mask for r in stages.limits.results}
    if not stages.limits.reportable:
        # Hanya fakta terlarang YANG DINYATAKAN yang menghentikan - bukan data
        # yang belum diisi. Lihat catatan modul limits.
        stages.stopped_at = "9_batas"
        stages.messages.append(
            f"Tahap 9: {len(stages.limits.blockers)} penggugur KCMI 4.6 - "
            "status yang dinyatakan memang melarang pelaporan di area ini.")
        return stages
    if stages.limits.readiness:
        stages.messages.append(
            f"Tahap 9: {len(stages.limits.readiness)} butir kesiapan pelaporan "
            "belum dinyatakan. Estimasi TETAP berjalan; yang tertahan hanya "
            "kelengkapan pernyataan keprospekan beralasan.")

    # 10 estimasi
    from ..errors import MissingDataError
    try:
        stages.estimate = estimate_grid.run(
            stages.models, stages.masks, stages.observation.frame,
            stages.intersections, collars, stages.radii, cfg,
            quality=stages.dataset.quality)
    except MissingDataError as exc:
        stages.stopped_at = "10_densitas"
        stages.messages.append(f"Tahap 10 densitas: {exc}")
        return stages
    if not stages.estimate.passed:
        stages.stopped_at = "10_swauji"
        stages.messages.append(
            f"Tahap 10 swauji peta: {len(stages.estimate.failures)} kegagalan.")
    return stages


def export_all(stages: Stages, output_dir: Path | str, project_name: str):
    """Tulis kedua paket keluaran dari satu run yang sudah selesai."""
    from ..export import bab5, handover

    if not stages.complete:
        raise ValueError(
            f"run belum selesai (berhenti di {stages.stopped_at}); "
            "keluaran tidak ditulis.")

    cfg = stages.cfg
    run_obj = project.start(output_dir, project_name)
    for topic, statement, basis in (
        ("Zona pelapukan",
         f"Konstanta {cfg.weathering.constant_depth_m} m, provenance "
         f"'{cfg.weathering.provenance}'.", cfg.weathering.provenance_basis),
        ("Batas IUP", "Batas IUP TIDAK diterapkan.",
         cfg.limits.iup_boundary_basis or "Keputusan proyek yang tercatat."),
        ("Batas kedalaman",
         "Tidak diterapkan." if cfg.max_depth_m is None
         else f"{cfg.max_depth_m:g} m.",
         cfg.limits.depth.no_depth_limit_basis or cfg.limits.depth.basis_note),
        ("Titik observasi",
         f"Kualitas {'dituntut' if cfg.observation_point.requires_quality else 'tidak dituntut'}.",
         cfg.observation_point.basis),
    ):
        run_obj.assume(topic, statement, basis)
    for path in (cfg.minex.survey_file, cfg.minex.lithology_file,
                 cfg.minex.quality_file, cfg.minex.topography_file):
        if path:
            run_obj.digest_input(Path(path))

    handover.write_grids(run_obj, stages.models, stages.masks)
    handover.write_contours(run_obj, stages.models,
                            interval_m=cfg.maps.dxf_export.contour_interval_m)
    handover.write_tables(run_obj, stages.estimate, cfg.resource_label)

    weathering = (stages.dataset.intervals[stages.dataset.intervals["is_marker"]]
                  .groupby("depth_to")["hole_id"].nunique()
                  .rename("n_lubang").rename_axis("tebal_pelapukan_m").reset_index())
    bab5.tabel_5_1(run_obj, weathering)
    _, stats = bab5.tabel_5_2(run_obj, stages.intersections)
    bab5.tabel_deviasi(run_obj, stages.validation)
    bab5.tabel_5_32(run_obj, stages.assessment)
    bab5.tabel_5_33(run_obj, stages.radii, stages.assessment.condition)
    bab5.tabel_5_35_36(run_obj, stages.estimate, cfg.resource_label)
    bab5.gambar_5_1(run_obj, stats)
    bab5.gambar_kontur_floor(run_obj, stages.models)
    bab5.gambar_peta_sumberdaya(run_obj, stages.estimate.estimates,
                                stages.models, stages.observation.frame)
    bab5.write_manual_slots(run_obj)
    bab5.write_docx(run_obj, cfg.resource_label, stages.assessment.condition,
                    stages.estimate, stages.assessment)
    run_obj.write_assumptions()
    run_obj.write_manifest()

    problems = handover.verify_markers(run_obj)
    return run_obj, problems
