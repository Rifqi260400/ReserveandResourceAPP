"""Paket laporan Bab V."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import (complexity, estimate_grid, limits, model, observation,
                     project, radius, validate)
from coalres.config import Config
from coalres.export import bab5
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame
from coalres.topo import build_surface

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"
INTRUSI = ("intrusi", "sederhana",
           "Tidak ada indikasi batuan beku pada seluruh 60 lubang maupun peta regional.")


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    if not CONFIG.exists():
        pytest.skip("dataset minex tidak tersedia")
    cfg = Config.load(CONFIG)
    dataset = load_minex(cfg)
    collars = dataset.collars.set_index("hole_id")
    intersections = to_frame(build_intersections_from_dataset(dataset, cfg))
    models = model.build(intersections, collars, cfg, spacing=25.0)
    t = dataset.topo_points
    topo = build_surface(t[:, 0], t[:, 1], t[:, 2], spacing=25.0, name="topo")
    declared = cfg.model_copy(update={"limits": cfg.limits.model_copy(update={
        "legal": cfg.limits.legal.model_copy(
            update={"permit_type": "IUP", "permit_covers_mine_life": True}),
        "land": cfg.limits.land.model_copy(
            update={"forest_category": "APL", "rtrw_allows_mining": True})})})
    lim = limits.run(models, declared, topo=topo, weathering_depth_m=3.0,
                     quality=dataset.quality)
    masks = {r.key: r.mask for r in lim.results}
    points = observation.build(intersections, dataset, declared)
    assessment = complexity.assess(dataset, declared, {INTRUSI[0]: INTRUSI[1:]})
    radii = radius.from_assessment(assessment)
    estimate = estimate_grid.run(models, masks, points.frame, intersections,
                                 collars, radii, declared, quality=dataset.quality)
    validation = validate.run(models, intersections, collars, declared,
                              quality=dataset.quality, spacing=25.0)

    run = project.start(tmp_path_factory.mktemp("bab5"), "UJI")
    run.assume("Zona pelapukan", "Konstanta 3,00 m.", "Dinyatakan asumsi.")
    weathering = (dataset.intervals[dataset.intervals["is_marker"]]
                  .groupby("depth_to")["hole_id"].nunique()
                  .rename("n_lubang").rename_axis("tebal_pelapukan_m").reset_index())
    bab5.tabel_5_1(run, weathering)
    _, stats = bab5.tabel_5_2(run, intersections)
    bab5.tabel_deviasi(run, validation)
    bab5.tabel_5_32(run, assessment)
    bab5.tabel_5_33(run, radii, assessment.condition)
    bab5.tabel_5_35_36(run, estimate, declared.resource_label)
    bab5.gambar_5_1(run, stats)
    bab5.gambar_kontur_floor(run, models)
    bab5.gambar_peta_sumberdaya(run, estimate.estimates, models, points.frame)
    bab5.write_manual_slots(run)
    bab5.write_docx(run, declared.resource_label, assessment.condition,
                    estimate, assessment)
    run.write_assumptions()
    run.write_manifest()
    return run, radii, assessment


def test_every_figure_ships_pdf_png_and_its_numbers(produced):
    """Pembaca harus dapat memeriksa gambar tanpa menjalankan ulang program."""
    run, *_ = produced
    folder = run.root / "01_bab5_laporan" / "gambar"
    stems = {p.stem for p in folder.glob("*.pdf")}
    assert stems
    for stem in stems:
        assert (folder / f"{stem}.png").exists()
        assert (folder / f"{stem}.csv").exists()


def test_figure_numbers_follow_the_reference_report(produced):
    run, *_ = produced
    names = {p.stem for p in (run.root / "01_bab5_laporan" / "gambar").glob("*.png")}
    assert "Gambar_5-1" in names
    # Kontur floor mulai 5.4, peta sumberdaya mulai 5.25 - sesuai laporan rujukan.
    assert "Gambar_5-4" in names
    assert "Gambar_5-25" in names
    # Tidak ada nomor bertingkat ganda seperti "Gambar_5-5-4".
    assert not [n for n in names if n.count("-") > 1]


def test_tables_ship_as_csv_beside_excel(produced):
    run, *_ = produced
    folder = run.root / "01_bab5_laporan" / "tabel"
    for csv in folder.glob("*.csv"):
        assert (folder / f"{csv.stem}.xlsx").exists()
    assert (folder / "Tabel_5-1.csv").exists()
    assert (folder / "Tabel_5-33.csv").exists()


def test_tabel_5_33_reproduces_the_sni_bands(produced):
    run, radii, assessment = produced
    frame = pd.read_csv(run.root / "01_bab5_laporan" / "tabel" / "Tabel_5-33.csv")
    row = frame.iloc[0]
    assert row["Kondisi Geologi"].lower() == assessment.condition
    assert f"x <= {radii['terukur']:.0f}" == row["Terukur"]
    assert str(int(radii["tereka"])) in row["Tereka"]


def test_the_deviation_tables_come_from_cross_validation(produced):
    """Tabel 5.3..5.30 melaporkan galat RAMALAN, bukan galat pembulatan grid."""
    run, *_ = produced
    folder = run.root / "01_bab5_laporan" / "tabel"
    deviations = sorted(folder.glob("Tabel_5-[0-9]*.csv"))
    assert len(deviations) >= 8
    frame = pd.read_csv(folder / "Tabel_5-3.csv")
    assert {"hole_id", "data", "model", "deviasi"} <= set(frame.columns)
    # Galat validasi silang jauh lebih besar daripada galat diskretisasi.
    assert frame["deviasi"].abs().max() > 0.5


def test_unbuildable_slots_are_printed_not_dropped(produced):
    """Laporan yang kehilangan gambar tanpa jejak lebih berbahaya."""
    run, *_ = produced
    text = (run.root / "01_bab5_laporan" / "SLOT_MANUAL.md").read_text()
    for number in list(bab5.CLASS_B_SLOTS) + list(bab5.CLASS_C_SLOTS):
        assert number in text
    assert text.count(bab5.MANUAL) == len(bab5.CLASS_B_SLOTS) + len(bab5.CLASS_C_SLOTS)


def test_the_resource_class_ramp_is_ordinal_not_categorical(produced):
    """Kelas adalah peringkat keyakinan, jadi satu hue menggelap.

    Ramp ini lolos uji ordinal: monoton, jarak lightness cukup, dan ujung
    terangnya masih 2,06:1 terhadap permukaan terang.
    """
    from matplotlib.colors import to_rgb
    import colorsys
    lightness = [colorsys.rgb_to_hls(*to_rgb(bab5.CLASS_COLORS[k]))[1]
                 for k in ("terukur", "tertunjuk", "tereka")]
    assert lightness == sorted(lightness)          # menggelap searah keyakinan
    assert all(b - a > 0.06 for a, b in zip(lightness, lightness[1:]))
    hues = {round(colorsys.rgb_to_hls(*to_rgb(v))[0], 2)
            for v in bab5.CLASS_COLORS.values()}
    assert len(hues) == 1                          # satu hue


def test_the_docx_draft_names_itself_a_draft(produced):
    docx = pytest.importorskip("docx")
    run, *_ = produced
    path = run.root / "01_bab5_laporan" / "draf_bab5.docx"
    assert path.exists()
    text = "\n".join(p.text for p in docx.Document(path).paragraphs)
    assert "DRAF OTOMATIS" in text
    assert "BELUM memuat penilaian Competent Person" in text
    # Kedua peringatan wajib kompleksitas ikut ke dokumen.
    assert "PERINGATAN WAJIB" in text


def test_the_manifest_covers_the_bab5_package_too(produced):
    run, *_ = produced
    payload = json.loads(
        (run.root / "02_reserve_handover" / "manifest.json").read_text())
    kinds = {entry["kind"] for entry in payload["files"]}
    assert {"tabel_csv", "gambar_pdf", "gambar_png", "gambar_data",
            "slot_manual"} <= kinds
    for entry in payload["files"]:
        assert (run.root / entry["path"]).exists()
