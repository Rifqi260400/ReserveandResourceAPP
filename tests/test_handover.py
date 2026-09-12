"""Paket serah-terima cadangan: grid, kontur, manifest."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coalres import (complexity, estimate_grid, limits, model, observation,
                     project, radius)
from coalres.config import Config
from coalres.export import handover
from coalres.io.minex import load_minex
from coalres.seams import build_intersections_from_dataset, to_frame
from coalres.topo import build_surface

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "minex_dummy.yaml"
INTRUSI = ("intrusi", "sederhana",
           "Tidak ada indikasi batuan beku pada seluruh 60 lubang maupun peta regional.")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
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
    report = limits.run(models, declared, topo=topo, weathering_depth_m=3.0,
                        quality=dataset.quality)
    masks = {r.key: r.mask for r in report.results}
    points = observation.build(intersections, dataset, declared)
    assessment = complexity.assess(dataset, declared, {INTRUSI[0]: INTRUSI[1:]})
    radii = radius.from_assessment(assessment)
    estimate = estimate_grid.run(models, masks, points.frame, intersections,
                                 collars, radii, declared, quality=dataset.quality)

    run = project.start(tmp_path_factory.mktemp("out"), "UJI")
    run.assume("Zona pelapukan", "Konstanta 3,00 m.", "Dinyatakan asumsi.")
    handover.write_grids(run, models, masks)
    handover.write_contours(run, models, interval_m=2.0)
    handover.write_tables(run, estimate, declared.resource_label)
    run.write_assumptions()
    run.write_manifest()
    return run, models, masks


def test_every_grid_carries_its_uncut_or_ltd_marker(exported):
    """Keduanya identik saat dibuka; penanda satu-satunya pertahanan."""
    run, *_ = exported
    assert handover.verify_markers(run) == []
    grids = [f for f in run.files if f["kind"].startswith("grid_")]
    assert grids
    for entry in grids:
        name = Path(entry["path"]).name
        assert ("_uncut" in name) ^ ("_ltd" in name)


def test_both_variants_are_always_written(exported):
    """Paket yang kadang memuat limited memaksa penerimanya menebak."""
    run, models, _ = exported
    for variant in ("uncut", "limited"):
        for surface in ("roof", "floor", "thickness"):
            kinds = [f for f in run.files
                     if f["kind"] == f"grid_{variant}_{surface}"]
            assert len(kinds) == len(models), (variant, surface)


def test_esri_ascii_round_trips_to_the_model_exactly(exported):
    rasterio = pytest.importorskip("rasterio")
    run, models, _ = exported
    path = run.root / "02_reserve_handover" / "uncut" / "SGBST_uncut.asc"
    with rasterio.open(path) as src:
        grid = src.read(1, masked=True)
        assert src.transform.a == 25.0
    z = models["B"].isopach.z
    assert grid.count() == int(np.isfinite(z).sum())
    assert float(grid.min()) == pytest.approx(float(np.nanmin(z)), abs=1e-3)
    assert float(grid.max()) == pytest.approx(float(np.nanmax(z)), abs=1e-3)


def test_ascii_geotiff_and_csv_agree(exported):
    rasterio = pytest.importorskip("rasterio")
    run, *_ = exported
    folder = run.root / "02_reserve_handover" / "uncut"
    with rasterio.open(folder / "SGBST_uncut.asc") as a, \
            rasterio.open(folder / "SGBST_uncut.tif") as b:
        assert np.allclose(a.read(1), b.read(1), atol=1e-3)
        assert np.allclose([a.transform.a, a.transform.c, a.transform.f],
                           [b.transform.a, b.transform.c, b.transform.f])
        count = a.read(1, masked=True).count()
    assert len(pd.read_csv(folder / "SGBST_uncut.csv")) == count


def test_ascii_rows_run_north_to_south(exported):
    """Ketentuan format. Terbalik, grid tampak wajar tapi tercermin."""
    rasterio = pytest.importorskip("rasterio")
    run, models, _ = exported
    with rasterio.open(run.root / "02_reserve_handover" / "uncut"
                       / "SGBSR_uncut.asc") as src:
        first_row = src.read(1, masked=True)[0]
    model_top = models["B"].roof.z[-1]
    finite = np.isfinite(model_top)
    assert np.allclose(first_row[finite], model_top[finite], atol=1e-3)


def test_limited_grids_are_never_larger_than_uncut(exported):
    rasterio = pytest.importorskip("rasterio")
    run, *_ = exported
    base = run.root / "02_reserve_handover"
    with rasterio.open(base / "uncut" / "SGBST_uncut.asc") as u, \
            rasterio.open(base / "limited" / "SGBST_ltd.asc") as l:
        assert l.read(1, masked=True).count() <= u.read(1, masked=True).count()


def test_contours_are_three_dimensional_polylines(exported):
    ezdxf = pytest.importorskip("ezdxf")
    run, *_ = exported
    path = run.root / "02_reserve_handover" / "kontur" / "Seam_B_Floor.dxf"
    document = ezdxf.readfile(path)
    polylines = list(document.modelspace().query("POLYLINE"))
    assert polylines
    heights = {round(v.dxf.location.z, 1)
               for p in polylines for v in p.vertices}
    # Ber-Z sungguhan, bukan datar di elevasi nol.
    assert len(heights) > 3
    assert heights != {0.0}


def test_the_manifest_records_every_file_written(exported):
    run, *_ = exported
    payload = json.loads(
        (run.root / "02_reserve_handover" / "manifest.json").read_text())
    assert payload["n_files"] == len(run.files)
    for entry in payload["files"]:
        assert (run.root / entry["path"]).exists()
        assert entry["kind"]
    assert payload["assumptions"]


def test_assumptions_reach_the_written_document(exported):
    run, *_ = exported
    text = (run.root / "03_audit_and_provenance" / "asumsi.md").read_text()
    assert "Zona pelapukan" in text and "Dinyatakan asumsi" in text


def test_the_run_root_is_named_for_its_time(exported):
    run, *_ = exported
    assert run.root.name.startswith("run_UJI_")
    for folder in project.SUBDIRS:
        assert (run.root / folder).is_dir()


def test_a_grid_without_a_marker_is_caught(exported):
    """Swauji dijalankan atas manifest, bukan atas niat penulisnya."""
    run, *_ = exported
    run.files.append({"path": "02_reserve_handover/uncut/SGXST.asc",
                      "kind": "grid_uncut_thickness", "bytes": 0, "note": ""})
    try:
        problems = handover.verify_markers(run)
        assert len(problems) == 1 and "tanpa penanda" in problems[0]
    finally:
        run.files.pop()


def test_non_square_cells_are_refused(tmp_path):
    """ESRI ASCII tidak dapat menyatakan sel persegi panjang."""
    with pytest.raises(ValueError, match="bujur sangkar"):
        handover.esri_ascii(np.array([0.0, 10.0]), np.array([0.0, 25.0]),
                            np.zeros((2, 2)), tmp_path / "x.asc")
