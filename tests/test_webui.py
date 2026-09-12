"""Antarmuka web: endpoint, manifest ekspor, dan pemuatan konfigurasi."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from coalres.webui.app import EXPORT_KINDS, create_app


@pytest.fixture(scope="module")
def client():
    from conftest import ROOT

    if not (ROOT / "config" / "minex_dummy_resolved.yaml").exists():
        pytest.skip("konfigurasi Minex tidak tersedia")
    return TestClient(create_app())


@pytest.fixture(scope="module")
def ran(client):
    client.post("/api/config/load", json={"path": "config/minex_dummy_resolved.yaml"})
    response = client.post("/api/run")
    assert response.status_code == 200, response.text
    return response.json()


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Estimasi Sumberdaya Batubara" in response.text


def test_endpoints_refuse_before_config_is_loaded():
    fresh = TestClient(create_app())
    assert fresh.post("/api/audit").status_code == 400
    assert fresh.get("/api/summary").status_code == 400
    assert fresh.get("/api/exports").status_code == 400


def test_bad_config_path_is_rejected(client):
    response = client.post("/api/config/load", json={"path": "tidak/ada.yaml"})
    assert response.status_code == 400


def test_summary_carries_the_resource_label(ran):
    assert ran["label"] in {"Sumberdaya", "Inventori Batubara"}
    assert ran["total"][0]["tonnes"] > 0
    assert set(ran["seams"]) == {"A", "A1", "A2", "B"}


def test_class_labels_follow_the_resource_label(ran):
    prefix = ran["label"].split()[0]
    assert all(row["class"].startswith(prefix) for row in ran["by_seam_class"])


def test_map_returns_contours_polygons_and_holes(client, ran):
    response = client.get("/api/map/A?interval=5")
    assert response.status_code == 200
    data = response.json()
    assert data["contours"]["roof"] and data["contours"]["floor"]
    assert data["polygons"] and data["holes"]
    assert all("z" in c and "points" in c for c in data["contours"]["roof"])


def test_unknown_seam_is_404(client, ran):
    assert client.get("/api/map/ZZ").status_code == 404


def test_all_five_export_kinds_are_available(client, ran):
    data = client.get("/api/exports").json()
    assert set(data["kinds"]) == set(EXPORT_KINDS)
    for kind, info in data["kinds"].items():
        assert info["count"] > 0, f"{kind} kosong"


def test_exports_are_driven_by_manifest_not_filename_guessing(client, ran):
    """Menebak kategori dari akhiran nama berkas rapuh: kode kualitas dan kode
    struktur bisa bertabrakan begitu konvensi penamaan diubah pengguna."""
    import json
    from pathlib import Path

    manifest = json.loads((Path(ran["output_dir"]) / "manifest.json").read_text())
    assert {"uncut_grid", "quality_grid", "contours_dxf",
            "resource_table", "bow_recap"} <= set(manifest)
    # Uncut memuat SR/SF/ST, bukan ketebalan cut.
    names = " ".join(manifest["uncut_grid"])
    assert "SR" in names and "SF" in names and "ST" in names
    assert "CST" not in " ".join(
        n for n in manifest["uncut_grid"] if n.endswith(".grid"))


def test_each_export_downloads(client, ran):
    for kind in EXPORT_KINDS:
        response = client.get(f"/api/export/{kind}")
        assert response.status_code == 200, kind
        assert len(response.content) > 0


def test_unknown_export_kind_is_404(client, ran):
    assert client.get("/api/export/tidak_ada").status_code == 404
