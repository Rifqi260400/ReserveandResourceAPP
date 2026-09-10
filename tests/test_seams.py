"""Modul seams: aturan tebal batubara dan perlakuan core loss."""
import pandas as pd
import pytest

from coalres.config import Config
from coalres.errors import MissingDataError
from coalres.io.excel import HeaderMap, SheetTable, Workbook
from coalres.seams import assumptions, build_intersections, to_frame

LIBRARY = {
    "CO": "Coal Undifferentiated", "C3": "Coal, 40-60% bright",
    "CS": "Clay Stone", "XC": "Carbonaceous CS", "SS": "Sandstone",
    "KL": "Core Loss", "SO": "Soil",
}


def make_workbook(rows: list[tuple[float, float, str, str | None]]) -> Workbook:
    """Workbook sintetis: (from, to, kode litologi, nama seam atau None)."""
    frame = pd.DataFrame(
        [{"hole_id": "SYN01" if i == 0 else None, "depth_from": f, "depth_to": t,
          "lithology": lith, "seam": seam} for i, (f, t, lith, seam) in enumerate(rows)]
    )
    header = HeaderMap("SLL_Reconciled", 10, [6, 7, 8], list(frame.columns),
                       {c: i for i, c in enumerate(frame.columns)})
    return Workbook(
        path=pd.io.common.Path("synthetic.xlsx"), hole_id="SYN01",
        sheets={"SLL_Reconciled": SheetTable(header=header, frame=frame, raw_rows=len(frame))},
        lithology_library=LIBRARY,
    )


@pytest.fixture
def cfg(base_config_dict, write_config):
    base_config_dict["coal_thickness_source"] = "lithology"
    base_config_dict["core_loss_treatment"] = "as_coal"
    return Config.load(write_config(base_config_dict))


def _one(wb, cfg):
    items = build_intersections(wb, cfg)
    assert len(items) == 1
    return items[0]


# --------------------------------------------------------------------------- #
# Core loss
# --------------------------------------------------------------------------- #
def test_core_loss_inside_seam_counts_as_coal(cfg):
    wb = make_workbook([
        (0.0, 10.0, "SS", None),
        (10.0, 12.0, "CO", "S1"),
        (12.0, 12.1, "KL", "S1"),      # core loss beratribut seam
        (12.1, 14.0, "CO", "S1"),
        (14.0, 20.0, "CS", None),
    ])
    item = _one(wb, cfg)
    assert item.coal_thickness_m == pytest.approx(4.0)     # 2.0 + 1.9 batubara + 0.1 core loss
    assert item.core_loss_in_coal_m == pytest.approx(0.1)
    assert item.parting_thickness_m == pytest.approx(0.0)


def test_core_loss_outside_seam_never_counts(cfg):
    """Aturan dibatasi pada atribusi geolog.

    Tanpa batasan ini, 0,445 m core loss di batuan penutup dan interburden
    DH09_05C1 (70,070-70,270 dan 83,470-83,715) akan terhitung batubara.
    """
    wb = make_workbook([
        (0.0, 5.0, "SS", None),
        (5.0, 5.5, "KL", None),        # core loss di waste
        (5.5, 10.0, "SS", None),
        (10.0, 14.0, "CO", "S1"),
        (14.0, 20.0, "CS", None),
    ])
    item = _one(wb, cfg)
    assert item.coal_thickness_m == pytest.approx(4.0)
    assert item.core_loss_in_coal_m == pytest.approx(0.0)


def test_core_loss_as_waste(base_config_dict, write_config):
    base_config_dict["core_loss_treatment"] = "as_waste"
    cfg = Config.load(write_config(base_config_dict))
    wb = make_workbook([
        (0.0, 10.0, "SS", None), (10.0, 12.0, "CO", "S1"),
        (12.0, 12.1, "KL", "S1"), (12.1, 14.0, "CO", "S1"), (14.0, 20.0, "CS", None),
    ])
    item = _one(wb, cfg)
    assert item.coal_thickness_m == pytest.approx(3.9)
    assert item.parting_thickness_m == pytest.approx(0.1)


def test_core_loss_excluded_shrinks_gross(base_config_dict, write_config):
    base_config_dict["core_loss_treatment"] = "exclude"
    cfg = Config.load(write_config(base_config_dict))
    wb = make_workbook([
        (0.0, 10.0, "SS", None), (10.0, 12.0, "CO", "S1"),
        (12.0, 12.1, "KL", "S1"), (12.1, 14.0, "CO", "S1"), (14.0, 20.0, "CS", None),
    ])
    item = _one(wb, cfg)
    assert item.coal_thickness_m == pytest.approx(3.9)
    assert item.gross_thickness_m == pytest.approx(3.9)     # 4.0 - 0.1


# --------------------------------------------------------------------------- #
# Parting
# --------------------------------------------------------------------------- #
def test_parting_below_cutoff_stays_in_gross_out_of_coal(cfg):
    wb = make_workbook([
        (0.0, 10.0, "SS", None), (10.0, 12.0, "CO", "S1"),
        (12.0, 12.2, "CS", None), (12.2, 14.0, "CO", "S1"), (14.0, 20.0, "CS", None),
    ])
    item = _one(wb, cfg)
    assert item.gross_thickness_m == pytest.approx(4.0)
    assert item.coal_thickness_m == pytest.approx(3.8)
    assert item.parting_thickness_m == pytest.approx(0.2)


def test_parting_above_cutoff_splits_the_seam(cfg):
    """max_parting_thickness_m = 0,3 pada konfigurasi uji."""
    wb = make_workbook([
        (0.0, 10.0, "SS", None), (10.0, 12.0, "CO", "S1"),
        (12.0, 12.6, "CS", None), (12.6, 14.0, "CO", "S1"), (14.0, 20.0, "CS", None),
    ])
    items = build_intersections(wb, cfg)
    assert len(items) == 2
    assert [i.seam for i in items] == ["S1_1", "S1_2"]
    assert all(i.split_from == "S1" for i in items)
    assert items[0].coal_thickness_m == pytest.approx(2.0)
    assert items[1].coal_thickness_m == pytest.approx(1.4)


def test_seam_below_minimum_thickness_is_dropped(cfg):
    """min_seam_thickness_m = 0,4 pada konfigurasi uji."""
    wb = make_workbook([
        (0.0, 10.0, "SS", None), (10.0, 10.3, "CO", "S1"), (10.3, 20.0, "CS", None),
    ])
    assert build_intersections(wb, cfg) == []


# --------------------------------------------------------------------------- #
# Data nyata
# --------------------------------------------------------------------------- #
@pytest.fixture
def real(cfg, workbook_dir):
    from coalres.io.excel import load_workbook
    rows = []
    for path in sorted(workbook_dir.glob("*.xlsx")):
        rows += build_intersections(load_workbook(path), cfg)
    return to_frame(rows).set_index(["hole_id", "seam"])


def test_real_s10b_matches_hand_calculation(real):
    """S10B: 11,770 gross - 0,410 parting = 11,360 dengan core loss ikut."""
    row = real.loc[("DH0905C1", "S10B")]
    assert row["gross_thickness_m"] == pytest.approx(11.770, abs=1e-3)
    assert row["coal_thickness_m"] == pytest.approx(11.360, abs=1e-3)
    assert row["parting_thickness_m"] == pytest.approx(0.410, abs=1e-3)
    assert row["core_loss_in_coal_m"] == pytest.approx(0.150, abs=1e-3)


def test_real_s10a_matches_hand_calculation(real):
    row = real.loc[("DH0905C1", "S10A")]
    assert row["coal_thickness_m"] == pytest.approx(9.680, abs=1e-3)
    assert row["core_loss_in_coal_m"] == pytest.approx(0.050, abs=1e-3)


def test_real_interburden_core_loss_excluded(real):
    """0,445 m core loss di 70,070-70,270 dan 83,470-83,715 tidak masuk seam mana pun."""
    total = real["core_loss_in_coal_m"].sum()
    assert total == pytest.approx(0.200, abs=1e-3)     # hanya 0,050 + 0,150


def test_real_s7b_parting_020_does_not_split(real):
    """Parting 0,20 m di S7B ada di bawah cutoff 0,30 m, jadi seam tetap utuh."""
    row = real.loc[("DH1101", "S7B")]
    assert row["max_parting_m"] == pytest.approx(0.200, abs=1e-3)
    assert row["coal_thickness_m"] == pytest.approx(2.880, abs=1e-3)
    assert pd.isna(row["split_from"]) or row["split_from"] is None


# --------------------------------------------------------------------------- #
def test_sampling_source_not_implemented_raises(base_config_dict, write_config):
    base_config_dict["coal_thickness_source"] = "sampling"
    cfg = Config.load(write_config(base_config_dict))
    wb = make_workbook([(0.0, 10.0, "SS", None), (10.0, 14.0, "CO", "S1")])
    with pytest.raises(MissingDataError, match="belum"):
        build_intersections(wb, cfg)


def test_assumptions_stamped_for_as_coal(cfg):
    text = " ".join(assumptions(cfg))
    assert "ASUMSI" in text and "as_coal" in text and "litologi" in text
    assert "di luar amplop seam selalu dihitung waste" in text
