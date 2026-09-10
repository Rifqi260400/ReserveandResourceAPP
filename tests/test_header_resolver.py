"""Resolver header terhadap workbook BGG yang sebenarnya."""
import pytest

from coalres.io.excel import load_workbook, normalise_hole_id


@pytest.fixture(scope="module")
def workbooks(request):
    from conftest import WORKBOOKS
    if not WORKBOOKS.exists():
        pytest.skip("workbook referensi tidak tersedia")
    return {p.stem: load_workbook(p) for p in sorted(WORKBOOKS.glob("*.xlsx"))}


def test_all_sheets_resolve(workbooks):
    for name, wb in workbooks.items():
        assert {"Collar", "SLL_Reconciled", "SLL_Wellsite", "BHC", "Sampling"} <= set(wb.sheets)


def test_sll_header_block_excludes_data_rows(workbooks):
    """Kegagalan yang ditemukan saat resolver diuji: baris data SLL hanya
    membawa dua angka, sehingga ambang '>=3 angka' menariknya menjadi header."""
    for wb in workbooks.values():
        h = wb.sheets["SLL_Reconciled"].header
        assert h.header_rows == [6, 7, 8]      # 0-based; baris 7-9 di Excel
        assert h.data_start_row == 10          # baris 11 di Excel
        assert h.flattened[0] == "Hole_ID"     # bukan 'Hole_ID DH09_05C1'


def test_alias_layer_absorbs_header_differences(workbooks):
    """Header berbeda antar workbook untuk kolom yang sama.

    DH09_05C1 menulis 'Lithologi' dan 'Continuity'; DH11_01 menulis 'Lith' dan
    'Hole Type'. Keduanya harus memetakan ke nama kanonik yang sama.
    """
    variants = set()
    for wb in workbooks.values():
        h = wb.sheets["SLL_Reconciled"].header
        assert "lithology" in h.canonical and "hole_type" in h.canonical
        variants.add(h.flattened[h.canonical["lithology"]])
    assert len(variants) > 1, "fixture seharusnya memuat dua ejaan header berbeda"


def test_depth_columns_not_stolen_by_core_loss_columns(workbooks):
    """Alias 'From' harus mengenai Depth From, bukan Core Loss From."""
    for wb in workbooks.values():
        h = wb.sheets["SLL_Reconciled"].header
        assert h.flattened[h.canonical["depth_from"]].endswith("Depth From")
        assert h.flattened[h.canonical["depth_to"]].endswith("Depth To")


def test_lithology_library_read_not_hardcoded(workbooks):
    for wb in workbooks.values():
        lib = wb.lithology_library
        assert lib["CO"].lower().startswith("coal")
        assert lib["KL"].lower() == "core loss"
        assert lib["C3"] == "Coal, 40-60% bright"


def test_sampling_adjusted_depths_present(workbooks):
    wb = workbooks["DH09_05C1"]
    frame = wb.sheets["Sampling"].frame
    assert {"adjusted_from", "adjusted_to", "drilling_from"} <= set(frame.columns)
    assert frame["adjusted_from"].notna().sum() >= 18


def test_hole_id_normalisation_joins_source_variants():
    assert len({normalise_hole_id(v) for v in
                ["DH-09-05C1", "DH09_05C1", "DH09 05C1", "dh09_05c1"]}) == 1
