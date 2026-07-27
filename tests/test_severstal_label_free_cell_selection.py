import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "_severstal_label_free_cell_selection.py"
SPEC = importlib.util.spec_from_file_location("selection", MODULE_PATH)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def write_cell(root, tag, *, dial=None):
    suffix = "base_mcs20" if tag == "base" else f"{tag}_mcs20"
    payload = {
        "dial": selection.EXPECTED_DIAL if dial is None else dial,
        "selected_ep": 20,
        "selected": {"off": {"P1": "3/4", "ARI": 0.5, "P3_comp": 0.6, "P4_hom": 0.7},
                     "lf": {"noise_pct": 50.0, "k": 5, "over_merge": 0,
                            "stability": 0.8, "coherence": 0.9}},
    }
    (root / f"severstal_pilot_{suffix}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_cells_accepts_only_complete_full20_mcs20_set(tmp_path):
    for tag in selection.REQUIRED_CELL_TAGS:
        write_cell(tmp_path, tag)
    cells = selection.collect_cells(str(tmp_path))
    assert tuple(cells) == selection.REQUIRED_CELL_TAGS
    assert len(cells) == 9
    assert all(cell["dial"] == selection.EXPECTED_DIAL for cell in cells.values())
    assert all(len(cell["sha256"]) == 64 for cell in cells.values())


def test_collect_cells_rejects_mixed_dial_and_missing_required_cell(tmp_path):
    for tag in selection.REQUIRED_CELL_TAGS[:-1]:
        write_cell(tmp_path, tag)
    write_cell(tmp_path, "nolocal")
    with pytest.raises(ValueError, match="exactly"):
        selection.collect_cells(str(tmp_path))

    (tmp_path / "severstal_pilot_nolocal_mcs20.json").unlink()
    write_cell(tmp_path, selection.REQUIRED_CELL_TAGS[-1], dial={"mcs": 6, "ms": 3, "eps": 0.06, "method": "leaf"})
    with pytest.raises(ValueError, match="expected dial"):
        selection.collect_cells(str(tmp_path))


def test_label_free_whitelist_excludes_frag_and_silhouette():
    assert selection.LABEL_FREE_CANDIDATES == (
        "seed_noise", "k", "stability", "coherence", "over_merge",
    )
