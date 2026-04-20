"""Unit tests for data_prep.wm811k_to_palette (fail-map format).

Runnable via:
    python -m pytest tests/test_wm811k_convert.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the project root importable when running this file directly.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_prep.wm811k_to_palette import (  # noqa: E402
    assign_grades_by_neighbor_density,
    save_palette_png,
    wm_to_palette,
)
from common.palette_io import (  # noqa: E402
    get_failmap_palette,
    load_palette_indices,
)


def test_outside_and_normal_mapping():
    """바깥(0)→31 transparent, 정상(1)→grade 0. 결함은 density-dependent."""
    wm = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    out = assign_grades_by_neighbor_density(wm)
    assert out.dtype == np.uint8
    assert out.shape == (3, 3)
    assert (out[wm == 0] == 31).all(), "outside → 31"
    assert (out[wm == 1] == 0).all(), "normal → 0"


def test_isolated_defect_is_grade_3():
    """고립된 결함(이웃 결함 0)은 grade 3."""
    wm = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    out = assign_grades_by_neighbor_density(wm)
    assert out[1, 1] == 3


def test_dense_defect_cluster_reaches_grade_7():
    """3x3 전부 결함이면 center 의 이웃 결함 8 → grade 7."""
    wm = np.full((3, 3), 2, dtype=np.int32)
    out = assign_grades_by_neighbor_density(wm)
    assert out[1, 1] == 7


def test_intermediate_density_grades():
    """일부 이웃만 결함 → grade 3..7 범위 내."""
    # 중심 (1,1) 주변 이웃 4개만 결함
    wm = np.array([
        [1, 2, 1],
        [2, 2, 2],
        [1, 2, 1],
    ])
    out = assign_grades_by_neighbor_density(wm)
    # 중심의 이웃 결함 수 = 4 → (4+1)//2 = 2 → grade 3+2 = 5
    assert out[1, 1] == 5
    # 모든 결함 grade 는 3..7 범위
    grades = out[wm == 2]
    assert grades.min() >= 3 and grades.max() <= 7


def test_wm_to_palette_alias_works():
    """Back-compat alias wm_to_palette returns same as new function."""
    wm = np.array([[0, 1, 2], [2, 1, 0]])
    a = wm_to_palette(wm)
    b = assign_grades_by_neighbor_density(wm)
    assert np.array_equal(a, b)


def test_accepts_list_of_list():
    """pkl waferMap cells can be list-of-list; np.asarray must handle."""
    wm = [[0, 1, 2], [2, 1, 0]]
    out = assign_grades_by_neighbor_density(wm)
    assert out.dtype == np.uint8
    assert out.shape == (2, 3)
    assert out[0, 0] == 31 and out[1, 2] == 31
    assert out[0, 1] == 0 and out[1, 1] == 0


def test_save_and_roundtrip_with_failmap_palette():
    """저장 → 재로드해도 grade index 가 그대로 보존되어야 함."""
    wm = np.zeros((20, 20), dtype=np.int32)
    wm[5:15, 5:15] = 1
    wm[10, 10] = 2  # isolated defect → grade 3
    idx_arr = assign_grades_by_neighbor_density(wm)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.png"
        save_palette_png(idx_arr, p, get_failmap_palette())
        assert p.exists()

        loaded = load_palette_indices(p)
        assert (loaded == 0).sum() > 0, "normal die (grade 0) present"
        assert (loaded == 3).sum() == 1, "single isolated defect → grade 3"
        assert (loaded == 31).sum() > 0, "outside-wafer background preserved"


def test_save_creates_parent_dirs():
    wm = np.array([[1, 2], [0, 0]])
    idx_arr = assign_grades_by_neighbor_density(wm)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a" / "b" / "c" / "wafer.png"
        save_palette_png(idx_arr, p, get_failmap_palette())
        assert p.exists()
