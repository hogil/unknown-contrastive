"""Unit tests for data_prep.wm811k_to_palette.

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

from data_prep.wm811k_to_palette import wm_to_palette, save_palette_png  # noqa: E402
from common.palette_io import get_default_palette, load_palette_indices  # noqa: E402


def test_wm_to_palette_mapping():
    wm = np.array([[0, 1, 2], [2, 1, 0]])
    out = wm_to_palette(wm)
    assert out.dtype == np.uint8
    assert out.shape == (2, 3)
    assert out[0, 0] == 31  # outside -> transparent
    assert out[0, 1] == 0   # normal die
    assert out[0, 2] == 7   # defect die
    assert out[1, 0] == 7
    assert out[1, 1] == 0
    assert out[1, 2] == 31


def test_wm_to_palette_accepts_list_of_list():
    """pkl waferMap cells can be list-of-list; np.asarray must handle it."""
    wm = [[0, 1, 2], [2, 1, 0]]
    out = wm_to_palette(wm)
    assert out.dtype == np.uint8
    assert out.shape == (2, 3)
    assert (out == np.array([[31, 0, 7], [7, 0, 31]], dtype=np.uint8)).all()


def test_save_and_roundtrip():
    wm = np.zeros((20, 20), dtype=np.int32)
    wm[5:15, 5:15] = 1
    wm[10, 10] = 2
    idx_arr = wm_to_palette(wm)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.png"
        save_palette_png(idx_arr, p, get_default_palette())
        assert p.exists()

        loaded = load_palette_indices(p)
        # Inside-wafer normal die should round-trip to grade 0.
        assert (loaded == 0).sum() > 0
        # Single defect die should round-trip to grade 7.
        assert (loaded == 7).sum() == 1
        # Outside-wafer background should stay at 31 (via alpha → 31 rule).
        assert (loaded == 31).sum() > 0


def test_save_creates_parent_dirs():
    wm = np.array([[1, 2], [0, 0]])
    idx_arr = wm_to_palette(wm)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "a" / "b" / "c" / "wafer.png"
        save_palette_png(idx_arr, p, get_default_palette())
        assert p.exists()
