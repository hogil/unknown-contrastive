"""Unit tests for ``data_prep.wm811k_to_palette`` (random-skewed grade mode).

Runnable via::

    python -m pytest tests/test_wm811k_convert.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable when running this file directly.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_prep.wm811k_to_palette import (  # noqa: E402
    assign_grades_random_skewed,
    assign_grades_textured,
    save_palette_png,
    split_indices_disjoint,
    split_indices_multi,
    _validate_config,
)
from common.palette_io import (  # noqa: E402
    get_failmap_palette,
    load_palette_indices,
)


# ---------------------------------------------------------------------------
# Deterministic mapping rules
# ---------------------------------------------------------------------------


def test_outside_deterministically_maps_to_31():
    wm = np.array([[0, 0, 0], [0, 0, 0]])
    rng = np.random.default_rng(0)
    out = assign_grades_random_skewed(wm, rng)
    assert out.dtype == np.uint8
    assert (out == 31).all()


def test_normal_deterministically_maps_to_0():
    wm = np.array([[1, 1, 1], [1, 1, 1]])
    rng = np.random.default_rng(0)
    out = assign_grades_random_skewed(wm, rng)
    assert (out == 0).all()


def test_defect_maps_into_grade_range():
    wm = np.full((20, 20), 2, dtype=np.int32)
    rng = np.random.default_rng(0)
    out = assign_grades_random_skewed(wm, rng)
    grades = np.unique(out)
    assert grades.min() >= 1 and grades.max() <= 7


def test_mixed_regions_preserved():
    wm = np.array([
        [0, 1, 2],
        [2, 1, 0],
    ])
    rng = np.random.default_rng(42)
    out = assign_grades_random_skewed(wm, rng)
    assert out[0, 0] == 31
    assert out[1, 2] == 31
    assert out[0, 1] == 0
    assert out[1, 1] == 0
    assert 1 <= out[0, 2] <= 7
    assert 1 <= out[1, 0] <= 7


def test_custom_grades_and_weights():
    """Custom grade list / weights are honoured."""
    wm = np.full((10, 10), 2, dtype=np.int32)
    rng = np.random.default_rng(0)
    out = assign_grades_random_skewed(
        wm, rng, grades=[3, 5], weights=[1, 0]
    )
    assert (out == 3).all(), "weight 0 for grade 5 must never be picked"


def test_rejects_malformed_wafer():
    with pytest.raises(ValueError):
        assign_grades_random_skewed(np.zeros((2, 2, 2)), np.random.default_rng(0))


def test_rejects_mismatched_weights():
    wm = np.full((2, 2), 2, dtype=np.int32)
    with pytest.raises(ValueError):
        assign_grades_random_skewed(
            wm, np.random.default_rng(0), grades=[1, 2, 3], weights=[1, 1]
        )


def test_rejects_negative_weights():
    wm = np.full((2, 2), 2, dtype=np.int32)
    with pytest.raises(ValueError):
        assign_grades_random_skewed(
            wm, np.random.default_rng(0), grades=[1], weights=[-1]
        )


# ---------------------------------------------------------------------------
# Reproducibility + statistical weight honouring
# ---------------------------------------------------------------------------


def test_seed_fixed_is_reproducible():
    wm = np.full((50, 50), 2, dtype=np.int32)
    a = assign_grades_random_skewed(wm, np.random.default_rng(123))
    b = assign_grades_random_skewed(wm, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_weight_distribution_matches_exponential_decay():
    """Large sample should reproduce the exponential weights within +-2%."""
    n = 200_000
    wm = np.full((1, n), 2, dtype=np.int32)
    rng = np.random.default_rng(2026)
    out = assign_grades_random_skewed(wm, rng)

    weights = np.array([1.0 / (2 ** k) for k in range(7)])  # exponential
    expected = weights / weights.sum()

    counts = np.array([(out == g).sum() for g in range(1, 8)])
    observed = counts / counts.sum()

    for g_idx, (obs, exp) in enumerate(zip(observed, expected)):
        assert abs(obs - exp) < 0.02, (
            f"grade {g_idx + 1}: observed {obs:.4f}, expected {exp:.4f}"
        )


# ---------------------------------------------------------------------------
# Train / val disjoint split
# ---------------------------------------------------------------------------


def test_split_is_disjoint():
    rng = np.random.default_rng(0)
    train, val = split_indices_disjoint(rng, n_have=100, n_train=20, n_val=30)
    assert len(train) == 20 and len(val) == 30
    assert len(set(train.tolist()) & set(val.tolist())) == 0


def test_split_raises_if_insufficient_pool():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        split_indices_disjoke = split_indices_disjoint(rng, n_have=10, n_train=8, n_val=5)


def test_split_seed_determines_permutation():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    t1, v1 = split_indices_disjoint(rng1, n_have=200, n_train=50, n_val=50)
    t2, v2 = split_indices_disjoint(rng2, n_have=200, n_train=50, n_val=50)
    assert np.array_equal(t1, t2)
    assert np.array_equal(v1, v2)


def test_split_multi_three_way_disjoint():
    rng = np.random.default_rng(7)
    out = split_indices_multi(
        rng,
        n_have=200,
        splits={"train": 50, "val": 30, "test": 30},
    )
    assert set(out.keys()) == {"train", "val", "test"}
    assert len(out["train"]) == 50
    assert len(out["val"]) == 30
    assert len(out["test"]) == 30
    # Pairwise disjoint
    tr, va, te = set(out["train"].tolist()), set(out["val"].tolist()), set(out["test"].tolist())
    assert not (tr & va) and not (tr & te) and not (va & te)
    # All from valid index space
    combined = tr | va | te
    assert combined.issubset(set(range(200)))


def test_split_multi_raises_when_insufficient():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        split_indices_multi(rng, n_have=10, splits={"train": 5, "val": 4, "test": 3})


# ---------------------------------------------------------------------------
# YAML config validation
# ---------------------------------------------------------------------------


_MINIMAL_CFG = {
    "version": 1,
    "seed": 42,
    "size": [4000, 4000],
    "upscale": "nearest",
    "mapping": {
        "outside": 31,
        "normal": 0,
        "defect": {
            "mode": "random_skewed",
            "grades": [1, 2, 3, 4, 5, 6, 7],
            "weights": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.007812],
        },
    },
    "split": {
        "train": {"defect_n_per_class": 50, "normal_n": 1600, "total": 2000},
        "val": {"defect_n_per_class": 20, "normal_n": 640, "total": 800},
        "test": {"defect_n_per_class": 20, "normal_n": 640, "total": 800},
    },
    "paths": {
        "pkl": "x",
        "train_out": "y",
        "val_out": "z",
        "test_out": "w",
    },
}


def test_validate_config_ok_on_minimal():
    _validate_config(dict(_MINIMAL_CFG))  # shallow copy ok


def test_validate_config_rejects_non_nearest_upscale():
    cfg = {**_MINIMAL_CFG, "upscale": "bilinear"}
    with pytest.raises(ValueError):
        _validate_config(cfg)


def test_validate_config_rejects_unknown_defect_mode():
    cfg = {**_MINIMAL_CFG}
    cfg["mapping"] = dict(cfg["mapping"])
    cfg["mapping"]["defect"] = {
        "mode": "fixed",
        "grades": [1],
        "weights": [1.0],
    }
    with pytest.raises(ValueError):
        _validate_config(cfg)


def test_validate_config_rejects_missing_keys():
    cfg = {**_MINIMAL_CFG}
    cfg.pop("seed")
    with pytest.raises(ValueError):
        _validate_config(cfg)


# ---------------------------------------------------------------------------
# File I/O + NEAREST resize preserves indices
# ---------------------------------------------------------------------------


def test_save_and_roundtrip_preserves_indices():
    wm = np.zeros((20, 20), dtype=np.int32)
    wm[5:15, 5:15] = 1
    wm[10, 10] = 2
    idx_arr = assign_grades_random_skewed(wm, np.random.default_rng(7))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "out.png"
        save_palette_png(idx_arr, p, get_failmap_palette())
        assert p.exists()
        loaded = load_palette_indices(p)
        assert (loaded == 31).sum() > 0
        assert (loaded == 0).sum() > 0
        # defect exactly 1 pixel, grade in 1..7
        defect_vals = loaded[10, 10]
        assert 1 <= int(defect_vals) <= 7


def test_nearest_resize_does_not_introduce_new_indices():
    wm = np.zeros((10, 10), dtype=np.int32)
    wm[3:7, 3:7] = 1
    wm[5, 5] = 2
    idx_arr = assign_grades_random_skewed(wm, np.random.default_rng(9))
    original_indices = set(int(v) for v in np.unique(idx_arr))

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "out.png"
        save_palette_png(
            idx_arr,
            p,
            get_failmap_palette(),
            target_size=(80, 80),
        )
        loaded = load_palette_indices(p)
        after = set(int(v) for v in np.unique(loaded))
        assert after.issubset(original_indices), (
            f"NEAREST introduced new indices: {after - original_indices}"
        )


def test_accepts_list_of_list():
    wm = [[0, 1, 2], [2, 1, 0]]
    out = assign_grades_random_skewed(wm, np.random.default_rng(0))
    assert out.shape == (2, 3)
    assert out.dtype == np.uint8
    assert out[0, 0] == 31 and out[1, 2] == 31
    assert out[0, 1] == 0 and out[1, 1] == 0


# ---------------------------------------------------------------------------
# Textured mode (option C)
# ---------------------------------------------------------------------------


def _simple_wafer(h: int = 10, w: int = 10) -> np.ndarray:
    """Disk-shaped wafer with defect dies scattered inside."""
    wm = np.zeros((h, w), dtype=np.int32)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= (min(h, w) // 2) ** 2
    wm[inside] = 1
    wm[cy, cx] = 2
    wm[cy + 1, cx + 1] = 2
    wm[cy - 1, cx - 1] = 2
    return wm


def test_textured_preserves_outside_mask():
    wm = _simple_wafer()
    rng = np.random.default_rng(0)
    out = assign_grades_textured(wm, rng, target_size=(100, 100))
    # Outside in original -> must remain 31 in upscaled output at the same
    # die block.
    assert out.shape == (100, 100)
    # Sample an outside corner pixel (original (0, 0) was outside)
    assert out[0, 0] == 31


def test_textured_shape_matches_target():
    wm = _simple_wafer()
    rng = np.random.default_rng(0)
    out = assign_grades_textured(wm, rng, target_size=(200, 150))
    assert out.shape == (200, 150)


def test_textured_grade_range():
    wm = _simple_wafer()
    rng = np.random.default_rng(0)
    out = assign_grades_textured(wm, rng, target_size=(80, 80))
    uniq = set(int(v) for v in np.unique(out))
    # All values must be within the allowed palette set
    assert uniq.issubset({0, 1, 2, 3, 4, 5, 6, 7, 31})


def test_textured_defect_perturb_zero_gives_uniform_die():
    """With defect_perturb_p=0, each defect die block holds one grade."""
    wm = _simple_wafer(h=8, w=8)
    rng = np.random.default_rng(11)
    out = assign_grades_textured(
        wm, rng, target_size=(80, 80),
        defect_perturb_p=0.0,
        normal_scatter_p=0.0,
    )
    # Die at (4, 4) covers pixels rows 40:50, cols 40:50 (8 scales x10).
    block = out[40:50, 40:50]
    assert block.min() == block.max(), (
        f"defect die block expected uniform, got range "
        f"[{block.min()}, {block.max()}]"
    )
    assert 1 <= int(block[0, 0]) <= 7


def test_textured_normal_scatter_introduces_grade_1():
    wm = np.ones((10, 10), dtype=np.int32)  # all-normal wafer
    rng = np.random.default_rng(3)
    out = assign_grades_textured(
        wm, rng, target_size=(100, 100),
        defect_perturb_p=0.0,
        normal_scatter_p=0.2,
        normal_scatter_grades=(1,),
    )
    uniq, counts = np.unique(out, return_counts=True)
    d = dict(zip(uniq.tolist(), counts.tolist()))
    # Expect both 0 and 1 present
    assert 0 in d and 1 in d
    # Fraction of 1's should be approximately 20% (+-5%).
    frac = d[1] / (d[0] + d[1])
    assert 0.15 <= frac <= 0.25, f"normal scatter ratio {frac} out of tolerance"


def test_textured_defect_perturb_introduces_neighbor_grades():
    # Single defect die in a 1x1 wafer -> entire output has the same base
    # grade, perturbation should introduce +-1 / +-2 neighbors.
    wm = np.array([[2]], dtype=np.int32)
    rng = np.random.default_rng(5)
    out = assign_grades_textured(
        wm, rng, target_size=(200, 200),
        defect_perturb_p=0.5,
    )
    unique = np.unique(out)
    # At least two grades present (base + at least one perturbed)
    assert len(unique) >= 2


def test_textured_seed_reproducible():
    wm = _simple_wafer()
    a = assign_grades_textured(wm, np.random.default_rng(77), target_size=(60, 60))
    b = assign_grades_textured(wm, np.random.default_rng(77), target_size=(60, 60))
    assert np.array_equal(a, b)


def test_validate_config_accepts_texture_section():
    cfg = {
        **_MINIMAL_CFG,
        "texture": {
            "mode": "synthetic_scatter",
            "defect_perturb_p": 0.3,
            "normal_scatter_p": 0.03,
            "global_scatter_n": 0,
        },
    }
    _validate_config(cfg)


def test_validate_config_rejects_unknown_texture_mode():
    cfg = {
        **_MINIMAL_CFG,
        "texture": {"mode": "swirl", "defect_perturb_p": 0.1, "normal_scatter_p": 0.1},
    }
    with pytest.raises(ValueError):
        _validate_config(cfg)


def test_validate_config_rejects_out_of_range_probability():
    cfg = {
        **_MINIMAL_CFG,
        "texture": {"mode": "none", "defect_perturb_p": 1.5, "normal_scatter_p": 0.1},
    }
    with pytest.raises(ValueError):
        _validate_config(cfg)
