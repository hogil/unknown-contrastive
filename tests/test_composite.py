"""Tests for common.composite and common.palette_io.

Runnable via:
    python -m pytest tests/test_composite.py -v
or directly:
    python tests/test_composite.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

# Make the project root importable when running this file directly.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.composite import (  # noqa: E402
    compute_base_indices,
    compute_grade_counts,
    compute_square_mean,
    render_composite_png,
)
from common.palette_io import (  # noqa: E402
    get_default_palette,
    stack_and_normalize,
)


def _write_palette_png(path: Path, index_arr: np.ndarray) -> None:
    """Write an (H, W) uint8 index array as a palette-mode PNG."""
    if index_arr.dtype != np.uint8:
        index_arr = index_arr.astype(np.uint8)
    img = Image.fromarray(index_arr, mode="P")
    img.putpalette(get_default_palette())
    img.save(path, format="PNG")


class TestComputeGradeCounts(unittest.TestCase):
    def test_compute_grade_counts_shape_and_values(self):
        # (N=3, H=4, W=4) with known contents — pixel (0,0) has grades [0, 1, 2]
        stacked = np.array(
            [
                [  # image 0
                    [0, 1, 2, 3],
                    [0, 0, 0, 0],
                    [5, 5, 5, 5],
                    [13, 13, 13, 13],
                ],
                [  # image 1
                    [1, 1, 1, 1],
                    [0, 0, 0, 0],
                    [5, 6, 7, 8],
                    [13, 0, 0, 0],
                ],
                [  # image 2
                    [2, 2, 2, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [13, 13, 13, 13],
                ],
            ],
            dtype=np.uint8,
        )

        counts = compute_grade_counts(stacked)
        self.assertEqual(counts.shape, (14, 4, 4))
        self.assertEqual(counts.dtype, np.uint32)

        # Pixel (0,0) had values [0, 1, 2] -> one of each
        self.assertEqual(counts[0, 0, 0], 1)
        self.assertEqual(counts[1, 0, 0], 1)
        self.assertEqual(counts[2, 0, 0], 1)

        # Row 1 is all zeros across all images
        self.assertEqual(counts[0, 1, 0], 3)
        self.assertEqual(counts[0, 1, 3], 3)

        # Pixel (3,0): three images all 13
        self.assertEqual(counts[13, 3, 0], 3)
        # Pixel (3,1): images had [13, 0, 13]
        self.assertEqual(counts[13, 3, 1], 2)
        self.assertEqual(counts[0, 3, 1], 1)

        # Sum over grade axis == N everywhere
        np.testing.assert_array_equal(counts.sum(axis=0), np.full((4, 4), 3, np.uint32))


class TestSquareMean(unittest.TestCase):
    def test_square_mean_known_input(self):
        # N=1, single image [[0,1],[2,3]] -> square_mean == [[0,1],[4,9]]
        stacked = np.array([[[0, 1], [2, 3]]], dtype=np.uint8)
        counts = compute_grade_counts(stacked)
        sm = compute_square_mean(counts, image_count=1)
        expected = np.array([[0.0, 1.0], [4.0, 9.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(sm, expected)
        self.assertEqual(sm.dtype, np.float32)

    def test_square_mean_divides_by_image_count(self):
        # N=2: both images grade 3 everywhere -> count[3]=2; square_mean = 2*9/2 = 9
        stacked = np.full((2, 3, 3), 3, dtype=np.uint8)
        counts = compute_grade_counts(stacked)
        sm = compute_square_mean(counts, image_count=2)
        np.testing.assert_array_almost_equal(sm, np.full((3, 3), 9.0, np.float32))


class TestInvalidMaskPropagation(unittest.TestCase):
    def test_invalid_mask_propagation(self):
        """Pixels where every input image is 14+ must end up invalid_mask=True
        and square_mean==0 there."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            # Two images, both have pixel (0,0) == 15 (invalid) and (0,1) == 3 (valid)
            arr_a = np.zeros((2, 2), dtype=np.uint8)
            arr_b = np.zeros((2, 2), dtype=np.uint8)
            arr_a[0, 0] = 15
            arr_b[0, 0] = 20
            arr_a[0, 1] = 3
            arr_b[0, 1] = 3

            path_a = tmpdir_p / "a.png"
            path_b = tmpdir_p / "b.png"
            _write_palette_png(path_a, arr_a)
            _write_palette_png(path_b, arr_b)

            stacked, invalid_mask = stack_and_normalize([path_a, path_b])

            # invalid pixel must be marked
            self.assertTrue(bool(invalid_mask[0, 0]))
            # valid pixel must not be marked
            self.assertFalse(bool(invalid_mask[0, 1]))

            counts = compute_grade_counts(stacked)
            sm = compute_square_mean(counts, image_count=stacked.shape[0], invalid_mask=invalid_mask)

            self.assertEqual(float(sm[0, 0]), 0.0)
            # (0,1) was 3 in both images -> count[3] = 2 -> sm = 2*9/2 = 9
            self.assertAlmostEqual(float(sm[0, 1]), 9.0, places=5)


class TestRenderComposite(unittest.TestCase):
    def test_render_composite_produces_png(self):
        """Synthetic palette PNGs → render_composite_png writes a valid RGB PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            h, w = 8, 12

            images: list = []
            rng = np.random.default_rng(42)
            for i in range(3):
                # mostly zeros with a few scattered defects
                arr = np.zeros((h, w), dtype=np.uint8)
                # Give each image a unique defect patch
                arr[2:5, 2 + i : 5 + i] = i + 1
                # a low-grade region
                arr[6, :] = 10
                # a bit of noise
                noise_idx = rng.integers(0, h * w, size=4)
                for ni in noise_idx:
                    arr[ni // w, ni % w] = rng.integers(1, 8)

                p = tmpdir_p / f"img_{i}.png"
                _write_palette_png(p, arr)
                images.append(p)

            stacked, invalid_mask = stack_and_normalize(images)
            self.assertEqual(stacked.shape, (3, h, w))

            out_path = tmpdir_p / "composite.png"
            info = render_composite_png(stacked, out_path, title="cluster0")

            # File exists
            self.assertTrue(out_path.exists())
            self.assertEqual(info["path"], str(out_path))
            self.assertEqual(info["image_count"], 3)
            self.assertEqual(info["width"], w)
            self.assertEqual(info["height"], h)
            self.assertIn("square_mean_stats", info)
            for key in ("min", "max", "p95"):
                self.assertIn(key, info["square_mean_stats"])

            # Open and check mode / size
            with Image.open(out_path) as rendered:
                self.assertEqual(rendered.mode, "RGB")
                self.assertEqual(rendered.size, (w, h))


class TestStackAndNormalize(unittest.TestCase):
    def test_stack_and_normalize_invalid_to_zero(self):
        """14+ input values must not survive in the output stacked array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)

            arr0 = np.array(
                [
                    [0, 1, 2, 3],
                    [14, 15, 20, 25],  # all invalid in this row for image 0
                    [5, 6, 7, 8],
                ],
                dtype=np.uint8,
            )
            arr1 = np.array(
                [
                    [1, 1, 1, 1],
                    [14, 14, 14, 14],  # also invalid
                    [9, 10, 11, 12],
                ],
                dtype=np.uint8,
            )
            p0 = tmpdir_p / "a.png"
            p1 = tmpdir_p / "b.png"
            _write_palette_png(p0, arr0)
            _write_palette_png(p1, arr1)

            stacked, invalid_mask = stack_and_normalize([p0, p1])

            # No value should exceed 13
            self.assertTrue(stacked.max() <= 13)
            self.assertEqual(stacked.dtype, np.uint8)

            # Row 1 was fully invalid in both images -> invalid_mask True
            self.assertTrue(invalid_mask[1].all())
            # Other rows should not be flagged as invalid
            self.assertFalse(invalid_mask[0].any())
            self.assertFalse(invalid_mask[2].any())

    def test_stack_and_normalize_8_13_only_rule(self):
        """A pixel that only sees grades 8..13 (no 0..7, no invalid) is forced to 8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            arr0 = np.full((2, 2), 10, dtype=np.uint8)
            arr1 = np.full((2, 2), 12, dtype=np.uint8)
            p0 = tmpdir_p / "a.png"
            p1 = tmpdir_p / "b.png"
            _write_palette_png(p0, arr0)
            _write_palette_png(p1, arr1)

            stacked, invalid_mask = stack_and_normalize([p0, p1])

            # All pixels should have been clamped to 8 across all images
            self.assertTrue((stacked == 8).all())
            self.assertFalse(invalid_mask.any())


if __name__ == "__main__":
    unittest.main()
