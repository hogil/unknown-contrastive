"""Tests for cluster_composite.generate_cluster_composites.

Runnable via:
    python -m pytest tests/test_cluster_composite.py -v
or directly:
    python tests/test_cluster_composite.py
"""

from __future__ import annotations

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

from cluster_composite import generate_cluster_composites  # noqa: E402
from common.palette_io import get_default_palette  # noqa: E402


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _write_palette_png(path: Path, index_arr: np.ndarray) -> None:
    if index_arr.dtype != np.uint8:
        index_arr = index_arr.astype(np.uint8)
    img = Image.fromarray(index_arr, mode="P")
    img.putpalette(get_default_palette())
    img.save(path, format="PNG")


class TestGenerateClusterComposites(unittest.TestCase):
    def test_generates_png_for_each_kept_cluster(self):
        rng = np.random.default_rng(3)
        D = 16
        # Two "clusters" around distinct unit-vector directions.
        dir_a = np.zeros(D, dtype=np.float32); dir_a[0] = 1.0
        dir_b = np.zeros(D, dtype=np.float32); dir_b[1] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            run_dir = tmpdir_p / "run"
            run_dir.mkdir()

            paths = []
            emb_list = []
            labels = []
            # 6 members in cluster 0, 5 in cluster 1
            for i in range(6):
                arr = np.zeros((8, 10), dtype=np.uint8)
                arr[2:5, 2:5] = (i % 5) + 1
                p = run_dir / f"a_{i}.png"
                _write_palette_png(p, arr)
                paths.append(str(p))
                emb_list.append(_l2(dir_a + rng.standard_normal(D).astype(np.float32) * 0.03))
                labels.append(0)
            for i in range(5):
                arr = np.zeros((8, 10), dtype=np.uint8)
                arr[4:7, 4:8] = (i % 4) + 1
                p = run_dir / f"b_{i}.png"
                _write_palette_png(p, arr)
                paths.append(str(p))
                emb_list.append(_l2(dir_b + rng.standard_normal(D).astype(np.float32) * 0.03))
                labels.append(1)

            emb = np.asarray(emb_list, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int64)

            out_dir = run_dir / "cluster_summary" / "composite"
            generated = generate_cluster_composites(
                run_dir=run_dir,
                emb=emb,
                files=paths,
                cluster_labels=labels,
                kept_labels=[0, 1],
                out_dir=out_dir,
                n_per_cluster=4,
                logger=None,
            )

            self.assertEqual(len(generated), 2)
            for gp in generated:
                self.assertTrue(gp.exists())
                with Image.open(gp) as im:
                    self.assertEqual(im.mode, "RGB")
                    self.assertEqual(im.size, (10, 8))

            # Expected filenames
            names = sorted(p.name for p in generated)
            self.assertEqual(
                names,
                ["cluster_000_composite.png", "cluster_001_composite.png"],
            )

    def test_empty_kept_labels_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            emb = np.zeros((3, 4), dtype=np.float32)
            labels = np.array([-1, -1, -1])
            generated = generate_cluster_composites(
                run_dir=tmpdir_p,
                emb=emb,
                files=["a.png", "b.png", "c.png"],
                cluster_labels=labels,
                kept_labels=[],
                out_dir=tmpdir_p / "out",
                n_per_cluster=5,
                logger=None,
            )
            self.assertEqual(generated, [])

    def test_kept_cluster_with_no_members_is_skipped(self):
        rng = np.random.default_rng(7)
        D = 8

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            # Cluster 0 exists; cluster 99 is in kept_labels but has no members.
            arr = np.zeros((6, 6), dtype=np.uint8)
            arr[1:3, 1:3] = 4
            p = tmpdir_p / "only.png"
            _write_palette_png(p, arr)

            emb = _l2(rng.standard_normal((3, D)).astype(np.float32))
            labels = np.array([0, 0, 0], dtype=np.int64)
            out_dir = tmpdir_p / "out"

            generated = generate_cluster_composites(
                run_dir=tmpdir_p,
                emb=emb,
                files=[str(p), str(p), str(p)],
                cluster_labels=labels,
                kept_labels=[0, 99],
                out_dir=out_dir,
                n_per_cluster=3,
                logger=None,
            )

            # Only cluster 0 should yield a composite.
            self.assertEqual(len(generated), 1)
            self.assertEqual(generated[0].name, "cluster_000_composite.png")


if __name__ == "__main__":
    unittest.main()
