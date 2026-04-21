"""Round-trip tests for save_embedding_artifacts.save_embedding_artifacts.

Runnable via:
    python -m pytest tests/test_save_embedding_artifacts.py -v
or directly:
    python tests/test_save_embedding_artifacts.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from save_embedding_artifacts import save_embedding_artifacts  # noqa: E402


class TestSaveEmbeddingArtifacts(unittest.TestCase):
    def test_round_trip_matches_cluster_composite_cli_schema(self):
        rng = np.random.default_rng(0)
        n, d = 12, 8
        emb = rng.standard_normal((n, d)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = np.array([0, 0, 0, 1, 1, 1, 1, -1, 2, 2, -1, 2], dtype=np.int64)
        files = [f"/tmp/data/img_{i:02d}.png" for i in range(n)]
        kept = [0, 2]

        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            paths = save_embedding_artifacts(rd, emb, labels, files, kept)

            self.assertEqual(paths["emb"], rd / "emb.npy")
            self.assertEqual(paths["cluster_labels"], rd / "cluster_labels.npy")
            self.assertEqual(paths["files"], rd / "files.txt")
            self.assertEqual(paths["kept_labels"], rd / "kept_labels.txt")
            for p in paths.values():
                self.assertTrue(p.exists(), f"missing {p}")

            emb_r = np.load(rd / "emb.npy")
            labels_r = np.load(rd / "cluster_labels.npy")
            files_r = (rd / "files.txt").read_text(encoding="utf-8").splitlines()
            kept_r = [
                int(x)
                for x in (rd / "kept_labels.txt").read_text().split()
                if x.strip()
            ]

            np.testing.assert_allclose(emb_r, emb)
            np.testing.assert_array_equal(labels_r, labels)
            self.assertEqual(files_r, files)
            self.assertEqual(kept_r, kept)

    def test_creates_missing_run_dir(self):
        emb = np.zeros((2, 3), dtype=np.float32)
        labels = np.array([0, 0])
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td) / "does" / "not" / "exist"
            self.assertFalse(rd.exists())
            save_embedding_artifacts(rd, emb, labels, ["a.png", "b.png"], [0])
            self.assertTrue((rd / "emb.npy").exists())

    def test_shape_mismatch_raises(self):
        emb = np.zeros((3, 4), dtype=np.float32)
        labels = np.array([0, 0])  # wrong length
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                save_embedding_artifacts(
                    Path(td), emb, labels, ["a.png", "b.png", "c.png"], [0]
                )

    def test_empty_kept_labels_produces_empty_file(self):
        emb = np.zeros((1, 2), dtype=np.float32)
        labels = np.array([-1])
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            save_embedding_artifacts(rd, emb, labels, ["x.png"], [])
            self.assertEqual((rd / "kept_labels.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
