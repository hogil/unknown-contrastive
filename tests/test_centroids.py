"""Tests for common.centroids.

Runnable via:
    python -m pytest tests/test_centroids.py -v
or directly:
    python tests/test_centroids.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make the project root importable when running this file directly.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.centroids import (  # noqa: E402
    assign_to_cluster,
    load_cluster_centroids,
    save_cluster_centroids,
)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class TestSaveAndLoadRoundtrip(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        rng = np.random.default_rng(0)
        D = 128
        # Build 3 distinct clusters in a 128-dim space plus noise points
        cluster_centers = _l2(rng.standard_normal((3, D)).astype(np.float32))
        emb_list: list = []
        labels: list = []
        for cid, center in enumerate(cluster_centers):
            for _ in range(15):
                jitter = rng.standard_normal(D).astype(np.float32) * 0.05
                emb_list.append(_l2((center + jitter).reshape(1, -1))[0])
                labels.append(cid)
        # 5 noise points
        for _ in range(5):
            emb_list.append(_l2(rng.standard_normal(D).reshape(1, -1))[0])
            labels.append(-1)

        emb = np.asarray(emb_list, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)
        self.assertEqual(emb.shape[0], 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "centroids"
            save_cluster_centroids(emb, labels_arr, out_dir)

            self.assertTrue((out_dir / "centroids.npy").exists())
            self.assertTrue((out_dir / "centroids_meta.json").exists())

            pack = load_cluster_centroids(out_dir)
            centroids = pack["centroids"]
            cluster_ids = pack["cluster_ids"]
            meta = pack["meta"]

            # Shape and dtype
            self.assertEqual(centroids.shape, (3, D))
            self.assertEqual(centroids.dtype, np.float32)

            # Centroids are L2-normalized
            norms = np.linalg.norm(centroids, axis=1)
            np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)

            # Cluster ids and sizes
            self.assertEqual(cluster_ids, [0, 1, 2])
            for cid in [0, 1, 2]:
                self.assertEqual(meta["sizes"][str(cid)], 15)
                self.assertIn("p95", meta["dist_percentiles"][str(cid)])

            # Each centroid must be closest to its own cluster center
            for cid, truth in enumerate(cluster_centers):
                sims = centroids @ truth
                self.assertEqual(int(np.argmax(sims)), cid)


class TestAssignNearest(unittest.TestCase):
    def test_assign_nearest(self):
        centroids = _l2(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        cluster_ids = [0, 1]
        thresholds = np.array([0.5, 0.5], dtype=np.float32)

        queries = _l2(
            np.array([[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]], dtype=np.float32)
        )

        assigned, nn_idx, dist = assign_to_cluster(
            queries, centroids, cluster_ids, thresholds
        )
        np.testing.assert_array_equal(assigned, np.array([0, 1], dtype=np.int64))
        np.testing.assert_array_equal(nn_idx, np.array([0, 1], dtype=np.int64))
        # Both should be very close (cosine distance tiny)
        self.assertTrue((dist < 0.05).all())


class TestAssignThresholdRejects(unittest.TestCase):
    def test_assign_threshold_rejects(self):
        centroids = _l2(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        )
        cluster_ids = [0, 1]
        thresholds = np.array([0.05, 0.05], dtype=np.float32)  # strict

        # A query far from both (orthogonal-ish) — cosine distance ≈ 1.0
        far_query = _l2(np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
        assigned, nn_idx, dist = assign_to_cluster(
            far_query, centroids, cluster_ids, thresholds
        )
        self.assertEqual(int(assigned[0]), -1)
        # Nearest index is still populated even though rejected
        self.assertIn(int(nn_idx[0]), (0, 1))
        self.assertGreater(float(dist[0]), 0.05)


class TestPercentileThresholds(unittest.TestCase):
    def test_percentile_thresholds_correct(self):
        """A cluster whose members land at cosine distance 0.01..0.10 from the
        centroid should yield p95 ≈ 0.095.

        Members are constructed so their mean (hence the recomputed centroid)
        lies exactly along the first basis vector: for every member tilted
        toward ``+orth`` we include a pair tilted toward ``-orth``.
        """
        D = 16
        # Centroid: first basis vector.
        centroid = np.zeros(D, dtype=np.float32)
        centroid[0] = 1.0

        # Desired cosine distances: 0.01, 0.02, ..., 0.10. Each shows up twice
        # (once on each side of the orthogonal axis) so the mean cancels along
        # `orth` and points purely along `centroid`.
        target_d = np.arange(0.01, 0.11, 0.01, dtype=np.float64)  # 10 values
        orth = np.zeros(D, dtype=np.float32)
        orth[1] = 1.0

        members = []
        for d in target_d:
            cos_t = 1.0 - d          # dot product with centroid
            sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t * cos_t)))
            members.append(cos_t * centroid + sin_t * orth)
            members.append(cos_t * centroid - sin_t * orth)
        members = _l2(np.asarray(members, dtype=np.float32))

        labels = np.full(len(members), 7, dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "c"
            save_cluster_centroids(members, labels, out_dir)
            pack = load_cluster_centroids(out_dir)
            meta = pack["meta"]

            # Centroid should point along the true direction (within fp tol).
            self.assertEqual(pack["cluster_ids"], [7])
            recovered = pack["centroids"][0]
            self.assertGreater(float(recovered @ centroid), 0.999)

            # With the doubled distance set (each of 0.01..0.10 present twice)
            # np.percentile(..., 95) ≈ 0.10; a conservative sanity band of
            # ±0.01 keeps the check close to the informal "~0.095" target.
            p95 = meta["dist_percentiles"]["7"]["p95"]
            self.assertAlmostEqual(p95, 0.095, delta=0.01)

            # Stricter: p95 must equal np.percentile of the actual within-
            # cluster distances that `save_cluster_centroids` records.
            empirical_dists = 1.0 - members @ recovered
            expected_p95 = float(np.percentile(empirical_dists, 95))
            self.assertAlmostEqual(p95, expected_p95, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
