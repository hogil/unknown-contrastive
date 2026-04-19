#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Synthetic sanity tests for experiments.eval_metrics."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.eval_metrics import (  # noqa: E402
    compute_cluster_metrics,
    compute_subset_selection,
)


class TestClusterMetrics(unittest.TestCase):
    def test_perfect_clustering(self):
        # 3 classes, 10 samples each, cluster ids == gt labels
        gt = np.repeat(np.arange(3), 10)
        cl = gt.copy()
        m = compute_cluster_metrics(cl, gt)
        self.assertEqual(m["n_samples"], 30)
        self.assertEqual(m["n_clusters_found"], 3)
        self.assertEqual(m["n_noise"], 0)
        self.assertAlmostEqual(m["ari"], 1.0, places=6)
        self.assertAlmostEqual(m["nmi"], 1.0, places=6)
        self.assertAlmostEqual(m["cluster_purity"], 1.0, places=6)

    def test_all_noise(self):
        gt = np.array([0, 1, 2, 0, 1, 2])
        cl = np.full_like(gt, -1)
        m = compute_cluster_metrics(cl, gt)
        self.assertEqual(m["n_noise"], 6)
        self.assertEqual(m["n_clusters_found"], 0)
        self.assertEqual(m["cluster_purity"], 0.0)
        self.assertEqual(m["noise_ratio"], 1.0)

    def test_with_noise_mixed(self):
        # 2 clusters perfectly matched + 2 noise points
        gt = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        cl = np.array([0, 0, 0, 1, 1, 1, -1, -1])
        m = compute_cluster_metrics(cl, gt)
        self.assertEqual(m["n_noise"], 2)
        self.assertEqual(m["n_clusters_found"], 2)
        self.assertAlmostEqual(m["cluster_purity"], 1.0, places=6)
        self.assertAlmostEqual(m["ari"], 1.0, places=6)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            compute_cluster_metrics(np.array([0, 1]), np.array([0]))


class TestSubsetSelection(unittest.TestCase):
    def _make_fake_dataset(self, root: Path):
        # Create 5 classes with varying sizes: 300, 250, 150, 50, 20
        sizes = {"A": 300, "B": 250, "C": 150, "D": 50, "E": 20}
        for cls, n in sizes.items():
            d = root / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"img_{i:04d}.png").write_bytes(b"\x89PNG\r\n")
        return sizes

    def test_largest_selection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_fake_dataset(root)
            items = compute_subset_selection(
                root, n_classes=3, n_per_class=100,
                selection="largest", seed=0,
            )
            classes_used = sorted({c for _, c in items})
            self.assertEqual(classes_used, ["A", "B", "C"])
            # 100 each from A,B,C
            self.assertEqual(len(items), 300)

    def test_capped_per_class(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_fake_dataset(root)
            items = compute_subset_selection(
                root, n_classes=5, n_per_class=200,
                selection="largest", seed=0,
            )
            # A:200 B:200 C:150 D:50 E:20 = 620
            self.assertEqual(len(items), 200 + 200 + 150 + 50 + 20)

    def test_explicit_classes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_fake_dataset(root)
            items = compute_subset_selection(
                root, n_per_class=10,
                explicit_classes=["D", "B"],
                seed=0,
            )
            classes_used = sorted({c for _, c in items})
            self.assertEqual(classes_used, ["B", "D"])
            self.assertEqual(len(items), 20)

    def test_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_fake_dataset(root)
            a = compute_subset_selection(
                root, n_classes=3, n_per_class=50, seed=123,
            )
            b = compute_subset_selection(
                root, n_classes=3, n_per_class=50, seed=123,
            )
            self.assertEqual(
                [(str(p), c) for p, c in a],
                [(str(p), c) for p, c in b],
            )


if __name__ == "__main__":
    unittest.main()
