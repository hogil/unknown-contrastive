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
    compute_silhouette_cosine,
    compute_subset_selection,
)
from experiments.eval_val import (  # noqa: E402
    read_kept_cluster_count,
    resolve_output_path,
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


class TestSilhouetteCosine(unittest.TestCase):
    def test_perfect_separation_gives_positive_score(self):
        rng = np.random.default_rng(0)
        # Two tight clusters far apart on unit sphere
        a = np.array([1.0, 0.0]) + rng.normal(scale=0.01, size=(30, 2))
        b = np.array([-1.0, 0.0]) + rng.normal(scale=0.01, size=(30, 2))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        emb = np.vstack([a, b])
        lbl = np.array([0] * 30 + [1] * 30)
        out = compute_silhouette_cosine(emb, lbl, sample_size=None)
        self.assertIsNotNone(out["silhouette"])
        self.assertGreater(out["silhouette"], 0.8)
        self.assertEqual(out["n_clusters"], 2)

    def test_noise_is_excluded(self):
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(20, 4))
        lbl = np.array([0] * 8 + [1] * 8 + [-1] * 4)
        out = compute_silhouette_cosine(emb, lbl, sample_size=None)
        self.assertEqual(out["n_used"], 16, "noise must not be scored")

    def test_single_cluster_returns_none(self):
        emb = np.random.default_rng(2).normal(size=(10, 3))
        lbl = np.zeros(10, dtype=int)
        out = compute_silhouette_cosine(emb, lbl, sample_size=None)
        self.assertIsNone(out["silhouette"])

    def test_subsample_is_applied(self):
        rng = np.random.default_rng(3)
        emb = rng.normal(size=(10_000, 8))
        lbl = rng.integers(0, 5, size=10_000)
        out = compute_silhouette_cosine(emb, lbl, sample_size=500, seed=0)
        self.assertLessEqual(out["n_used"], 500)
        self.assertIsNotNone(out["silhouette"])


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


class TestResolveOutputPath(unittest.TestCase):
    def test_default_uses_run_dir(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            p = resolve_output_path(rd, None)
            self.assertEqual(p, rd / "eval_summary.json")

    def test_explicit_file_path(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            target = rd / "eval_summary_test.json"
            p = resolve_output_path(rd, target)
            self.assertEqual(p, target)

    def test_existing_directory_gets_default_filename(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            sub = rd / "eval_out"
            sub.mkdir()
            p = resolve_output_path(rd, sub)
            self.assertEqual(p, sub / "eval_summary.json")

    def test_trailing_separator_treated_as_dir(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            # Directory doesn't exist yet, but trailing sep signals intent.
            raw = str(rd / "not_yet") + os.sep
            p = resolve_output_path(rd, raw)
            self.assertEqual(p.name, "eval_summary.json")
            self.assertEqual(p.parent, rd / "not_yet")

    def test_val_test_split_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            rd = Path(td)
            val = resolve_output_path(rd, rd / "eval_summary_val.json")
            tst = resolve_output_path(rd, rd / "eval_summary_test.json")
            self.assertNotEqual(val, tst)
            default = resolve_output_path(rd, None)
            self.assertNotEqual(val, default)
            self.assertNotEqual(tst, default)


class TestReadKeptClusterCount(unittest.TestCase):
    def test_missing_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(read_kept_cluster_count(Path(td) / "nope.json"), 0)

    def test_populated_cluster_ids(self):
        import json as _json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "centroids_meta.json"
            p.write_text(_json.dumps({"cluster_ids": [0, 3, 7]}))
            self.assertEqual(read_kept_cluster_count(p), 3)

    def test_empty_cluster_ids(self):
        import json as _json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "centroids_meta.json"
            p.write_text(_json.dumps({"cluster_ids": []}))
            self.assertEqual(read_kept_cluster_count(p), 0)

    def test_malformed_json_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "centroids_meta.json"
            p.write_text("{not valid json")
            self.assertEqual(read_kept_cluster_count(p), 0)

    def test_missing_key_returns_zero(self):
        import json as _json
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "centroids_meta.json"
            p.write_text(_json.dumps({"sizes": {}}))
            self.assertEqual(read_kept_cluster_count(p), 0)


class TestEvalSummaryMetaFields(unittest.TestCase):
    """Sanity-check that the real eval_summary files carry the new meta keys."""

    REQUIRED = {
        "cluster_assignment_source",
        "approximate_predict_used",
        "filter_cascaded",
        "kept_cluster_count",
    }

    def test_meta_fields_present_if_file_exists(self):
        import json as _json
        candidates = [
            _ROOT / "outputs_cpu_ep10" / "baseline_260420_201121" / "eval_summary_val.json",
            _ROOT / "outputs_cpu_ep10" / "baseline_260420_201121" / "eval_summary_test.json",
            _ROOT / "outputs_cpu_smoke" / "baseline_260420_181745" / "eval_summary_test.json",
        ]
        checked = False
        for p in candidates:
            if not p.exists():
                continue
            data = _json.loads(p.read_text(encoding="utf-8"))
            missing = self.REQUIRED - set(data.keys())
            # Legacy files (from before this commit) may not have the fields yet;
            # only assert on files that already carry at least one of them,
            # otherwise skip so the test doesn't spuriously fail on old runs.
            if self.REQUIRED & set(data.keys()):
                self.assertFalse(
                    missing,
                    f"{p.name} missing meta keys: {missing}",
                )
                checked = True
        if not checked:
            self.skipTest("no eval_summary with new meta fields present yet")


if __name__ == "__main__":
    unittest.main()
