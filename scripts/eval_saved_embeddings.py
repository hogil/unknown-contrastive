#!/usr/bin/env python3
"""Evaluate an existing embedding .npy with the project open-set metrics.

This is intentionally eval-only. It does not extract images or load a model.
Use it to compare checkpoints after their embeddings have already been saved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    v_measure_score,
)

from eval_open_set_embeddings import (
    class_distance_metrics,
    knn_same_rates,
    l2_normalize,
    merge_clusters_by_centroid,
    purity_score,
    reassign_noise_to_nearest_cluster,
    to_same_dim,
)
from cluster_metrics import capture_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", required=True, help="Saved embedding .npy")
    p.add_argument("--paths-json", required=True, help="JSON with paths/labels, or list of paths")
    p.add_argument("--label", default="", help="Stage label for CSV output")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--pca-dim", type=int, default=0, help="0 keeps raw L2-normalized embedding")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-cluster-size", type=int, default=12)
    p.add_argument("--min-samples", type=int, default=15)
    p.add_argument("--cluster-selection-method", choices=["leaf", "eom"], default="leaf")
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    p.add_argument(
        "--noise-reassign",
        choices=["none", "nearest_q80", "nearest_q90", "assign_all"],
        default="assign_all",
    )
    p.add_argument("--cluster-merge-centroid-sim", type=float, default=None)
    return p.parse_args()


def load_paths_and_labels(path: Path) -> tuple[list[str], list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        paths = [str(x) for x in raw.get("paths", [])]
        labels = [str(x) for x in raw.get("labels", [])]
    elif isinstance(raw, list):
        paths = [str(x) for x in raw]
        labels = []
    else:
        raise SystemExit(f"unsupported paths json format: {path.resolve()}")

    if labels and len(labels) != len(paths):
        raise SystemExit(f"labels length != paths length: {path.resolve()}")
    if not labels:
        labels = [Path(p).parent.name for p in paths]
    return paths, labels


def hdbscan_details(emb: np.ndarray, y: np.ndarray, class_names: list[str], args) -> dict[str, Any]:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.cluster_selection_method,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        allow_single_cluster=False,
    )
    seed_pred = clusterer.fit_predict(emb)
    pred, reassign_info = reassign_noise_to_nearest_cluster(emb, seed_pred, args.noise_reassign)
    pred, merge_info = merge_clusters_by_centroid(emb, pred, args.cluster_merge_centroid_sim)

    clusters = sorted(int(x) for x in np.unique(pred) if int(x) >= 0)
    counts = {int(c): int(np.sum(pred == c)) for c in clusters}
    label_names = [class_names[int(label)] for label in y]
    capture = capture_metrics(pred, label_names)
    dominant_by_cluster = capture["dominant_by_cluster"]
    found_classes = capture["captured_classes"]

    largest = max(counts.values()) if counts else 0
    return {
        "seed_noise_pct": float(np.mean(seed_pred == -1)),
        "noise_pct": float(np.mean(pred == -1)),
        "noise_reassigned": int(reassign_info["noise_reassigned"]),
        "merged_pairs": int(merge_info["merged_pairs"]),
        "clusters": int(len(clusters)),
        "largest_cluster": int(largest),
        "found_total": f"{capture['capture_count']}/{capture['target_class_count']}",
        "capture": float(capture["capture_rate"]),
        "image_cap": float(capture["dominant_image_capture_rate"]),
        "class_coverage": float(capture["class_coverage_rate"]),
        "found_classes": ",".join(sorted(found_classes)),
        "ari": float(adjusted_rand_score(y, pred)),
        "ami": float(adjusted_mutual_info_score(y, pred)),
        "homogeneity": float(homogeneity_score(y, pred)),
        "completeness": float(completeness_score(y, pred)),
        "v_measure": float(v_measure_score(y, pred)),
        "purity": float(purity_score(y, pred)),
        "dominant_by_cluster": json.dumps(dominant_by_cluster, ensure_ascii=False, sort_keys=True),
    }


def main():
    args = parse_args()
    emb_path = Path(args.embedding).resolve()
    paths_path = Path(args.paths_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    emb = np.load(emb_path).astype(np.float32, copy=False)
    paths, labels = load_paths_and_labels(paths_path)
    if len(labels) != emb.shape[0]:
        raise SystemExit(f"embedding rows != labels: {emb.shape[0]} vs {len(labels)}")

    class_names = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_idx[x] for x in labels], dtype=np.int64)

    if int(args.pca_dim) > 0:
        eval_emb, pca_info = to_same_dim(emb, int(args.pca_dim), int(args.seed))
    else:
        eval_emb = l2_normalize(emb)
        pca_info = {
            "method": "raw_l2",
            "input_dim": int(emb.shape[1]),
            "target_dim": int(emb.shape[1]),
            "explained_variance_ratio_sum": None,
        }

    rates = knn_same_rates(eval_emb, y)
    dist = class_distance_metrics(eval_emb, y)
    cluster = hdbscan_details(eval_emb, y, class_names, args)

    row = {
        "label": args.label or emb_path.stem,
        "embedding": str(emb_path),
        "paths_json": str(paths_path),
        "n_images": int(emb.shape[0]),
        "n_classes": int(len(class_names)),
        "classes": ",".join(class_names),
        "pca_method": pca_info["method"],
        "pca_dim": int(eval_emb.shape[1]),
        "pca_evr": pca_info["explained_variance_ratio_sum"],
        "top1": rates["1"],
        "top3": rates["3"],
        "top5": rates["5"],
        "top7": rates["7"],
        "top9": rates["9"],
        "dist_ratio": dist["centroid_nearest_other_over_intra"],
        "method": args.cluster_selection_method,
        "mcs": args.min_cluster_size,
        "ms": args.min_samples,
        "eps": args.cluster_selection_epsilon,
        "noise_reassign": args.noise_reassign,
        "merge": args.cluster_merge_centroid_sim,
        **cluster,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print(f"[OUT] {out_csv}")


if __name__ == "__main__":
    main()
