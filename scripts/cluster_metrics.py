"""Canonical label-only metrics for clustering evaluation.

P1 capture is intentionally discrete: a target class is captured only when it
is the deterministic dominant class of at least one non-noise cluster.  The
dominant class is computed from the full clustered pool, including excluded
background labels; ``excluded`` only removes labels from the P1 denominator.
The continuous largest-cluster share is a separate diagnostic, never P1.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np


def _dominant_label(counts: Counter) -> Any | None:
    """Return a unique main class, or None when a cluster has a tied maximum."""
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    return winners[0] if len(winners) == 1 else None


def capture_metrics(pred: Iterable[int], labels: Iterable[Any], excluded: Iterable[Any] = ()) -> dict[str, Any]:
    """Return the canonical P1 capture and separate continuous diagnostics."""
    pred_array = np.asarray(pred)
    labels_array = np.asarray(labels)
    if pred_array.shape[0] != labels_array.shape[0]:
        raise ValueError("pred and labels must have the same length")
    excluded_set = set(excluded)
    target_mask = ~np.isin(labels_array, list(excluded_set))
    target_classes = sorted(set(labels_array[target_mask].tolist()), key=str)
    cluster_counts: dict[int, Counter] = defaultdict(Counter)
    # Cluster dominance must be decided on the full pool.  Otherwise a
    # Normal-dominant cluster can incorrectly make a minor defect look captured.
    for cluster_id, label in zip(pred_array, labels_array):
        if int(cluster_id) >= 0:
            cluster_counts[int(cluster_id)][label] += 1
    dominant_by_cluster = {}
    for cluster_id, counts in cluster_counts.items():
        dominant = _dominant_label(counts)
        if dominant is not None:
            dominant_by_cluster[cluster_id] = dominant
    captured_classes = sorted(set(dominant_by_cluster.values()), key=str)
    # This is the older May implementation: it counts a class as captured if
    # any one of its samples is merely non-noise.  Keep it only to explain
    # historical rows; it is deliberately not named P1.
    legacy_present_classes = sorted(
        set(labels_array[target_mask & (pred_array >= 0)].tolist()), key=str
    )
    class_totals = Counter(labels_array[target_mask].tolist())
    coverage = {}
    dominant_image_capture = {}
    for label, total in class_totals.items():
        coverage[label] = max((counts.get(label, 0) for counts in cluster_counts.values()), default=0) / total
        dominant_image_capture[label] = sum(
            counts.get(label, 0)
            for cluster_id, counts in cluster_counts.items()
            if dominant_by_cluster.get(cluster_id) == label
        ) / total
    target_count = len(target_classes)
    capture_count = sum(label in set(captured_classes) for label in target_classes)
    legacy_presence_count = sum(label in set(legacy_present_classes) for label in target_classes)
    cluster_count = len(cluster_counts)
    dominant_cluster_count = len(dominant_by_cluster)
    # A captured class must be the dominant class of at least one cluster.
    # Keeping this invariant explicit prevents a coverage-style quantity from
    # silently being presented as P1.
    if capture_count > dominant_cluster_count or dominant_cluster_count > cluster_count:
        raise RuntimeError("P1 capture count cannot exceed the non-noise cluster count")
    return {
        "capture_rate": capture_count / max(1, target_count),
        "capture_count": int(capture_count),
        "target_class_count": int(target_count),
        "cluster_count": int(cluster_count),
        "dominant_cluster_count": int(dominant_cluster_count),
        "captured_classes": captured_classes,
        "dominant_by_cluster": dominant_by_cluster,
        "legacy_presence_count": int(legacy_presence_count),
        "legacy_presence_rate": legacy_presence_count / max(1, target_count),
        "class_coverage_rate": float(np.mean(list(coverage.values()))) if coverage else 0.0,
        "dominant_image_capture_rate": float(np.mean(list(dominant_image_capture.values())))
        if dominant_image_capture else 0.0,
    }
