"""Shared full-pool cluster scoring primitives.

This module intentionally contains only the evaluation contract used by the
saved-embedding scorers. It does not import a training pipeline.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
)

try:
    from .cluster_metrics import capture_metrics
except ImportError:  # Executed as a top-level module by _score_umapfree.py.
    from cluster_metrics import capture_metrics


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def tier1(
    pred: Iterable[int],
    true_idx: Iterable[int],
    labs: Iterable[Any],
    classes: Iterable[Any],
    excluded: Iterable[Any] = (),
) -> dict[str, Any]:
    """Score target labels without removing background from cluster dominance."""
    pred_array = np.asarray(pred)
    labels_array = np.asarray(labs)
    true_array = np.asarray(true_idx)
    target = ~np.isin(labels_array, list(excluded))
    pred_target = pred_array[target]
    true_target = true_array[target]
    noise = pred_target == -1
    non_noise = ~noise
    capture = capture_metrics(pred_array, labels_array, excluded)
    result: dict[str, Any] = {
        "noise_pct": round(float(noise.mean() * 100), 2),
        "n_clusters": capture["cluster_count"],
        "capture": round(float(capture["capture_rate"]), 4),
        "recov": round(float(capture["dominant_image_capture_rate"]), 4),
        "capture_count": capture["capture_count"],
        "target_class_count": capture["target_class_count"],
        "legacy_presence_count": capture["legacy_presence_count"],
        "legacy_presence_rate": round(float(capture["legacy_presence_rate"]), 4),
        "class_coverage": round(float(capture["class_coverage_rate"]), 4),
    }
    if non_noise.sum() > 1 and len(set(pred_target[non_noise].tolist())) > 1:
        result.update(
            completeness=round(float(completeness_score(true_target[non_noise], pred_target[non_noise])), 4),
            homogeneity=round(float(homogeneity_score(true_target[non_noise], pred_target[non_noise])), 4),
            ari=round(float(adjusted_rand_score(true_target[non_noise], pred_target[non_noise])), 4),
            ami=round(float(adjusted_mutual_info_score(true_target[non_noise], pred_target[non_noise])), 4),
        )
    else:
        result.update(completeness=0.0, homogeneity=0.0, ari=0.0, ami=0.0)
    captured = set(capture["captured_classes"])
    result["capture_detail"] = {
        str(label): float(label in captured)
        for label in sorted(set(labels_array[target].tolist()), key=str)
    }
    return result
