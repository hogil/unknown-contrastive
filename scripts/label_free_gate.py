#!/usr/bin/env python3
"""Label-free frozen/adapted grouping gate.

The path manifest is treated as an opaque, ordered list.  This module never
reads parent directories, class names, labels, or majority labels.

Example:
    python scripts/label_free_gate.py \
        --paths-manifest paths.txt \
        --frozen-embedding frozen.npy \
        --adapted-embedding adapted.npy \
        --frozen-aug-embedding frozen_aug.npy \
        --adapted-aug-embedding adapted_aug.npy \
        --out gate.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Fixed production clustering recipe.  These values are deliberately not
# exposed as per-dataset tuning flags.
HDBSCAN_CONFIG = {
    "min_cluster_size": 12,
    "min_samples": 15,
    "metric": "euclidean",
    "cluster_selection_method": "leaf",
    "cluster_selection_epsilon": 0.06,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _opaque_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path manifest entries must be non-empty strings")
    return value.rstrip("\r\n")


def read_path_manifest(path: Path) -> list[str]:
    """Read ordered paths without interpreting their contents or parents."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if set(payload) != {"paths"} or not isinstance(payload["paths"], list):
                raise ValueError("JSON manifest must be a string list or {'paths': [...]}")
            payload = payload["paths"]
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must contain an ordered string list")
        return [_opaque_path(item) for item in payload]

    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return []
        header = [cell.strip().lower() for cell in rows[0]]
        if header in (["path"], ["image_path"]):
            rows = rows[1:]
        elif len(rows[0]) != 1:
            raise ValueError("CSV manifest must have exactly one path column")
        if any(len(row) != 1 for row in rows):
            raise ValueError("CSV manifest must have exactly one path column")
        return [_opaque_path(row[0]) for row in rows if row and row[0].strip()]

    return [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_embedding(path: Path) -> np.ndarray:
    """Load a finite 2-D embedding array from .npy or a simple .npz."""
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = list(loaded.files)
        preferred = [key for key in ("embedding", "embeddings", "arr_0") if key in keys]
        if len(preferred) != 1:
            loaded.close()
            raise ValueError(f"{path} must contain exactly one embedding array")
        array = loaded[preferred[0]]
        loaded.close()
    else:
        array = loaded
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{path} must be a non-empty 2-D array")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path} must contain numeric values")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{path} contains non-finite values")
    return array


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embedding contains a zero-norm row")
    return embedding / norms


def hdbscan_fit(embedding: np.ndarray) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("hdbscan is required for the label-free gate") from exc
    clusterer = hdbscan.HDBSCAN(**HDBSCAN_CONFIG)
    return clusterer.fit_predict(embedding.astype(np.float64, copy=False)).astype(np.int64)


def _cluster_pairs(prediction: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left = prediction[pairs[:, 0]]
    right = prediction[pairs[:, 1]]
    return (left >= 0) & (right >= 0) & (left == right)


def bootstrap_stability(
    embedding: np.ndarray,
    base_prediction: np.ndarray,
    *,
    replicates: int,
    fraction: float,
    pair_samples: int,
    seed: int,
) -> float:
    """Mean co-cluster pair Jaccard under bootstrap re-clustering.

    Cluster ids are arbitrary, so stability compares pairwise co-membership,
    not numeric cluster ids.  Noise pairs are excluded from both sets.
    """
    rng = np.random.default_rng(seed)
    n = len(embedding)
    sample_size = max(2, min(n, int(round(fraction * n))))
    scores: list[float] = []
    for _ in range(replicates):
        indices = np.sort(rng.choice(n, size=sample_size, replace=False))
        boot_prediction = hdbscan_fit(embedding[indices])
        base = base_prediction[indices]
        if sample_size < 2:
            continue
        pair_count = min(pair_samples, sample_size * (sample_size - 1) // 2)
        pair_indices = rng.integers(0, sample_size, size=(pair_count * 2, 2))
        pair_indices = pair_indices[pair_indices[:, 0] != pair_indices[:, 1]][:pair_count]
        if len(pair_indices) == 0:
            continue
        base_same = _cluster_pairs(base, pair_indices)
        boot_same = _cluster_pairs(boot_prediction, pair_indices)
        union = np.count_nonzero(base_same | boot_same)
        intersection = np.count_nonzero(base_same & boot_same)
        scores.append(float(intersection / union) if union else 1.0)
    return float(np.mean(scores)) if scores else 0.0


def within_group_coherence(embedding: np.ndarray, prediction: np.ndarray) -> float:
    """Weighted mean pairwise cosine similarity for non-noise groups."""
    values: list[tuple[float, int]] = []
    for cluster_id in sorted(set(prediction.tolist())):
        if cluster_id < 0:
            continue
        group = embedding[prediction == cluster_id]
        if len(group) < 2:
            continue
        similarity = group @ group.T
        upper = similarity[np.triu_indices(len(group), k=1)]
        if len(upper):
            values.append((float(np.mean(upper)), len(upper)))
    if not values:
        return 0.0
    return float(sum(value * weight for value, weight in values) / sum(weight for _, weight in values))


def augmentation_consistency(main: np.ndarray, augmented: np.ndarray) -> float:
    if main.shape != augmented.shape:
        raise ValueError("augmentation embedding must have the same shape as the main embedding")
    main = l2_normalize(main)
    augmented = l2_normalize(augmented)
    return float(np.mean(np.sum(main * augmented, axis=1)))


def summarize(
    embedding: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_fraction: float,
    bootstrap_pairs: int,
    seed: int,
    augmented: np.ndarray | None,
) -> dict[str, Any]:
    embedding = l2_normalize(embedding)
    prediction = hdbscan_fit(embedding)
    total = len(prediction)
    cluster_ids = sorted(int(value) for value in set(prediction.tolist()) if value >= 0)
    sizes = [int(np.count_nonzero(prediction == cluster_id)) for cluster_id in cluster_ids]
    over_merge_cutoff = 0.20
    over_merge_count = sum(1 for size in sizes if size >= over_merge_cutoff * total)
    non_noise = total - int(np.count_nonzero(prediction == -1))
    fragmentation = float(len(cluster_ids) / max(non_noise, 1))
    result: dict[str, Any] = {
        "n": total,
        "embedding_dim": int(embedding.shape[1]),
        "cluster_count": len(cluster_ids),
        "noise_pct": 100.0 * float(np.count_nonzero(prediction == -1)) / total,
        "largest_group_pct": 100.0 * (max(sizes) if sizes else 0) / total,
        "over_merge_count": int(over_merge_count),
        "over_merge_cutoff_pct": 100.0 * over_merge_cutoff,
        "bootstrap_stability": bootstrap_stability(
            embedding,
            prediction,
            replicates=bootstrap_replicates,
            fraction=bootstrap_fraction,
            pair_samples=bootstrap_pairs,
            seed=seed,
        ),
        "within_group_cosine_coherence": within_group_coherence(embedding, prediction),
        "fragmentation_proxy": fragmentation,
        "fragmentation_formula": "non_noise_cluster_count / non_noise_sample_count",
        "non_noise_count": non_noise,
        "augmentation_consistency": (
            augmentation_consistency(embedding, augmented) if augmented is not None else None
        ),
        "augmentation_consistency_available": augmented is not None,
    }
    return result


def _comparison(
    frozen: dict[str, Any],
    adapted: dict[str, Any],
    *,
    noise_tolerance_pct: float,
    stability_tolerance: float,
    coherence_tolerance: float,
    augmentation_tolerance: float,
    fragmentation_tolerance_pct: float,
    max_noise_pct: float,
    allow_missing_augmentation: bool,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, actual: Any, required: Any) -> None:
        checks[name] = {"passed": bool(passed), "actual": actual, "required": required}

    check(
        "over_merge_count",
        adapted["over_merge_count"] == 0,
        adapted["over_merge_count"],
        "== 0",
    )
    check(
        "noise_pct",
        adapted["noise_pct"] <= frozen["noise_pct"] + noise_tolerance_pct,
        adapted["noise_pct"],
        f"<= frozen + {noise_tolerance_pct} percentage points",
    )
    check(
        "max_noise_pct",
        adapted["noise_pct"] <= max_noise_pct,
        adapted["noise_pct"],
        f"<= {max_noise_pct} percentage points",
    )
    check(
        "bootstrap_stability",
        adapted["bootstrap_stability"] + stability_tolerance >= frozen["bootstrap_stability"],
        adapted["bootstrap_stability"],
        f">= frozen - {stability_tolerance}",
    )
    check(
        "within_group_cosine_coherence",
        adapted["within_group_cosine_coherence"] + coherence_tolerance
        >= frozen["within_group_cosine_coherence"],
        adapted["within_group_cosine_coherence"],
        f">= frozen - {coherence_tolerance}",
    )

    frozen_aug = frozen["augmentation_consistency"]
    adapted_aug = adapted["augmentation_consistency"]
    if frozen_aug is None and adapted_aug is None:
        checks["augmentation_consistency"] = {
            "passed": bool(allow_missing_augmentation),
            "status": "not_applicable",
            "actual": None,
            "required": "both augmentation pairs supplied for comparison",
        }
    elif frozen_aug is not None and adapted_aug is not None:
        check(
            "augmentation_consistency",
            adapted_aug + augmentation_tolerance >= frozen_aug,
            adapted_aug,
            f">= frozen - {augmentation_tolerance}",
        )
    else:
        checks["augmentation_consistency"] = {
            "passed": bool(allow_missing_augmentation),
            "status": "not_applicable" if allow_missing_augmentation else "comparison_unavailable",
            "actual": adapted_aug,
            "required": "both frozen and adapted augmentation pairs supplied",
        }

    allowed_fragmentation = frozen["fragmentation_proxy"] * (1.0 + fragmentation_tolerance_pct)
    check(
        "fragmentation_proxy",
        adapted["fragmentation_proxy"] <= allowed_fragmentation,
        adapted["fragmentation_proxy"],
        f"<= frozen * (1 + {fragmentation_tolerance_pct}) = {allowed_fragmentation}",
    )
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def _absolute_gate(
    summary: dict[str, Any],
    *,
    min_stability: float,
    min_coherence: float,
    max_noise_pct: float,
) -> dict[str, Any]:
    """Apply the production absolute gate without labels or relative baselines."""
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, actual: Any, required: Any) -> None:
        checks[name] = {"passed": bool(passed), "actual": actual, "required": required}

    check("min_stability", summary["bootstrap_stability"] >= min_stability,
          summary["bootstrap_stability"], f">= {min_stability}")
    check(
        "min_coherence",
        summary["within_group_cosine_coherence"] >= min_coherence,
        summary["within_group_cosine_coherence"],
        f">= {min_coherence}",
    )
    check("max_noise_pct", summary["noise_pct"] <= max_noise_pct,
          summary["noise_pct"], f"<= {max_noise_pct}")
    check("over_merge_count", summary["over_merge_count"] == 0,
          summary["over_merge_count"], "== 0")
    check("cluster_count", summary["cluster_count"] >= 2,
          summary["cluster_count"], ">= 2")
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a label-free frozen/adapted HDBSCAN approval gate."
    )
    parser.add_argument("--paths-manifest", required=True, type=Path)
    parser.add_argument("--frozen-embedding", required=True, type=Path)
    parser.add_argument("--adapted-embedding", type=Path)
    parser.add_argument("--frozen-aug-embedding", type=Path)
    parser.add_argument("--adapted-aug-embedding", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--bootstrap-replicates", type=int, default=5)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-tolerance-pct", type=float, default=0.0)
    parser.add_argument("--stability-tolerance", type=float, default=0.0)
    parser.add_argument("--coherence-tolerance", type=float, default=0.0)
    parser.add_argument("--augmentation-tolerance", type=float, default=0.0)
    parser.add_argument("--fragmentation-tolerance-pct", type=float, default=0.10)
    parser.add_argument("--max-noise-pct", type=float, default=70.0)
    parser.add_argument("--min-stability", type=float, default=0.75)
    parser.add_argument("--min-coherence", type=float, default=0.80)
    parser.add_argument(
        "--allow-missing-augmentation",
        action="store_true",
        help="Allow adapted approval without both frozen/adapted augmentation pairs.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_replicates < 1 or not 0 < args.bootstrap_fraction <= 1 or args.bootstrap_pairs < 1:
        raise ValueError("bootstrap settings must be positive and fraction must be in (0, 1]")
    if not 0.0 <= args.noise_tolerance_pct <= 100.0:
        raise ValueError("noise tolerance must be in [0, 100] percentage points")
    if not 0.0 <= args.max_noise_pct <= 100.0:
        raise ValueError("max noise must be in [0, 100] percent")
    for name in ("min_stability", "min_coherence"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in ("stability_tolerance", "coherence_tolerance", "augmentation_tolerance"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0.0 <= args.fragmentation_tolerance_pct <= 1.0:
        raise ValueError("fragmentation tolerance must be in [0, 1] as a fraction")
    manifest_paths = read_path_manifest(args.paths_manifest)
    frozen = load_embedding(args.frozen_embedding)
    adapted = load_embedding(args.adapted_embedding) if args.adapted_embedding else None
    frozen_aug = load_embedding(args.frozen_aug_embedding) if args.frozen_aug_embedding else None
    adapted_aug = load_embedding(args.adapted_aug_embedding) if args.adapted_aug_embedding else None

    expected_n = len(manifest_paths)
    for name, embedding in (
        ("frozen", frozen),
        ("adapted", adapted),
        ("frozen_aug", frozen_aug),
        ("adapted_aug", adapted_aug),
    ):
        if embedding is not None and len(embedding) != expected_n:
            raise ValueError(f"{name} row count {len(embedding)} != manifest row count {expected_n}")
    if (frozen_aug is None) != (adapted_aug is None) and adapted is not None:
        # A one-sided augmentation input cannot establish non-degradation.
        # It is retained as a recorded failure rather than silently ignored.
        pass

    frozen_summary = summarize(
        frozen,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_fraction=args.bootstrap_fraction,
        bootstrap_pairs=args.bootstrap_pairs,
        seed=args.seed,
        augmented=frozen_aug,
    )
    frozen_approval = _absolute_gate(
        frozen_summary,
        min_stability=args.min_stability,
        min_coherence=args.min_coherence,
        max_noise_pct=args.max_noise_pct,
    )
    adapted_summary = None
    adapted_absolute = None
    comparison = None
    rescue_approval = None
    selection_reason = "frozen_only"
    selected_mode = "frozen"
    if adapted is None:
        workflow_action = "use_frozen" if frozen_approval["passed"] else "adapt_required"
    else:
        workflow_action = "pseudo_tapt_review_required"
    if adapted is not None:
        adapted_summary = summarize(
            adapted,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_fraction=args.bootstrap_fraction,
            bootstrap_pairs=args.bootstrap_pairs,
            seed=args.seed,
            augmented=adapted_aug,
        )
        comparison = _comparison(
            frozen_summary,
            adapted_summary,
            noise_tolerance_pct=args.noise_tolerance_pct,
            stability_tolerance=args.stability_tolerance,
            coherence_tolerance=args.coherence_tolerance,
            augmentation_tolerance=args.augmentation_tolerance,
            fragmentation_tolerance_pct=args.fragmentation_tolerance_pct,
            max_noise_pct=args.max_noise_pct,
            allow_missing_augmentation=args.allow_missing_augmentation,
        )
        adapted_absolute = _absolute_gate(
            adapted_summary,
            min_stability=args.min_stability,
            min_coherence=args.min_coherence,
            max_noise_pct=args.max_noise_pct,
        )
        if frozen_approval["passed"]:
            if comparison["passed"] and adapted_absolute["passed"]:
                selected_mode = "adapted"
                workflow_action = "use_adapted"
                selection_reason = "adapted_nonworse_than_valid_frozen"
            else:
                selected_mode = "frozen"
                workflow_action = "use_frozen"
                selection_reason = "valid_frozen_retained"
        else:
            # Collapse can make a degenerate frozen result look perfectly stable.
            # Once frozen fails the absolute gate, compare adapted against fixed
            # absolute requirements instead of requiring it to beat those
            # collapse-inflated relative metrics.
            frozen_aug = frozen_summary["augmentation_consistency"]
            adapted_aug = adapted_summary["augmentation_consistency"]
            if frozen_aug is not None and adapted_aug is not None:
                augmentation_passed = (
                    adapted_aug + args.augmentation_tolerance >= frozen_aug
                )
                augmentation_required = (
                    f">= frozen - {args.augmentation_tolerance}"
                )
            else:
                augmentation_passed = bool(args.allow_missing_augmentation)
                augmentation_required = "both augmentation pairs supplied"
            rescue_approval = {
                "passed": bool(adapted_absolute["passed"] and augmentation_passed),
                "checks": {
                    "adapted_absolute": {
                        "passed": bool(adapted_absolute["passed"]),
                        "actual": bool(adapted_absolute["passed"]),
                        "required": True,
                    },
                    "augmentation_consistency": {
                        "passed": bool(augmentation_passed),
                        "actual": adapted_aug,
                        "required": augmentation_required,
                    },
                },
                "reason": "frozen_failed_absolute_gate",
            }
            if rescue_approval["passed"]:
                selected_mode = "adapted"
                workflow_action = "use_adapted"
                selection_reason = "adapted_rescued_invalid_frozen"
            else:
                selected_mode = "frozen"
                workflow_action = "pseudo_tapt_review_required"
                selection_reason = "adapted_failed_rescue_gate"

    input_paths = {
        "paths_manifest": args.paths_manifest,
        "frozen_embedding": args.frozen_embedding,
        "adapted_embedding": args.adapted_embedding,
        "frozen_aug_embedding": args.frozen_aug_embedding,
        "adapted_aug_embedding": args.adapted_aug_embedding,
    }
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(value) for value in sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "input_files": {
            key: (
                {"path": str(value.resolve()), "sha256": _sha256(value)}
                if value is not None
                else None
            )
            for key, value in input_paths.items()
        },
        "manifest_row_count": expected_n,
        "manifest_order_used": True,
        "labels_read": False,
        "folder_names_read": False,
        "majority_label_used": False,
    }
    output = {
        "schema_version": "label_free_gate.v1",
        "selected_mode": selected_mode,
        "workflow_action": workflow_action,
        "frozen": frozen_summary,
        "frozen_approval": frozen_approval,
        "adapted": adapted_summary,
        "adapted_absolute": adapted_absolute,
        "approval": comparison,
        "rescue_approval": rescue_approval,
        "selection_reason": selection_reason,
        "thresholds": {
            "over_merge_group_pct": 20.0,
            "min_stability": args.min_stability,
            "min_coherence": args.min_coherence,
            "cluster_count_min": 2,
            "noise_tolerance_pct": args.noise_tolerance_pct,
            "max_noise_pct": args.max_noise_pct,
            "stability_tolerance": args.stability_tolerance,
            "coherence_tolerance": args.coherence_tolerance,
            "augmentation_tolerance": args.augmentation_tolerance,
            "fragmentation_tolerance_pct": args.fragmentation_tolerance_pct,
            "fragmentation_tolerance_unit": "fraction",
            "allow_missing_augmentation": args.allow_missing_augmentation,
        },
        "hdbscan": HDBSCAN_CONFIG,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "fraction": args.bootstrap_fraction,
            "pair_samples": args.bootstrap_pairs,
            "seed": args.seed,
            "stability_metric": "co-cluster pair Jaccard",
        },
        "provenance": provenance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_mode": selected_mode, "workflow_action": workflow_action,
                      "frozen_approval": frozen_approval, "approval": comparison}, indent=2))
    print(f"[OUT] {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
