#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation utilities for contrastive/HDBSCAN sweeps.

Provides:
    - compute_cluster_metrics: ARI/NMI/purity given cluster + GT labels
    - compute_subset_selection: pick top-N classes x N_PER_CLASS samples from an
      ImageFolder-structured root for consistent cross-variant evaluation.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================
# Cluster metrics
# =========================================================
def compute_cluster_metrics(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
) -> Dict:
    """
    Compute clustering quality metrics.

    Parameters
    ----------
    cluster_labels : np.ndarray of shape (N,)
        HDBSCAN cluster ids. -1 = noise.
    gt_labels : np.ndarray of shape (N,)
        Ground truth integer labels (e.g. class_to_idx indices).

    Returns
    -------
    dict with keys:
        n_samples, n_clusters_found, n_noise, noise_ratio,
        ari, nmi, cluster_purity, per_cluster_purity
    ARI/NMI are computed over the NON-NOISE subset (noise excluded),
    cluster_purity is Sum(max class count per cluster) / N_clustered.
    """
    # Lazy imports — avoid forcing sklearn at module import time.
    try:
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for compute_cluster_metrics "
            "(pip install scikit-learn)"
        ) from e

    cluster_labels = np.asarray(cluster_labels)
    gt_labels = np.asarray(gt_labels)

    if cluster_labels.shape != gt_labels.shape:
        raise ValueError(
            f"shape mismatch: cluster_labels={cluster_labels.shape} vs "
            f"gt_labels={gt_labels.shape}"
        )

    n_samples = int(cluster_labels.shape[0])
    noise_mask = (cluster_labels == -1)
    n_noise = int(noise_mask.sum())
    noise_ratio = float(n_noise) / float(max(1, n_samples))

    keep_mask = ~noise_mask
    c_kept = cluster_labels[keep_mask]
    g_kept = gt_labels[keep_mask]

    uniq_clusters = sorted(int(u) for u in np.unique(c_kept))
    n_clusters_found = len(uniq_clusters)

    if c_kept.size == 0:
        ari = 0.0
        nmi = 0.0
        cluster_purity = 0.0
        per_cluster_purity: Dict[int, float] = {}
    else:
        ari = float(adjusted_rand_score(g_kept, c_kept))
        nmi = float(normalized_mutual_info_score(g_kept, c_kept))

        per_cluster_purity = {}
        total_correct = 0
        for cid in uniq_clusters:
            idx = np.where(c_kept == cid)[0]
            if idx.size == 0:
                per_cluster_purity[cid] = 0.0
                continue
            counts = Counter(int(x) for x in g_kept[idx])
            majority = max(counts.values())
            per_cluster_purity[cid] = float(majority) / float(idx.size)
            total_correct += majority
        cluster_purity = float(total_correct) / float(c_kept.size)

    return {
        "n_samples": n_samples,
        "n_clusters_found": n_clusters_found,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "ari": ari,
        "nmi": nmi,
        "cluster_purity": cluster_purity,
        "per_cluster_purity": per_cluster_purity,
    }


# =========================================================
# Subset selection
# =========================================================
def _list_class_images(class_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def compute_subset_selection(
    dataset_root: Path,
    n_classes: int = 10,
    n_per_class: int = 200,
    selection: str = "largest",
    explicit_classes: Optional[List[str]] = None,
    seed: int = 42,
    # Accept the dict-style keys from EVAL_SUBSET for convenience.
    **kwargs,
) -> List[Tuple[Path, str]]:
    """
    Pick a deterministic subset from an ImageFolder-style root.

    Parameters
    ----------
    dataset_root : Path
        Root dir; immediate sub-directories are class folders.
    n_classes : int
        Top-N classes to keep (only used when explicit_classes is None).
    n_per_class : int
        Per-class image cap (fewer taken if the class has less).
    selection : {"largest", "random", "explicit"}
        How to choose classes. "explicit" requires explicit_classes.
    explicit_classes : list[str] | None
        If given, overrides `selection` and uses exactly these class names
        (preserving their order).
    seed : int
        RNG seed used for within-class sampling (and for "random" selection).

    Returns
    -------
    list[(Path, class_name)] — shuffled deterministically for reproducibility.
    """
    # Allow callers to pass EVAL_SUBSET dict directly (upper-case keys).
    if "N_CLASSES" in kwargs:
        n_classes = kwargs["N_CLASSES"]
    if "N_PER_CLASS" in kwargs:
        n_per_class = kwargs["N_PER_CLASS"]
    if "SELECTION" in kwargs:
        selection = kwargs["SELECTION"]
    if "EXPLICIT_CLASSES" in kwargs and kwargs["EXPLICIT_CLASSES"] is not None:
        explicit_classes = kwargs["EXPLICIT_CLASSES"]
    if "SEED" in kwargs:
        seed = kwargs["SEED"]

    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not a directory: {dataset_root}")

    all_class_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if len(all_class_dirs) == 0:
        raise RuntimeError(f"No class sub-directories under {dataset_root}")

    class_to_images: Dict[str, List[Path]] = {}
    for cdir in all_class_dirs:
        imgs = _list_class_images(cdir)
        if len(imgs) > 0:
            class_to_images[cdir.name] = imgs

    # ---- choose classes ----
    if explicit_classes is not None:
        missing = [c for c in explicit_classes if c not in class_to_images]
        if missing:
            raise KeyError(f"Explicit classes missing from dataset: {missing}")
        chosen_names = list(explicit_classes)
    elif selection == "largest":
        ranked = sorted(
            class_to_images.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )
        chosen_names = [name for name, _ in ranked[:n_classes]]
    elif selection == "random":
        rng = random.Random(seed)
        chosen_names = rng.sample(
            list(class_to_images.keys()),
            k=min(n_classes, len(class_to_images)),
        )
    elif selection == "explicit":
        raise ValueError(
            "selection='explicit' requires explicit_classes to be set."
        )
    else:
        raise ValueError(f"Unknown selection strategy: {selection!r}")

    # ---- per-class sampling ----
    rng = random.Random(seed)
    items: List[Tuple[Path, str]] = []
    for cname in chosen_names:
        imgs = sorted(class_to_images[cname])  # deterministic order
        if len(imgs) <= n_per_class:
            picked = list(imgs)
        else:
            picked = rng.sample(imgs, n_per_class)
        for p in picked:
            items.append((p, cname))

    # Final deterministic shuffle so loaders don't see class-sorted batches
    rng2 = random.Random(seed + 1)
    rng2.shuffle(items)
    return items
