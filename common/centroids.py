"""Cluster centroid persistence + nearest-centroid assignment.

Pairs with `contrastive.py`'s HDBSCAN output.  Usage:

    # Training end (after HDBSCAN fit):
    from common.centroids import save_cluster_centroids
    save_cluster_centroids(emb_norm, labels, run_dir / "centroids",
                           clusterer=clusterer, kept_labels=kept_cluster_ids)

    # Prediction time:
    from common.centroids import load_cluster_centroids, assign_to_cluster
    pack = load_cluster_centroids(centroids_dir)
    assigned, nn_idx, dist = assign_to_cluster(
        query_emb, pack["centroids"], pack["cluster_ids"], thresholds
    )

All embeddings are assumed L2-normalized; distance used is cosine
distance = 1 - dot.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_cluster_centroids(
    emb: np.ndarray,
    cluster_labels: np.ndarray,
    out_dir: Path,
    *,
    clusterer=None,
    kept_labels: Optional[list] = None,
    percentiles: tuple = (50, 90, 95, 99),
) -> Path:
    """Persist centroids + per-cluster distance percentiles.

    Args:
        emb: (N, D) L2-normalized embeddings.
        cluster_labels: (N,) HDBSCAN labels.  -1 = noise.
        out_dir: directory to write into (created if missing).
        clusterer: fitted `hdbscan.HDBSCAN` (optional, pickled for
            `approximate_predict` at inference time).
        kept_labels: restrict centroids to these cluster ids.  If None,
            all non-noise labels are used.
        percentiles: percentiles of within-cluster cosine distance to save
            as candidate thresholds.

    Returns:
        Path to the written `centroids.npy`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = np.asarray(emb, dtype=np.float32)
    labels = np.asarray(cluster_labels).astype(int)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape}")
    if labels.shape[0] != emb.shape[0]:
        raise ValueError("emb and cluster_labels length mismatch")

    if kept_labels is None:
        uniq = sorted(int(l) for l in np.unique(labels) if int(l) != -1)
    else:
        uniq = sorted(int(l) for l in kept_labels if int(l) != -1)

    D = emb.shape[1]
    centroids = np.zeros((len(uniq), D), dtype=np.float32)
    sizes: dict = {}
    dist_pcts: dict = {}

    for i, lab in enumerate(uniq):
        mask = labels == lab
        if not mask.any():
            # requested label has no members — skip gracefully
            centroids[i] = 0.0
            sizes[str(lab)] = 0
            dist_pcts[str(lab)] = {f"p{int(p)}": 0.0 for p in percentiles}
            continue

        members = emb[mask]
        mean = members.mean(axis=0)
        mean = _l2_normalize(mean.reshape(1, -1))[0]
        centroids[i] = mean

        dists = 1.0 - members @ mean  # cosine distance; (S,)
        dists = np.clip(dists, 0.0, 2.0)
        sizes[str(lab)] = int(mask.sum())
        dist_pcts[str(lab)] = {
            f"p{int(p)}": float(np.percentile(dists, p)) for p in percentiles
        }

    np.save(out_dir / "centroids.npy", centroids)

    meta = {
        "cluster_ids": uniq,
        "sizes": sizes,
        "dist_percentiles": dist_pcts,
        "embedding_dim": int(D),
        "percentiles_saved": list(percentiles),
    }
    with (out_dir / "centroids_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if clusterer is not None:
        try:
            with (out_dir / "clusterer.pkl").open("wb") as f:
                pickle.dump(clusterer, f)
        except Exception as e:  # pragma: no cover — best-effort
            with (out_dir / "clusterer_pkl_error.txt").open("w", encoding="utf-8") as f:
                f.write(f"pickle failed: {e!r}\n")

    return out_dir / "centroids.npy"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_cluster_centroids(centroids_dir: Path) -> dict:
    """Inverse of :func:`save_cluster_centroids`.

    Returns a dict with keys ``centroids`` (np.ndarray[K,D]), ``cluster_ids``
    (list[int]), ``meta`` (dict), and ``clusterer`` (object or None).
    """
    centroids_dir = Path(centroids_dir)
    centroids = np.load(centroids_dir / "centroids.npy")
    centroids = centroids.astype(np.float32, copy=False)

    with (centroids_dir / "centroids_meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    cluster_ids = [int(x) for x in meta.get("cluster_ids", [])]

    clusterer = None
    pkl_path = centroids_dir / "clusterer.pkl"
    if pkl_path.exists():
        try:
            with pkl_path.open("rb") as f:
                clusterer = pickle.load(f)
        except Exception:
            clusterer = None

    return {
        "centroids": centroids,
        "cluster_ids": cluster_ids,
        "meta": meta,
        "clusterer": clusterer,
    }


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------
def assign_to_cluster(
    emb: np.ndarray,
    centroids: np.ndarray,
    cluster_ids: list,
    thresholds: np.ndarray,
):
    """Assign each row of `emb` to its nearest centroid (cosine distance).

    Rows whose distance exceeds the threshold for that centroid are marked
    as unknown (``-1``).

    Args:
        emb: (M, D) L2-normalized query embeddings.
        centroids: (K, D) L2-normalized centroids.
        cluster_ids: length-K list aligning centroid rows to cluster ids.
        thresholds: (K,) per-cluster max allowed cosine distance.

    Returns:
        (assigned_cluster_id [M], nearest_idx [M], distance [M]).  When
        ``centroids`` is empty every query is assigned ``-1`` with a
        nearest index of ``-1`` and distance ``inf``.
    """
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape}")

    M = emb.shape[0]
    centroids = np.asarray(centroids, dtype=np.float32)
    cluster_ids = [int(c) for c in cluster_ids]
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)

    if centroids.size == 0 or centroids.shape[0] == 0:
        return (
            np.full((M,), -1, dtype=np.int64),
            np.full((M,), -1, dtype=np.int64),
            np.full((M,), np.inf, dtype=np.float32),
        )

    if len(cluster_ids) != centroids.shape[0]:
        raise ValueError("cluster_ids length must match centroid rows")
    if thresholds.shape[0] != centroids.shape[0]:
        raise ValueError("thresholds length must match centroid rows")

    sims = emb @ centroids.T                 # (M, K)
    dists = 1.0 - sims                       # cosine distance
    nearest_idx = np.argmin(dists, axis=1)   # (M,)
    nearest_dist = dists[np.arange(M), nearest_idx].astype(np.float32)

    nn_thr = thresholds[nearest_idx]
    cluster_arr = np.asarray(cluster_ids, dtype=np.int64)
    assigned = np.where(nearest_dist <= nn_thr, cluster_arr[nearest_idx], -1).astype(np.int64)

    return assigned, nearest_idx.astype(np.int64), nearest_dist
