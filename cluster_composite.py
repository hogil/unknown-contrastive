"""Cluster composite map generation.

For each KEPT HDBSCAN cluster:
    1) Pick the top-N medoid samples (closest to cluster centroid by cosine distance).
    2) Load the corresponding palette-index PNGs via `common.palette_io`.
    3) Render `cluster_XXX_composite.png` via `common.composite.render_composite_png`.

Output path layout:
    run_dir/cluster_summary/composite/cluster_XXX_composite.png

Main entry point: `generate_cluster_composites(...)`. Intended to be called from
`contrastive.py`'s `main()` immediately after HDBSCAN clustering. A CLI is
provided as a convenience wrapper but it only works when embeddings have been
persisted separately (re-embedding from scratch is out of scope).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def _select_medoid_indices(
    emb: np.ndarray,
    member_idxs: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return the indices (into the full `emb` array) of the `n` members
    closest to the L2-normalized cluster centroid (cosine distance).
    """
    if member_idxs.size == 0:
        return member_idxs
    members = emb[member_idxs]
    mean = members.mean(axis=0, keepdims=True)
    mean = _l2_normalize(mean)[0]
    # cosine distance (emb is assumed L2-normalized; stay safe and re-normalize)
    mem_norm = _l2_normalize(members, axis=-1)
    dists = 1.0 - (mem_norm @ mean)
    order = np.argsort(dists)
    take = order[: max(1, min(n, member_idxs.size))]
    return member_idxs[take]


def generate_cluster_composites(
    run_dir: Path,
    emb: np.ndarray,
    files: List[str],
    cluster_labels: np.ndarray,
    kept_labels: List[int],
    out_dir: Path,
    n_per_cluster: int = 10,
    logger: Optional[Any] = None,
    overlay_dir: Optional[str] = None,  # unused — composites must come from palette PNGs
    target_size: Optional[tuple] = None,  # force-resize before stacking; None = use first image size
) -> List[Path]:
    """Generate one composite PNG per KEPT cluster.

    Parameters
    ----------
    run_dir : Path
        Root of the current training run (for logging only; outputs go to
        ``out_dir`` which the caller typically sets to
        ``run_dir / "cluster_summary" / "composite"``).
    emb : np.ndarray
        ``(N, D)`` L2-normalized embeddings used for clustering.
    files : list[str]
        Length-N list of palette-PNG paths (UNKNOWN_DIR-based paths).
    cluster_labels : np.ndarray
        ``(N,)`` HDBSCAN labels (``-1`` = noise).
    kept_labels : list[int]
        Subset of cluster ids considered "KEPT" by contrastive.py's filter.
    out_dir : Path
        Destination directory; created if needed.
    n_per_cluster : int
        Top-N medoid samples to stack per cluster.
    logger : optional
        Standard `logging.Logger` or any object with `.info` / `.warning`.
    overlay_dir : str, optional
        Ignored; composite always uses palette PNGs (the ``files`` paths).

    Returns
    -------
    list[Path]
        Paths of generated composite PNGs (may be empty if `kept_labels`
        is empty or no cluster has enough members).
    """
    _ = overlay_dir  # reserved for API symmetry
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []

    if kept_labels is None or len(kept_labels) == 0:
        if logger is not None:
            logger.info(f"[cluster_composite] kept_labels empty; nothing to do.")
        return generated

    # Lazy import so test discovery doesn't force PIL requirements on callers.
    from common.palette_io import stack_and_normalize
    from common.composite import render_composite_png

    cluster_labels = np.asarray(cluster_labels)

    for lab in sorted(int(x) for x in kept_labels):
        member_mask = cluster_labels == lab
        member_idxs = np.where(member_mask)[0]
        if member_idxs.size == 0:
            if logger is not None:
                logger.warning(
                    f"[cluster_composite] cluster {lab:03d} has no members; skipped."
                )
            continue

        top_idxs = _select_medoid_indices(emb, member_idxs, n_per_cluster)
        paths = [files[int(i)] for i in top_idxs]
        if not paths:
            continue

        try:
            stacked, invalid_mask = stack_and_normalize(paths, target_size=target_size)
        except Exception as e:
            if logger is not None:
                logger.warning(
                    f"[cluster_composite] stack failed for cluster {lab:03d}: {e}"
                )
            continue

        out_path = out_dir / f"cluster_{lab:03d}_composite.png"
        try:
            info = render_composite_png(
                stacked,
                out_path,
                title=f"cluster_{lab:03d} (n={stacked.shape[0]})",
            )
        except Exception as e:
            if logger is not None:
                logger.warning(
                    f"[cluster_composite] render failed for cluster {lab:03d}: {e}"
                )
            continue

        if logger is not None:
            logger.info(
                f"[cluster_composite] cluster {lab:03d}: "
                f"n={info['image_count']} -> {out_path.name}"
            )
        generated.append(out_path)

    return generated


# ---------------------------------------------------------------------------
# CLI (optional; only works when a run directory carries pre-saved emb.npy).
# ---------------------------------------------------------------------------
def _cli() -> int:
    # TODO: full CLI requires re-running the contrastive encoder — out of scope.
    # We support the narrow case where `run_dir/emb.npy`, `run_dir/files.txt`, and
    # `run_dir/cluster_labels.npy` have been pre-saved by some other tool.
    parser = argparse.ArgumentParser(description="Generate cluster composite PNGs.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--n-per-cluster", type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run_dir
    emb_path = run_dir / "emb.npy"
    labels_path = run_dir / "cluster_labels.npy"
    files_path = run_dir / "files.txt"
    kept_path = run_dir / "kept_labels.txt"

    missing = [p for p in (emb_path, labels_path, files_path) if not p.exists()]
    if missing:
        print(
            f"[cluster_composite] CLI requires pre-saved embeddings. Missing: "
            f"{[str(p) for p in missing]}"
        )
        return 2

    emb = np.load(emb_path)
    cluster_labels = np.load(labels_path)
    files = files_path.read_text(encoding="utf-8").splitlines()
    if kept_path.exists():
        kept_labels = [int(x) for x in kept_path.read_text().split() if x.strip()]
    else:
        kept_labels = sorted(int(x) for x in np.unique(cluster_labels) if int(x) != -1)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("cluster_composite")
    out_dir = run_dir / "cluster_summary" / "composite"
    paths = generate_cluster_composites(
        run_dir=run_dir,
        emb=emb,
        files=files,
        cluster_labels=cluster_labels,
        kept_labels=kept_labels,
        out_dir=out_dir,
        n_per_cluster=args.n_per_cluster,
        logger=logger,
    )
    logger.info(f"[cluster_composite] generated {len(paths)} composite(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
