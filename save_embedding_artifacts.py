"""Embedding + clustering artifact persistence for contrastive runs.

Standalone helper kept out of ``contrastive.py`` so the main training file
stays untouched. Produces four files in a run_dir:

* ``emb.npy``            - L2-normalized embedding array (N, D)
* ``cluster_labels.npy`` - HDBSCAN cluster id per sample (N,)
* ``files.txt``          - image path per sample, parallel to ``emb.npy``
* ``kept_labels.txt``    - cluster ids that passed the KEEP filter

These filenames match what ``cluster_composite._cli``, ``predict.py``, and
the recovery scripts (``scripts/materialize_centroids.py``,
``scripts/recover_clustering.py``) expect, so a finished run can be
replayed for composite/predict/eval without re-embedding.

Usage as library::

    from save_embedding_artifacts import save_embedding_artifacts
    save_embedding_artifacts(run_dir, emb, cluster_labels, files, kept_labels)

Usage as CLI (materialize from a completed run_dir that already has the
clustering in memory only)::

    # Typically called from a short post-training wrapper — see
    # scripts/materialize_centroids.py for a related pattern.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def save_embedding_artifacts(
    run_dir: Path,
    emb: np.ndarray,
    cluster_labels: np.ndarray,
    files: List[str],
    kept_labels: List[int],
) -> Dict[str, Path]:
    """Persist embedding + clustering artifacts for downstream re-use.

    Parameters
    ----------
    run_dir : Path
        Target directory; created if missing.
    emb : np.ndarray
        (N, D) float32 embedding matrix (expected L2-normalized by caller).
    cluster_labels : np.ndarray
        (N,) integer HDBSCAN labels (-1 for noise).
    files : list of str
        Per-sample image path, length must equal N.
    kept_labels : list of int
        Cluster ids that passed the KEEP filter (may be empty).

    Returns
    -------
    dict[str, Path]
        Keys ``emb``, ``cluster_labels``, ``files``, ``kept_labels`` mapping
        to the four written paths.

    Raises
    ------
    ValueError
        If ``emb``, ``cluster_labels``, and ``files`` disagree on sample count.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    emb_arr = np.ascontiguousarray(emb)
    labels_arr = np.ascontiguousarray(cluster_labels)
    if emb_arr.shape[0] != labels_arr.shape[0] or emb_arr.shape[0] != len(files):
        raise ValueError(
            f"shape mismatch: emb={emb_arr.shape}, labels={labels_arr.shape}, "
            f"files={len(files)}"
        )

    paths = {
        "emb": run_dir / "emb.npy",
        "cluster_labels": run_dir / "cluster_labels.npy",
        "files": run_dir / "files.txt",
        "kept_labels": run_dir / "kept_labels.txt",
    }
    np.save(paths["emb"], emb_arr)
    np.save(paths["cluster_labels"], labels_arr)
    paths["files"].write_text("\n".join(str(f) for f in files), encoding="utf-8")
    paths["kept_labels"].write_text(
        "\n".join(str(int(x)) for x in kept_labels) + ("\n" if kept_labels else ""),
        encoding="utf-8",
    )
    return paths
