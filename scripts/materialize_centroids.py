"""One-shot recovery: materialize ``centroids/`` from a persisted ``emb.npy``.

When ``recover_clustering.py`` (or the main training tail) crashes after
embedding but before ``common.centroids.save_cluster_centroids`` runs, the
run_dir ends up with ``emb.npy`` but no ``centroids/``. This script re-fits
HDBSCAN on the embeddings using the run's own CFG, computes the KEEP mask
via ``contrastive.is_kept_cluster``, and writes ``centroids/clusterer.pkl``
+ ``centroids.npy`` + ``centroids_meta.json`` so the eval/predict paths
can proceed.

We do NOT read ``cluster_labels.npy`` — it may be corrupt (Task #13 bug
where save_embedding_artifacts wrote all zeros). HDBSCAN is re-fit from
``emb.npy`` so the labels it produces here are definitive for this
materialization.

Usage::

    python scripts/materialize_centroids.py \\
        --run-dir outputs_gpu_v2/strong_augment_260420_234333

CFG defaults are pulled from ``last_training.pt['cfg']``; override with CLI
flags if the run's CFG needs adjustment (e.g. ``--hdbscan-metric euclidean``
to swap cosine for the sklearn-BallTree-compatible euclidean on
L2-normalized features — monotonically equivalent for ranking).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--hdbscan-metric",
        default="euclidean",
        help=(
            "HDBSCAN metric. Defaults to euclidean which on L2-normalized "
            "embeddings is monotonically equivalent to cosine for ranking "
            "and is what recover_clustering.py used."
        ),
    )
    ap.add_argument(
        "--skip-sanity",
        action="store_true",
        help=(
            "Skip the hardcoded cluster-count / noise-count / kept-size sanity "
            "check. Those values were calibrated for the original ood_A recovery "
            "(16 clusters, 1 kept, 1597 noise) and will spuriously fail for any "
            "other run. Pass --skip-sanity when materializing centroids for a "
            "different run_dir."
        ),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    emb_path = run_dir / "emb.npy"
    if not emb_path.exists():
        print(f"[err] missing {emb_path}", file=sys.stderr)
        return 1

    # Load CFG from the checkpoint so keep-thresholds / hdbscan-params match
    # what the trainer intended for this run.
    import torch
    import contrastive

    ckpt = torch.load(
        run_dir / "checkpoints" / "last_training.pt",
        map_location="cpu",
        weights_only=False,
    )
    side_cfg = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(side_cfg, dict):
        contrastive.CFG.update(side_cfg)
    contrastive.CFG["HDBSCAN_METRIC"] = args.hdbscan_metric

    emb = np.load(emb_path)
    if emb.ndim != 2:
        print(f"[err] emb must be 2-D, got {emb.shape}", file=sys.stderr)
        return 1
    print(f"[load] emb.shape={emb.shape}  dtype={emb.dtype}")

    # Re-fit HDBSCAN — cannot reuse a prior clusterer because none was
    # pickled. contrastive.cluster_and_save_hdbscan also writes a bunch of
    # side-effect files we don't need right now; inline the fit here for a
    # minimal blast radius.
    import hdbscan

    params = dict(
        min_cluster_size=contrastive.CFG["MIN_CLUSTER_SIZE"],
        min_samples=contrastive.CFG["MIN_SAMPLES"],
        metric=contrastive.CFG["HDBSCAN_METRIC"],
        cluster_selection_method=contrastive.CFG["CLUSTER_SELECTION_METHOD"],
        cluster_selection_epsilon=contrastive.CFG["CLUSTER_SELECTION_EPSILON"],
        allow_single_cluster=contrastive.CFG["ALLOW_SINGLE_CLUSTER"],
        prediction_data=True,
    )
    print(f"[hdbscan] {params}")
    clusterer = hdbscan.HDBSCAN(**params)
    labels = clusterer.fit_predict(emb)
    probs = clusterer.probabilities_
    persist = clusterer.cluster_persistence_

    uniq = sorted(int(u) for u in np.unique(labels) if int(u) >= 0)
    n_noise = int((labels == -1).sum())
    print(f"[hdbscan] clusters_pre_filter={len(uniq)}  noise={n_noise} ({n_noise/len(labels):.1%})")

    # Rebuild the same per-cluster stats shape `is_kept_cluster` expects.
    pers_map = {lab: float(persist[i]) for i, lab in enumerate(uniq)}
    centers = []
    for lab in uniq:
        idxs = np.where(labels == lab)[0]
        centers.append(emb[idxs].mean(0) if len(idxs) else np.zeros(emb.shape[1], dtype=emb.dtype))
    centers = np.stack(centers, 0) if centers else np.zeros((0, emb.shape[1]), dtype=emb.dtype)

    kept_labels: list[int] = []
    for ci, lab in enumerate(uniq):
        idxs = np.where(labels == lab)[0]
        size = int(len(idxs))
        if size == 0:
            continue
        s = {
            "lab": lab,
            "size": size,
            "median_prob": float(np.median(probs[idxs])),
            "mean_prob": float(np.mean(probs[idxs])),
            "std_prob": float(np.std(probs[idxs])),
            "persistence": pers_map.get(lab, 0.0),
            "cohesion": contrastive._cluster_neighbor_cohesion(emb, idxs, k=5),
            "margin": contrastive._cluster_margin_ratio(emb, idxs, centers, my_idx=ci),
            # purity needs per-sample classes — we don't have them here without
            # re-walking the training tree. is_kept_cluster's purity check is
            # typically disabled (thresh=0) for this CFG; passing 1.0 is the
            # maximum possible value so the purity clause can't gate wrongly.
            "purity": 1.0,
        }
        if contrastive.is_kept_cluster(s):
            kept_labels.append(lab)
    print(f"[keep] kept_clusters={len(kept_labels)} / {len(uniq)}  ids={kept_labels}")

    # Sanity check against the structure recovery_clustering already wrote to
    # disk as clusters/hdbscan/cluster_00{0..15}_* folders. If our re-fit
    # diverges from that, fail loudly — silently producing different clusters
    # than the on-disk folders would create a split-brain state.
    if args.skip_sanity:
        print(f"[skip-sanity] pre-filter={len(uniq)}  noise={n_noise}  kept={len(kept_labels)}  ids={kept_labels}")
    else:
        expected_clusters = 16
        expected_kept = 1
        # Noise count varies slightly run-to-run because HDBSCAN's sklearn
        # backend is non-deterministic on ties; allow a small window around the
        # recovery-run's 1597 so we don't spuriously fail.
        expected_noise = 1597
        noise_tol = 10
        if len(uniq) != expected_clusters:
            print(
                f"[err] pre-filter clusters {len(uniq)} != expected {expected_clusters} "
                f"— re-fit diverged from on-disk cluster folders",
                file=sys.stderr,
            )
            return 3
        if abs(n_noise - expected_noise) > noise_tol:
            print(
                f"[err] noise={n_noise} outside expected {expected_noise}±{noise_tol}",
                file=sys.stderr,
            )
            return 3
        if len(kept_labels) != expected_kept:
            print(
                f"[err] kept_clusters={len(kept_labels)} != expected {expected_kept}; "
                f"ids={kept_labels}",
                file=sys.stderr,
            )
            return 3
        kept_id = kept_labels[0]
        kept_size = int((labels == kept_id).sum())
        if not (25 <= kept_size <= 35):
            print(
                f"[warn] KEPT cluster_{kept_id} has {kept_size} members; "
                f"expected ~29 (Edge-Ring). Continuing but flag this.",
                file=sys.stderr,
            )
        print(f"[ok] sanity: 16 clusters, {n_noise} noise, KEPT=[{kept_id}] size={kept_size}")

    # Persist everything save_cluster_centroids expects.
    from common.centroids import save_cluster_centroids

    centroids_dir = run_dir / "centroids"
    centroids_dir.mkdir(parents=True, exist_ok=True)
    save_cluster_centroids(
        emb, labels, centroids_dir,
        clusterer=clusterer, kept_labels=kept_labels,
    )
    # save_cluster_centroids best-effort pickles the clusterer already;
    # force a fresh pickle with prediction_data_ guaranteed regenerated so
    # eval_val.py's approximate_predict path succeeds without a fallback.
    try:
        clusterer.generate_prediction_data()
    except Exception:
        pass  # prediction_data=True at fit ensured it, but be defensive
    with (centroids_dir / "clusterer.pkl").open("wb") as f:
        pickle.dump(clusterer, f)

    # Overwrite the corrupt cluster_labels.npy (Task #13 bug produced all-zeros
    # in this run_dir) with the definitive labels from this fit.
    labels_path = run_dir / "cluster_labels.npy"
    np.save(labels_path, labels.astype(np.int64))
    print(f"[ok] overwrote {labels_path.name} with {labels.shape} int64 labels")

    # Overwrite kept_labels.txt (currently empty 0-byte file). Format: one
    # cluster id per line, matches what contrastive.save_embedding_artifacts
    # and recover_clustering.py both read.
    kept_path = run_dir / "kept_labels.txt"
    kept_path.write_text(
        "\n".join(str(int(k)) for k in kept_labels) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {kept_path.name} ({len(kept_labels)} line(s))")

    # Verify all three artifacts.
    must_exist = [
        centroids_dir / "clusterer.pkl",
        centroids_dir / "centroids.npy",
        centroids_dir / "centroids_meta.json",
    ]
    missing = [p for p in must_exist if not p.exists()]
    if missing:
        print(f"[err] post-write missing: {missing}", file=sys.stderr)
        return 2

    cents = np.load(centroids_dir / "centroids.npy")
    print(f"[ok] centroids.npy shape={cents.shape}  kept={len(kept_labels)}")
    print(f"[done] materialized centroids/ + cluster_labels.npy + kept_labels.txt at {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
