#!/usr/bin/env python3
"""Plot t-SNE from saved contrastive/grouping embeddings.

Examples:
  python scripts/plot_tsne.py --run runs/260528_065354_evalonly_e21k
  python scripts/plot_tsne.py --run result_grouping/<TS>_grouping/<folder> --label-mode cluster
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _resolve_run_dir(p: str) -> Path:
    root = Path(p).expanduser().resolve()
    if (root / "contrastive" / "embeddings.npy").exists():
        return root / "contrastive"
    return root


def _load_labels(run_dir: Path, label_mode: str):
    paths_json = run_dir / "paths.json"
    if not paths_json.exists():
        raise SystemExit(f"paths.json not found: {paths_json}")
    meta = json.loads(paths_json.read_text(encoding="utf-8"))
    paths = meta.get("paths", [])
    labels = meta.get("labels")
    if label_mode == "class":
        if labels is None:
            labels = [Path(p).parent.name for p in paths]
        return paths, [str(x) for x in labels]

    clusters_csv = run_dir / "clusters.csv"
    clusters_txt = run_dir / "clusters_global_list.txt"
    if clusters_csv.exists():
        import csv
        rows = list(csv.DictReader(clusters_csv.open(newline="", encoding="utf-8")))
        key = "cluster_id" if rows and "cluster_id" in rows[0] else "cluster"
        return paths, [str(r.get(key, "-1")) for r in rows]
    if clusters_txt.exists():
        labels = []
        for line in clusters_txt.read_text(encoding="utf-8").splitlines()[1:]:
            labels.append(line.split("\t", 1)[0])
        return paths, labels
    raise SystemExit(f"cluster labels not found: {clusters_csv} or {clusters_txt}")


def _stratified_indices(labels: list[str], max_points: int, seed: int) -> np.ndarray:
    n = len(labels)
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        buckets[label].append(i)
    classes = sorted(buckets)
    base = max(1, max_points // max(1, len(classes)))
    selected = []
    leftovers = []
    for label in classes:
        idx = np.array(buckets[label])
        rng.shuffle(idx)
        selected.extend(idx[:base].tolist())
        leftovers.extend(idx[base:].tolist())
    if len(selected) < max_points and leftovers:
        leftovers = np.array(leftovers)
        rng.shuffle(leftovers)
        selected.extend(leftovers[:max_points - len(selected)].tolist())
    selected = np.array(selected[:max_points], dtype=int)
    selected.sort()
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 폴더 또는 contrastive/grouping 결과 폴더")
    ap.add_argument("--label-mode", choices=["class", "cluster"], default="class")
    ap.add_argument("--max-points", type=int, default=5000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    emb_path = run_dir / "embeddings.npy"
    if not emb_path.exists():
        raise SystemExit(f"embeddings.npy not found: {emb_path}")

    emb = np.load(emb_path)
    paths, labels = _load_labels(run_dir, args.label_mode)
    if len(emb) != len(labels):
        raise SystemExit(f"embedding/label length mismatch: {len(emb)} vs {len(labels)}")

    idx = _stratified_indices(labels, args.max_points, args.seed)
    x = emb[idx].astype(np.float32)
    y = np.array(labels, dtype=object)[idx]

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    n_comp = min(50, x.shape[1], max(2, x.shape[0] - 1))
    x_pca = PCA(n_components=n_comp, random_state=args.seed).fit_transform(x)
    perplexity = min(args.perplexity, max(5.0, (len(x_pca) - 1) / 3.0))
    xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        metric="euclidean",
    ).fit_transform(x_pca)

    uniq = sorted(set(y.tolist()), key=lambda v: (v == "-1", str(v)))
    cmap = plt.get_cmap("gist_ncar", max(1, len(uniq)))
    colors = {label: cmap(i) for i, label in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(16, 11), dpi=160)
    for label in uniq:
        m = y == label
        size = 5 if label != "-1" else 3
        alpha = 0.78 if label != "-1" else 0.25
        ax.scatter(xy[m, 0], xy[m, 1], s=size, c=[colors[label]], label=str(label),
                   linewidths=0, alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"t-SNE ({args.label_mode}) | {run_dir.parent.name if run_dir.name == 'contrastive' else run_dir.name} "
        f"| n={len(idx):,}/{len(emb):,} | perplexity={perplexity:g}",
        fontsize=13,
    )
    if len(uniq) <= 60:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=6, markerscale=2, frameon=False, ncol=1)
    ax.grid(False)
    fig.tight_layout()

    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        out = (run_dir / "plots" / f"tsne_{args.label_mode}.png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OUT] {out}")


if __name__ == "__main__":
    main()
