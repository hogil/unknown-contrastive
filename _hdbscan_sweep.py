"""HDBSCAN ε × mcs sweep on existing contrastive embedding (no encoder retrain).

Usage:
    python _hdbscan_sweep.py <run_dir> [--mcs-grid 12,8] [--eps-grid 0.06,0.08,0.10]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


def metrics(y, c, emb):
    out = {
        "n": int(len(y)),
        "n_noise": int((c == -1).sum()),
        "noise_pct": float((c == -1).mean() * 100),
        "n_clusters": int(len(set(c)) - (1 if -1 in c else 0)),
        "Completeness": float(completeness_score(y, c)),
        "Homogeneity": float(homogeneity_score(y, c)),
        "AMI": float(adjusted_mutual_info_score(y, c)),
        "NMI": float(normalized_mutual_info_score(y, c)),
        "ARI": float(adjusted_rand_score(y, c)),
    }
    mc = c >= 0
    if mc.sum() > 10 and len(set(c[mc])) > 1:
        emb_n = normalize(emb[mc], axis=1)
        out["Sil"] = float(silhouette_score(emb_n, c[mc], metric="euclidean"))
    classes_seen = sum(1 for cls in set(y) if ((y == cls) & (c >= 0)).any())
    out["capture"] = float(classes_seen / len(set(y)))
    out["n_classes_captured"] = classes_seen
    out["n_classes_total"] = len(set(y))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--mcs-grid", default="12,8")
    p.add_argument("--eps-grid", default="0.06,0.08,0.10")
    p.add_argument("--ms", type=int, default=4)
    p.add_argument("--method", default="leaf")
    args = p.parse_args()

    run = Path(args.run_dir)
    emb = np.load(run / "eval/embeddings/embedding.npy")
    files = (run / "eval/embeddings/files.txt").read_text(encoding="utf-8").splitlines()
    classes = (run / "eval/embeddings/classes.txt").read_text(encoding="utf-8").splitlines()
    labels = np.array([Path(f).parent.name for f in files])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[l] for l in labels])
    mask_def = labels != "Normal"
    y_d = y[mask_def]
    e_d = emb[mask_def]

    mcs_grid = [int(x) for x in args.mcs_grid.split(",")]
    eps_grid = [float(x) for x in args.eps_grid.split(",")]

    rows = []
    print(f"=== HDBSCAN sweep on {run.name} ===")
    print(f"defect-only n={len(y_d)}, ms={args.ms}, method={args.method}")
    print()
    print(f"{'mcs':>4} {'eps':>5} | {'cl':>3} {'noise%':>7} {'Comp':>6} {'AMI':>6} {'cap':>5} {'ARI':>6} {'Sil':>6}")
    print("-" * 70)
    for mcs in mcs_grid:
        for eps in eps_grid:
            cl = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=args.ms,
                metric="euclidean",
                cluster_selection_method=args.method,
                cluster_selection_epsilon=eps,
                allow_single_cluster=False,
            ).fit_predict(e_d)
            m = metrics(y_d, cl, e_d)
            row = {"mcs": mcs, "eps": eps, **m}
            rows.append(row)
            print(f"{mcs:>4} {eps:>5.2f} | {m['n_clusters']:>3} "
                  f"{m['noise_pct']:>6.2f}% {m['Completeness']:>6.3f} "
                  f"{m['AMI']:>6.3f} {m['capture']:>5.3f} "
                  f"{m['ARI']:>6.3f} {m.get('Sil', 0):>6.3f}")

    out_path = run / "hdbscan_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
