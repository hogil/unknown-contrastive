#!/usr/bin/env python
"""중간/최종 임베딩 채점기 — 신규 UMAP-free 잣대(FINCH/Louvain) + 기존 잣대(UMAP+HDBSCAN) 병행.

사용: python _score_umapfree.py <emb.npy> [<emb.npy> ...]
라벨 유도: --labels-from {pool,cache} (pool=eval_dir rglob 순서, cache=npz 파일명 Class__stem)
채점 정책: Normal/Random/R 은 클러스터링엔 포함, 채점 제외 + nz→noise% 별도.
"""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fp", REPO / "_field_pipeline.py")
FP = importlib.util.module_from_spec(spec); sys.modules["fp"] = FP; spec.loader.exec_module(FP)

EXCL = {"Normal", "Random", "R"}


def labels_pool(pool_dir: Path):
    e = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in pool_dir.rglob("*") if p.is_file() and p.suffix.lower() in e)
    return [p.parent.name for p in paths]


def labels_cache(cache_dir: Path):
    import glob
    files = sorted(glob.glob(str(cache_dir / "*.npz")))
    return [Path(f).name.split("__")[0] for f in files]


def score(pred, labs):
    labs = np.asarray(labs)
    keep = ~np.isin(labs, list(EXCL))
    classes = sorted(set(labs[keep]))
    true_idx = np.array([classes.index(c) if c in classes else -1 for c in labs])
    r = FP.tier1(np.asarray(pred)[keep], true_idx[keep], list(labs[keep]), classes)
    nz = ~keep
    r["nz_to_noise_pct"] = round(float((np.asarray(pred)[nz] == -1).mean() * 100), 1) if nz.any() else None
    r.pop("capture_detail", None)
    return r


def run_finch(z):
    from finch import FINCH
    c, num_clust, _ = FINCH(z, verbose=False)
    out = {}
    for p in range(min(3, c.shape[1])):
        out[f"finch_p{p}(k{num_clust[p]})"] = c[:, p]
    return out


def run_louvain(z, res=6.0, k=15):
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph
    g = kneighbors_graph(z, k, metric="cosine", mode="distance")
    G = nx.Graph()
    G.add_nodes_from(range(z.shape[0]))
    cx = g.tocoo()
    for i, j, d in zip(cx.row, cx.col, cx.data):
        G.add_edge(int(i), int(j), weight=float(1.0 - d))
    comms = nx.community.louvain_communities(G, resolution=res, seed=42)
    pred = np.full(z.shape[0], -1)
    for ci, mem in enumerate(comms):
        for m in mem:
            pred[m] = ci
    # 크기 1-2 커뮤니티는 noise 취급 (기존 res6 운영점과 동일 후처리)
    from collections import Counter
    cnt = Counter(pred.tolist())
    pred = np.array([p if cnt[p] > 2 else -1 for p in pred])
    return pred


def run_umap_hdb(z):
    import umap, hdbscan
    u = umap.UMAP(n_components=10, n_neighbors=10, min_dist=0.0, metric="cosine",
                  random_state=42).fit_transform(z)
    cl = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3,
                         cluster_selection_method="leaf",
                         cluster_selection_epsilon=0.15).fit(u.astype(np.float64))
    return cl.labels_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embs", nargs="+")
    ap.add_argument("--labels-from", choices=["pool", "cache"], default="pool")
    ap.add_argument("--pool", default="data/images/mixedwm38_pool_mixed29")
    ap.add_argument("--cache", default="cache_fmap/mixed29_clean")
    ap.add_argument("--skip-umap", action="store_true")
    a = ap.parse_args()
    labs = labels_pool(REPO / a.pool) if a.labels_from == "pool" else labels_cache(REPO / a.cache)
    for ef in a.embs:
        z = FP.l2(np.load(ef).astype(np.float32))
        assert len(labs) == z.shape[0], f"label/emb mismatch {len(labs)} vs {z.shape[0]} ({ef})"
        rows = {}
        rows.update({k: score(v, labs) for k, v in run_finch(z).items()})
        rows["louvain_res6"] = score(run_louvain(z), labs)
        if not a.skip_umap:
            rows["umap_hdbscan(고정잣대)"] = score(run_umap_hdb(z), labs)
        print(f"\n=== {Path(ef).name} ===")
        hdr = ["method", "capture", "recov", "noise%", "nz→noise%", "k", "Comp", "Hom"]
        print(" | ".join(hdr))
        for m, r in rows.items():
            print(" | ".join(str(x) for x in [m, r["capture"], r["recov"], r["noise_pct"],
                                              r["nz_to_noise_pct"], r["n_clusters"],
                                              r["completeness"], r["homogeneity"]]))


if __name__ == "__main__":
    main()
