#!/usr/bin/env python
"""중간/최종 임베딩 채점기 — 신규 UMAP-free 잣대(FINCH/Louvain) + 기존 잣대(UMAP+HDBSCAN) 병행.

사용: python _score_umapfree.py <emb.npy> [<emb.npy> ...]
라벨 유도: --labels-from {pool,cache} (pool=eval_dir rglob 순서, cache=npz 파일명 Class__stem)
채점 정책: Normal/Random/R 은 클러스터링엔 포함, 채점 제외 + nz→noise% 별도.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
import cluster_scoring as FP
from _common import load_pool_manifest  # noqa: E402  (manifest-based --pool selection, backward-compat)

EXCL = {"Normal", "Random", "R"}


def labels_pool(pool_dir: Path):
    e = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if pool_dir.is_file() and pool_dir.suffix.lower() == ".json":
        # .json manifest (make_pool_manifest.py 생성) — flat sorted-by-path 순서, 기존과 동일 의미.
        kept = [(p, lab) for p, lab in load_pool_manifest(pool_dir) if p.suffix.lower() in e]
        kept.sort(key=lambda t: t[0])
        return [(lab if lab is not None else p.parent.name) for p, lab in kept]
    paths = sorted(p for p in pool_dir.rglob("*") if p.is_file() and p.suffix.lower() in e)
    return [p.parent.name for p in paths]


def labels_cache(cache_dir: Path):
    import glob
    files = sorted(glob.glob(str(cache_dir / "*.npz")))
    return [Path(f).name.split("__")[0] for f in files]


def score(pred, labs, excl=None):
    excl = set(EXCL if excl is None else excl)
    labs = np.asarray(labs)
    keep = ~np.isin(labs, list(excl))
    classes = sorted(set(labs[keep]))
    true_idx = np.array([classes.index(c) if c in classes else -1 for c in labs])
    # Cluster labels come from the full pool.  Passing the full arrays is
    # essential: a Normal-dominant group must not falsely capture a minor
    # defect after background rows have been sliced away.
    r = FP.tier1(np.asarray(pred), true_idx, list(labs), classes, excluded=excl)
    r["n_classes"] = len(classes)
    nz = ~keep
    r["nz_to_noise_pct"] = round(float((np.asarray(pred)[nz] == -1).mean() * 100), 1) if nz.any() else None
    # ★ 클러스터 수 분리 (사용자 260612): noise(Normal/R) 가 뭉친 클러스터도 클러스터다.
    #   k_def = defect 주류 클러스터, k_tot = 전체 클러스터 (nz-주류 포함)
    pf = np.asarray(pred)
    ids = sorted(set(pf.tolist()) - {-1})
    n_def = sum(1 for c in ids if np.isin(labs[pf == c], list(excl)).mean() < 0.5)
    r["k_def"], r["k_tot"] = n_def, len(ids)
    r["k_noise"] = len(ids) - n_def  # noise(Normal/R) 주류 클러스터 수
    r.pop("capture_detail", None)
    return r


def retrieval(z, labs):
    """보조 지표 (옛 트랙 top1/top5 계승): defect-only kNN precision@1/@5 — 클러스터러 무관 임베딩 품질."""
    labs = np.asarray(labs)
    keep = ~np.isin(labs, list(EXCL))
    zk, lk = z[keep], labs[keep]
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(zk)
    _, idx = nn.kneighbors(zk)
    same = (lk[idx[:, 1:]] == lk[:, None])
    return float(same[:, 0].mean()), float(same.mean())


def run_finch(z):
    from finch import FINCH
    c, num_clust, _ = FINCH(z, verbose=False)
    out = {}
    for p in range(min(4, c.shape[1])):
        out[f"finch_p{p}(k{num_clust[p]})"] = c[:, p]
    return out


def _gpu_knn_cosine(z, k):
    """GPU cosine kNN (self 제외) — sklearn kneighbors_graph 와 동일 이웃/거리. CUDA 없으면 None."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        t = torch.from_numpy(np.ascontiguousarray(z)).float().cuda()
        t = torch.nn.functional.normalize(t, dim=1)
        sim = t @ t.T
        sim.fill_diagonal_(-2.0)
        vals, idx = sim.topk(k, dim=1)
        return (1.0 - vals).cpu().numpy(), idx.cpu().numpy()
    except Exception:
        return None


def run_louvain(z, res=6.0, k=15):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(z.shape[0]))
    gpu = _gpu_knn_cosine(z, k)
    if gpu is not None:  # ★ 260720 GPU 가속 (동일 이웃 구조)
        dists, idxs = gpu
        for i in range(z.shape[0]):
            for j, d in zip(idxs[i], dists[i]):
                G.add_edge(int(i), int(j), weight=float(1.0 - d))
    else:
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(z, k, metric="cosine", mode="distance")
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


def run_hdbscan_raw(z):
    """옛 검증 레시피 (scripts/train_contrastive.py): UMAP 없이 raw 위 직접 — mcs12/ms15/leaf/eps0.06.
    260720: 거리행렬을 GPU 로 사전계산 (euclidean, 수학 동일) — 1024-d 트리 대비 대폭 가속. CUDA 없으면 기존 경로."""
    import hdbscan
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.from_numpy(np.ascontiguousarray(z)).float().cuda()
            d2 = torch.cdist(t, t, p=2.0).double().cpu().numpy()
            cl = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=15,
                                 cluster_selection_method="leaf",
                                 cluster_selection_epsilon=0.06,
                                 metric="precomputed").fit(d2)
            return cl.labels_
    except Exception:
        pass
    cl = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=15,
                         cluster_selection_method="leaf",
                         cluster_selection_epsilon=0.06).fit(z.astype(np.float64))
    return cl.labels_


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
    ap.add_argument("--pool", default="data/pools/unknown_eval100.json")
    ap.add_argument("--cache", default="cache_fmap/mixed29_clean")
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--exclude-classes", default="Normal,Random,R",
                    help="comma-separated background/noise classes excluded from P1/P3/P4/ARI scoring")
    ap.add_argument("--out-csv", default=None,
                    help="append machine-readable rows with P1/P2/P3/P4/ARI/Sil/k/fragment")
    a = ap.parse_args()
    excl = {x.strip() for x in str(a.exclude_classes).split(",") if x.strip()}
    labs = labels_pool(REPO / a.pool) if a.labels_from == "pool" else labels_cache(REPO / a.cache)
    csv_rows = []
    for ef in a.embs:
        z = FP.l2(np.load(ef).astype(np.float32))
        assert len(labs) == z.shape[0], f"label/emb mismatch {len(labs)} vs {z.shape[0]} ({ef})"
        rows, preds_map = {}, {}
        for k, v in run_finch(z).items():
            rows[k] = score(v, labs, excl); preds_map[k] = v
        lv = run_louvain(z); rows["louvain_res6"] = score(lv, labs, excl); preds_map["louvain_res6"] = lv
        hb = run_hdbscan_raw(z); rows["hdbscan_raw(옛다이얼)"] = score(hb, labs, excl); preds_map["hdbscan_raw(옛다이얼)"] = hb
        if not a.skip_umap:
            uh = run_umap_hdb(z); rows["umap_hdbscan(고정잣대)"] = score(uh, labs, excl); preds_map["umap_hdbscan(고정잣대)"] = uh
        # Silhouette (cosine, defect-only non-noise) — 보조
        labs_arr = np.asarray(labs)
        keep = ~np.isin(labs_arr, list(excl))
        from sklearn.metrics import silhouette_score
        for m, r in rows.items():
            p = np.asarray(preds_map[m])[keep]
            msk = p != -1
            try:
                r["sil"] = round(float(silhouette_score(z[keep][msk], p[msk], metric="cosine")), 4) \
                    if msk.sum() > 10 and len(set(p[msk])) > 1 else None
            except Exception:
                r["sil"] = None
        t1, t5 = retrieval(z, labs)
        print(f"\n=== {Path(ef).name} === (retrieval top1={t1:.3f} top5={t5:.3f})")
        # ★★ 절대규칙 (사용자 260615): 성능은 항상 P1 P2 P3 P4 ARI Sil + k(전체/클래스수/noise) + 파편비(전체÷클래스, 1=이상)
        hdr = ["method", "P1 capture(found/total)", "recov", "P2 noise%", "P3 Comp", "P4 Hom", "ARI", "Sil",
               "k(전체/클래스수/noise)", "파편비(전체÷클래스)"]
        print(" | ".join(hdr))
        for m, r in rows.items():
            frag = round(r["k_tot"] / max(1, r["n_classes"]), 2)
            p1 = f"{r['capture_count']}/{r['target_class_count']} ({r['capture']:.4f})"
            print(" | ".join(str(x) for x in [m, p1, r["recov"], r["noise_pct"],
                                              r["completeness"], r["homogeneity"], r["ari"], r["sil"],
                                              f"{r['k_tot']}/{r['n_classes']}/{r['k_noise']}", frag]))
            csv_rows.append({
                "embedding": str(Path(ef).resolve()),
                "embedding_name": Path(ef).stem,
                "method": m,
                "P1_capture": r["capture"],
                "P1_capture_count": r["capture_count"],
                "P1_target_class_count": r["target_class_count"],
                "legacy_presence_count": r["legacy_presence_count"],
                "legacy_presence_rate": r["legacy_presence_rate"],
                "recov": r["recov"],
                "P2_noise_pct": r["noise_pct"],
                "P3_completeness": r["completeness"],
                "P4_homogeneity": r["homogeneity"],
                "ARI": r["ari"],
                "AMI": r.get("ami"),
                "Sil": r["sil"],
                "k_total": r["k_tot"],
                "k_classes": r["n_classes"],
                "k_noise": r["k_noise"],
                "fragment_ratio": frag,
                "exclude_classes": ",".join(sorted(excl)),
            })
    if a.out_csv:
        import csv
        out = Path(a.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "embedding", "embedding_name", "method", "P1_capture", "P1_capture_count",
            "P1_target_class_count", "legacy_presence_count", "legacy_presence_rate", "recov",
            "P2_noise_pct", "P3_completeness", "P4_homogeneity", "ARI", "AMI",
            "Sil", "k_total", "k_classes", "k_noise", "fragment_ratio",
            "exclude_classes",
        ]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"[CSV] {out.resolve()}")


if __name__ == "__main__":
    main()
