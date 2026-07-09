#!/usr/bin/env python
"""EAC (Evidence Accumulation Clustering, Fred & Jain TPAMI 2005) — co-association + lifetime cut.

SOTA 부품 임베딩들의 FINCH p0/p1/p2 partition 을 base 로 co-association 행렬 누적
→ average-link 계층 클러스터링 → maximum lifetime cut (다이얼 0, k 자동).
GPU 0 / 학습 0 — 기존 임베딩 위 CPU 후처리.

사용: python _ens_coassoc.py <emb1.npy> <emb2.npy> ... [--parts 0,1,2]
"""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sc", REPO / "_score_umapfree.py")
SC = importlib.util.module_from_spec(spec); sys.modules["sc"] = SC; spec.loader.exec_module(SC)


def finch_partitions(z, which):
    from finch import FINCH
    c, num_clust, _ = FINCH(z, verbose=False)
    return [(c[:, p], num_clust[p]) for p in which if p < c.shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embs", nargs="+")
    ap.add_argument("--parts", default="0,1,2")
    ap.add_argument("--pool", default="data/images/mixedwm38_pool_mixed29")
    a = ap.parse_args()
    which = [int(x) for x in a.parts.split(",")]
    labs = SC.labels_pool(REPO / a.pool)
    n = len(labs)

    base = []
    for ef in a.embs:
        z = SC.FP.l2(np.load(ef).astype(np.float32))
        assert z.shape[0] == n
        for part, k in finch_partitions(z, which):
            base.append(part)
            print(f"[base] {Path(ef).name} k={k}")
    m = len(base)

    # co-association: 같은 클러스터로 묶인 비율 (n=1550 → 19MB)
    C = np.zeros((n, n), dtype=np.float32)
    for part in base:
        C += (part[:, None] == part[None, :])
    C /= m

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    # maximum lifetime cut: 병합 높이 사이 최대 gap 지점에서 자르기 (다이얼 0)
    h = Z[:, 2]
    gaps = np.diff(h)
    idx = int(np.argmax(gaps))
    thr = (h[idx] + h[idx + 1]) / 2.0
    pred = fcluster(Z, t=thr, criterion="distance") - 1
    print(f"[lifetime] cut={thr:.3f} (gap {gaps[idx]:.3f}) → k={len(set(pred))}")
    r = SC.score(pred, labs)
    print("EAC | " + " | ".join(str(x) for x in [r["capture"], r["recov"], r["noise_pct"],
                                                 r["nz_to_noise_pct"],
                                                 f"{r['n_clusters']}/{r['n_classes']}",
                                                 r["completeness"], r["homogeneity"]]))
    # 보조: lifetime 상위 5 cut 후보도 채점 (cap 1.0 후보 탐색 — 선택은 무라벨 lifetime 순)
    for rank in np.argsort(gaps)[::-1][1:5]:
        t2 = (h[rank] + h[rank + 1]) / 2.0
        p2 = fcluster(Z, t=t2, criterion="distance") - 1
        r2 = SC.score(p2, labs)
        print(f"EAC_alt(cut={t2:.3f},k={r2['n_clusters']}) | {r2['capture']} | {r2['recov']} | "
              f"{r2['n_clusters']}/{r2['n_classes']} | {r2['completeness']} | {r2['homogeneity']}")


if __name__ == "__main__":
    main()
