#!/usr/bin/env python3
"""grouping 결과 진단 — embedding 이 문제인지 HDBSCAN 파라미터가 문제인지 자동 판별.

재학습/재임베딩 안 함. 이미 저장된 embeddings.npy 만 읽어서 초 단위로 판정.

사용:
    python scripts/_diagnose_embeddings.py <경로>

<경로> 는 다음 중 아무거나:
    runs/<run>/contrastive/embeddings.npy        (파일 직접)
    runs/<run>/contrastive/                       (폴더 — embeddings.npy 자동 탐색)
    result_grouping/<run>/<folder>/               (grouping 결과 폴더)
    result_grouping/<run>/                        (하위 첫 embeddings.npy)

출력:
    1) embedding 퍼짐 통계 → collapse(모델 이상) 여부
    2) HDBSCAN 파라미터 조합별 그룹수/noise% 표
    3) 평이한 결론 ("embedding 이상" or "파라미터만 바꾸면 됨")
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np


def find_npy(path: Path) -> Path:
    if path.is_file() and path.suffix == ".npy":
        return path
    if path.is_dir():
        cands = sorted(path.rglob("embeddings.npy"), key=lambda p: len(p.parts))
        if cands:
            return cands[0]
    raise SystemExit(f"embeddings.npy 를 못 찾음: {path}\n"
                     f"  예: python scripts/_diagnose_embeddings.py runs/<run>/contrastive/")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("사용: python scripts/_diagnose_embeddings.py <embeddings.npy 또는 폴더>")
    npy = find_npy(Path(sys.argv[1]))
    emb = np.load(npy).astype(np.float32)
    n, dim = emb.shape
    print(f"[embeddings] {npy}")
    print(f"  N={n} 개,  dim={dim}")

    # ---------- 1) embedding 퍼짐 (collapse 진단) ----------
    # L2 정규화 (cosine 기준)
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(0)
    m = min(2000, n)                                   # 표본 (쌍거리는 N^2 라 표본)
    idx = rng.choice(n, size=m, replace=False)
    sub = norm[idx]
    sims = sub @ sub.T                                 # cosine 유사도 행렬
    iu = np.triu_indices(m, k=1)
    pair_sim = sims[iu]
    mean_sim = float(pair_sim.mean())
    # 최근접 이웃 유사도 (자기 자신 제외 max)
    np.fill_diagonal(sims, -1.0)
    nn_sim = sims.max(axis=1)
    per_dim_std = float(norm.std(axis=0).mean())

    print("\n[1] embedding 퍼짐 (collapse 여부)")
    print(f"  쌍 cosine 유사도 평균 : {mean_sim:.3f}   (1.0=전부 동일, 0=직교)")
    print(f"  최근접 이웃 유사도 평균: {float(nn_sim.mean()):.3f}")
    print(f"  차원별 std 평균        : {per_dim_std:.4f}")

    if mean_sim > 0.90:
        verdict_emb = "COLLAPSE 의심 (embedding 이 다 비슷 → 모델이 실데이터 변별 못 함)"
        collapsed = True
    elif mean_sim > 0.70:
        verdict_emb = "부분 collapse (퍼짐 약함 — 파라미터로 일부 가능하나 모델 개선 권장)"
        collapsed = False
    else:
        verdict_emb = "정상 (충분히 퍼짐 → 파라미터만 맞추면 그룹 갈라짐)"
        collapsed = False
    print(f"  → {verdict_emb}")

    # ---------- 2) HDBSCAN 파라미터 sweep ----------
    print("\n[2] HDBSCAN 파라미터별 그룹수 / noise%  (embedding 재사용, 빠름)")
    try:
        import hdbscan
    except Exception:
        raise SystemExit("hdbscan 미설치: pip install hdbscan")

    combos = []
    for method in ("eom", "leaf"):
        for mcs in (10, 25, 50):
            for ms in (3, 5, 10):
                for eps in (0.0, 0.02, 0.05):
                    combos.append((method, mcs, ms, eps))

    print(f"  {'method':>5} {'min_clu':>7} {'min_smp':>7} {'eps':>5} | {'groups':>6} {'noise%':>7}")
    best = None
    for method, mcs, ms, eps in combos:
        try:
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                 cluster_selection_method=method,
                                 cluster_selection_epsilon=eps, metric="euclidean")
            pred = cl.fit_predict(emb)
        except Exception as e:
            continue
        ng = len(set(int(p) for p in pred if p >= 0))
        noise = float((pred == -1).mean() * 100)
        flag = ""
        # "좋은" 후보: 그룹 2개 초과 + noise 80% 미만
        if ng >= 3 and noise < 80:
            score = ng - (noise / 20.0)
            if best is None or score > best[0]:
                best = (score, method, mcs, ms, eps, ng, noise)
            flag = " ←"
        print(f"  {method:>5} {mcs:>7} {ms:>7} {eps:>5} | {ng:>6} {noise:>6.1f}%{flag}")

    # ---------- 3) 결론 ----------
    print("\n[3] 결론")
    if collapsed:
        print("  ✗ embedding 이 collapse 됐다 (모델 문제).")
        print("    → 파라미터 바꿔도 제대로 안 갈라진다.")
        print("    → 해결: 실데이터(또는 실데이터 섞어) contrastive 재학습 필요.")
    elif best is not None:
        _, method, mcs, ms, eps, ng, noise = best
        print(f"  ✓ embedding 은 정상. HDBSCAN 파라미터만 바꾸면 갈라진다.")
        print(f"    추천: method={method}, min_cluster_size={mcs}, min_samples={ms}, epsilon={eps}")
        print(f"          → 그룹 {ng}개, noise {noise:.1f}%")
        print(f"    grouping 재실행 (재임베딩 없이 이 파라미터로):")
        print(f"      --min-cluster-size {mcs} --min-samples {ms}")
        print(f"      (CLUSTER_SELECTION_METHOD/EPSILON 은 predict_grouping_prod.py CONFIG 에서 {method}/{eps})")
    else:
        print("  △ 적정 그룹(3개 이상, noise<80%) 조합을 못 찾음.")
        print("    embedding 이 약하게 뭉쳐있을 가능성 → 모델 개선 권장.")


if __name__ == "__main__":
    main()
