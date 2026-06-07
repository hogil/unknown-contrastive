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

출력 (벡터간 거리 분석 중심, HDBSCAN 안 함):
    [A] 벡터 크기(L2 norm) 통계
    [B] 쌍거리 분포 (cosine/euclid 분위수 + 텍스트 히스토그램)
    [C] 최근접 이웃 거리 (얼마나 빽빽한가 — epsilon 에 다 들어가는지)
    [D] 차원 활용도 / PCA 분산 (소수 차원 집중 = collapse)
    [판정] embedding 이 collapse 인지 정상인지
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

    rng = np.random.default_rng(0)

    # raw norm 통계 (정규화 전 — 벡터 크기 자체가 다 같은지)
    raw_norms = np.linalg.norm(emb, axis=1)
    print("\n[A] 벡터 크기(L2 norm, 정규화 전)")
    print(f"  mean={raw_norms.mean():.4f}  std={raw_norms.std():.4f}  "
          f"min={raw_norms.min():.4f}  max={raw_norms.max():.4f}")

    # L2 정규화 (cosine/거리 기준 통일)
    norm = emb / (raw_norms[:, None] + 1e-9)

    # ---------- [B] 쌍거리 분포 (표본) ----------
    m = min(3000, n)
    idx = rng.choice(n, size=m, replace=False)
    sub = norm[idx]
    sims = sub @ sub.T
    iu = np.triu_indices(m, k=1)
    pair_sim = sims[iu]                                   # cosine 유사도
    pair_cos_dist = 1.0 - pair_sim                        # cosine 거리 (0=동일, 2=정반대)
    # 유클리드 거리 (정규화 벡터에선 sqrt(2-2cos))
    pair_eu = np.sqrt(np.clip(2.0 - 2.0 * pair_sim, 0, None))

    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    cos_q = np.percentile(pair_cos_dist, qs)
    eu_q = np.percentile(pair_eu, qs)
    print(f"\n[B] 쌍거리 분포 (표본 {m}개, {len(pair_sim):,} 쌍)")
    print(f"  cosine 유사도  mean={pair_sim.mean():.3f}  (1=동일, 0=직교)")
    print(f"  {'분위수':>6} | " + " ".join(f"{q:>5}%" for q in qs))
    print(f"  {'cos거리':>6} | " + " ".join(f"{v:>6.3f}" for v in cos_q))
    print(f"  {'eu거리 ':>6} | " + " ".join(f"{v:>6.3f}" for v in eu_q))
    # 텍스트 히스토그램 (cosine 거리)
    print("  cos거리 히스토그램:")
    hist, edges = np.histogram(pair_cos_dist, bins=12, range=(0, max(0.001, pair_cos_dist.max())))
    hmax = max(1, hist.max())
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        bar = "#" * int(40 * h / hmax)
        print(f"    {e0:5.3f}~{e1:5.3f} | {bar} {h}")

    # ---------- [C] 최근접 이웃 거리 (얼마나 빽빽한가) ----------
    np.fill_diagonal(sims, -1.0)
    nn_sim = sims.max(axis=1)
    nn_cos_dist = 1.0 - nn_sim
    print(f"\n[C] 최근접 이웃 cosine 거리 (작을수록 빽빽 = epsilon 에 다 들어감)")
    nnq = np.percentile(nn_cos_dist, [0, 25, 50, 75, 95, 100])
    print(f"  min={nnq[0]:.4f}  q25={nnq[1]:.4f}  median={nnq[2]:.4f}  "
          f"q75={nnq[3]:.4f}  q95={nnq[4]:.4f}  max={nnq[5]:.4f}")
    frac_lt_eps = float((nn_cos_dist < 0.06).mean() * 100)
    print(f"  최근접거리 < 0.06(현 epsilon) 비율: {frac_lt_eps:.1f}%   "
          f"(높으면 epsilon 이 다 합침)")

    # ---------- [D] 차원 활용도 / PCA 분산 ----------
    per_dim_std = norm.std(axis=0)
    eff_dims = float((per_dim_std > 0.01).sum())
    # PCA — 상위 분산 비중
    try:
        c = norm - norm.mean(0, keepdims=True)
        cov = (c.T @ c) / max(1, len(c) - 1)
        evals = np.linalg.eigvalsh(cov)[::-1]
        evals = np.clip(evals, 0, None)
        tot = evals.sum() + 1e-12
        pc1 = float(evals[0] / tot * 100)
        pc10 = float(evals[:10].sum() / tot * 100)
        # 분산 90% 설명에 필요한 차원 수
        cum = np.cumsum(evals) / tot
        d90 = int(np.searchsorted(cum, 0.90) + 1)
    except Exception:
        pc1 = pc10 = d90 = -1
    print(f"\n[D] 차원 활용도 (collapse 면 소수 차원에 분산 몰림)")
    print(f"  std>0.01 인 차원 수 : {eff_dims:.0f} / {dim}")
    print(f"  PC1 분산 비중       : {pc1:.1f}%   (한 축에 몰리면 collapse)")
    print(f"  상위 10 PC 분산 비중: {pc10:.1f}%")
    print(f"  분산 90% 설명 차원수: {d90}  (작을수록 collapse)")

    # ---------- 종합 판정 ----------
    mean_sim = float(pair_sim.mean())
    print("\n[판정] embedding 상태")
    print("  (핵심: 전체 쌍거리 = global 퍼짐. 최근접거리 작은 건 cluster tight 라 오히려 좋음 — collapse 근거 아님)")
    reasons = []
    score_collapse = 0
    # ★ collapse 핵심 = 전체 쌍 유사도(global)가 높음 = 다 비슷한 방향
    if mean_sim > 0.90:   score_collapse += 3; reasons.append(f"쌍 cosine 유사도 {mean_sim:.2f} 매우 높음(거의 동일)")
    elif mean_sim > 0.70: score_collapse += 2; reasons.append(f"쌍 cosine 유사도 {mean_sim:.2f} 높음")
    elif mean_sim > 0.50: score_collapse += 1; reasons.append(f"쌍 cosine 유사도 {mean_sim:.2f} 다소 높음")
    # 소수 차원/축에 분산 집중
    if pc1 > 50:   score_collapse += 2; reasons.append(f"PC1 분산 {pc1:.0f}% 한 축 집중")
    elif pc1 > 30: score_collapse += 1; reasons.append(f"PC1 분산 {pc1:.0f}%")
    if eff_dims < dim * 0.1: score_collapse += 1; reasons.append(f"활성 차원 {eff_dims:.0f}/{dim} 극소")
    if 0 < d90 <= 3:         score_collapse += 1; reasons.append(f"분산 90% 가 {d90} 차원에 집중")

    if score_collapse >= 4:
        print("  [XX] 심한 COLLAPSE — embedding 이 거의 한 점. 모델이 변별 못 함.")
    elif score_collapse >= 2:
        print("  [X]  COLLAPSE 의심 — embedding 이 좁게 뭉침.")
    else:
        print("  [OK] 정상 — embedding 충분히 퍼짐 (그룹 못 갈라지면 HDBSCAN 파라미터 문제).")
    if reasons:
        print("  근거: " + ", ".join(reasons))
    print(f"\n  (collapse 면 모델 재학습, 정상이면 다음 단계로 HDBSCAN 파라미터 — "
          f"python scripts/_diagnose_embeddings.py {npy} 의 [A~D] 수치로 판단)")


if __name__ == "__main__":
    main()
