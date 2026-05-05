# Results

매 iteration 마다 paper-recorder agent 가 새 row 추가.

## 표 1 — Tier 1 metric 비교 (전체 run)

| Run | Date | Completeness | AMI | noise_pct (def) | class_capture | frac_single_cluster |
|---|---|---|---|---|---|---|
| **Iter 0** baseline | 2026-05-05 | **0.9466** | **0.9288** | **0.71%** | **38/38 (1.000)** | **34/38 (0.8947)** |

**해석 (Iter 0)**:
- Completeness 0.95 — 같은 class 가 같은 cluster 에 모이는 정도 매우 높음
- AMI 0.93 — chance 대비 매우 정렬된 clustering
- noise 0.71% — defect 격리 거의 완벽
- 모든 38 defect class 가 적어도 1 cluster 에 잡힘
- 89.5% class 가 단일 cluster — 4 class 만 split (Full_×3 + Thick-Edge_fork)

## 표 2 — Tier 2 보조 metric

| Run | Homogeneity | Silhouette (cosine) | ARI |
|---|---|---|---|
| Iter 0 | 0.9154 | 0.5664 | 0.7002 |

**ARI 가 낮은 이유**: HDBSCAN 가 39 cluster 산출 (= 39 GT class 와 같으나 매핑 안 일치).
ARI 는 over-cluster 페널티 강함. Tier 1 (Completeness / AMI) 가 더 적합.

## 표 3 — class_fragmentation_summary detail

| Run | n_total | captured | uncaptured | mean_coverage | weighted_coverage | mean_n_clusters | single | split_2 | split_3+ |
|---|---|---|---|---|---|---|---|---|---|
| Iter 0 | 38 | 38 | 0 | 0.9904 | 0.9929 | 1.105 | 34 | 4 | 0 |

## 표 4 — Split classes detail (Iter 0)

| class | n | n_clusters | dom_recall | coverage | 진단 |
|---|---|---|---|---|---|
| `Full_bank_boundary` | 200 | 2 | 0.555 | 1.000 | 200 → 111+89 split (no noise). intra/inter ratio **8.72** ★ |
| `Full_scratch_rot` | 200 | 2 | 0.505 | 0.990 | 101+97+2noise. ratio 4.44 |
| `Full_fork` | 200 | 2 | 0.540 | 0.985 | 108+89+3noise. ratio 2.66 |
| `Thick-Edge_fork` | 50 | 2 | 0.480 | 0.860 | 24+19+7noise. ratio 2.13 (작은 class) |

**발견**: 4 class 모두 합성 데이터의 진짜 두 sub-style.
검증: HDBSCAN sweep (모든 hyperparameter 에서 동일 split) + GMM BIC bimodal +
intra/inter ratio 2-9×. 자세히 `docs/contrastive-eval/DECISIONS.md` D-10.

## 표 5 — Wang-Isola alignment + uniformity

| Run | alignment ↓ | uniformity ↓ (negative) | method | 해석 |
|---|---|---|---|---|
| Iter 0 | 0.3018 | -2.4955 | intra_class_proxy | 정상 학습. uniformity 약간 부족 (target -3+). 학습 epoch ↑ 또는 augment 다양화 여지 |

## 표 6 — Retrieval (val embedding nearest neighbor)

| Run | recall@1 | recall@5 | recall@10 | mAP@10 | min recall@1 (worst class) |
|---|---|---|---|---|---|
| Iter 0 | 0.9936 | 0.9923 | 0.9910 | 0.9938 | 0.935 (Edge-Bottom_scratch) |

**해석**: embedding quality 우수. 같은 class 끼리 매우 잘 모음. 다만 HDBSCAN clustering
시 Full_*** sub-style 분리 같은 over-cluster 영향. Retrieval 99.36% 와 ARI 0.70 의 gap 이
이를 반영.

## 다음 iteration 계획

| Iter | 변경 | 예상 효과 |
|---|---|---|
| Iter 1 | per-class sample 수 random (50~200+) | production 비율 흉내, generalization 검증 |
| Iter 2 | alignment + uniformity epoch monitoring | collapse 감지, plateau 시 stop |
| Iter 3 | Hard Negative Mining (β=1.0) | Edge-Top × bank_boundary 4-class merge 분리 |
| Iter 4 | (optional) USE_LOCAL=True | grid spatial contrast — Full sub-style 추가 신호 |

각 iteration 진행 시 위 표 row 추가 + ITERATIONS.md 에 변경 history 추가.
