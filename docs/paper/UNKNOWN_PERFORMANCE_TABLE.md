# Unknown contrastive 구성요소 성능표 (mixed29 생성 평가데이터)

portfolio.md 의 옛 38-class 표와 동일 컬럼 형식. 현 트랙 = MixedWM38 혼합 29-combo 평가.
- 데이터: 학습 WM-811K 단일 8종+Normal (5,149장, 무라벨) → 평가 MixedWM38 혼합 29종 (1,550장)
- 채점: FINCH p1 (parameter-free), 학습-평가 클래스·성분 무겹침
- M1 capture = 메인 클러스터로 등장한 불량 class 수 / 전체 (이진), M2 noise%, M3 Completeness, M4 Homogeneity

## 누적 ablation (cap-1 라인)
| # | Recipe (mixed29, FINCH p1) | M1 (capture) | M2 (noise %) | M3 (Completeness) | M4 (Homogeneity) | ARI | AMI | Sil | k(불량/전체/29) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Global InfoNCE only (q4k baseline) | 1.0000 | 0.00% | 0.5402 | 0.6396 | 0.2281 | 0.4970 | 0.099 | 66/71/29 |
| 2 | + Label Smoothing 0.02 | 1.0000 | 0.00% | 0.5602 | 0.6883 | 0.2336 | 0.5261 | 0.106 | 75/80/29 |
| 3 | + SoftNCE top-20 | 1.0000 | 0.00% | 0.5693 | 0.7043 | 0.2550 | 0.5391 | 0.081 | 77/82/29 |
| 4 | + KoLeo 0.1 (최종 cap-1 SOTA) | 1.0000 | 0.00% | 0.5619 | 0.7114 | 0.2473 | 0.5310 | 0.093 | 82/86/29 |
| 5 | 최종 recipe + kNN 2% 후처리 | 1.0000 | 0.34% | 0.5623 | 0.7119 | 0.2482 | 0.5312 | 0.095 | 82/86/29 |

## 보조 운영점 (cap 0.966, 다른 지표 우위)
| # | Recipe | capture | noise% | Comp | Hom | ARI | AMI | Sil | k(불량/전체/29) |
|---|---|---|---|---|---|---|---|---|---|
| 4a | + mass0.2 (recov-max, recov 0.5966) | 0.9655 | 0.00% | 0.5791 | 0.7087 | 0.2850 | 0.5500 | 0.113 | 72/77/29 |
| 4b | + Barlow (ARI/파편-max) | 0.9655 | 0.00% | 0.5833 | 0.6963 | 0.2912 | 0.5530 | 0.106 | 65/70/29 |

## Baseline 경쟁 (SSL method 선택, mixed29 FINCH p1 best-ep recov)
| method | capture | recov |
|---|---|---|
| SimCLR q4k (선택) | 1.000 | 0.5014 |
| Barlow Twins | 1.000 | 0.4959 |
| MoCo | 0.966 | 0.477 |
| SimSiam | 0.931 | 0.428 |
| VICReg | 1.000 | 0.405 |
| BYOL | 1.000 | 0.370 |
| DINO | 0.862 | 0.352 |

## unknown 합성 데이터셋 (E:/data/images/unknown, held-out 21/21) — 대조
| 운영점 | capture | recov | ARI | k(불량/전체/21) |
|---|---|---|---|---|
| duo frozen (학습 0) | 1.000 | 0.970 | 0.628 | 52/58/21 |
| duo frozen p2 (해석) | 0.905 | 0.867 | 0.846 | 19/21/21 |
- single-label 합성은 held-out 이어도 frozen 0.970 — 학습 불필요. 난이도 본질 = multi-label 중첩(mixed29).

## 음성 결과 (안 통한 옵션)
local-grid(DenseCL) / NeCo / MoCo / NV-filter / ig72 / dual-temperature / SCE / NNCLR / τ스윕 /
queue↑ / mass0.1·0.25 / topk15·25·30 / linear·ad head / FCMAE-on-mixed29 / 5-way 희석 / TEMI(클러스터러로는 FINCH 미달).
