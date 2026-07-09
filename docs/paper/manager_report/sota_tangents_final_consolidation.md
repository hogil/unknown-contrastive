# SOTA Tangents — Final Consolidation (paper finalization phase)

생성: 2026-05-13 (read-only research, no code changes, no training dispatch)

본 baseline: ConvNeXtV2-base + TAPT backbone + 5-epoch contrastive (Global InfoNCE + Local DenseCL + MoCo Queue 4096 + NV-Retriever NEG 0.72 + NeCo).
Best paper claim: NEW (3-comp, no Local) + HDBSCAN(eom, mcs=12, ms=3) → **ARI 0.859 ± 0.018** (3-seed).
Oracle ceiling: NEW + Agglo Ward K=42 → **0.901 ± 0.022**.
데이터: WM-811K wafer 합성 (32 obj-active class + 9 wafer-canvas + Normal) = **43 class**.

목표: paper finalization 단계에서 **추가 학습 minimal** (eval-only / post-hoc 우선) SOTA tangents 정리. 인용 가치 + 우리 도메인 적합성 검증.

---

## 1. Cluster post-processing (eval-only / 추가 학습 최소)

### 1.1 Iterative Cluster Harvesting (ICH) — 우리 도메인 직격탄

| 항목 | 값 |
|---|---|
| arxiv | [2404.15436](https://arxiv.org/abs/2404.15436) — Pleli et al., 2024-04-23 |
| backbone (paper) | Xception (ImageNet pretrained), **frozen** |
| clustering | PCA dim-reduce → Agglomerative (Ward, Euclidean) → silhouette filter loop |
| dataset (paper) | WM1K (1,302 wafer, 8 class) + WM811K_sub (923 wafer, 8 class) |
| 핵심 수치 | Homogeneity WM1K **0.90 (partial assignment)** vs OTC baseline 0.77 |

**method 요약**: 한 iteration 마다 (1) PCA → AC, (2) **silhouette score 가장 높은 cluster 한 개 만 confirm + 분리**, (3) 남은 sample 로 다시 PCA + AC. 정지 조건 `|FD'| > nPCA`. → **partial assignment** (전수 라벨 X) 산출 — 즉 confident sample 만 group 화하고 noise 는 미할당.

**우리 baseline 에 적용 가능성**: ★★★ (post-hoc, encoder 동결, eval-only)
- 학습 변경 없음. NEW 변종 embedding 위에 ICH 를 HDBSCAN 대안으로 시도 가능.
- 우리 데이터 = 43 class (≫ paper 8 class) — silhouette 분포 분산 더 크다. **early stopping criterion 튜닝 필요**.
- 다만 paper 의 평가는 Homogeneity 단일 — 우리 P1 (capture_rate) / P3 (Completeness) 호환성 직접 비교 X. **partial assignment 면 capture_rate 가 낮아질 위험** → 우리 P1 = 1.000 기준에 직접 부합 X.
- 권고: **paper 의 reference 로 인용** + ablation 1 row ("ICH post-hoc on NEW") — Oracle Ward K=42 와 비교축. 자동 mcs 튜닝 없는 alternative 라는 angle.

**예상 gain (paper claim 인용 only, 우리 도메인 측정 X)**: WM1K 8 class 에서 OTC 0.77 → ICH 0.90 homogeneity (+0.13). 우리 43 class 에는 직접 transfer 불명.

### 1.2 ODAR — outlier 와 normal 을 별도 cluster 로 분리하는 feature transform

| 항목 | 값 |
|---|---|
| arxiv | [2412.05669](https://arxiv.org/abs/2412.05669) — Li & Wang, 2024-12-07 |
| 핵심 | feature space 를 outlier 와 inlier 가 자동 두 cluster 로 분리되도록 변환 → 어떤 clustering 알고리즘이든 outlier 식별 가능 |
| paper 결과 | 10 dataset 중 7 best, 최소 +5% accuracy improvement |

**우리 baseline 적용**: ★★ (wrapper 한 단 추가)
- 우리 HDBSCAN noise (P2) 의 일부 = **Normal 으로 분류돼야 할 wafer** + **defect 인데 노이즈 처리된 wafer**. 후자가 capture_rate 손실 원흉.
- ODAR transform 을 HDBSCAN 전 단계로 끼우면 noise(def) 줄일 가능성. 단 paper 가 ARI 가 아닌 accuracy 기준 — 우리 metric 직접 매핑 X.
- 권고: paper inclusion 대신 **discussion 한 줄** 로 reference (post-hoc noise reduction 방향).

### 1.3 TANGO — typicality-aware mode-seeking (DPC 계열)

| 항목 | 값 |
|---|---|
| arxiv | [2408.10084](https://arxiv.org/abs/2408.10084) — PMLR 2025 |
| 핵심 | typicality-aware nonlocal mode-seeking — Density Peak Clustering 의 deep feature 적응 |

**우리 baseline 적용**: ★ (큰 변경 필요, DPC 자체 도입 학습 비용)
- HDBSCAN 대안으로 DPC + deep features 라인. 단 우리 NEW + HDBSCAN(eom mcs=12) 이 이미 0.859 도달 — DPC 가 큰 gain 줄지 불명.
- 권고: **related work 한 단락**만 — DPC 계열은 paper 의 "discussed alternatives" 로.

---

## 2. HDBSCAN dependent K-discovery 개선

### 2.1 Hybrid DBSCAN*/HDBSCAN cluster selection

| 항목 | 값 |
|---|---|
| arxiv | [1911.02282](https://arxiv.org/abs/1911.02282) — Malzer & Baum, v4 |
| 핵심 | EOM / leaf 외에 **threshold ε** 추가 → DBSCAN* 과 HDBSCAN 의 hybrid cluster 산출 |
| 효과 | 낮은 mcs 환경에서 micro-cluster 폭발 억제 + 다양 density 처리 |

**우리 baseline 적용**: ★★★ (HDBSCAN 인자 단순 추가)
- `hdbscan` 라이브러리에 `cluster_selection_epsilon` 인자 이미 존재. 우리 cfg `mcs=12` 에 `cluster_selection_epsilon=0.05~0.10` 추가 sweep 가능.
- 우리 noise(def) 6% lock-in 환경에서 noise 더 줄일 수 있음 (paper hybrid 의 강점). 단 P4 (Homogeneity) 침범 위험 — sweep 필요.
- 권고: **ablation 1-2 row** ("HDBSCAN + eps=0.05", "HDBSCAN + eps=0.10") — 별도 학습 X, eval only.

### 2.2 SNC (Selective Neighbor Clustering, CiPR/TMLR)

| 항목 | 값 |
|---|---|
| arxiv | [2304.06928](https://arxiv.org/abs/2304.06928) — Hao, Han, Wong, TMLR |
| github | https://github.com/haoosz/CiPR |
| 핵심 | semi-supervised hierarchical clustering — **selective neighbors 의 connected component** 로 hierarchy 구축 + joint reference score 로 unknown K 추정 |

**우리 baseline 적용**: ★★ (GCD 세팅 가정 X, hierarchy build 부분만 발췌)
- CiPR 는 GCD (partial label) 가정. 우리는 label 없이 (SSL only) — 단 SNC 의 cluster hierarchy + K estimation 부분만 떼서 적용은 가능.
- 우리 baseline 의 K 발견 = HDBSCAN 자동 (mcs 의존). SNC 로 K 추정한 후 우리 Agglo Ward Oracle 자리에 SNC K 를 대신 넣는 ablation 가능 → "Oracle K 없이 SNC K = ?" 비교.
- 권고: paper appendix 의 **K-discovery alternatives** 표 1 row 추가 가능. 우리 보다 좋아질 보장 X — paper claim 우리 도메인 직접 transfer 안 됨.

### 2.3 K-estimation (MTMC, model-bias dropout)

| arxiv | 핵심 |
|---|---|
| [2505.14044](https://arxiv.org/html/2505.14044) MTMC | maximum token manifold capacity — class token 다양성으로 K 추정 |
| [2412.12501](https://arxiv.org/abs/2412.12501) Model Bias GCD | dropout 으로 K estimation, "ground truth 와 근사" |

**적용성**: ★ (둘 다 GCD 세팅 + ViT class token 의존 — 우리 ConvNeXtV2 backbone 직접 transfer X)
- 권고: 인용 only (related work).

---

## 3. Unknown defect detection in wafer manufacturing — 2024-2025 ARI/F1 SOTA

### 3.1 DECOR (Deep Embedding Clustering with Orientation Robustness)

| 항목 | 값 |
|---|---|
| arxiv | [2510.03328](https://arxiv.org/abs/2510.03328) — Jothiraj et al., AAAI 2026 KGML Bridge (non-archival) |
| backbone | R2Conv (D4-equivariant CAE) + GroupPooling |
| clustering | DeepDPM (Dirichlet Process Mixture) |
| outlier | Isolation Forest + LOF ensemble, MAD threshold |
| dataset | MixedWM38 (multi-label split) |
| **수치** | **NMI 0.543 ± 0.03, ARI 0.296 ± 0.00** |

**우리 baseline 과 직접 비교**:
- 우리 NEW + HDBSCAN(eom mcs=12) ARI = **0.859** ≫ DECOR ARI 0.296 (MixedWM38).
- 단 **dataset 다름** — MixedWM38 = 38 mixed-pattern class (multi-label), 우리 = 43 single-label synthetic. 직접 비교 부정확.
- DECOR 의 R2Conv equivariance 는 우리 합성 데이터에 적합하지 않음 (우리는 회전 augment 자체를 금기 — scratch_rot 등 angle = class identity).
- 권고: **related work 인용** + "우리 도메인의 angle-aware class (e.g. scratch_rot) 때문에 D4 equivariance 부적합" 한 줄 contrast. 우리 ARI 0.859 가 압도적으로 보이지만 dataset 차이 강조 (paper integrity).

### 3.2 ViT-Tiny (지도학습 SOTA, MixedWM38)

| 항목 | 값 |
|---|---|
| arxiv | [2504.02494](https://arxiv.org/abs/2504.02494) — Mohammad & Ryu, 2025-04 |
| dataset | WM-38k (= MixedWM38) |
| **수치** | **F1 98.4%** (4-defect), MSF-Trans 대비 +2.94% |

**적용성**: ★ (지도학습 ceiling — unsupervised 비교 부적절)
- 우리 task = unsupervised clustering, ViT-Tiny = supervised classification — 직접 비교 X.
- 권고: **supervised upper bound** 로 한 줄 인용. "with labels, ViT-Tiny F1 98.4%; ours is label-free clustering, ARI 0.859" — fairness 강조.

### 3.3 Autoencoder + CNN (WM-811K, 98.56%)

| arxiv | [2411.11029](https://arxiv.org/abs/2411.11029) — Bao et al., 2024-11 |
| 핵심 | latent noise injection + CNN, **WM-811K 정확도 98.56%** |

**적용성**: ★ — 지도학습. 우리 unsupervised baseline 과 paradigm 다름. 인용 only.

### 3.4 Mean Teacher + SupCon (WM-811K semi-supervised)

| 항목 | 값 |
|---|---|
| arxiv | [2411.18533](https://arxiv.org/abs/2411.18533) |
| dataset | WM-811K (9 class) |
| **수치** | **F1 83.40%** (vs baseline ResNet 78.87%, +4.53%) |
| 핵심 | Mean Teacher EMA + SupCon loss |

**적용성**: ★★ (SupCon 이지만 label 있는 semi-supervised — 우리 SSL only 와 다름)
- 우리 feedback memory: **SupCon 거부** (`feedback_no_multicrop_no_supcon.md`) — unknown defect generalization 손상 우려. SSL InfoNCE 만.
- 권고: paper related work — "SupCon-based methods (Mean Teacher F1 83.4%) require labels; ours is fully unsupervised". 인용 only, 도입 X.

---

## 4. Contrastive feature quality metric (post-hoc 적용 가능)

### 4.1 RankMe — effective rank (★★★ 우리 paper 에 즉시 추가 가능)

| 항목 | 값 |
|---|---|
| arxiv | [2210.02885](https://arxiv.org/abs/2210.02885) — Garrido, Balestriero, Najman, LeCun, ICML 2023 |
| 핵심 | embedding matrix 의 **effective rank** (Shannon entropy of singular values 의 exp) |
| 적용 | **fully post-hoc**, label X, hyperparam X |
| github | https://github.com/facebookresearch/active-self-supervised-learning (관련) |

**우리 baseline 적용**: ★★★ (eval-only, label X, 추가 학습 X)
- 우리 4 variant (NEW, FUSION, …) 의 128-dim embedding 위에서 effective rank 직접 계산 가능.
- 우리 paper 의 alignment+uniformity 표에 **RankMe column** 한 개 추가하면 representation quality 다각도 비증 강화. SOTA 평가 표준이라 reviewer 가 기대.
- **예상 gain**: ARI 같은 downstream 변화 X (post-hoc metric), but **paper 신뢰도 + reviewer 수용도** 상승. Iter table 의 추가 column.

### 4.2 NESum / α-ReQ / stable rank (Tsitsulin 2023, "Unsupervised Embedding Quality Evaluation")

| 항목 | 값 |
|---|---|
| arxiv | [2305.16562](https://arxiv.org/abs/2305.16562) — Tsitsulin et al., PMLR 221 |
| 비교 metric | RankMe, α-ReQ, NESum, stable rank, self-clustering, coherence, condition number |
| paper 주장 | **NESum + stable rank** 가 RankMe 보다 더 안정적으로 downstream 성능과 상관 |
| 비고 | NESum, Stable rank 한 cluster / Coherence, α-ReQ, RankMe, condition number 다른 cluster (paper의 PCA 분석 결과) |

**적용성**: ★★★ (RankMe 와 동일하게 post-hoc, embedding matrix 만 있으면 됨)
- 권고: **RankMe + NESum 두 metric 같이 산출** + paper 보조 표에 추가. 단일 metric 의존 X — Tsitsulin paper 가 RankMe 의 outlier 이슈 지적.

### 4.3 alignment + uniformity (이미 우리 사용 중)

| arxiv | [2005.10242](https://arxiv.org/abs/2005.10242) — Wang & Isola, ICML 2020 |
| status | 우리 baseline 의 monitoring metric. 추가 X. |

권고: paper 의 RankMe + NESum 추가 후 alignment/uniformity 와 4-metric panel 로 representation quality 종합 평가.

---

## 5. Anchor-based / k-NN based clustering refinement (HDBSCAN noise 감축)

### 5.1 Soft HDBSCAN (probabilistic membership, scikit-learn-contrib)

| 항목 | 값 |
|---|---|
| 출처 | hdbscan 라이브러리 native API + RAPIDS cuML 가속 |
| 핵심 | `all_points_membership_vectors()` — 각 sample 의 모든 cluster 소속 확률 산출 |

**우리 baseline 적용**: ★★★ (HDBSCAN config 변경 만)
- 현재 noise(def) 처리된 wafer 중 일부는 boundary case — soft membership 으로 가장 가까운 cluster 에 reassign 가능 (확률 threshold τ).
- 우리 P1 (capture_rate) 손실 wafer (현재 0 — paper claim 이미 1.000) 의 robustness 강화 + P2 (noise) 6% → 더 낮춰질 가능성.
- 권고: **paper ablation 한 row** — "soft HDBSCAN with τ=0.3 reassignment" — 우리 ARI 0.859 와 비교.

### 5.2 k-NN voting on HDBSCAN noise points (post-hoc, label propagation)

- HDBSCAN 노이즈 처리된 sample 에 대해 k-NN (k=5~15) 로 가장 빈도 높은 cluster label 부여 — graph-prop 방식의 가장 단순한 형태.
- arxiv: explicit paper 없음 (BERTopic / UMAP+HDBSCAN 워크플로우의 common practice). 우리 baseline 의 sanity-check ablation 으로 충분.

**적용성**: ★★★ (10 줄 wrapper, eval only)
- 권고: paper 의 "noise reduction" ablation — pre-soft 0.859 vs post-knn 0.86x. gain 작더라도 reviewer-safe.

### 5.3 ODAR (재방문, 1.2 절)

- §1.2 참조. wafer outlier 와 normal 분리 framework — k-NN refinement 와 결합 가능.

---

## Recommended action items (paper finalization)

| # | item | 적용성 | 추가 학습 | paper 가치 |
|---|---|---|---|---|
| A1 | **RankMe + NESum post-hoc metric** column 을 4 variant 표에 추가 | ★★★ | X | high (representation quality 종합 평가, reviewer-expected) |
| A2 | **HDBSCAN `cluster_selection_epsilon` sweep** (0.05, 0.10) 1-2 row | ★★★ | X | medium (noise 추가 감축 가능, eom 외 alternative) |
| A3 | **Soft HDBSCAN τ-reassignment ablation** 1 row | ★★★ | X | medium (P2 noise 감축, paper robustness) |
| A4 | **ICH (2404.15436) related work + ablation** 1 row (post-hoc Agglo + silhouette filter, no mcs) | ★★ | X | high (직접 wafer 도메인 paper, 우리 비교축) |
| A5 | **DECOR (2510.03328) related work**: ARI 0.296 (MixedWM38) vs ours 0.859 — dataset 차이 강조 | ★ | X | high (2025-2026 wafer SOTA 비교점) |
| A6 | **Mean Teacher + SupCon related work**: F1 83.4% with labels vs ours unsupervised | ★ | X | medium (semi-supervised 대비 unsupervised positioning) |
| A7 | **ViT-Tiny supervised ceiling**: F1 98.4% as upper bound — 우리 unsup ARI 0.859 의 context | ★ | X | medium (fairness 강조) |

**도입 권고 (paper revision)**:
1. METHOD 또는 EVAL 섹션에 RankMe/NESum 산출식 + 우리 4 variant 결과 표 추가 (A1).
2. RESULTS ablation 표에 HDBSCAN eps sweep + soft τ row 추가 (A2, A3).
3. RELATED WORK 섹션에 ICH/DECOR/MeanTeacher+SupCon/ViT-Tiny 4 paper 단락 1 개 추가 (A4-A7).
4. DISCUSSION 에 "우리 ARI 0.859 의 위치" — DECOR 0.296 (다른 데이터셋) + ViT-Tiny 98.4% F1 (지도학습) 의 두 reference 사이.

**도입 비권고**:
- SupCon — 우리 feedback memory 명시 거부.
- DPC / TANGO — HDBSCAN 이 이미 충분, paradigm 변경 비용 큼.
- D4-equivariance (DECOR) — 우리 angle-class 정체성과 충돌.
- GCD K-estimation (MTMC, model-bias dropout) — partial label 가정 X.

---

## Fetch / search 실패 / 부분 실패 기록 (fact gate)

- `https://arxiv.org/pdf/2305.16562` PDF binary fetch 실패 — Tsitsulin "Unsupervised Embedding Quality Evaluation" 의 정확한 NESum 수식 / dataset 추출 미완료. 대신 WebSearch 의 paper 요약 인용 ("NESum + stable rank 가 RankMe 보다 안정") 만 사용.
- `https://arxiv.org/abs/2411.18533` 초기 fetch 정보 부족 → `/html/2411.18533v1` 으로 재시도, F1 수치 (83.40%) 정상 추출.
- DECOR (2510.03328) abstract: 첫 fetch ARI/F1 미상, `/html` 재시도로 NMI 0.543 / ARI 0.296 정상 추출.
- 그 외 모든 paper 는 첫 fetch 에서 정상 데이터 확보.

## 결론 한 줄

A1 (RankMe+NESum) + A2 (eps sweep) + A3 (soft τ) + A4-A7 (related work 단락) = 추가 학습 0 + paper 신뢰도 + SOTA positioning. ARI gain 은 A2/A3 에서만 가능, A1/A4-A7 은 paper integrity 강화축.

[OUT] D:\project\unknown-contrastive\docs\paper\manager_report\sota_tangents_final_consolidation.md
