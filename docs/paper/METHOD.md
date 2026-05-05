# Method

## 1. Data synthesis (sister repo `known-cnn`)

### 1.1 Source distribution
- **WM-811K** wafer-map dataset (Wu et al. 2015).
- 8 base distribution classes learned via per-class heatmap (`dist_learn/_dist_learn.py`).

### 1.2 Wafer composition (6400 × 6400 palette PNG)
- 32 × 32 grid of 200 × 200 chips.
- Per-class: distribution of {normal / defect / invalid} chips drawn from learned heatmap.
- Defect chips: 75% primary object + 25% mixed (other chip-objects from neighboring distribution).
- Palette: 8-bit indexed PNG, indices 0-7 (defect grades 0=clean → 7=severe), 8-23 (border/special), 24-31 (background/invalid).

### 1.3 Object catalog (round-26)
| obj | description |
|---|---|
| `bank_boundary` | small geometric boundary mark, main grade 1+sub 2 |
| `fork` | branching scratch pattern, main grade 3 |
| `scratch` | linear scratch, main grade 3 |
| `scratch_rot` | rotated scratch (`scratch` 의 angle 변형), main grade 3 |
| `invalid_main` | invalid-pattern chip, used in `*_invalid_main` classes |

Plus **9 wafer-canvas patterns** (no per-chip object): `BrokenRing`, `CenterDonut`,
`CrescentArc`, `CrossScratch`, `DiagonalSmear`, `ParallelScratches`, `RingDots`, `Row`,
`Starburst` — generated via wafer-level alpha mechanism (`dist_apply/_sample_canvas_gen.py`).

## 2. Backbone

- **ConvNeXtV2-base** (Woo et al. 2023, ECCV).
- Pretraining: **FCMAE** (fully-convolutional MAE, Woo et al. 2023). Init weights at
  `models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth`.
- **TAPT (task-adaptive pre-training)**: 33-class supervised classifier trained on the
  same wafer dataset (sister repo `known-cnn`), state_dict extracted via `run_contrastive.py`
  → injected as `LOCAL_BACKBONE_WEIGHTS` for contrastive head training.
- Backbone is **frozen** during contrastive training (FREEZE_BACKBONE=True). Only the
  projection head and queue trainable.

## 3. Contrastive training

### 3.1 Architecture
- **Projection head**: 2-layer MLP, output dim **128**, L2-normalized (unit hypersphere).
- Input: 384 × 384 RGB (palette PNG converted to RGB at load).
- Encoder output → projection → InfoNCE.

### 3.2 Loss
**InfoNCE** (Oord et al. 2018, arxiv 1807.03748):
```
L = -log[ exp(sim(z₁,z₂)/τ) / Σ exp(sim(z₁,z_neg)/τ) ]
```
- `z₁, z₂` = augmented two views of same wafer (positive pair).
- `z_neg` = (a) other in-batch wafers, (b) **MoCo-style momentum queue** (size 4096).
- τ (temperature) = 0.07.
- sim = cosine similarity (since both L2-normalized).

Three components:
- **Global InfoNCE** (G) — wafer-level positive/negative.
- **Queue InfoNCE** (Q) — extends global with 4096-size memory bank (He et al. 2020).
- **Local InfoNCE** (L, optional) — grid-cell-level intra-image contrast (USE_LOCAL flag).
  Currently disabled (USE_LOCAL=False) in baseline; will be re-evaluated.

### 3.3 Augmentation (wafer-safe)
| op | range | note |
|---|---|---|
| RandomRotation | ±15° | 검사장비 stage 회전 오차 |
| RandomAffine translate/scale | ±3% | alignment / magnification |
| Gaussian noise | σ=0.01 | sensor noise |

**Forbidden** (wafer class identity 손상):
- HFlip (`scratch_rot` angle 반전 → 다른 sub-class 됨)
- VFlip / 180° rotation (Edge-Top ↔ Edge-Bottom)
- ColorJitter (palette grade 0-7 의미 손상)
- MixUp / CutMix / Cutout
- Multi-crop with random position (SwAV 풍 — wafer 위치 정보 손상, 채택 거부 D-4)

### 3.4 Hyperparameters (baseline)
- BATCH = 16, IMAGE_SIZE = 384.
- EPOCHS = 10 (warmup 1).
- LR_HEAD = 1e-3 (head only, backbone frozen).
- TRAIN_SAMPLING_RATIO = 1.0.

## 4. Clustering (post-training)

- **HDBSCAN** (McInnes et al. 2017, JOSS).
- Parameters (baseline):
  - `min_cluster_size=12`, `min_samples=4`
  - `cluster_selection_method='leaf'` (default eom 도 시험 — `eom` 이 평균 더 큰 cluster)
  - `cluster_selection_epsilon=0.06`
  - metric=euclidean (cosine 정규화 후 등가)

## 5. Inference

- Single forward through frozen encoder + projection head → 128-d embedding (L2 normalized).
- HDBSCAN `approximate_predict` (또는 fresh clustering).
- Cluster medoid 추출 (`cluster_summary/cluster_*_medoid_dist*.png`).

## 6. Composite visualization

- Per-cluster top-K=20 medoid wafers → 두 가지 RGB 합성 (`compose_clusters.py`):
  - **binary**: pixel 별 (1≤grade≤7) 인 wafer 비율, 0=white → 1=red
  - **grademean**: pixel 별 grade 합 / N / 7, 0=white → 1=red
- Invalid skip: 모든 K wafer 가 grade≥8 인 픽셀은 흰색 강제.
- 6400 × 6400 RGB PNG.

## 7. Evaluation metrics (Tier 1 + 2)

`docs/contrastive-eval/METRICS.md` 정책 준수. **공식 metric 만**, 커스텀 금지.

### Tier 1 (필수 표 1행)
- **Completeness** (Rosenberg-Hirschberg 2007) — P3
- **AMI** (Vinh et al. 2010) — chance-corrected
- **noise_pct (defect only)** (HDBSCAN 표준) — P2
- **class_capture_rate** = (n_clusters≥1 인 class) / 전체 — P1

### Tier 2 (보조)
- Homogeneity (Rosenberg 2007)
- Silhouette cosine (Rousseeuw 1987)
- ARI (Hubert & Arabie 1985) — over-cluster 페널티 inherent

### class_fragmentation_summary (사용자 criteria A/B/C)
- A. class_capture_rate
- B. weighted_cluster_coverage = (defect 가 cluster 에 속한 비율, sample-weighted)
- C. frac_single_cluster = (n_clusters==1 인 class 비율)

## 8. 학습 도중 monitoring (Iter 1+ 도입)

`docs/contrastive-eval/MONITORING.md`:
- **Alignment + Uniformity** (Wang & Isola 2020) — label 무관, 매 epoch
- **k-NN top-1** — label 있는 작은 subset 만, 옵션
- **Periodic HDBSCAN + Tier 1** — 5 epoch 마다, label 있을 때

## 9. 거부된 옵션 (사유 `docs/contrastive-eval/DECISIONS.md`)

- D-4: **Multi-crop** (SwAV) — wafer 위치 정보 손상
- D-5: **SupCon 주력** — production unknown defect generalization 위험

## 10. 향후 개선 (계획)

- **Hard Negative Mining** (Robinson 2021, ICLR) — InfoNCE β param 도입. label 무관, production 호환. `docs/contrastive-eval/HARD_NEGATIVE.md`.
- **Production-realistic sampling** — per-class sample 수 random (50~200+).
- 합성 sub-style 통일 또는 sub-class 분리 (Full_*, Thick-Edge_fork).
