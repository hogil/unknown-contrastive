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
- **Local InfoNCE** (L) — grid-cell-level intra-image contrast (`USE_LOCAL=True`),
  weighted by `LOCAL_WEIGHT`. DenseCL-style (Wang et al. 2021) dense feature alignment
  between two augmented views' grid cells. ★ **LOCAL_WEIGHT atomic sweep (Iter A0, 1, 3)**:
  - LW=0.5 (Iter A0): Completeness=0.9375, AMI=0.8946, noise=9.34%, capture=1.000
  - LW=0.7 (Iter 3): Completeness=0.936, AMI=0.893, noise=9.42%, capture=1.000 — **NULL** (A0 와 통계적 동일, sister-pair centroid 모두 소수점 셋째 자리 동일)
  - LW=1.0 (Iter 1): Completeness=0.9481, AMI=0.9040, noise=4.62%, capture=1.000 — **best (sweet spot)**
  
  **★ LW=1.0 sweet spot 가설**: 0.5 ↔ 0.7 plateau (null), 0.7 ↔ 1.0 비선형 jump
  (noise 9.42% → 4.62%, -50%). local contrast 가 일정 weight 임계 (≈0.85?) 넘어야
  효과 발현되는 비선형 구간 추정. weight ↑ 마다 monotonic 개선이 아니므로 fine-grained
  탐색 (LW=1.5) 이 다음 단계 후보. trade-off: `Donut_scratch_rot` noise 26.7% → 40%
  (Iter A0→1, 회전 sub-style 자체 학습 부족 가설).

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

### 3.4 Hyperparameters

**Iter 0 (historical, old anchor)**:
- BATCH = 16, IMAGE_SIZE = 384, EPOCHS = 10 (warmup 1)
- LR_HEAD = 1e-3 (head only, backbone frozen), TRAIN_SAMPLING_RATIO = 1.0
- USE_LOCAL = False

**Iter A0+ (new method-track, D-15 anchor)**:
- BATCH = **8**, IMAGE_SIZE = 384, EPOCHS = **5**
- GPU per-step throttle 5000ms (driver TDR 회피, EXPERIMENTS.md 참조)
- LR_HEAD = 1e-3, USE_LOCAL = **True**, LOCAL_WEIGHT (Iter A0=0.5, Iter 1=1.0)
- IGNORE_NEG_SIM = 0.72, QUEUE_SIZE = 4096, TEMP = 0.07

### 3.5 NeCo patch-neighbor consistency (lever 5, iter 37 SOTA, 2026-05-09)

**NeCo** (Pariza et al. 2024, arXiv:2408.11054) — patch-neighbor consistency loss
between two augmented views. For each spatial token in view 1, enforce ranking
consistency with its k-nearest neighbors in view 2's token set:

```
L_NeCo = Σ_i KL( softrank(sim(z₁ᵢ, neighbors of z₁ᵢ)) || softrank(sim(z₂ᵢ', neighbors of z₂ᵢ')) )
```

Combined loss:
```
L_total = L_global + L_queue + LOCAL_WEIGHT · L_local + NECO_WEIGHT · L_NeCo
```

**Sweep result (iter 37/38/39, new anchor)**:
- NECO_WEIGHT = **0.2** (iter 37): Comp=0.991, AMI=0.960, ARI=0.870, noise=0.61% — **★ SOTA**
- NECO_WEIGHT = 0.1 (iter 38, under): Comp=0.985, AMI=0.956, noise=0.52%, but mixed_clusters 6→7
- NECO_WEIGHT = 0.3 (iter 39, over): Comp=0.980, AMI=0.954, ARI=0.868, noise=1.05%

**★ NECO_WEIGHT=0.2 sweet spot lock-in** — 0.1 (under-signal, sister-pair partial separation)
와 0.3 (over-signal, dominate G/L) 모두 reject. Sweet-spot 좁음 (0.2 ± 0.1 가 가까운 reject)
→ fragile 한 lever 임을 명시.

**★ Base-cfg 의존성** — NeCo 는 P2 King base (LR=1e-3, NEG=0.72, TEMP=0.07) 위에서만 SOTA.
Quality King base (LR=5e-4, NEG=0.65, TEMP=0.05) + NeCo 0.2 (iter 40) = ARI -13pp regression
으로 negative interaction 확인. Lever 들이 독립적이지 않으며 base cfg 에 따라 부호가 바뀜.

**Variant (iter 43, in progress)**: Zone-Aware NeCo — `NECO_ZONE_VERTICAL=3` 로 wafer 를
top/middle/bottom 3 vertical zone 으로 분할, zone 내 patch-neighbor consistency 만 강제.
Edge-Top vs Edge-Bottom 같은 위치 sub-style 직격 의도.

### 3.7 Component Isolation Methodology (Real Baseline B0-B5, 2026-05-11) — ★ NEW

기존 atomic ablation (Iter A0 → iter 58) 의 limitation: A0 baseline 에 이미 Local InfoNCE
(DenseCL) / MoCo Queue / NEG filter 활성 → lever 효과는 그 위에서의 incremental tuning.
진짜 component-level contribution isolation 위해 **Real Baseline (B0)** 부터 component 단계별
추가하는 protocol 도입.

#### Protocol

```
B0 — Real Baseline      USE_LOCAL=false, USE_QUEUE=false, NEG=off (1.0), NeCo=0
B1 — + Local DenseCL    USE_LOCAL=true, LW=0.5
B2 — LW=1.0             LW=0.5 → 1.0 (lever 1 isolated)
B3 — + MoCo Queue       USE_QUEUE=true, QUEUE_SIZE=4096
B4 — + NEG filter       IGNORE_NEG_SIM=0.72
B5 — + NeCo 0.2         NECO_WEIGHT=0.2 (= iter 37 cfg)
```

**고정 cfg (모든 B 단계 동일)**: ConvNeXtV2-base FCMAE + TAPT, IMAGE_SIZE=384, BATCH=8, EPOCHS=5,
WARMUP=1, LR_HEAD=1e-3, NCE_TEMP=0.07, SEED=42, anchor avg30_new_260508_123037 (43 class, n=2146).
HDBSCAN eom mcs=12 ms=3 모든 row 동일 (encoder 학습 무관 axis 분리).

#### 주요 측정 (RESULTS 표 13)

| step | atomic 변경 | ΔARI | Δnoise | 판정 |
|:-:|---|---:|---:|---|
| B0 → B1 | + Local DenseCL | +0.028 | -2.27pp | ✓ Local 단독 효과 |
| B1 → B2 | LW=0.5 → 1.0 | **-0.028** | **+2.27pp** | **✗ LW isolated regression** |
| B2 → B3 | + MoCo Queue | **+0.023** | **-4.89pp** | **★ N6 Component Interaction** |
| B3 → B4 | + NEG=0.72 | +0.014 | -0.78pp | ✓ small clean |
| B4 → B5 | + NeCo 0.2 | **-0.004** | **+0.44pp** | **✗ NeCo isolated ≈ 0** |

**총 누적 B0 → B5**: ARI 0.8231 → 0.8564 (+0.033), noise 6.20% → 0.96% (-5.24pp).

#### Methodology 의미

1. **paper community 의 lever isolated 보고 함정 드러남** — atomic ablation 만으로
   contribution 분해 시 Component Interaction (N6) 누락.
2. **LW lever 의 진짜 contribution = Queue interaction** — isolated 로는 negative.
3. **NeCo (paper N1) contribution 재검토** — isolated effect ≈ 0, combined 효과만 인정.
   B5 same-seed reproduce 가 iter 37 보다 ΔARI -0.014 — run-to-run variance 가 N1 isolated
   effect 보다 큼 (multi-seed N2 강한 evidence).

상세: `RESULTS.md` 표 13, `ABLATION_PLAN.md`, `DISCUSSION.md` §7.9.

### 3.6 Backbone partial unfreeze — ★ 영구 reject

`BACKBONE_UNFREEZE_LAST_N=1` (last stage unfreeze) + `LR_SCALE` (backbone lr / head lr) 두
LR_SCALE 값 모두 reject:

| iter | LR_SCALE | 결과 | 판정 |
|---|---|---|---|
| iter 36 | 0.02 | capture 0.976 (P1 violation), Comp -0.025 | ✗ |
| iter 42 | 0.005 | noise 11.69% (×19), ARI -0.396 (huge) | ✗ |

**결론**: TAPT backbone (sister repo `known-cnn` supervised 33-class 학습 결과) 이 이미
도메인 정렬 충분. 추가 unfreeze 는 small-data (2,146 wafer) 위에서 supervised collapse 풍
over-fit 유도. **axis 영구 reject lock-in.**

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
- ★ **IGNORE_NEG_SIM 0.65** (Iter 2, 2026-05-06) — REJECT. NV-Retriever 풍 false-negative
  filter 강화가 모든 P1-P4 후퇴 (capture 1.000 → 0.976, mega-cluster 2 → 3, Donut_scratch_rot
  0% cov). 0.72 의 conservative threshold 가 이미 적정 — 0.65 로 더 공격적 filter
  시 진짜 hard negative 까지 제거되어 cluster boundary collapse. Iter 1 base 로 fallback,
  IGNORE_NEG_SIM=0.65 는 dead branch lock-in.

## 10. 향후 개선 (계획)

- **Hard Negative Mining** (Robinson 2021, ICLR) — InfoNCE β param 도입. label 무관, production 호환. `docs/contrastive-eval/HARD_NEGATIVE.md`.
- **Production-realistic sampling** — per-class sample 수 random (50~200+).
- 합성 sub-style 통일 또는 sub-class 분리 (Full_*, Thick-Edge_fork).
