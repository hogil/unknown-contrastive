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

### 3.6 NeCo vs DenseCL Local InfoNCE — complementary, not substitutable (★ N1 v6 FINAL, iter 67-77 + per-class K=42 breakdown 2026-05-12; ★ N1 v7 FINAL multi-seed correction 2026-05-13)

★ **v7 FINAL CORRECTION 2026-05-13 (iter 84 B5 seed=1 reproducibility)**:
The v6 "B5 absolute SOTA Agglo K=42 ARI 0.9358" claim is retracted on multi-seed
grounds — B5 seed=1 reproduce = 0.8482 (Δ −0.0876), B5 2-seed avg 0.8920 ± 0.062
< NEW 3-seed avg 0.9014 ± 0.022 (std 2.8× lower). The single-seed=42 per-class
purity flips below are preserved as observations, but the dual-cfg recipe collapses
to single-cfg (NEW) recommendation for **both** frontiers (HDBSCAN unknown-K AND
Agglo Ward K=42 known-K). Local DenseCL is **operationally optional, not required
for SOTA**. See §3.6 Conclusion block (v6 final → v7 final) and RESULTS §17 for
the v7 final recipe.

The 11-iteration four-component lattice exploration (iter 67-77, 2026-05-12) maps
NeCo (Pariza et al. 2024, arXiv:2408.11054) against DenseCL Local InfoNCE (Wang
et al. 2021) on identical Global-InfoNCE baselines. Coarse Tier 1 metrics
(ARI / noise / n_cluster) show four-decimal identity at iter 69 (NeCo only) vs B1
(Local only) and motivated the v5 "substitutable on partitioning" claim.

★ **N1 v6 FINAL CORRECTION 2026-05-12 (per-class K=42 Agglomerative Ward
breakdown)**: aggregate ARI identity hides a **per-class complementary inductive
bias** between the two mechanisms. Under identical Agglomerative Ward K=42
clustering on defect-only embeddings (RESULTS §16 NEW), per-GT-class dominant
cluster purity differs systematically:

- **Local DenseCL strength** — sub-pattern variant integration: when a class
  contains rotational / positional sub-style variants (`fork`, `scratch`,
  `scratch_rot` placed at different wafer sites), Local DenseCL's patch-grid
  contrast holds the sub-styles inside a single purity-100% cluster:
  `Edge-Ring_fork` (n=31) B5 100% vs NEW (NeCo-only) 64.5% (−35.5pp);
  `Center_scratch` (n=40) B5 95% vs NEW 75% (−20pp); `Donut_fork` (n=37) B5
  100% vs NEW 81.1% (−18.9pp); `Edge-Top_scratch` (n=19) B5 100% vs NEW 84.2%
  (−15.8pp).
- **NeCo strength** — uniform-pattern consolidation: when a class is
  visually uniform (round / symmetric canvas patterns without sub-style
  variation), NeCo's neighbor-rank consistency reaches purity 100% where
  Local-with-NeCo (B5) drops below: `CenterCircle` (n=42) B5 54.8% vs NEW
  100% (+45.2pp); `Edge-Top_fork` (n=20) B5 90% vs NEW 100% (+10pp).
- **Net average per-class purity (Agglomerative Ward K=42)**: **B5 = 97.0%,
  NEW = 96.2%, Δ −0.83pp** — B5 marginally better at micro-aggregate while
  individual class winners flip in both directions.

**Mechanism reading**: Local DenseCL contrasts every grid-cell against every
other grid-cell across the two augmented views, which forces wafer-position-aware
visual sub-style variants to align inside the same class manifold. NeCo's
neighbor-rank consistency operates on a softer "ranking of neighbors" signal,
which excels at consolidating uniform-pattern classes whose neighbor structure
is rotationally symmetric (CenterCircle round geometry), but fails to bind
together rotation/position variants that have different neighbor-rank profiles
across grid positions.

#### Partitioning identity (iter 69 vs B1) — preserved, but reinterpreted

| cfg | iter | ARI | Δ vs B0 | noise | n_cl |
|---|:-:|---:|---:|---:|---:|
| B0 (Global only) | — | 0.8231 | base | 6.20% | 37 |
| B1 (B0 + Local DenseCL LW=0.5) | — | **0.8514** | **+0.028** | **3.93%** | 37 |
| iter 69 (B0 + NeCo 0.2) | — | **0.8514** | **+0.028** | **3.93%** | 37 |

Four-decimal ARI identity, identical noise (3.93%), identical n_cluster (37) —
the two mechanisms produce indistinguishable **aggregate** partitions on the
B0-baseline HDBSCAN scope, but per-class breakdown above shows their
inductive biases differ class-by-class. The aggregate-identity finding is a
necessary-but-not-sufficient test for component substitutability; per-class
purity is the load-bearing evidence layer.

★ **Sub-correction (Sil claim still retracted)**: The originally-reported
"Silhouette +0.193 in favor of NeCo (geometry-only asymmetry)" came from
comparing B1 Sil 0.514 vs iter 69 Sil 0.707 under different HDBSCAN protocols.
Under apples-to-apples re-measurement (eom + mcs=12 + ms=3, defect-only scope)
NEW Sil 0.7860 is slightly worse than B5 Sil 0.7988 by −0.013, within seed
variance. The Sil retraction (paper N8) is unchanged in v6. The per-class
complementarity finding is a **separate axis** (Agglomerative Ward K=42 oracle
purity), independent of HDBSCAN Silhouette.

#### Partitioning equivalence (iter 69 vs B1)

| cfg | iter | ARI | Δ vs B0 | noise | n_cl |
|---|:-:|---:|---:|---:|---:|
| B0 (Global only) | — | 0.8231 | base | 6.20% | 37 |
| B1 (B0 + Local DenseCL LW=0.5) | — | **0.8514** | **+0.028** | **3.93%** | 37 |
| iter 69 (B0 + NeCo 0.2) | — | **0.8514** | **+0.028** | **3.93%** | 37 |

Four-decimal ARI identity, identical noise (3.93%), identical n_cluster (37) —
the two mechanisms produce indistinguishable partitions.

★ **CORRECTION 2026-05-12 (HDBSCAN protocol mismatch retraction)**:

The originally-reported "Silhouette +0.193 in favor of NeCo (geometry-only
asymmetry)" came from comparing B1 Sil 0.514 vs iter 69 Sil 0.707, but those
measurements were taken under different HDBSCAN protocols. Under apples-to-apples
re-measurement (eom + mcs=12 + ms=3, defect-only scope) NEW (NeCo+Queue+NEG) Sil
0.7860 is **slightly worse** than B5 (Local+Queue+NEG+NeCo) Sil 0.7988 by −0.013,
within seed variance. The "NeCo strictly superior on Silhouette / +30% Sil" claim
is therefore **retracted**. ARI / noise / n_cl equivalence (4-decimal identity) is
maintained.

#### Component complementarity — both required for absolute SOTA

The per-class breakdown above motivates a refined recipe. Adding NeCo on top of
an already-Local-active baseline (B5 = B4 + NeCo) gives **ARI −0.004, noise
+0.44pp** versus B4 on HDBSCAN — measured under unknown-K density clustering, the
two mechanisms read as redundant. But under known-K Agglomerative Ward K=42
(RESULTS §15 / §16), B5 achieves **single-seed ARI 0.9358** — strictly higher
than iter 70 NEW (NeCo only) at 0.9200 (Δ +0.0158). The five-component B5 cfg
(Local + Queue + NEG + NeCo) is therefore the **absolute SOTA under linkage
clustering with oracle K**, because the per-class complementarity (fork/scratch
sub-pattern variants integrated by Local + CenterCircle uniform pattern
consolidated by NeCo) only becomes recoverable when K is exposed to the
clustering algorithm.

The remaining differentiator (NeCo independent contribution beyond the
complementary per-class signal) is **Normal/defect boundary stability** (paper
N1 v5 unchanged): NeCo consolidates the Normal cluster (Normal noise 77.7% →
14.1%, 859 of 1000 Normals merge into 1 dense cluster), boosting full-set
Completeness 0.851 → 0.917 and full-set ARI 0.69 → 0.83. The defect-cluster
intra_p95 actually **widens** +26% under NeCo. NeCo's mechanism on this domain
is therefore **two-pronged**: (a) uniform-pattern consolidation visible per-class
(N1 v6) and (b) Normal/defect boundary stability visible at full-set scope
(N1 v5).

#### Conclusion (METHOD-level, v6 final)

```
Patch-neighbor consistency mechanism class:
  - Local DenseCL InfoNCE (Wang 2021, grid-cell intra-image contrast)
  - NeCo (Pariza 2024, patch-neighbor rank consistency)

  Aggregate (HDBSCAN, density clustering, unknown-K):
    Functional substitutes on ARI, noise, n_cluster (iter 69 vs B1 4-decimal identity)

  Per-class (Agglomerative Ward K=42, linkage clustering, oracle-K):
    COMPLEMENTARY, NOT SUBSTITUTABLE.
    - Local DenseCL strength: sub-pattern variant integration
      (fork / scratch / scratch_rot rotational+positional variants).
      Edge-Ring_fork 100% (B5) vs 64.5% (NEW), Center_scratch 95% vs 75%,
      Donut_fork 100% vs 81.1%, Edge-Top_scratch 100% vs 84.2%.
    - NeCo strength: uniform-pattern consolidation
      (CenterCircle round-shape, Edge-Top_fork).
      CenterCircle 100% (NEW) vs 54.8% (B5), Edge-Top_fork 100% vs 90%.
    - Net B5 marginally better aggregate (97.0% vs 96.2%, Δ −0.83pp avg)
      while individual class winners flip on both sides.

  Differentiator (NeCo only, full-set scope):
    Normal/defect boundary stability — Normal noise 77.7% → 14.1%,
    full-set ARI 0.69 → 0.83 (Normal-cluster consolidation paper N1 v5).

  Recommendation (★ v7 FINAL 2026-05-12, replaces v6 dual-cfg):
    - **Use iter 70 NEW (4-component, NeCo + Queue + NEG, no Local) for BOTH frontiers**:
        Unknown-K real-world (HDBSCAN): multi-seed avg ARI 0.859 ± 0.018
        Known-K oracle (Agglo Ward K=42): multi-seed avg ARI 0.9014 ± 0.022
    - **The v6 "B5 absolute SOTA on Agglo K=42 ARI 0.9358" claim is retracted**.
      B5 seed=1 reproducibility (iter 84, same cfg, same protocol) gave Agglo ARI
      0.8482 (Δ −0.088 from seed=42 0.9358). B5 2-seed avg = 0.8920 ± 0.062 is
      BELOW NEW 3-seed avg 0.9014 ± 0.022 (Δ −0.0094) with std 2.8× higher.
    - Local DenseCL is **operationally optional, not required for SOTA**.
      Per-class purity flips at single-seed (RESULTS §16) preserved as observation
      but DO NOT propagate to multi-seed averages.
    - **Always report multi-seed average ± std** — single-seed comparisons across
      cfg families can flip winner claims by ±0.09 ARI (B5 seed=42 vs seed=1 = Δ −0.088
      is the strongest reproducibility evidence in this paper).
```

This is a more careful claim than the original NeCo paper (Pariza 2024) makes on
wafer data: NeCo and DenseCL **share a partitioning function (HDBSCAN aggregate
ARI) and per-class winners flip on single-seed Agglo K=42 but NEW (NeCo only) ≥
B5 (Local + NeCo) on multi-seed average across all benchmarked clustering
methods**. The single-cfg recipe (v7) replaces the v6 dual-cfg framing: NEW
encoder cfg is used for both unknown-K HDBSCAN and known-K Agglomerative Ward
frontiers, and Local DenseCL is operationally optional rather than required for
SOTA. NeCo is preserved for Normal/defect boundary stability (paper N1 v5
unchanged).

### 3.6b Backbone partial unfreeze — ★ 영구 reject

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

- **HDBSCAN** (McInnes et al. 2017, JOSS) — primary, **unknown-K real-world frontier**.
- Parameters (baseline):
  - `min_cluster_size=12`, `min_samples=4`
  - `cluster_selection_method='leaf'` (default eom 도 시험 — `eom` 이 평균 더 큰 cluster)
  - `cluster_selection_epsilon=0.06`
  - metric=euclidean (cosine 정규화 후 등가)

### 3.7 Clustering algorithm selection — dual-frontier rationale (★ NEW, 2026-05-12)

A five-method clustering benchmark (iter 82-83, 2026-05-12) measures the same three
contrastive embeddings (B4 / B5 / iter 70 NEW) under five clustering algorithms:
HDBSCAN, DP-GMM, KMeans K=42, Agglomerative Ward K=42, and Spectral K=42. The
empirical finding (RESULTS §15) is that **ARI at fixed embedding differs by +0.04
to +0.10 magnitude across clustering algorithms**, and the cfg ranking flips
between density-based (HDBSCAN / DP-GMM) and centroid/linkage-based (KMeans / Agglo)
families. This motivates a **dual-frontier framework** for clustering algorithm
selection:

```
Frontier 1 — Unknown-K, real-world deployment (HDBSCAN)
  Use HDBSCAN with eom + mcs=12 + ms=3 + defect-only scope.
  Encoder cfg: iter 70 NEW (Global + NeCo 0.2 + Queue 4096 + NEG 0.72, no Local).
  3-seed ARI 0.859 ± 0.018 (HDBSCAN, defect-only).
  Differentiator: Normal-cluster consolidation (Normal noise 77.7% → 14.1%,
                  full-set ARI 0.83 vs B5 0.69, paper N1 v5).

Frontier 2 — Known-K, oracle lab benchmark (Agglomerative Ward) ★ v7 revised
  Use Agglomerative Ward with K=K_gt (here 42, defect-only).
  Encoder cfg: **iter 70 NEW (same as Frontier 1, no separate B5 retrain)**.
  NEW 3-seed avg ARI 0.9014 ± 0.022 (Agglomerative Ward K=42 defect-only).
  B5 / iter 37 cfg 2-seed avg ARI 0.8920 ± 0.062 (BELOW NEW, std 2.8× higher).
  ★ v6 "B5 single-seed=42 ARI 0.9358 absolute SOTA" retracted — that 0.9358
    reading was a cherry-picked outlier; seed=1 reproduction gave 0.8482.
  Differentiator: linkage-based fine sub-structure recovery when K is known.

Avoid: Spectral K=42 (graph-not-fully-connected warnings, ARI variance 0.23 to 0.79
       across cfg families). DP-GMM is acceptable as a Tier 2 supplementary check
       (variational K-discovery, K_discovered 46-47 here vs K_gt 42).
```

**Methodology disclosure obligation**: any ARI number reported on a
contrastive-clustering pipeline must specify the clustering algorithm AND the
K-discovery regime (unknown-K density-based vs known-K oracle). Otherwise the same
embedding produces headline numbers that differ by 0.04 to 0.10 depending on the
clustering algorithm choice alone — a magnitude larger than most encoder lever
contributions in this paper.

The empirical reason: NeCo (in NEW cfg) widens defect-cluster intra_p95 by +26%
relative to B5, which **benefits density-based clustering** (more separation between
density modes, lower noise-cliff hazard) but **hurts oracle-K linkage clustering**
(harder to recover fine sub-structure when intra-cluster scatter is larger). The
two clustering algorithm families read the same embedding through different
geometric lenses; the dual-frontier framework makes this explicit.

## 4b. Computational Requirements (★ NEW 2026-05-13)

All numbers below are direct measurements from the Claude Code execution log on
the working dataset (`docs/paper/manager_report/performance_data_260513.md`,
user directive: "여기 있는 건 모두 claude code 로 직접 실험한 것들이다 그래서
사실들이다" — every figure is an executed fact, not an estimate).

### 4b.1 Hardware

| component | spec |
|---|---|
| GPU | NVIDIA RTX 4060 Ti (16 GB VRAM) |
| backbone params | **87.7 M** (ConvNeXtV2-base FCMAE + TAPT) |
| projection head | 2-layer MLP, 128-D output, trainable (~ 0.2 M) |
| backbone state | **frozen** during contrastive training |

### 4b.2 Training hyperparameters (NEW cfg, iter 70+)

| field | value |
|---|---|
| epochs | 5 |
| batch | 8 (default); 4 in iter 84 due to concurrent GPU jobs (D-9: BATCH variation = same-condition) |
| optimizer | AdamW, cosine warmup 3-step |
| LR_HEAD | 1e-3 (head only; backbone frozen) |
| LOCAL_WEIGHT (NEW) | 0 (no Local DenseCL) |
| LOCAL_WEIGHT (B5) | 1.0 (Local DenseCL enabled) |
| NECO_WEIGHT | 0.2 |
| QUEUE_SIZE | 4096 |
| TEMP | 0.07 |
| IGNORE_NEG_SIM | 0.72 |
| augmentation | rotate ±15°, translate/scale ±3%, gaussian σ=0.01 (no flip / colorjitter / mixup / cutmix — domain-safe) |

### 4b.3 Training time (single RTX 4060 Ti, 5-epoch, n=2146)

| recipe | components | n_seed | min / run | std |
|---|---|---:|---:|---|
| **NEW** ★ | Global + NeCo + Queue + NEG (no Local) | 3 | **23.7** | ± 0.01 (highly reproducible, CV ≈ 7.7%) |
| B5 | Local + Queue + NEG + NeCo (= iter 37 cfg) | 3 | 28 – 49 | wide (Local DenseCL forward + backward variance) |

→ **NEW is ~30 % faster than B5** at single-run wall-time, attributable to Local
DenseCL grid-cell forward + backward removal. Across a 3-seed sweep: NEW
total ≈ 71 min, B5 total 85 – 147 min. The wall-time advantage compounds
with the multi-seed reproducibility advantage (RESULTS §17, std ratio 2.8×
lower under Agglomerative Ward K=42).

### 4b.4 Evaluation pipeline time (n=2146, single GPU)

| stage | time |
|---|---|
| embedding extraction (frozen encoder forward) | ≈ 3 min |
| HDBSCAN fit + Tier 1+2 metric + per-class fragmentation report | 4 – 7 min |
| composite-cluster PNG rendering (6400 × 6400, binary + grademean) | included above |
| **total eval pass** | **7 – 10 min** |

### 4b.5 Inference latency (paper claim, single RTX 4060 Ti)

| mode | latency / image | throughput |
|---|---:|---:|
| **single wafer** (BATCH = 1) | **14.3 ms** | **70 wafers/sec** |
| batch 8 | 17.2 ms | 58 wafers/sec |
| batch 32 (amortized) | 18.5 ms | 54 wafers/sec |

### 4b.6 HDBSCAN clustering time

| operation | time |
|---|---|
| full re-cluster (1146 defects, fresh fit) | **507 ms** (model-update path) |
| single-point predict (`approx_predict`) | **≈ 10 ms** (online use) |

### 4b.7 End-to-end production latency (per wafer)

```
encoder forward pass    14 ms   (BATCH = 1, frozen ConvNeXtV2-base + 128-D head)
HDBSCAN approx_predict  10 ms   (single-point, pre-fit model)
──────────────────────────────
total                  ~24 ms   →   ~ 40 wafers/sec deployable on a single RTX 4060 Ti
```

→ The pipeline is **real-time deployable** for in-fab wafer-by-wafer triage at
common semiconductor inspection throughputs (typical line: 1 – 10 wafers/sec).
The encoder is the bottleneck; HDBSCAN online predict is sub-half the encoder
cost.

## 4c. Post-process refinement — soft τ-reassignment (★ NEW 2026-05-13, Step 1c)

Source-of-truth: `docs/paper/manager_report/step1_eval_only_summary_260513.md`.
Full ablation matrix: RESULTS.md §19.

### 4c.1 Motivation

The NEW recipe's HDBSCAN clustering (defect-only, `eom mcs=12 ms=3`) produces
≈ 1.48 % residual noise (3-seed avg, RESULTS §19) — wafers that the density-based
selector cannot confidently assign to any cluster. For **production deployment**
where every incoming wafer must receive a cluster label (e.g., for routing to
defect-class-specific downstream analyses), residual noise needs a principled
reassignment policy that does not retrain the encoder.

### 4c.2 Method — KNN-softmax τ-thresholded reassignment

For each HDBSCAN noise point `xᵢ` (predicted label = −1):

1. Compute `k = 10` nearest neighbors of `xᵢ` from the set of non-noise points,
   using cosine similarity on the 128-D L2-normalized embedding.
2. Build a label distribution over the k neighbors' cluster IDs.
3. Apply softmax with temperature `T = 0.1` over per-class neighbor-similarity sums:

   `p(c | xᵢ) = softmax_T(Σⱼ∈Nₖ(xᵢ) sim(xᵢ, xⱼ) · 𝟙[label(xⱼ) = c])`

4. Predicted label = `argmax_c p(c | xᵢ)`, but accept the reassignment only when
   `max_c p(c | xᵢ) ≥ τ`; otherwise keep the noise label.

The threshold τ controls the **confidence floor** for reassignment.

### 4c.3 τ trade-off (paper Step 1c)

| τ | ARI Δ vs ∞ baseline | noise % | std improvement | use case |
|:-:|---:|---:|---|---|
| ∞ (no reassign) | 0 | 1.48 % | 0.0140 (baseline) | strict density-based, default research metric |
| 0.90 | −0.0022 | 0.49 % | −5.7 % | conservative reassignment, evaluation parity |
| **0.70** | −0.0035 | 0.15 % | **−12 %** (0.0140 → 0.0123) | **best reproducibility**, soft-label triage |
| **0.50** ★ | −0.0050 | **0.00 %** | −11 % (0.0140 → 0.0125) | **★ production cfg lock** — every wafer labeled |

Tier 1 P1 class capture rate = 1.000 across all τ values (no class is lost by
reassignment). The headline P2 priority metric (noise rate, see
`feedback_priority_p1_to_p4.md`) collapses from 1.48 % → 0 % at τ = 0.5 with
an ARI cost (Δ = −0.005) well within 3-seed std (0.014).

### 4c.4 Production deployment recommendation

**τ = 0.5 is the recommended production configuration** for in-fab triage:

```
post_process:
  type:               knn_softmax_tau_reassign
  k:                  10
  similarity:         cosine
  softmax_temperature: 0.1
  tau:                0.5         # ★ every wafer receives a cluster label
  source:             defect-only HDBSCAN eom mcs=12 ms=3 baseline
```

Under this configuration, 17 / 1146 wafers (3-seed avg) initially flagged as
HDBSCAN noise are reassigned to their nearest cluster via the KNN-softmax vote,
eliminating the residual noise rate entirely while paying a marginal ARI cost
(−0.005). The latency overhead is ≈ 1 ms / wafer (10-NN lookup against a
pre-fitted FAISS index on 1146 embedding points), negligible compared to the
14 ms encoder forward pass (METHOD §4b.7).

For **research evaluation**, the ∞ baseline (no reassignment) remains the
default — paper's headline metrics (Tier 1, §17b multi-seed avg) are reported
without reassignment to preserve apples-to-apples comparison with prior
HDBSCAN-only literature.

### 4c.5 Relation to HDBSCAN cluster_selection_epsilon (paper N9)

A natural alternative would be the HDBSCAN `cluster_selection_epsilon` parameter,
which similarly absorbs marginal-density points. We measured this parameter
across 21 cells (3 seeds × 7 ε ∈ {0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15})
in Step 1b (RESULTS §19.2) and observed **zero effect** — all 21 measurements
yielded identical Tier 1+2 metrics to 4 decimal places. The NEW embedding's
cluster tree is fully saturated under the `(eom, mcs=12, ms=3)` triple.

Soft τ-reassignment is therefore the **only** post-hoc lever available on NEW
for noise control without retraining; the cluster_selection_epsilon parameter
is **deprecated** for this recipe (paper N9 reinforcement).

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

### 7.1 Headline ARI provenance — defect-only post-hoc HDBSCAN (★ NEW 2026-05-13, paper N8 reinforcement)

★ **Critical reviewer-facing disclosure**: Paper headline ARI claim **NEW + HDBSCAN
ARI 0.859 ± 0.018** (3-seed mean ± ddof=1 std, equivalently `0.859 ± 0.018`) is
**NOT** the number a reviewer would obtain by re-running the default eval pipeline
`_eval_contrastive_unknown_n50.py` on the published checkpoints. The headline comes
from a **separate post-hoc HDBSCAN re-cluster** with the following explicit protocol:

| axis | headline (paper) | default `eval_summary.json` |
|---|---|---|
| scope | **defect-only** (n ≈ 1146, Normal class excluded) | full-set (n = 2146 inc. Normal) |
| `cluster_selection_method` | `eom` | `eom` |
| `min_cluster_size` (mcs) | **12** | 5 |
| `min_samples` (ms) | **3** | 5 |
| `cluster_selection_epsilon` (eps) | **NOT set (none)** | 0.06 |
| metric | cosine (L2-normalized 128-d projection) | cosine |
| K-discovery | density (unknown-K) | density (unknown-K) |

**Source of the 3 seed measurements** (Tier 1 JSON, defect-only re-cluster):

| seed | run_dir | tier1 file | ARI | Comp | AMI | noise_pct | n_cl |
|---|---|---|---:|---:|---:|---:|---:|
| 42 | `outputs_contrastive_260512_001719/` | `tier1_neco_replaces_local.json` | 0.8797 | 0.9872 | 0.9594 | 0.87% | 37 |
| 1 | `outputs_contrastive_260512_010113/` | `tier1_sota_seed1.json` | 0.8491 | 0.9856 | 0.9488 | 1.05% | 35 |
| 2 | `outputs_contrastive_260512_014507/` | `tier1_sota_seed2.json` | 0.8475 | 0.9747 | 0.9428 | 2.53% | 36 |
| **avg** | — | — | **0.8588** | 0.9825 | 0.9503 | 1.48% | — |
| **std (ddof=1)** | — | — | **0.0181** | 0.0070 | 0.0091 | 0.84pp | — |

A reviewer reproducing via the **default** `_eval_contrastive_unknown_n50.py` eval
pipeline will see a **different (lower) ARI** because (a) the default eval includes
the Normal class in scope and (b) the default eval uses `cluster_selection_epsilon
= 0.06`, both of which inflate the cluster-boundary count and re-include Normal
density mass. Specifically: same iter 70 NEW checkpoint, **default eval**
`eval_summary.json` returns full-set ARI ≈ 0.74; **defect-only re-cluster without
eps** returns 0.880 (seed=42), of which the 3-seed mean is the 0.8588 headline.

**This is not a metric inflation** — both numbers are valid Tier 1 ARI outputs
under their respective scopes, and the 0.859 ± 0.018 is mathematically traceable
to the 3 tier1 JSON files above. **But the protocol gap must be disclosed** so
reviewers can reproduce exactly the headline number. The same protocol gap is
also the operational basis of paper **N8 (HDBSCAN Protocol Mismatch Methodology)**.

Same protocol applies to the **B5 multi-seed** comparison (RESULTS §17b): B5
seed=42 ARI 0.8564 / seed=1 ARI 0.8122 are both **defect-only eom+mcs=12+ms=3
without eps** measurements, apples-to-apples with NEW. The Agglomerative Ward
K=42 and KMeans K=42 multi-seed numbers (RESULTS §15c, §17b) are likewise on
defect-only scope (Normal excluded so K=42 = GT defect class count).

Provenance trace: `docs/paper/manager_report/claim_0859_origin_trace.md`.

★ **All ARI / Sil / NMI / Hom / Comp / noise / capture numbers reported in this
paper headline tables (RESULTS §15-17, ABSTRACT v0.9, README current SOTA,
DISCUSSION §7.10.7 / §7.12.4) are on this defect-only protocol unless explicitly
marked otherwise (e.g., "full-set" qualifier).**

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
