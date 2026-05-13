# Contrastive Wafer Group Detection — 초보자 친화 요약

> 이 문서는 AI/통계 비전공자도 이해할 수 있게 시각적으로 설명한 프로젝트 종합 요약이다.
> 자세한 수식·논문 인용은 `METHOD.md`, `REFERENCES.md` 참조.

---

## 0. 한눈에 보기 — 입력 wafer → 자동 그룹핑

### 입력: 다양한 wafer 결함 패턴 (43 종 중 6 종 예시)

| Center_fork | Edge-Top_scratch | Edge-Bottom_scratch_rot |
|:---:|:---:|:---:|
| ![](figs/wafer_01_Center_fork.png) | ![](figs/wafer_02_EdgeTop_scratch.png) | ![](figs/wafer_03_EdgeBottom_scratch_rot.png) |

| Edge-Ring_scratch | BrokenRing | RingDots |
|:---:|:---:|:---:|
| ![](figs/wafer_04_EdgeRing_scratch.png) | ![](figs/wafer_05_BrokenRing.png) | ![](figs/wafer_06_RingDots.png) |

### 출력: HDBSCAN 자동 grouping (같은 결함 → 같은 cluster)

| Group 1 — Center_fork | Group 2 — Edge-Top_scratch |
|:---:|:---:|
| ![](figs/group_01_Center_fork.png) | ![](figs/group_02_EdgeTop_scratch.png) |

| Group 3 — BrokenRing | Group 4 — RingDots |
|:---:|:---:|
| ![](figs/group_03_BrokenRing.png) | ![](figs/group_04_RingDots.png) |

★ **iter 37 SOTA**: 43 class 모두 group 1+ 형성 (capture 1.000), noise(def) 0.61% (1146 중 7 wafer 만 어디도 못 묶임)

---

## 0.5 Real Baseline Component Isolation (★ 2026-05-11 NEW)

> 사용자 지적: "기존 Iter A0 baseline 에 이미 Local/Queue/NEG 활성. 진짜 component 단독
> 효과 (isolated effect) 분리 필요" → Real Baseline B0 (Global InfoNCE only) 부터 단계별
> component 추가하는 ablation 6 step (B0→B5) 실시.

### 측정 결과 표 (eom mcs=12 ms=3, seed=42)

| step | cfg | P1 cap | P2 noise | P3 Comp | P4 Hom | AMI | ARI | Sil(cos) | n_cl |
|:-:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | Global InfoNCE only | 1.000 | 6.20% | 0.9602 | 0.929 | 0.9290 | 0.8231 | 0.582 | 37 |
| B1 | + Local DenseCL (LW=0.5) | 1.000 | 3.93% | 0.9665 | 0.9351 | 0.9387 | 0.8514 | 0.5139 | 37 |
| B2 | LW=1.0 (lever 1 isolated) | 1.000 | 6.20% | 0.9602 | 0.9257 | 0.9290 | 0.8231 | 0.5089 | 37 |
| B3 | + MoCo Queue 4096 | 1.000 | 1.31% | 0.9828 | 0.9365 | 0.9496 | 0.8464 | 0.5727 | 36 |
| **B4** ★ | + NEG=0.72 | 1.000 | **0.52%** | **0.9852** | 0.9439 | 0.9557 | **0.8605** | 0.6109 | 37 |
| B5 | + NeCo 0.2 (=iter 37 cfg) | 1.000 | 0.96% | 0.9801 | 0.9403 | 0.9503 | 0.8564 | 0.6104 | 37 |

### Component isolated effect (Δ vs 직전 step)

```
B0 → B1   + Local DenseCL  ΔARI +0.028 / Δnoise -2.27pp   ✓ Local 단독 효과
B1 → B2   + LW strong       ΔARI -0.028 / Δnoise +2.27pp   ✗ LW 단독 regression!
B2 → B3   + MoCo Queue      ΔARI +0.023 / Δnoise -4.89pp   ★★★ N6 huge (LW+Queue 결합)
B3 → B4   + NEG=0.72        ΔARI +0.014 / Δnoise -0.78pp   ✓ NEG 단독 효과
B4 → B5   + NeCo 0.2        ΔARI -0.004 / Δnoise +0.44pp   ✗ NeCo isolated ≈ 0!

누적 B0 → B5:   ΔARI +0.033 / Δnoise -5.24pp
```

### 5가지 paper-grade 핵심 발견

```
1. TAPT backbone 의 강력함
   B0 (Global InfoNCE only) 이미 ARI 0.823 = iter 37 ARI (0.870) 의 94.6%
   → 우리 contrastive head + HDBSCAN tuning total isolated effect = 5%

2. LW=1.0 lever isolated regression
   B1 (LW=0.5) → B2 (LW=1.0) ARI -0.028, noise +2.27pp
   = "Iter A0→1 의 LW lever 효과 -50%" 는 Local+Queue+NEG 활성 위에서만

3. ★★★ Component Interaction (paper N6 NEW)
   B2→B3 + Queue: ARI +0.023 / noise -4.89pp
   = Queue 가 LW over-emphasis 흡수
   = paper community "lever isolated" 보고 함정 입증

4. NeCo (paper N1) isolated effect ≈ 0
   B4→B5 + NeCo 0.2: ARI -0.004 / noise +0.44pp
   B4 > B5 (NeCo 없는 cfg 가 NeCo 있는 cfg 보다 우위)
   = 기존 "iter 35→37 NeCo -70% noise" claim 은 cross-run variance

5. same-seed run-to-run variance
   B5 (seed=42) vs iter 37 (seed=42, same cfg):
      iter 37: ARI 0.870 / noise 0.61%
      B5:      ARI 0.856 / noise 0.96%
      ΔARI 0.014 (multi-seed std 만큼!)
   = paper N2 (multi-seed honesty) 강한 evidence
```

### paper contribution 갱신: N1-N5 → N1-N6

```
N1 NeCo (Pariza 2024) — wafer first      ← reframe: isolated ≈ 0, combined 효과만
N2 Multi-seed honesty                     ← B5 reproduce variance 강화
N3 HDBSCAN eom + ms=3 tuning              ← encoder 무관 (paper N3)
N4 NeCo mechanism reinterpretation        ← Normal-defect boundary repulsion
N5 6-axis saturation point                ← iter 50-58 sweep
★ N6 (NEW) Component Interaction Matters  ← Real Baseline B0-B5 isolation
```

상세: `RESULTS.md` 표 13, `ABLATION_PLAN.md`, `DISCUSSION.md` §7.9, `ITERATIONS.md` iter 60-65.

---

## 0.7 ★★★★★ Iter 84 — B5 seed=1 reproducibility retracts v6 absolute SOTA (★ N1 v7 FINAL, 2026-05-12)

> iter 84 (`outputs_contrastive_260512_114525/`) — B5 reproducibility test under seed=1.
> 결과: v6 "B5 Agglomerative Ward K=42 ARI 0.9358 absolute SOTA" claim **retracted**.
> 진짜 multi-seed SOTA = NEW cfg (NeCo only, no Local) on both Unknown-K AND Oracle-K frontiers.

### Multi-seed ARI summary (★ N1 v7 핵심 table)

| Method | B5 2-seed avg ± std | NEW 3-seed avg ± std | Δ (NEW − B5) | B5/NEW std ratio |
|---|---:|---:|---:|---:|
| HDBSCAN | 0.8343 ± 0.031 | **0.859 ± 0.018** | **+0.0245** | 1.7× |
| Agglo Ward K=42 | 0.8920 ± 0.062 | **0.9014 ± 0.022** | **+0.0094** | **2.8×** |
| KMeans K=42 | 0.8540 ± 0.044 | **0.8678 ± 0.026** | **+0.0138** | 1.7× |

→ **NEW > B5 on multi-seed avg across all 3 clustering methods**.

### B5 single-seed lucky outlier evidence

```
B5 cfg (Local + Queue + NEG + NeCo) — same cfg, same anchor, same HDBSCAN protocol:
   seed=42 (iter 83):  Agglo K=42 ARI = 0.9358  ← v6 absolute SOTA claim
   seed=1  (iter 84):  Agglo K=42 ARI = 0.8482  ← reproducibility
   Δ (seed=1 - seed=42) = -0.0876   ★ LARGEST cross-seed flip in 84-iter cycle

NEW cfg (NeCo + Queue + NEG, no Local) — same protocol:
   seed=42 (iter 70):  Agglo K=42 ARI = 0.9200
   seed=1  (iter 71):  Agglo K=42 ARI = 0.8854
   Δ (seed=1 - seed=42) = -0.0346   ← 2.5x smaller than B5 reproducibility drop
```

### v6 → v7 retraction map

| v6 claim (2026-05-12 morning) | v7 status (2026-05-12 afternoon) |
|---|---|
| "B5 absolute SOTA at Agglo Ward K=42 ARI 0.9358" | ★ RETRACTED — lucky outlier (B5 2-seed avg 0.8920 ± 0.062 < NEW 3-seed avg 0.9014 ± 0.022) |
| "Local DenseCL NOT deprecated, absolute SOTA keeps both Local + NeCo" | ★ RETRACTED — Local operationally optional, NEW (no Local) dominates multi-seed avg |
| "Dual-cfg dual-frontier recipe" | Single-cfg recommendation (NEW) + dual-clustering-target (HDBSCAN unknown-K, Agglo Ward K=42 oracle-K) |
| "Complementary inductive biases B5 vs NEW Agglo K=42" | Per-class flips observed at single-seed (RESULTS §16 preserved) — do NOT propagate to multi-seed averages |

### Single-cfg practitioner recipe (★ v7 final)

```
Recipe (v7 FINAL):
  Encoder cfg:   iter 70 NEW (Global + NeCo 0.2 + Queue 4096 + NEG 0.72, no Local)
                 → single cfg covers BOTH frontiers

  Frontier 1 (Unknown-K real-world):
    Clustering:  HDBSCAN eom mcs=12 ms=3 defect-only
    Multi-seed ARI: 0.859 ± 0.018 (3-seed)

  Frontier 2 (Known-K oracle):
    Clustering:  Agglomerative Ward K=42 defect-only
    Multi-seed ARI: 0.9014 ± 0.022 (3-seed)

  Methodology obligation:
    - Always report multi-seed average ± std
    - Disclose clustering algorithm + K-discovery regime (paper N9)
    - Disclose HDBSCAN protocol (selection_method, mcs, ms, scope; paper N8)
```

### paper contribution 갱신: N1-N8 → N1-N9 with N1 v7 FINAL

```
N1 v7 (FINAL) — NEW cfg unified multi-seed SOTA on both frontiers
              ← v5 (substitutable HDBSCAN aggregate) preserved
              ← v6 (complementary single-seed Agglo K=42 per-class) preserved as observation
              ← v7 (dual-cfg recipe → single-cfg) multi-seed evidence

N2 Multi-seed honesty — strongest evidence: B5 seed=42 → seed=1 Δ ARI -0.088 (Agglo K=42)
                        = largest cross-seed flip in 84-iter cycle, larger than any encoder lever

N3-N6 (Real Baseline, NeCo mechanism, saturation, Component Interaction) unchanged

N7 Component Dependency Hierarchy — refined v7:
   - NEG requires Queue (unchanged)
   - Local DenseCL: operationally optional (v7 replaces v6 "complementary required for SOTA")
   - TEMP sign flip across cfg families (unchanged, B5-family vs NEW-family)

N8 HDBSCAN Protocol Mismatch Methodology (unchanged) — Sil +30% retracted

N9 Clustering Algorithm Dependency (unchanged) — dual-frontier framing preserved at
   clustering-target axis, single-cfg encoder (v7 refinement)
```

상세: ABSTRACT v0.9 (CURRENT), RESULTS §17 (NEW B5 seed=1 table + multi-seed comparison),
DISCUSSION §7.10.7 (v7 retraction + single-cfg recipe), §7.12.4 (single-cfg revised),
CONCLUSION §8.6 + §8.8 (single-cfg closing recommendation),
METHOD §3.6 + §3.7 Frontier 2 (NEW cfg for both frontiers),
INTRODUCTION C7 (v6 retraction + v7 introduction),
ITERATIONS iter 84 entry (append-only).

---

## 0.6 Phase 2 — 4-component lattice + alternative NEW cfg (★ 2026-05-12 NEW; ★★★★★ revised 2026-05-12)

> §0.5 N6 (Component Interaction) 발견 후 follow-up: 4-component lattice (Local /
> Queue / NEG / NeCo) 16-cell 중 12 cell 측정 (iter 67-77) → 더 parsimonious
> **NEW cfg** (4-component, drop Local DenseCL) + paper N7 (Component Dependency
> Hierarchy) 도출.
>
> **★★★★★ CORRECTION 2026-05-12 (HDBSCAN protocol mismatch retraction)**:
> 본 §0.6 의 이전 "Sil +30% robust ★★★" / "+0.184 (+30%)" / "geometry King" 표현
> 모두 **retract**. apples-to-apples (eom + mcs=12 + ms=3, defect-only) 재측정 결과
> B5 Sil = 0.7988, NEW Sil = 0.7860 — **Sil equivalent (−0.013 within seed variance)**.
> 진짜 NEW vs B5 차이 = ARI marginal (+0.003 3-seed avg) + Normal-cluster consolidation
> (paper N1 v5: Normal noise 77.7% → 14.1%, 859/1000 Normals → 1 dense cluster).
> 자세한 retraction: RESULTS §14c / §14h / §14i / §14k, ABSTRACT v0.6.

### 4-component lattice 12-cell 측정 (seed=42, HDBSCAN eom mcs=12 ms=3)

| iter | cfg (Local / Queue / NEG / NeCo) | ARI | noise | Sil | n_cl |
|:-:|---|---:|---:|---:|---:|
| B0 | 0/0/0/0 | 0.8231 | 6.20% | 0.582 | 37 |
| B1 | 1/0/0/0 (LW=0.5) | 0.8514 | 3.93% | 0.514 | 37 |
| B2 | 1*/0/0/0 (LW=1.0) | 0.8231 | 6.20% | 0.509 | 37 |
| iter 69 | 0/0/0/1 (NeCo only) | **0.8514** | 3.93% | **0.707** | 37 |
| iter 67 | 1*/0/0/1 | 0.8510 | 3.93% | n/a | 37 |
| iter 74 | 0/0/1/1 (no Queue) | **0.8514** | 3.93% | 0.707 | 37 |
| B3 | 1*/1/0/0 | 0.8464 | 1.31% | 0.573 | 36 |
| iter 68 | 1*/1/0/1 | 0.8464 | 1.31% | 0.756 | 36 |
| iter 75 | 0/1/0/1 | 0.8822 | 1.31% | 0.785 | 36 |
| B4 | 1*/1/1/0 | **0.8605** | **0.52%** | 0.611 | 37 |
| B5 | 1*/1/1/1 (= iter 37 cfg) | 0.8564 | 0.96% | 0.610 | 37 |
| **iter 70** ★★ | **0/1/1/1 (NEW SOTA)** | **0.8797** | 0.87% | **0.786** | 37 |
| iter 77 | 0/1/1/1 NeCo=0.4 | 0.8605 | 0.52% | **0.801** | 37 |

★ iter 70 = NEW SOTA single-seed. NeCo 가 Local DenseCL **대체**.

### NEW vs B5 multi-seed (iter 70/71/72 vs iter-37 family) — ★ CORRECTED 2026-05-12

| cfg | components | ARI 3-seed avg | Sil seed=42 (apples) | noise mean |
|---|:-:|---:|---:|---:|
| B5 (Local + Queue + NEG + NeCo) | 5 | 0.856 +/- 0.012 | **0.7988** | 0.96% |
| NEW (no Local, NeCo + Queue + NEG) | 4 | 0.859 +/- 0.018 | **0.7860** | 1.48% |
| Δ (NEW - B5) | -1 | +0.003 (within std) | **−0.013 (equivalent, slight regression)** | +0.52pp |

★ CORRECTED: 이전 "Sil +0.184 / +30% robust" 는 cross-protocol artefact (B5 leaf+ms=4
vs NEW eom+ms=3). apples-to-apples (eom+mcs=12+ms=3, defect-only) 재측정 후 Sil
equivalent within seed variance. 자세한 retraction: RESULTS §14c, ABSTRACT v0.6.

### Phase 2 핵심 발견 5 (paper N1 v6 FINAL + N7 + N8 NEW) — ★ N1 v6 REFINED 2026-05-12

```
1. NeCo ↔ Local DenseCL: aggregate-substitutable on HDBSCAN, COMPLEMENTARY per-class on Agglo K=42 (★ N1 v6 FINAL)
   - iter 69 (NeCo only) ARI 0.8514 = B1 (Local only) ARI 0.8514 (소수점 4자리 동일, aggregate HDBSCAN)
   - 그러나 per-class Agglomerative Ward K=42 purity 에서 winner flips on both sides:
     B5 win: Edge-Ring_fork 100% vs NEW 64.5%, Center_scratch 95% vs 75%,
              Donut_fork 100% vs 81.1%, Edge-Top_scratch 100% vs 84.2% (sub-pattern integration)
     NEW win: CenterCircle 100% vs B5 54.8%, Edge-Top_fork 100% vs 90% (uniform consolidation)
     Net avg: B5 97.0% vs NEW 96.2% (Δ −0.83pp, B5 marginal win on aggregate)
   - Absolute SOTA Agglo K=42 single-seed=42: B5 0.9358 > NEW 0.9200 (Δ +0.0158)
   → 두 mechanism 은 **complementary, NOT substitutable**. v5 "substitutable" framing 은
     HDBSCAN aggregate 에서만 valid. Local DenseCL는 **NOT deprecated**.

2. ★ RETRACTED: "NeCo > Local DenseCL on geometry (Silhouette +30%)" — protocol mismatch
   이전 측정: B1 Sil 0.514 vs iter 69 Sil 0.707 (cross-protocol)
   apples-to-apples (eom mcs=12 ms=3, defect-only):
     B5 Sil = 0.7988, NEW Sil = 0.7860 → equivalent (−0.013 within seed variance)
   → "+30% Sil" / "Sil monotonic ↑" / "geometry-vs-partitioning Pareto" 모두 retract.

3. ★ N1 v5 reframe (NEW) — 진짜 NeCo gain channel
   defect-cluster intra_p95 NeCo 추가 시 +26% (오히려 widening, NOT 압축)
   진짜 gain = Normal-cluster consolidation:
     Normal noise 77.7% → 14.1% (859/1000 Normals → 1 dense cluster)
     full-set Completeness 0.851 → 0.917 (+0.066)
     full-set ARI 0.69 → 0.83 (+0.14)
   → NeCo's wafer-domain mechanism = Normal/defect boundary stability,
     NOT defect-cluster compactness.

4. ★ NEW cfg = 4 components only (no Local) — alternative, not strict SOTA
   iter 70: Global + NeCo (0.2) + Queue (4096) + NEG (0.72)
   seed=42 single: ARI 0.8797 (vs iter 37 single 0.870, +0.023)
   3-seed mean ARI 0.859 +/- 0.018 (vs B5 0.856 ± 0.012, +0.003 marginal)
   apples Sil 0.7860 (vs B5 0.7988, equivalent)
   → simpler architecture + equivalent ARI + Normal/defect boundary stability
     (NOT strict superiority, operational choice)

5. ★ N7 v6 — Component Dependency Hierarchy (★ refined 2026-05-12, complementary)
   Required:     Global + (Local + NeCo combined for SOTA, or NeCo alone for density)
   Significant:  MoCo Queue (Queue-on adds +0.029 ARI with NeCo)
   Conditional:  NEG filter requires Queue       ← iter 74 (NeCo + NEG no Queue) ARI 0.8514
                                                   = iter 69 (NeCo only) ARI 0.8514 (4자리 동일)
                                                   → NEG 가 8-batch 만으로는 통계 부족
   Complementary: Local DenseCL ↔ NeCo (★ N1 v6) — aggregate-identical HDBSCAN ARI but
                   complementary per-class purity under Agglo Ward K=42.
                   Local 은 sub-pattern variant integration (fork/scratch rotational+
                   positional), NeCo 는 uniform-pattern consolidation (CenterCircle round).
                   B5 (both) Agglo K=42 ARI 0.9358 = absolute SOTA (Δ +0.0158 vs NEW 0.9200).

6. NeCo weight sweep — ARI inverse-U only (Sil pattern retracted)
   w=0.0 (B4 apples):     ARI 0.8605 / Sil 0.8012
   w=0.2 (iter 70 apples) ARI 0.8797 / Sil 0.7860  ← ARI peak, Sil 약간 낮음
   w=0.4 (iter 77 apples) ARI 0.8605 / Sil 0.8012
   → ARI inverse-U peak at 0.2 유지. Sil monotonic ascent / Pareto frontier 모두 retract.

7. ★ N8 NEW — HDBSCAN Protocol Mismatch Methodology
   본 §0.6 의 "Sil +30%" headline 이 cross-protocol artefact 였던 사례를
   paper-grade methodology contribution 으로 포함 (negative methodology evidence).
   향후 contrastive-clustering paper 는 selection_method / mcs / ms / epsilon /
   metric scope (full-set vs defect-only) 를 모두 명시 + 통일해야 cross-cfg
   비교가 의미 있다.
```

### TEMP cross-cfg interaction (N6 강화)

```
TEMP 0.07 vs 0.05 sign reverses across cfg families:
   B5 family (with Local):  TEMP 0.05 > TEMP 0.07 (+0.014 ARI, iter 37 vs 65)
   NEW family (no Local):   TEMP 0.05 < TEMP 0.07 (-0.024 ARI, iter 73 vs 70)

→ hparam optimum 은 component context 에 dependent
→ paper N6 cross-cfg dependency 추가 evidence
```

### paper contribution 갱신: N1-N7 → N1-N8 (★ revised 2026-05-12)

```
N1 v5 (final)            → NeCo gain channel = Normal/defect boundary stability,
                            NOT defect-cluster compactness. Defect-only metrics
                            (ARI / Sil / noise) functionally equivalent to DenseCL.
                            Full-set ARI 0.83 (NEW) vs 0.69 (B5) via Normal-cluster
                            consolidation. (이전 v4 "geometry Pareto" retract.)
N2 Multi-seed honesty    → iter 70/71/72 + iter 75 evidence 강화
N3 HDBSCAN eom + ms=3    → encoder 무관, fixed across all 12 lattice cells
N4 NeCo mechanism        → Normal-defect boundary repulsion (cluster-analyzer 발견)
N5 6-axis saturation     → iter 50-58 sweep (B5 family)
N6 Component Interaction → §0.5 + TEMP cross-cfg flip (§0.6 추가)
N7 Component Dependency Hierarchy
                         → 4-component lattice (iter 67-77)
                         → NEG requires Queue
                         → Local DenseCL ↔ NeCo substitutable on partitioning
                            (apples-to-apples Sil equivalent; "deprecated Local"
                             표현 retract)
★ N8 NEW HDBSCAN Protocol Mismatch Methodology
                         → 본 §0.6 의 "+30% Sil" retracted-headline 이 worked example
                         → cross-cfg 비교는 mcs/ms/method/epsilon/scope 통일 필수
                         → multi-seed robustness within protocol ≠ cross-protocol valid
```

### NEW Methods cfg (post-iter 77, multi-seed validated)

```python
# Encoder (frozen TAPT)
BACKBONE       = "ConvNeXtV2-base FCMAE + supervised TAPT (sister repo known-cnn)"
PROJ_DIM       = 128
IMAGE_SIZE     = 384

# Loss (NEW: 4-component, drop Local)
USE_LOCAL      = False         # ★ DROP (Phase 2 N7)
LOCAL_WEIGHT   = 0             # n/a
NCE_TEMP       = 0.07          # ★ 0.05 가 NEW cfg 에선 negative (N6)
IGNORE_NEG_SIM = 0.72
NECO_WEIGHT    = 0.2           # ★ kept (paper N1)
USE_QUEUE      = True
QUEUE_SIZE     = 4096

# HDBSCAN unchanged
MIN_CLUSTER_SIZE         = 12
MIN_SAMPLES              = 3
CLUSTER_SELECTION_METHOD = "eom"
```

상세: `RESULTS.md` 표 14, `DISCUSSION.md` §7.10, `ITERATIONS.md` iter 67-77,
`FIGURES.md` F-N7-lattice / F-N7-multiseed-Sil / F-N7-neco-pareto.

---

## 1. 프로젝트 한 줄 설명

> **반도체 wafer 사진 수천 장을 컴퓨터에게 "비슷한 것끼리 묶어라" 시키는 시스템.**
> 새로운 결함 패턴이 들어오면 자동으로 "이건 처음 보는 그룹이네!" 알려주는 게 목표.

---

## 2. 컴퓨터가 wafer 를 어떻게 다루는가? (embedding)

```
사진 (384×384 픽셀 RGB)
       ↓
   [신경망 (ConvNeXtV2-base, 88M params)]
       ↓
   숫자 128개 = 점 1개
   = "embedding"
```

각 wafer 사진이 **128차원 공간의 점 1개** 가 됨. 비슷한 wafer = 가까운 점.

**비유**: 사람을 "키, 몸무게, 나이" 3차원 점으로 표현한다고 치면,
키 175cm·70kg·30살 두 사람이면 가까운 점. 우리는 3차원 대신 128차원.

---

## 3. Contrastive Learning 이 뭐야?

> **"같은 종류는 가깝게, 다른 종류는 멀게"** 점들을 배치하도록 신경망 훈련.

```
훈련 전:                       훈련 후:

점들이 마구 섞여있음           같은 색끼리 모임

  🔴 🔵 🔴 🔵                    🔴🔴🔴   🔵🔵🔵
   🔵🔴🔵🔴      →               🔴🔴      🔵🔵
  🔴🔵🔴🔵
```

방법: anchor 한 점 잡고 → "같은 종류 점은 끌어당겨라 (positive), 다른 종류 점은 밀어내라 (negative)".

**InfoNCE loss 직관**:
```
                exp( sim(anchor, positive) / τ )
L = -log ─────────────────────────────────────────
                Σ exp( sim(anchor, candidate) / τ )

읽기: "분자 (positive 점수) 가 분모 (전체 점수 합) 의 몇 % 인가 → log"
```

---

## 4. Global vs Local feature

```
입력 384×384 → ConvNeXt → feature [12, 12, 1024]   12×12 = 144 patch grid

384×384 입력             12×12 patch (32×32 픽셀씩)
┌──────────┐             ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│          │             │1 │2 │3 │4 │5 │6 │7 │8 │9 │10│11│12│
│  wafer   │             ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  image   │   →         │13│14│15│16│ ... 144 patch          │
│          │             ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│          │             │  ...                              │
└──────────┘             └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

```
Global feature:                  Local feature:

12×12 patch 들을               12×12 patch 각각이
모두 평균해서 1개 vector        독립적으로 contrast 참여

   pooled = wafer 전체 평균       각 patch 가 [128] vector
   [128]                          → 144 × 128 만큼 신호

"이 wafer 가 Donut 이다"         "이 patch 위치에 scratch 가 있다"
큰 정보                           위치-민감 정보
```

---

## 5. Hyperparameter 시각화 (4개 핵심)

### 5-1. LR_HEAD — "걸음 크기" (불 세기)

```
산 (loss 곡면) 위에서 한 발씩 골짜기 (정답) 로 내려가기:

LR=1e-3 (큰 걸음)              LR=5e-4 (작은 걸음, 우리 best)

⛰️ start                            ⛰️ start
    ╲                                   ╲
     🔴────────╲                          🔵
              🔴────────╲                  ╲
                       🔴 ← 너무 큼!         🔵
                        ╲ valley 지나침       ╲
                         🔴                    🔵 ← valley 정확히
                                                  도달

   ✗ 빨리 가지만 흔들림                  ✓ 느리지만 정확
   ✗ collapse 위험 (모델 폭발)           ✓ Comp 0.83 → 0.948 (+12pp) ★

너무 크면 (1e-2): 한 발에 산 너머로 → 발산 → 학습 망가짐 (collapse)
너무 작으면 (1e-6): 1cm 씩 이동 → 골짜기 도달 못 함
```

### 5-2. NCE_TEMP (τ) — "양념 진하기" (softmax sharpness)

```
softmax( sim/τ ) 의 효과:

τ=0.5 (양념 묽다 = 부드럽다):
   exp(0.92/0.5) = 1.82
   exp(0.30/0.5) = 1.22
   exp(0.10/0.5) = 1.10
   ↓
   ┌─────────────────────────────┐
   │ pos:    44%  ███████████░░  │
   │ neg_A:  29%  ████████░░░░░  │  → 모든 negative 골고루 push
   │ neg_B:  27%  ███████░░░░░░  │  → boundary 흐림
   └─────────────────────────────┘

τ=0.05 (양념 진하다, 우리 best ★):
   exp(0.92/0.05) = 90,000,000
   exp(0.30/0.05) = 400
   exp(0.10/0.05) = 7.4
   ↓
   ┌─────────────────────────────┐
   │ pos:    99.9% ████████████  │
   │ neg_A:   0.03%░░░░░░░░░░░░  │  → 가장 비슷한 hard neg 만 push
   │ neg_B:    0.0%░░░░░░░░░░░░  │  → sister-class 분리 sharp ↑
   └─────────────────────────────┘
```

**비유**: 햇볕 (τ 큼, 모든 곳 미지근) vs 돋보기 (τ 작음, 한 점만 뜨겁게).

```
시험 채점 비유:
τ 큰 (관대): 70점, 80점 — 다 비슷 평가 → 학생 노력 약함
τ 작은 (엄격): 80점=C, 81점=B, 82점=A → 1점 차이가 huge gap
              → 학생 1점 더 받으려 죽기살기 노력 → 미세한 차이도 잡음
```

### 5-3. IGNORE_NEG_SIM (NEG) — "false negative 보호장치"

```
contrastive 의 가정: "같은 wafer aug = positive, 나머지 = negative"
   ↑ 라벨 정보 안 씀 (SSL)
   ↑ 그래서 같은 종류 wafer 도 negative 처럼 처리

batch 안에 같은 종류 wafer 가 우연히 있으면 → false negative 발생:

anchor 와 각 candidate 의 sim:
   sim ─────────────── 1.0 (자기 자신)
   0.95 ███████████  Edge-Top_scratch_2  ← false neg!
   0.85 ██████████   Edge-Top_scratch_rot ← sister
   ─── NEG=0.65 ────────────── threshold 라인
   0.60 ████         Donut_scratch
   0.30 ██           Center_invalid

   NEG 위 sim → ❌ 분모에서 빼버림 (false neg 의심)
   NEG 아래 sim → ✓ 정상 negative push
```

**NEG=0.72 (Iter 1) vs NEG=0.65 (Iter 14)**:
- 0.72: 더 많은 neg 살림 → strong gradient → noise 4.62% (P2 King)
- 0.65: 더 많이 제외 → sister-class 보존 → Comp 0.952 (Quality King)

### 5-4. LOCAL_WEIGHT (LW) — "전체 보기 vs 부분 보기"

```
L_total = L_global + LW × L_local

LW=0.5 (Iter A0 baseline):                LW=1.0 ★ (Iter 1 P2 King):

   Global 위주 + local 약간                   Global = local

   ┌──────────────┐                          ┌──┬──┬──┬──┬──┬──┐
   │   pooled     │                          ├──┼──┼──┼──┼──┼──┤
   │   wafer      │                          │  ... 144 patch   │
   │   전체 평균  │            vs            ├──┼──┼──┼──┼──┼──┤
   │   [128]      │                          │  각 patch 독립    │
   └──────────────┘                          │   contrast       │
                                             └──┴──┴──┴──┴──┴──┘

   ✓ Center/Donut 큰 패턴                    ✓ Edge-Top scratch 위치 ↑
   ✗ Edge-Top scratch 위치 약함              ✓ fork 작은 결함 ↑
   noise 9.34%                              noise 4.62% ★ (-50%)
```

★ **LW 가 가장 큰 lever** — 0.5 → 1.0 만으로 noise 절반 감소.

---

## 6. NeCo (Neighborhood-aware Cluster Order) — lever 5번째

> 출처: Pariza et al. 2024, arXiv:2408.11054
> 우리 iter 37 에서 noise 2.01 → 0.61% (**-70%**) huge 효과

### 한 줄 정의

> **두 augmentation view 의 같은 patch 가 보는 "이웃 patch 들의 순위" 가 같아야 한다**

### 친구 관계 비유

```
민수 (anchor patch) 의 친구 관계:
   1순위: 영희 / 2순위: 철수 / 3순위: 동수

이 순위가 어디서 봐도 같아야:

   학교 view A:                  집 view B:
   1순위 영희   sim 0.92         1순위 영희   sim 0.95  ✓
   2순위 철수   sim 0.85         2순위 철수   sim 0.82  ✓
   3순위 동수   sim 0.60         3순위 동수   sim 0.65  ✓

   → 두 view 의 이웃 순위 같음 → loss 작음 ✓

순위 어긋나면:
   학교: 1순위 영희 / 2순위 철수
   집:   1순위 철수 / 2순위 영희   ← 순위 뒤바뀜 ✗
   → loss 커짐 → 학습 신호 발생 → spatial 관계 안정화 학습
```

### 우리 wafer 에 적용

```
view A patch [12,12,128]            view B patch [12,12,128]
       ↓                                   ↓
P_A[i, :] = softmax(sim(p_i, p_j)/τ)    P_B[i, :] = softmax(...)
   144 차원 분포                          144 차원 분포
       ↓                                   ↓
       └─ symmetric KL divergence ────────┘
                  ↓
              L_neco

L_total = L_global + LW×L_local + NECO_WEIGHT × L_neco
                                        ↑
                                    우리 sweep (0.0 ~ 0.3)
```

### NeCo weight sweet spot

```
NeCo=0.0  →  Comp 0.978 / AMI 0.946  (iter 35 baseline)
NeCo=0.1  →  Comp 0.985 / AMI 0.956  ↗ 약간 ↑
NeCo=0.2  →  Comp 0.991 / AMI 0.960  ★★★ peak (sweet spot, iter 37)
NeCo=0.3  →  Comp 0.980 / AMI 0.954  ↘ 후퇴
NeCo=0.5  →  미실시 (양쪽 후퇴 → lock)

너무 작 (0.1): "이웃 순위 일관성" 약함 → cluster 흔들림
너무 크 (0.3): NeCo 강제가 InfoNCE 신호 덮어버림 → AMI ↓
```

### NeCo 의 진짜 메커니즘 (cluster-analyzer 발견)

```
가설:  cluster 응집 (intra distance) 줄임
실측:  intra 거리 변화 거의 없음 (0.0206 → 0.0212)

★ 진짜 효과: Normal-defect boundary 재배치
   Full_scratch ↔ Normal centroid 거리 0.27 → 0.32 (+0.05)
   Full_scratch_rot ↔ Normal centroid 거리 0.29 → 0.36 (+0.07)

   → iter 35 에서 Normal supercluster 에 흡수됐던 54 wafer 가
     iter 37 에서 own pure cluster (size 30, 20) 로 분리
   → 이게 noise 2.01 → 0.61% 의 진짜 원인
```

---

## 7. HDBSCAN (점들을 자동 그룹핑)

훈련 끝나면 점들 모인 모양에서 cluster 자동 추출:

```
embedding 공간 (점들이 흩어져있음):

  🔴🔴       🔵🔵🔵          .    ← 외톨이 점 = "noise"
   🔴       🔵🔵🔵🔵
  🔴🔴🔴      🔵🔵            🟢🟢
                              🟢🟢🟢
                .             🟢

HDBSCAN 결과:
  [Cluster 1: 7개]   [Cluster 2: 7개]   [Cluster 3: 5개]   noise: 2개
```

### HDBSCAN 옵션

| param | 비유 | 효과 |
|---|---|---|
| **mcs** (min_cluster_size) | "최소 N개 모여야 cluster" | mcs=12: 12개 미만은 noise. ↑ 큰 cluster 만 ↓ tiny 도 살림 |
| **ms** (min_samples) | 빽빽함 기준 | ↑ 보수적 noise ↑ / ↓ 관대 cluster 잡 |
| **method=leaf** | "가지 끝까지 세분화" | sub-cluster 다 살림 → over-segment 위험 |
| **method=eom** ★ | "큰 안정 덩어리만 픽" | 자연스러운 sub-style 흡수 → noise -58% |

### eom vs leaf 시각화

```
hierarchical cluster tree:
                       [전체 wafer 1146]
                       /              \
                 [stable A: 800]    [stable B: 250]   ← eom 가 pick
                 /        \           /          \      "큰 stable mass"
            [a1: 80]  [a2: 90]   [b1: 50]   [b2: 30]  ← leaf 가 pick
                                                          "모든 leaf"
─────────────────────────────────────────────────────────────────
leaf method:                           eom method (★ 새 발견):
  41 cluster (over-segmentation)       35 cluster (clean stable)
  같은 class → 여러 sub-cluster        sub-style 자동 통합
  noise 6.54%                          noise 2.79%  ★ -58%
```

---

## 8. 진짜 lever 5개 — 효과 size 정리

```
40+ iter 결과 — 의미 있는 axis 만:

LW (0.5→1.0):       noise [████████████████████] 9.34→4.62%   -50%   ★★★ Iter 1
LR_HEAD (1e-3→5e-4): Comp [████████████]         0.83→0.948   +12pp  ★★ Iter 11
NEG (0.72→0.65):    sister 분리 ↑                              ↑     ★ Iter 13
NCE_TEMP (0.07→0.05): AMI [████]                 0.91→0.913   +0.3pp ★ Iter 14
NeCo (0→0.2):       noise [██████████████████]   2.01→0.61%   -70%   ★★★★ iter 37
HDBSCAN leaf→eom:   noise [██████████████████]   6.72→2.79%   -58%   ★★★ encoder 무관
HDBSCAN ms 4→3:     noise [██████████]           1.22→0.61%   -50%   ★ encoder 무관

dead axes (모두 reject):
   PercPos α / EPOCHS↑ / WARMUP↑ / TOPK≠12 / QUEUE≠4096 / BATCH≠8 /
   LW 작은 변화 / NEG 사촌 / multi-axis combo / HDBSCAN ε / backbone unfreeze
```

---

## 9. 진화 history

```
A0 baseline                    9.34% noise
   │ + LW=1.0 (lever 1)
Iter 1                         4.62% (P2 King)
   │ + LR/NEG/TEMP (lever 2-4)
Iter 14                        6.63% / Comp 0.952 (Quality King)
   │ + new anchor v19o chip
iter 34                        6.72% / Comp 0.951
   │ Iter 1 cfg back
iter 35                        4.19%
   │ + HDBSCAN eom (encoder 무관)
iter 35 + eom                  2.01% / Comp 0.978
   │ + NeCo 0.2 (lever 5)
iter 37 + eom ms=3             0.61% / Comp 0.991 ★★★★★ SOTA (defect-only)
   │ NeCo 0.2 sweet spot 확정
   │ multi-seed (iter 44-46): 0.866 ± 0.014 ARI (N2)
   │ 6-axis saturation (iter 50-58): all within ± std (N5)
   │ Real Baseline isolation (B0-B5): LW interaction, NeCo isolated ≈ 0 (N6)
   │ 4-component lattice (iter 67-77): NeCo ≡ DenseCL, NEG requires Queue (N7)
   │ + drop Local DenseCL (NEW 4-component cfg, alternative)
iter 70 NEW                    0.87% / apples Sil 0.7860 (vs B5 apples 0.7988)
   │ ARI 0.859 ± 0.018 (3-seed) — marginal +0.003 vs B5 0.856 ± 0.012
   │ apples Sil equivalent (−0.013 within seed variance) — "+30% Sil" RETRACTED (N8)
   │ 진짜 NEW gain channel = Normal-cluster consolidation (paper N1 v5):
   │    Normal noise 77.7% → 14.1%, full-set ARI 0.69 → 0.83

★ 총 noise 감소 (defect-only): 9.34% → 0.61% (-93.5%, iter 37 base)
★ NEW vs B5: ARI marginal +0.003 (within std), Sil equivalent (apples), noise +0.5pp
★ NEW 진짜 gain: Normal/defect boundary stability (full-set Completeness +0.066)
```

---

## 0.7 Phase 3 — Clustering Algorithm Dependency (★ 2026-05-12 NEW, paper N9)

> §0.6 N7 (Component Dependency Hierarchy) 후속: 동일 contrastive embedding (B4 / B5 /
> iter 70 NEW) 위에서 **5 가지 clustering 알고리즘 벤치마크** → ARI 가 clustering
> method 에 따라 +0.04 ~ +0.10 변동 → dual-frontier framework 도출.

### 5-method × 3-cfg ARI matrix (single-seed=42)

| cfg | HDBSCAN | DP-GMM | KMeans-42 (oracle) | Agglo-Ward-42 (oracle) | Spectral-42 |
|---|---:|---:|---:|---:|---:|
| B4 Local-based | 0.8605 | 0.8344 | 0.8876 | **0.9055** | 0.4046 |
| B5 (= iter 37) | 0.8564 | 0.8369 | 0.8854 | **★ 0.9358** | 0.7898 |
| iter 70 NEW | **0.8797** | **0.8413** | 0.8798 | 0.9200 | 0.2289 |

JSON: `tier1_clustering_benchmark.json`.

### NEW 3-seed multi-seed (clustering method 별)

| Method | NEW 3-seed avg | NEW std | B5 single-seed=42 |
|---|---:|---:|---:|
| HDBSCAN | 0.8588 | 0.018 | 0.8564 |
| **Agglomerative K=42** | **0.9014** | 0.022 | **★ 0.9358** |
| KMeans K=42 | 0.8678 | 0.026 | 0.8854 |

### 핵심 발견 5 (paper N9 NEW)

```
1. Cfg ranking flip across clustering method families
   density-based (HDBSCAN, DP-GMM):  NEW > B5 ≈ B4
   centroid/linkage with oracle K:   B5 > NEW ≈ B4

2. ARI magnitude shift +0.04~+0.10 across methods at fixed embedding
   B5 HDBSCAN → Agglo: +0.079 — 단일 encoder lever 보다 큼
   → clustering method choice = encoder lever 급 변수

3. ★★★ Dual-frontier framework (paper-grade deliverable, ★ v7 revised 2026-05-13)
   Unknown-K (real-world):
     iter 70 NEW + HDBSCAN = 0.859 ± 0.018 (3-seed)
     rationale: Normal/defect boundary stability (paper N1 v5)
   Known-K (oracle benchmark):
     iter 70 NEW + Agglomerative K=42 = 0.9014 ± 0.022 (3-seed) ★ multi-seed SOTA
     rationale: linkage-based recovers fine sub-structure
     ★ v7 retracted: v6 recommended "B5 + Agglo K=42 = 0.9358 single-seed".
     B5 seed=1 reproduce = 0.8482 (Δ −0.0876), 2-seed avg 0.8920 ± 0.062
     BELOW NEW 3-seed 0.9014 ± 0.022 with std 2.8× higher. RESULTS §17, iter 84.

4. N2 lucky-pattern (seed=42 vs seed=1) 가 method axis 까지 확장
   HDBSCAN +0.030 / Agglo +0.035 / KMeans +0.034 — 모두 동일 magnitude
   → variance source = embedding 자체 (not method)
   → multi-seed protocol obligation 유지

5. Spectral K=42 instability
   B5 0.79 / B4 0.40 / NEW 0.23 — variance 0.56, graph-disconnect warnings
   → 실용 추천 제외 (oracle K 만으로 안전 보장 안 됨)
```

### paper N9 deliverable

> **모든 ARI claim 은 clustering algorithm + K-discovery regime 명시 의무**.
> HDBSCAN 단독 결과로 SOTA 발표 시 oracle K Agglomerative 가 같은 embedding 위에서
> +0.04~+0.10 더 나올 수 있다. methodology disclosure 의 일부.

### paper contributions 갱신: N1-N8 → N1-N9 (★ 2026-05-12, ★ N1 v6 refined)

```
N1 v6 FINAL  NeCo + Local DenseCL = COMPLEMENTARY per-class (v5 "substitutable" refined)
              + NeCo's full-set gain channel = Normal/defect boundary stability (v5 preserved)
N2     Multi-seed honesty (이제 method axis 까지 확장 — Phase 3 evidence)
N3     5 encoder levers + 3 HDBSCAN axes + 14 dead axes
N4     NeCo mechanism = Normal-defect boundary repulsion (v5) + uniform-pattern consolidation (v6)
N5     iter 37 multi-axis saturation point
N6     Component Interaction (Real Baseline B0-B5)
N7     Component Dependency Hierarchy (4-component lattice) — Local NOT deprecated v6
N8     HDBSCAN Protocol Mismatch Methodology
N9     Clustering Algorithm Dependency — dual-frontier framework
       ★ paired with N1 v6: dual-cfg recipe
         - unknown-K + HDBSCAN → NEW (NeCo only, Normal-stream + uniform-pattern)
         - known-K + Agglo Ward → B5 (Local + NeCo combined, complementary per-class)
```

---

## 10. 현재 SOTA — two operational configurations (★ corrected 2026-05-12)

### Configuration A — iter 37 / B5 (5-component, defect-only optimized)

| 지표 | 값 | 의미 |
|---|---:|---|
| **noise(def, P2)** | **0.61%** | 1146 중 7 wafer 만 어디도 못 묶임 |
| **Completeness (P3)** | **0.991** | cluster 응집 거의 perfect |
| **Homogeneity (P4)** | 0.978 | cluster 안 순도 |
| **AMI** | **0.960** | balanced 측정 |
| **ARI** | **0.870** (single-seed) / 0.866 ± 0.014 (3-seed) | over-cluster 페널티 포함 |
| **Silhouette (cos, apples eom+ms=3 def-only)** | **0.7988** | 응집/분리 |
| **capture (P1)** | **1.000** | 43/43 class 모두 group 1+ |
| **n_clusters** | 36 | compact |

### ★ Configuration B — iter 70 NEW (4-component, Normal/defect boundary focus)

| 지표 | 값 | iter 37 대비 (corrected 2026-05-12) |
|---|---:|---|
| **Silhouette (apples eom+ms=3 def-only)** | **0.7860** | **−0.013 (equivalent within seed variance)** |
| ARI | 0.8797 (single) / 0.859 ± 0.018 (3-seed) | +0.003 (3-seed, within std) |
| Completeness (defect-only) | 0.987 | tied |
| AMI | 0.959 | tied |
| noise(def, P2) | 1.48% (3-seed mean) | +0.87pp 후퇴 |
| **full-set Completeness (with Normal)** | **0.917** | **+0.066 ★ (Normal-cluster consolidation)** |
| **full-set ARI** | **0.83** | **+0.14 ★ (paper N1 v5 primary lift)** |
| capture (P1) | 1.000 | tied |
| components | **4** (drop Local DenseCL) | -1 component, simpler |

★ 이전 "Sil +30% robust" headline 은 cross-protocol artefact (B5 leaf+ms=4 vs NEW
eom+ms=3) 로 retracted (paper N8). apples-to-apples 재측정 후 Sil equivalent.

→ **운영자 선택지** (★ N1 v7 FINAL single-cfg recipe, 2026-05-13 revised; supersedes v6 dual-cfg):
- **K known + linkage clustering (oracle benchmark)** → **Config B (NEW, iter 70) + Agglomerative Ward K=42**
  = ARI **0.9014 ± 0.022** (3-seed avg, multi-seed SOTA). Local DenseCL operationally optional.
  ★ v7 retracted: v6 recommended Config A (B5, iter 37) = 0.9358 — single-seed cherry-picked
  outlier. B5 seed=1 reproduce = 0.8482 (Δ −0.0876), B5 2-seed avg 0.8920 ± 0.062 BELOW NEW
  3-seed 0.9014 ± 0.022 with std 2.8× higher. Per-class purity complementarity (v6) preserved
  as single-seed observation only — does NOT propagate to multi-seed avg. RESULTS §17.
- **K unknown + density clustering (real-world, Normal-dominant)** → **Config B (NEW, iter 70) + HDBSCAN**
  = ARI 0.859 ± 0.018 (3-seed). Normal/defect boundary stability (full-set ARI 0.83 vs A 0.69,
  Normal noise 77.7% → 14.1%).
- **defect-only clustering 만 필요한 lab pipeline** → Config B + HDBSCAN eom mcs=12 ms=3 (same
  as above) OR Config A (single-seed=42 noise floor 0.61% reading; multi-seed reproducibility
  worse than NEW per N1 v7).

### Configuration A (iter 37, B5 in lattice)

```python
# Encoder
BACKBONE       = "ConvNeXtV2-base FCMAE + supervised TAPT"
PROJ_DIM       = 128
IMAGE_SIZE     = 384
FREEZE_BACKBONE = True

# Loss (5-component)
USE_LOCAL      = True
LOCAL_WEIGHT   = 1.0
LOCAL_POS_TOPK = 12
NCE_TEMP       = 0.07
IGNORE_NEG_SIM = 0.72
NECO_WEIGHT    = 0.2
USE_QUEUE      = True
QUEUE_SIZE     = 4096

# Training
EPOCHS         = 5
BATCH          = 8
LR_HEAD        = 1e-3
SEED           = 42

# HDBSCAN
MIN_CLUSTER_SIZE = 12
MIN_SAMPLES      = 3
CLUSTER_SELECTION_METHOD = "eom"
CLUSTER_SELECTION_EPSILON = 0.06
```

### Configuration B (iter 70 NEW, post-Phase 2) ★

```python
# Encoder unchanged
BACKBONE       = "ConvNeXtV2-base FCMAE + supervised TAPT"
PROJ_DIM       = 128
IMAGE_SIZE     = 384

# Loss (4-component — drop Local DenseCL)
USE_LOCAL      = False         # ★ Phase 2 N7: NeCo subsumes Local
LOCAL_WEIGHT   = 0             # n/a
NCE_TEMP       = 0.07          # ★ 0.05 negative in NEW cfg (N6 cross-flip)
IGNORE_NEG_SIM = 0.72          # requires Queue (N7 dependency)
NECO_WEIGHT    = 0.2           # ARI peak; 0.4 for Sil-max
USE_QUEUE      = True
QUEUE_SIZE     = 4096

# Training/HDBSCAN unchanged
```

### Anchor 데이터

```
경로: D:/project/data/contrastive_anchor/avg30_new_260508_123037
n=2146 wafer (defect 42 class avg ~30 + Normal 1000)
chip 합성: v19o (260508 regen)
```

---

## 11. 진행 history (iter 34 → 77)

| # | atomic 변경 | noise(def) | Comp | AMI | ARI | Sil | cap | 판정 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 34 | new anchor + Iter 14 cfg (baseline) | 2.79% | 0.977 | 0.931 | 0.750 | — | 1.000 | base |
| 35 | + Iter 1 P2 King | 2.01% | 0.978 | 0.946 | 0.856 | — | 1.000 | ★ -28% noise |
| 36 | + backbone unfreeze (LR_SCALE 0.02) | 4.28% | 0.953 | 0.873 | 0.582 | — | 0.976 ❌ | ✗ REJECT |
| **37** | **+ NeCo 0.2** | **0.61%** | **0.991** | **0.960** | **0.870** | 0.610 | **1.000** | **★★★★★ SOTA A (noise King)** |
| 38 | NeCo 0.1 | 0.52% | 0.985 | 0.956 | 0.860 | — | 1.000 | ✗ regression |
| 39 | NeCo 0.3 | 1.05% | 0.980 | 0.954 | 0.868 | — | 1.000 | ✗ — 0.2 lock |
| 44-46 | iter 37 multi-seed | 0.6% mean | 0.991 | 0.960 | 0.866 ± 0.014 | — | 1.000 | ★ N2 protocol |
| 50-58 | 6-axis saturation sweep | — | — | — | within ± std | — | 1.000 | ★ N5 |
| **B0-B5** | Real Baseline isolation | varies | varies | varies | varies | — | 1.000 | ★ N6 (§0.5) |
| **iter 67** | B2 + NeCo (no Queue) | 3.93% | — | — | 0.8510 | — | — | ✓ NeCo×Queue evidence |
| iter 68 | B3 + NeCo (Queue, no NEG) | 1.31% | — | — | 0.8464 | 0.756 | — | tied (with-Queue) |
| **iter 69** | B0 + NeCo only | 3.93% | — | — | **0.8514** | 0.707 | — | ★★★ NeCo ≡ DenseCL |
| **iter 70** ★★ | **NEW cfg single-seed** | 0.87% | 0.987 | 0.959 | **0.8797** | **0.786** (apples) | 1.000 | **★ alternative (Normal/defect boundary focus, paper N1 v5)** |
| iter 71 | iter 70 seed=1 | — | — | — | 0.8491 | 0.7832 | 1.000 | multi-seed -0.031 |
| iter 72 | iter 70 seed=2 | — | — | — | 0.8475 | 0.8130 | 1.000 | multi-seed (3-seed avg) |
| iter 73 | NEW + TEMP 0.05 | — | — | — | 0.8555 | — | 1.000 | ✗ -0.024 (N6 cross-flip) |
| **iter 74** | NeCo + NEG (no Queue) | 3.93% | — | — | **0.8514** | 0.707 | — | ★★★ NEG requires Queue (N7) |
| iter 75 | NEW + no NEG | 1.31% | — | — | 0.8822 | 0.785 | 1.000 | (single-seed lucky) |
| iter 76 | NEW + NeCo=0.1 | — | — | — | 0.860 | 0.801 | 1.000 | sub-peak |
| iter 77 | NEW + NeCo=0.4 | 0.52% | — | — | 0.8605 | **0.801** | 1.000 | Sil max / ARI to B4 |

### 핵심 path

```
baseline (iter 34)              noise 2.79%, AMI 0.931
   │ ★ Iter 1 P2 King cfg
iter 35                         noise 2.01%, AMI 0.946  (-28%)
   │ ★★★★ NeCo 0.2 (lever 5)
iter 37 ★★★★★ 현 SOTA           noise 0.61%, AMI 0.960  (-70%)
```

### iter 43 자동 결정 logic

```
iter 42 결과 → ┬→ best 갱신 (Comp ≥ 0.991, AMI ≥ 0.960, cap=1.000)
              │      ↓
              │   iter 43 = LR_SCALE 0.01 (덜 보수)
              │   iter 44 = LOCAL_POS_TOPK 16 추가 combo
              │   iter 45 = multi-seed (variance 측정)
              │
              └→ regression
                     ↓
                 iter 43 = multi-seed iter 37 (seed 1, 2, 3)
                 iter 44 = LOCAL_POS_TOPK 16 (NeCo 추가 후 dead axis 재검증)
                 iter 45 = 합성 spec 강화 (TEF vs FF) — 사용자 승인
```

---

## 12. 절대 금기 / dead axes

```
✗ TTA (test-time augmentation): 사용자 정책 영구 금지
✗ SupCon: unknown class 일반화 약화 우려 — 사용자 정책 거부
✗ Multi-crop: wafer 위치 정보 손상 — 사용자 정책 거부
✗ EPOCHS > 5: over-fit (Iter 6, 12 reject)
✗ WARMUP > 1: P1 violation (Iter 17 reject)
✗ BATCH ≠ 8: sweet spot
✗ LOCAL_POS_TOPK ≠ 12: sweet spot (8, 16 reject)
✗ QUEUE_SIZE ≠ 4096: sweet spot (8192 reject)
✗ HDBSCAN eps sweep: dead axis (0.0~0.20 모두 동일)
✗ Backbone full unfreeze: collapse (Iter 31)
✗ Backbone partial unfreeze + LR_HEAD 1e-3: capture P1 violation (iter 36)
```

---

## 13. 관련 문서

- `ITERATIONS.md` — Iter A0 ~ 40 상세 history (append-only)
- `METHOD.md` — 수식·아키텍처 detail
- `RESULTS.md` — 표 정책 + 공식 결과
- `EXPERIMENTS.md` — ablation 설계
- `REFERENCES.md` — 인용 논문 (NeCo, DenseCL, NV-Retriever 등)
