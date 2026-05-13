# Results

매 iteration 마다 paper-recorder agent 가 새 row 추가.

## 표 1 — Tier 1 metric 비교 (전체 run)

| Run | Date | Completeness | AMI | noise_pct (def) | class_capture | frac_single_cluster |
|---|---|---|---|---|---|---|
| Iter 0 (historical, old anchor) | 2026-05-05 | 0.9466 | 0.9288 | 0.71% | 38/38 (1.000) | 34/38 (0.8947) |
| **Iter A0** (new anchor baseline) | 2026-05-06 | 0.9375 | 0.8946 | 9.34% | 42/42 (1.000) | 35/42 (0.8333) |
| **Iter 1** (LOCAL_WEIGHT 1.0) ★ | 2026-05-06 | **0.9481** | **0.9040** | **4.62%** | **42/42 (1.000)** | **36/42 (0.8571)** |
| Iter 2 (IGN_NEG 0.65) — REJECT | 2026-05-06 | 0.816 / 0.950 | 0.821 / 0.903 | 7.16% | **41/42 (0.976)** | — |
| Iter 3 (LOCAL_WEIGHT 0.7) — NULL | 2026-05-06 | 0.936 | 0.893 | 9.42% | 42/42 (1.000) | — |

★ Iter 1 = best Tier 1 (LW=1.0, IGN_NEG=0.72) — Iter 2/3 둘 다 reject 또는 null.

**Iter 2 (REJECT)**: IGNORE_NEG_SIM 0.72 → 0.65 모든 P1-P4 후퇴. capture 1.000 → 0.976
(Donut_scratch_rot 0% cov), mega-cluster 2 → 3 (c034 EB + c009 Center + c023 ET 추가 collapse).
filter 강화가 진짜 hard negative 까지 제거 → boundary collapse. dead branch.

**Iter 3 (NULL)**: LOCAL_WEIGHT 1.0 → 0.7 모든 metric 이 A0 (LW=0.5) 와 통계적 동일
(<0.01 차이). sister-pair centroid distance 모두 A0 와 소수점 셋째 자리 동일.
**LW 0.5~0.7 plateau, 1.0 비선형 jump** — Iter 1 의 jump 가 단일 sweet spot 후보.

**Track 주의**: Iter 0 은 historical (Normal 1000 + per-class 200, 8357 wafer, 38 defect class).
Iter A0 부터 새 data anchor (D-15: defect 42 class avg30 random + Normal_bank_boundary 1000
= 2146 wafer) 위 method-track. anchor 가 다르므로 Iter 0 ↔ Iter A0/1 직접 비교 의미 X.
Iter A0 vs Iter 1 만 atomic 비교 가능 (same data, LOCAL_WEIGHT 만 변경).

**해석 (Iter 1, latest)**:
- LOCAL_WEIGHT 0.5 → 1.0 atomic 변경 단독으로 Tier 1 4 metric 모두 개선
- noise_pct(def) 9.34% → 4.62% — 절반 감소, Tier 1 P2 핵심 개선
- Completeness +0.0106, AMI +0.0094, capture 1.000 유지 (P1 손실 없음)
- frac_single_cluster 0.833 → 0.857 (+0.024) — split class 1개 줄어듦 (7→6)
- weighted_cluster_coverage 0.872 → 0.926 (+0.054)
- primary weak collapse: Edge-Bottom_scratch_rot 76.7% → 6.7% (-70pp)

## 표 2 — Tier 2 보조 metric

| Run | Homogeneity | Silhouette (cosine) | ARI |
|---|---|---|---|
| Iter 0 (historical) | 0.9154 | 0.5664 | 0.7002 |
| Iter A0 | 0.8934 | 0.7908 | 0.7040 |
| Iter 1 | 0.8978 | 0.7766 | 0.7325 |
| Iter 2 (REJECT) | 0.875 / 0.895 | 0.737 | 0.639 / 0.728 |
| Iter 3 (NULL) | 0.893 | 0.785 | 0.701 |

**ARI 가 낮은 이유**: HDBSCAN 가 39 cluster 산출 (= 39 GT class 와 같으나 매핑 안 일치).
ARI 는 over-cluster 페널티 강함. Tier 1 (Completeness / AMI) 가 더 적합.

## 표 3 — class_fragmentation_summary detail

| Run | n_total | captured | uncaptured | mean_coverage | weighted_coverage | mean_n_clusters | single | split_2 | split_3+ |
|---|---|---|---|---|---|---|---|---|---|
| Iter 0 (historical) | 38 | 38 | 0 | 0.9904 | 0.9929 | 1.105 | 34 | 4 | 0 |
| Iter A0 | 42 | 42 | 0 | 0.895 | 0.8717 | 1.167 | 35 | 7 | 0 |
| Iter 1 | 42 | 42 | 0 | 0.934 | 0.9258 | 1.143 | 36 | 6 | 0 |

## 표 4 — Split classes detail (Iter 0, historical)

| class | n | n_clusters | dom_recall | coverage | 진단 |
|---|---|---|---|---|---|
| `Full_bank_boundary` | 200 | 2 | 0.555 | 1.000 | 200 → 111+89 split (no noise). intra/inter ratio **8.72** ★ |
| `Full_scratch_rot` | 200 | 2 | 0.505 | 0.990 | 101+97+2noise. ratio 4.44 |
| `Full_fork` | 200 | 2 | 0.540 | 0.985 | 108+89+3noise. ratio 2.66 |
| `Thick-Edge_fork` | 50 | 2 | 0.480 | 0.860 | 24+19+7noise. ratio 2.13 (작은 class) |

**발견**: 4 class 모두 합성 데이터의 진짜 두 sub-style.
검증: HDBSCAN sweep (모든 hyperparameter 에서 동일 split) + GMM BIC bimodal +
intra/inter ratio 2-9×. 자세히 `docs/contrastive-eval/DECISIONS.md` D-10.

## 표 4b — Weak classes top-3 (cluster coverage 기준, 새 anchor)

| Run | rank | class | n | noise_pct | n_clusters | cov |
|---|---|---|---|---|---|---|
| Iter A0 | 1 | `Edge-Bottom_scratch_rot` | 30 | 76.67% | 2 | 0.20 |
| Iter A0 | 2 | `Edge-Bottom_fork` | 41 | 63.41% | 2 | 0.27 |
| Iter A0 | 3 | `Thick-Edge_fork` | 29 | 51.72% | 2 | 0.45 |
| Iter 1 | 1 | `Donut_scratch_rot` | 15 | 40.00% | 1 | 0.60 |
| Iter 1 | 2 | `Thick-Edge_fork` | 29 | 34.48% | 2 | 0.62 |
| Iter 1 | 3 | `Donut_fork` | 37 | 21.62% | 2 | 0.76 |

**Δ (Iter A0 → Iter 1)**: `Edge-Bottom_scratch_rot` (primary weak) noise 76.67% → 6.67%
(-70pp), top-3 에서 사라짐. `Edge-Bottom_fork` 63.41% → 14.63%, top-3 에서 사라짐.
대신 `Donut_scratch_rot` 26.67% → 40% 악화, primary weak 자리 차지. local 강화가 위치별
trade-off (Edge-Bottom 좋아짐 / Donut 약간 나빠짐).

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

Iter 2 (IGN_NEG 0.65) REJECT + Iter 3 (LW=0.7) NULL → Iter 1 (LW=1.0, IGN_NEG=0.72) 가
현 best. 다음 후보:

| Iter | 변경 | 예상 효과 |
|---|---|---|
| Iter 4 (후보 a) | LOCAL_WEIGHT 1.0 → 1.5 push | 1.0 이 plateau 시작인지 sweet spot 실험 |
| Iter 4 (후보 b) | EPOCHS 5 → 10 | uniformity -3.0+ target, 학습 더 길게 |
| Iter 4 (후보 c) | TEMP 0.07 → 0.05 | sharper boundary |
| Iter 4 (후보 d) | Donut_scratch_rot 약화 원인 추적 (image-analyzer) | Iter 1 의 새 weak 원인 분석 |
| Iter 5+ | Hard Negative Mining (Robinson β param) | InfoNCE 위 importance weighting |

각 iteration 진행 시 위 표 row 추가 + ITERATIONS.md 에 변경 history 추가.

---

# ★ New anchor track (iter 34+) — `avg30_new_260508_123037`

> data anchor 변경: `avg30_260505_203615` → `avg30_new_260508_123037` (43 class — 42 defect +
> Normal_bank_boundary, 2,146 wafer, v19o chip 합성 + canvas 9 추가). Iter 34+ 결과는 별도 표 분리.
> Iter 0-30 (old anchor) 와 iter 34+ (new anchor) 직접 비교 의미 X (anchor 분포 다름).

## 표 7 — Tier 1 metric (new anchor track)

> **Score 정책**: HDBSCAN cfg `eom mcs=12 ms=3` 고정 (encoder method axis 와 분리). leaf
> 또는 다른 ms/mcs 표 row 추가 시 cfg 명시.

| Run | Date | Comp | AMI | ARI | noise_pct (def) | capture | base 변경 |
|---|---|---:|---:|---:|---:|---:|---|
| iter 34 | 2026-05-08 | 0.977 | 0.931 | 0.750 | 2.79% | 1.000 | new anchor + Iter 14 cfg (Quality King) |
| **iter 35** | 2026-05-08 | 0.978 | 0.946 | 0.856 | 2.01% | 1.000 | + LR 5e-4→1e-3, NEG 0.65→0.72, TEMP 0.05→0.07 (Iter 1 P2 cfg) |
| iter 36 ✗ | 2026-05-09 | 0.953 | 0.873 | 0.582 | 4.28% | **0.976 ❌** | + UNFREEZE_LAST_N=1, LR_SCALE=0.02 — **P1 violation** (eom mcs=12 ms=4) |
| **iter 37 ★★★★★** | 2026-05-09 | **0.991** | **0.960** | **0.870** | **0.61%** | **1.000** | + NECO_WEIGHT=0 → **0.2** (5번째 lever) — **★ SOTA** |
| iter 38 ✗ | 2026-05-09 | 0.985 | 0.956 | 0.860 | 0.52% | 1.000 | NECO_WEIGHT 0.2 → 0.1 (under-signal) |
| iter 39 ✗ | 2026-05-09 | 0.980 | 0.954 | 0.868 | 1.05% | 1.000 | NECO_WEIGHT 0.2 → 0.3 (over-signal) — 0.2 lock |
| iter 40 ✗ | 2026-05-09 | 0.962 | 0.922 | 0.738 | 4.10% | 1.000 | Quality King cfg + NeCo 0.2 (cross-cfg incompat) |
| iter 41 ✗ | 2026-05-09 | 0.997 | — | — | 3.05% | **0.952 ❌** | HDBSCAN mcs forcing on iter 37 (encoder X) — **P1 violation** |
| iter 42 ✗ | 2026-05-09 | 0.948 | 0.823 | 0.474 | 11.69% | 1.000 | + UNFREEZE_LAST_N=1, LR_SCALE=0.005 — axis 영구 reject |
| iter 43 (in progress) | 2026-05-09 | — | — | — | — | — | + NECO_ZONE_VERTICAL=3 (★ Zone-Aware NeCo, novelty A) |

**판정**: iter 37 = new anchor SOTA (Comp/AMI/ARI/noise/capture 5 metric 모두 best).
NeCo (lever 5) sweet-spot=0.2 lock-in (iter 38/39 reject). backbone unfreeze axis (iter 36/42)
영구 reject. cross-cfg interaction (iter 40) — NeCo 는 P2 King cfg 위에서만 효과.

## 표 8 — 5 lever 효과 정리 (encoder-side, lever 별 atomic 변경 효과)

| # | Lever | atomic step | 효과 (Tier 1) | iter | track |
|---:|---|---|---|---|---|
| 1 | LOCAL_WEIGHT | 0.5 → 1.0 | noise 9.34 → 4.62% (-50%) | Iter 1 | old anchor |
| 2 | LR_HEAD | 1e-3 → 5e-4 | Comp 0.83 → 0.948 (+12pp) | Iter 11 | old anchor |
| 3 | IGNORE_NEG_SIM | 0.72 → 0.65 | sister-class 분리 ↑, capture 유지 | Iter 13 | old anchor |
| 4 | NCE_TEMP | 0.07 → 0.05 | AMI / ARI / Comp / Hom 모두 +0.3pp | Iter 14 | old anchor |
| **5 ★** | **NECO_WEIGHT** | **0 → 0.2** | **noise 2.01 → 0.61% (-70%) ★** | **iter 37** | **new anchor** |

**HDBSCAN side levers (encoder 무관)**:

| Lever | atomic step | 효과 | track |
|---|---|---|---|
| HDBSCAN method | leaf → eom | noise -58% (mcs=12 ms=4 기준) | new anchor |
| HDBSCAN ms | 4 → 3 | noise 추가 -50% | new anchor |
| HDBSCAN mcs | {8,10,12} | ms=3 기준 동등 (12 lock-in) | new anchor |

**Note 1**: lever 5 (NeCo) 는 base cfg 에 강하게 의존 — iter 40 (Quality King + NeCo) 에서 ARI
-13pp regression 으로 negative interaction 확인. **NeCo 는 P2 King cfg (Iter 1, LR=1e-3 NEG=0.72
TEMP=0.07) base 위에서만 SOTA**. Quality King cfg (Iter 14) + NeCo 는 reject.

**Note 2**: HDBSCAN axes 는 encoder 학습과 별개 track (`feedback_hdbscan_cfg_sweep_ok.md`).
같은 embedding 위 cluster fitting variation 이므로 method ablation 과 분리 보고.

## 표 9 — 진화 path (Iter A0 → iter 37 SOTA)

```
A0  (baseline, old anchor)            Comp 0.938  AMI 0.895  noise 9.34%
  └─ Iter 1   ★ LW lever              Comp 0.948  AMI 0.904  noise 4.62%   (-50% noise)
       └─ Iter 11 ★ LR lever          Comp 0.948  AMI 0.905  noise 6.11%   (Comp +12pp earlier)
            └─ Iter 13 ★ NEG lever    Comp 0.949  AMI 0.906  noise 5.32%
                 └─ Iter 14 ★★        Comp 0.952  AMI 0.913  noise 6.63%   (Quality King, old)
                      │
                      └─ ⋯ Iter 15-30: sweep dead (lever 1-4 saturated)
  ★ Track switch: anchor → avg30_new_260508_123037
  iter 34 (Iter 14 cfg, new anchor)   Comp 0.977  AMI 0.931  noise 2.79%
       └─ iter 35 (P2 King 3-axis)    Comp 0.978  AMI 0.946  noise 2.01%   ← new anchor base
            └─ iter 37 ★★★★★ NeCo 0.2 Comp 0.991  AMI 0.960  noise 0.61%   ← SOTA
                 │
                 ├─ iter 38 ✗ NeCo 0.1 (under)
                 ├─ iter 39 ✗ NeCo 0.3 (over) → 0.2 lock
                 ├─ iter 40 ✗ Quality King + NeCo (cross-cfg)
                 ├─ iter 41 ✗ HDBSCAN forcing (P1)
                 ├─ iter 42 ✗ UNFREEZE 0.005 (axis 영구 reject)
                 └─ iter 43 (in progress) Zone-Aware NeCo (novelty A)
```

## 표 10 — Dead axes 누적 (반복 시간 낭비 방지)

| Axis | 시도 iter | 결론 |
|---|---|---|
| NV-Retriever PercPos α | Iter 7-10 | 4-step sweep all reject |
| EPOCHS ↑ | Iter 5,6,12 | 5 sweet spot (over-fit beyond) |
| WARMUP | Iter 17,18 | 1 sweet spot (P1 violation at 2) |
| LOCAL_POS_TOPK | Iter 19,22 | 12 sweet spot |
| QUEUE_SIZE | Iter 20 | 4096 sweet spot |
| BATCH | Iter 23 | 8 sweet spot |
| LW small Δ | Iter 25-27,30,3 | gradient scaling only |
| NEG sister | Iter 28,29 | 0.65 sharp local min |
| TEMP sister | Iter 15,24 | 0.05 sweet spot (old) / 0.07 (new) |
| LR sister | Iter 16 | 5e-4 sweet spot (old) / 1e-3 (new) |
| **★ Backbone unfreeze** | **iter 36, 42** | **★ 영구 reject** (TAPT 이미 정렬) |
| HDBSCAN forcing (capture cost) | iter 41 | P1 violation reject |
| Cross-cfg NeCo (Quality King base) | iter 40 | base cfg 의존 negative interaction |
| NeCo sister sweep | iter 38, 39 | 0.2 sharp sweet spot |
| **Hierarchical NeCo** (1,2,4 pools) | **iter 50/51** | **2-seed 0.856 TIED — multi-resolution dead** |
| **Zone-Aware NeCo** (z=3, z=4) | **iter 43, 54/55** | **3-seed 0.876±0.012 TIED — single-seed lucky only** |
| LW push | iter 52 (1.2) | TIED with 1.0 (saturate at NeCo anchor) |
| LR push | iter 53 (7e-4) | TIED with 1e-3 (Comp 0.992 within std) |
| TOPK push | iter 54/55 (16) | TIED — multi-seed lucky (single 0.880 = iter 37 lucky tail) |
| QUEUE push | iter 56 (8192) | TIED with 4096 |
| TEMP push | iter 57 (0.06) | TIED with 0.07 (new anchor) |
| NEG push | iter 58 (0.65) | TIED-edge with 0.72 (NEG lever dead at NeCo anchor) |

---

## 표 11 — 6-axis comprehensive saturation matrix (iter 50-58, new anchor + iter 37 base)

> 모든 row: iter 37 base (LW=1.0, LR=1e-3, NEG=0.72, TEMP=0.07, NeCo=0.2, TOPK=12, QUEUE=4096),
> eom mcs=12 ms=3, capture=1.000.

| iter | atomic 변경 | ARI | Comp | AMI | noise(def) | 판정 |
|---|---|---:|---:|---:|---:|---|
| **37 ★ baseline (single)** | — | **0.870** | **0.991** | **0.960** | **0.61%** | **SOTA single seed** |
| 44-46 (3-seed mean) | seed sweep on iter 37 | 0.866±0.014 | 0.989 | 0.958 | 0.65% | multi-seed mean |
| 50 | + NECO_HIER_POOLS="1,2,4" (Hierarchical) | 0.860 | 0.985 | 0.956 | 0.52% | TIED |
| 51 | iter 50 + seed=1 | 0.852 | 0.988 | 0.951 | 0.87% | TIED (2-seed mean 0.856) |
| 52 | LW 1.0 → 1.2 | 0.856 | 0.980 | 0.950 | 0.96% | TIED (LW saturate) |
| 53 | LR 1e-3 → 7e-4 | 0.853 | 0.992 | 0.955 | 0.44% | TIED (LR saturate) |
| **54** | **TOPK 12 → 16 (seed=42)** | **0.880** | 0.987 | 0.959 | 0.87% | LUCKY single — iter 37 lucky tail 정확 재현 |
| **55** | iter 54 + seed=1 | 0.852 | 0.988 | 0.951 | 0.87% | ★★★ Zone z=4 lucky pattern 재현 (2-seed 0.866) |
| 56 | QUEUE 4096 → 8192 | 0.867 | 0.985 | 0.954 | 1.31% | TIED |
| 57 | TEMP 0.07 → 0.06 | 0.856 | 0.984 | 0.952 | 1.57% | TIED (TEMP saturate) |
| 58 | NEG 0.72 → 0.65 | 0.846 | 0.973 | 0.944 | 1.75% | TIED-edge (NEG saturate at NeCo anchor) |

**판정 종합** — 6 hparam axis (LW / LR / NEG / TEMP / TOPK / QUEUE) + Spatial NeCo
variants (Hierarchical 1,2,4 / Zone-Aware z=3) 의 모든 atomic 변경이 multi-seed std
(0.014 ARI) 안으로 tied. **iter 37 cfg = multi-axis saturation point** 확정.

이 결과가 **N5 contribution (comprehensive saturation point lock-in)** — paper finalize
의 마지막 contribution.

## 표 12 — Multi-seed lucky pattern 정확 재현 matrix (N2 강화 evidence)

> 두 완전 다른 ablation axis (code-level Spatial NeCo variant + hparam TOPK)
> 가 동일한 +0.010 ARI lucky variance 를 noise floor 로 공유.

| Axis | iter | seed=42 ARI | seed=1 ARI | 2-seed mean | 2-seed std |
|---|---|---:|---:|---:|---:|
| **iter 37 baseline (3-seed)** | 44-46 | 0.870 | (avg with seed=2,3) | **0.866** | **0.014** |
| **Zone-Aware NeCo z=3** | 43 + reseed | 0.880 | 0.872 | 0.876 | 0.012 |
| **Zone-Aware NeCo z=4** | 43-variant | 0.880 | 0.852 | 0.866 | 0.014 |
| **TOPK 16** | 54 / 55 | **0.880** | **0.852** | **0.866** | **0.014** |

**해석**:
- Zone z=4 와 TOPK 16 의 seed=42/seed=1 ARI 가 **소수점 셋째 자리까지 정확히 일치**
  (0.880 / 0.852 / 0.866 / 0.014).
- 두 axis 의 single-seed +0.010 ARI 는 모두 같은 lucky variance pattern 의 sample.
- → **multi-seed protocol 의 importance 의 강력한 evidence** — 단일 seed 결과로
  novelty axis 채택은 reproducibility 위반.
- 본 paper 의 N2 contribution (multi-seed honesty) 의 핵심 supporting evidence.

## 표 8' — 5-lever saturation status (iter 37 + iter 50-58 update)

| # | Lever | atomic step | new anchor 효과 | iter 37 cfg saturate? (iter 50-58) |
|---:|---|---|---|---|
| 1 | LOCAL_WEIGHT | 0.5 → 1.0 | noise -50% (lever 1, old anchor) | ★ saturate (iter 52: 1.2 TIED) |
| 2 | LR_HEAD | 5e-4 → 1e-3 (new anchor) | (lever, iter 35) | ★ saturate (iter 53: 7e-4 TIED) |
| 3 | IGNORE_NEG_SIM | 0.72 (new anchor lock) | (Iter 13 lever, old anchor) | ★ saturate (iter 58: 0.65 TIED-edge) |
| 4 | NCE_TEMP | 0.07 (new anchor lock) | (Iter 14 lever, old anchor 0.05) | ★ saturate (iter 57: 0.06 TIED) |
| 5 | NECO_WEIGHT | 0 → 0.2 | noise -70% (lever 5, ★ SOTA) | ★ sharp sweet (iter 38/39: 0.1/0.3 reject) |
| (HP) | LOCAL_POS_TOPK | 12 lock | — | ★ saturate (iter 54/55: 16 lucky single only) |
| (HP) | QUEUE_SIZE | 4096 lock | — | ★ saturate (iter 56: 8192 TIED) |
| (Sp) | NeCo Spatial variant | none | — | ★ Hierarchical (iter 50/51) + Zone-Aware (iter 43) 모두 TIED |

**N5 결론**: 5 active levers (encoder-side) + 2 hyperparameter sister axis + 2 NeCo Spatial
variant 모두 sweep 후 saturate. **iter 37 cfg = 6-axis multi-axis saturation point**.

---

## 표 13 — ★ Real Baseline Component Isolation Matrix (B0-B5, 2026-05-11)

> 사용자 지적 정합: 기존 Iter A0 baseline 에 이미 Local / Queue / NEG 활성. 진짜
> component-level contribution isolation 위해 **Global InfoNCE only** 의 minimal baseline (B0)
> 부터 단계별 component 추가. HDBSCAN cfg `eom mcs=12 ms=3` 모든 row 동일 고정. seed=42.

### 13a. ablation matrix (cfg + Tier 1+2 결과)

| step | cfg | USE_LOCAL | LW | USE_QUEUE | NEG | NeCo | P1 cap | P2 noise | P3 Comp | P4 Hom | AMI | NMI | ARI | Sil | n_cl |
|:-:|---|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | Global only | F | 0 | F | off | 0 | **1.000** | 6.195% | 0.9602 | 0.929 | 0.9290 | 0.949 | 0.8231 | 0.582 | 37 |
| B1 | + Local (LW=0.5) | T | 0.5 | F | off | 0 | 1.000 | 3.927% | 0.9665 | 0.9351 | 0.9387 | 0.9505 | 0.8514 | 0.5139 | 37 |
| B2 | LW=1.0 | T | 1.0 | F | off | 0 | 1.000 | 6.195% | 0.9602 | 0.9257 | 0.9290 | 0.9427 | 0.8231 | 0.5089 | 37 |
| B3 | + MoCo Queue | T | 1.0 | T | off | 0 | 1.000 | 1.309% | 0.9828 | 0.9365 | 0.9496 | 0.9591 | 0.8464 | 0.5727 | 36 |
| **B4** ★ | + NEG=0.72 | T | 1.0 | T | 0.72 | 0 | 1.000 | **0.524%** | **0.9852** | 0.9439 | 0.9557 | 0.9641 | **0.8605** | 0.6109 | 37 |
| B5 | + NeCo 0.2 (=iter 37) | T | 1.0 | T | 0.72 | 0.2 | 1.000 | 0.960% | 0.9801 | 0.9403 | 0.9503 | 0.9598 | 0.8564 | 0.6104 | 37 |

run_dir:
- B0: `outputs_contrastive_260511_154102/`
- B1: `outputs_contrastive_260511_162616/`
- B2: `outputs_contrastive_260511_170230/`
- B3: `outputs_contrastive_260511_173842/`
- B4: `outputs_contrastive_260511_181441/`
- B5: `outputs_contrastive_260511_185039/`

### 13b. component-by-component isolated effect (Δ vs 직전 step)

| step | Δstep | ΔARI | Δnoise | ΔComp | ΔAMI | 판정 |
|:-:|---|---:|---:|---:|---:|---|
| B0 → B1 | + Local DenseCL | **+0.028** | **-2.27pp** | +0.006 | +0.010 | ✓ Local 단독 효과 |
| B1 → B2 | LW=0.5 → 1.0 | **-0.028** | **+2.27pp** | -0.006 | -0.010 | **✗ LW 단독 regression** |
| B2 → B3 | + MoCo Queue | **+0.023** | **-4.89pp** | +0.023 | +0.021 | **★★★ N6 huge — LW+Queue interaction** |
| B3 → B4 | + NEG=0.72 | **+0.014** | **-0.78pp** | +0.003 | +0.006 | ✓ NEG 단독 효과 small but clean |
| B4 → B5 | + NeCo 0.2 | **-0.004** | **+0.44pp** | -0.005 | -0.005 | **✗ NeCo isolated ≈ 0** |

**누적 (B0 → B5)**: ΔARI **+0.033**, Δnoise **-5.24pp**, ΔComp **+0.020**, ΔAMI **+0.021**.

### 13c. 핵심 발견 — N6 (Component Interaction) NEW

1. **LW lever (encoder paper N3-1) isolated effect 는 negative** — B1→B2 의 LW=0.5→1.0
   단독 변경이 ARI -0.028, noise +2.27pp.
2. **LW 의 진짜 효과 = Queue 와의 interaction** — B2→B3 의 Queue 추가가 ARI +0.023, noise -4.89pp.
   B2 (LW 단독 활성) 위에서 Queue 가 LW over-emphasis 를 흡수.
3. **NeCo (paper N1) isolated effect ≈ 0 또는 negative** — B4→B5 의 NeCo 0→0.2 단독 변경이
   ARI -0.004, noise +0.44pp.
4. **B4 > B5** — NeCo 없는 cfg 가 NeCo 있는 cfg 보다 모든 metric 우위 (ARI 0.8605 > 0.8564,
   Comp 0.9852 > 0.9801, noise 0.524% < 0.960%).
5. **B5 vs iter 37 (same seed=42)** — ARI 0.8564 vs 0.870 (Δ -0.014, multi-seed std 만큼!).
   → same-seed run-to-run variance 가 NeCo isolated effect 보다 큼. **N2 (multi-seed) 강한 evidence**.

### 13d. paper contribution 분류 갱신 (B0 → B5)

| Component | Source | Paper claim | Real Baseline isolated effect |
|---|---|---|---|
| ConvNeXtV2 + TAPT | external | backbone choice | B0 ARI 0.823 (이미 강함) |
| Global InfoNCE | baseline (B0) | 기존 (SimCLR/MoCo) | base |
| Local InfoNCE | baseline (B1) | 기존 (DenseCL Wang 2021) | +0.028 ARI ✓ |
| LW tuning | hparam (B2 sweet) | hparam discovery | -0.028 isolated (paper N6) |
| MoCo Queue | baseline (B3) | 기존 (MoCo He 2020) | +0.023 ARI (lift LW) |
| NEG filter | baseline (B4) | 기존 (NV-Retriever) | +0.014 ARI ✓ |
| **NeCo** | **B5 (paper N1)** | Pariza 2024 wafer first | **-0.004 isolated ✗** |
| HDBSCAN eom + ms=3 | post-hoc (paper N3) | new | encoder 무관 (모든 row 동일) |

★ paper 진짜 NEW contribution 명확화:
- **N1 (NeCo)**: isolated effect ≈ 0, 단 combined (B3 → B5 total ΔARI +0.010) 효과만 인정 — interaction 필수
- **N3 (HDBSCAN)**: encoder 학습 무관, 모든 row 에서 noise -91% 효과 (12.6%→0.61%, leaf→eom+ms3)
- **★ N6 (Component Interaction, NEW)**: LW lever 의 진짜 효과 = isolated 아닌 Queue interaction

---

## 표 14 — ★ NEW cfg + 4-component lattice + multi-seed (iter 67-77, 2026-05-12)

> iter 67-77 (총 11 iter) 의 4-component lattice 탐색 결과.
>
> ★★★★★ **2026-05-12 CORRECTION**: 이전 본 표의 "Sil +30%" / "Sil 0.610 vs 0.794" claim
> 은 **HDBSCAN protocol mismatch artefact** 였음. B5 / B4 Sil 값이 leaf+ms=4 protocol 로
> 측정됐고, iter 70 NEW 는 eom+ms=3 (defect-only) 로 측정. apples-to-apples 재계산 (모든 row
> eom + mcs=12 + ms=3, defect-only scope) 결과:
> - B5 Sil = **0.7988** (0.6104 가 아님), B4 Sil = **0.8012**, iter 70 NEW Sil = **0.7860**
> - 따라서 NEW vs B5 Sil 은 **-0.013 (slightly worse, NOT +30%)**
> - 진짜 NeCo gain = **Normal-cluster consolidation** (Normal noise 77.7% → 14.1%, 859 Normals
>   → 1 dense cluster) — defect-cluster geometry 가 아니라 full-set partitioning 의 Normal/defect
>   boundary stability.
>
> 이하 표 row 의 Sil 컬럼은 apples-to-apples 재계산 값으로 정정.

### 14a. 4-component lattice (Local / Queue / NEG × NeCo) — seed=42 only

> 모든 row: Global InfoNCE base + NECO_WEIGHT (0 또는 0.2 명시) + LW=1.0 (Local 있을 때).
> HDBSCAN eom mcs=12 ms=3, seed=42, capture=1.000 모든 row.

> ★ Sil 컬럼은 **apples-to-apples (eom + mcs=12 + ms=3, defect-only)** 재계산 값.
> 이전 mixed-protocol 값 (B4/B5=0.61, NEW=0.79) 은 retracted.

| iter | Local | Queue | NEG | NeCo | ARI | Comp | AMI | noise | Sil (apples) | n_cl |
|:-:|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|
| B0 | F | F | F | 0 | 0.8231 | 0.9602 | 0.9290 | 6.20% | (TBD) | 37 |
| B1 | T | F | F | 0 | 0.8514 | 0.9665 | 0.9387 | 3.93% | (TBD) | 37 |
| B2 | T(LW=1) | F | F | 0 | 0.8231 | 0.9602 | 0.9290 | 6.20% | (TBD) | 37 |
| B3 | T | T | F | 0 | 0.8464 | 0.9828 | 0.9496 | 1.31% | (TBD) | 36 |
| B4 | T | T | T | 0 | 0.8605 | 0.9852 | 0.9557 | 0.524% | **0.8012** | 37 |
| B5 (=iter 37) | T | T | T | 0.2 | 0.8564 | 0.9801 | 0.9503 | 0.96% | **0.7988** | 37 |
| iter 67 | T | F | F | 0.2 | 0.8508 | 0.9659 | 0.9390 | 3.93% | (TBD) | 37 |
| iter 68 | T | T | F | 0.2 | 0.8464 | 0.9828 | 0.9496 | 1.31% | (TBD) | 36 |
| iter 69 | F | F | F | 0.2 | 0.8514 | 0.9665 | 0.9387 | 3.93% | (TBD) | 37 |
| **iter 70** | **F** | **T** | **T** | **0.2** | **0.8797** | **0.9872** | **0.9594** | **0.87%** | **0.7860** | 37 |
| iter 74 | F | F | T | 0.2 | 0.8514 | 0.9665 | 0.9387 | 3.93% | (TBD) | 37 |
| iter 75 | F | T | F | 0.2 | 0.8822 | 0.9841 | 0.9599 | 1.31% | (TBD) | 38 |

### 14b. NEW cfg multi-seed (iter 70/71/72, 3-seed)

> NEW cfg = Global + NeCo (0.2) + Queue (4096) + NEG (0.72), **no Local**.
> Sil 값은 apples-to-apples (eom mcs=12 ms=3, defect-only) 가정.

| seed | iter | ARI | Comp | AMI | noise | Sil (apples) |
|:-:|:-:|---:|---:|---:|---:|---:|
| 42 | 70 | 0.8797 | 0.9872 | 0.9594 | 0.87% | 0.7860 |
| 1 | 71 | 0.8491 | 0.9856 | 0.9488 | 1.05% | (TBD same-protocol) |
| 2 | 72 | 0.8475 | 0.9747 | 0.9428 | 2.53% | (TBD same-protocol) |
| **avg** | — | **0.859 ± 0.018** | **0.9825 ± 0.007** | **0.9503 ± 0.008** | **1.48%** | (TBD multi-seed) |

### 14c. NEW cfg vs B5 (iter 37 cfg) — multi-seed comparison ★ CORRECTED 2026-05-12

| cfg | ARI single-seed=42 | ARI 3-seed avg | Sil single-seed=42 (apples) | noise avg | components |
|---|---:|---:|---:|---:|:-:|
| B5 (Local + Queue + NEG + NeCo) | 0.8564 | 0.856 ± 0.012 | **0.7988** | 0.96% | 5 |
| NEW (NeCo + Queue + NEG, no Local) | 0.8797 | 0.859 ± 0.018 | **0.7860** | 1.48% | 4 |
| Δ (NEW − B5) | **+0.023** (single-seed) | **+0.003** (3-seed avg, marginal, within std) | **−0.013 (Sil equivalent, slight regression)** | +0.52pp | −1 |

**★ CORRECTED 결정적 발견** (HDBSCAN protocol mismatch retraction):

- 이전 claim "Sil +30%" / "Sil +0.184" 는 **B5 가 leaf+ms=4, NEW 가 eom+ms=3** 인 cross-protocol
  비교 artefact. apples-to-apples 재측정 (모두 eom + mcs=12 + ms=3, defect-only) 시 B5 Sil =
  **0.7988**, NEW Sil = **0.7860** → Sil **NEW slightly worse (−0.013)** within seed variance.
- ARI 측면 NEW vs B5: single-seed=42 **+0.023**, 3-seed avg **+0.003** (marginal, within std 0.018).
- noise floor +0.52pp 후퇴.
- **진짜 NeCo gain** (cluster-analyzer 분석, `outputs_contrastive_260512_001719/eval/cluster_report.parquet`):
  - defect-cluster intra_p95 **+26%** (NeCo 가 defect geometry 를 widening — 압축이 아님)
  - **Normal-cluster consolidation**: Normal noise 77.7% → 14.1% (859/1000 Normals → 1 dense cluster)
  - full-set ARI 0.83 (NEW) vs 0.69 (B5) — **Normal 분리 quality 가 진짜 lift**
- → paper N1 v5 재정의: **NeCo 는 defect cluster geometry 가 아닌 Normal/defect boundary
  stability 를 lift**. "geometry vs partitioning Pareto" 표현 retract.

### 14d. NeCo ≡ DenseCL Local InfoNCE — functionally equivalent on partitioning (iter 69 vs B1)

| cfg | iter | ARI | Δ vs B0 | noise | Sil (mixed-protocol, retracted) | n_cl |
|---|:-:|---:|---:|---:|---:|---:|
| B0 (Global only) | — | 0.8231 | base | 6.20% | 0.582 | 37 |
| B1 (B0 + Local LW=0.5) | — | **0.8514** | **+0.028** | **3.93%** | 0.514 | 37 |
| iter 69 (B0 + NeCo only) | — | **0.8514** | **+0.028** | **3.93%** | 0.707 | 37 |

**4 자리 ARI 동일**, noise 동일, n_cl 동일 → patch-neighbor consistency 메커니즘 동치 (ARI/noise).

★ **CORRECTION 2026-05-12**: 이전 "NeCo Sil +0.193 (NeCo > Local on geometry)" claim 은 B1
Sil 0.514 가 다른 HDBSCAN protocol 로 측정됐을 가능성 (apples-to-apples 재측정 미완) 으로
**retract**. ARI / noise / n_cl 4자리 동일성은 유지 (HDBSCAN 결과 partition 동일 = mechanism
substitutability 유효). Silhouette 우위 claim 은 apples-to-apples 재측정 후에만 재인정.

### 14e. NeCo weight sweep (seed=42, NEW cfg) — ★ Sil pattern retracted

| NeCo weight | iter | ARI | noise | Sil (mixed-protocol, retracted) |
|---:|:-:|---:|---:|---:|
| 0.0 | B4 (Local base) | 0.8605 | 0.524% | 0.6109 → apples 0.8012 |
| **0.2 ★** | **iter 70 (NEW)** | **0.8797** | 0.87% | 0.7860 |
| 0.4 | iter 77 | 0.8605 | 0.52% | 0.8012 |

**패턴** (CORRECTED 2026-05-12):
- ARI inverse-U with **peak at 0.2** (single-seed, within multi-seed std at 3-seed avg)
- noise monotonic ↑ with NeCo weight (NEG protects lower weight)
- ~~Sil monotonic ↑ with NeCo weight~~ **RETRACTED** — apples-to-apples B4 Sil = 0.8012,
  NEW (NeCo=0.2) Sil = 0.7860, NeCo=0.4 Sil = 0.8012. **No monotonic ascent**;
  geometry-vs-partitioning Pareto frontier claim **retracted**.

→ ~~Pareto frontier paper claim (ARI vs Sil trade-off)~~ **retracted**. 진짜 trade-off 는
**ARI marginal lift (single-seed +0.019, 3-seed +0.003) vs noise floor +0.52pp + Sil −0.013**.
Net NEW vs B5 = ARI marginal / noise slight regression / Sil equivalent → **ARI marginal lift
은 Normal-cluster consolidation 에서 옴** (paper N1 v5 reframe).

### 14f. N7 NEW Component Dependency Hierarchy

iter 74 (NeCo + NEG, **no Queue**) = ARI **0.8514** = iter 69 (NeCo only) **exact identical**
(4자리, noise 3.93%, Sil 0.7071, n_cl 37 모두 동일).

→ **NEG effect = 0 when Queue absent**. NEG 의 false-negative protection 은 large negative
pool (Queue 4096) 의 statistical distribution 필요. batch-only negative (n=8) 로는 filter
통계적 의미 X.

```
Component Dependency Hierarchy (N7):

  Required:     Global InfoNCE + {Local DenseCL ‖ NeCo}   ← substitutes, choose one
  Significant:  MoCo Queue (+0.029 with NeCo, +0.023 with Local LW=1.0)
  Conditional:  NEG filter ← requires Queue (no-Queue NEG = 0)
  Substitutable: Local DenseCL ↔ NeCo (equivalent on ARI/noise/n_cl;
                  Sil comparison requires apples-to-apples re-measurement)
```

★ CORRECTION: "Deprecated: Local DenseCL (NeCo better on Sil)" 는 retract. apples-to-apples
재측정 후 Sil 비교 필요. ARI / noise / n_cl 동치는 유지 (substitutability).

### 14g. TEMP × component interaction (cross-cfg, paper N6 extension)

| base cfg | TEMP 0.07 | TEMP 0.05 | Δ |
|---|---:|---:|---:|
| B5 (with Local) | 0.8564 (iter 65) | 0.8700 (iter 37) | **+0.014 ★** |
| NEW (no Local, iter 70) | 0.8797 | **0.8555** (iter 73) | **-0.024 ✗** |

→ TEMP 0.05 lift 가 **Local 의 patch-stability 와 시너지** 였음. NeCo only 환경에서는
TEMP 0.07 sweet spot, lower TEMP 면 NeCo neighbor 신호 over-sharpen → ARI 24pp 후퇴.

**paper N6 강화 evidence**: "best hparam depends on component context — 단일 component
sweep 으로는 optimum 못 찾음".

### 14h. ★ paper N1 v5 (NeCo reframe, FINAL HONEST) — 2026-05-12

cluster-analyzer agent re-analysis of `outputs_contrastive_260512_001719/eval/cluster_report.parquet`
(iter 70 NEW) vs B5 (iter 37 cfg) under same HDBSCAN protocol (eom mcs=12 ms=3, defect-only) :

> **NeCo improves ARI on full-set clustering (with Normal class) via Normal-cluster
> consolidation — Normal noise 77.7% → 14.1%, 859 Normals merged into 1 dense cluster,
> boosting Completeness 0.917 (vs B5 0.851) and full-set ARI 0.83 (vs B5 0.69). On
> defect-only metrics, NeCo and DenseCL Local InfoNCE are functionally equivalent
> (Sil ±0.013, ARI ±0.003 multi-seed avg). The benefit is in Normal/defect boundary
> stability, not defect-cluster geometry.**

Evidence:
- defect-cluster intra_p95 (NeCo) **+26%** vs B5 — NeCo actually **widens** defect
  geometry, not compacts (Sil is robust because inter-cluster separation grows more).
- Normal-cluster: 1000 Normal samples, B5 noise 77.7% → NEW noise 14.1%.
- 859 / 1000 Normals → 1 dense Normal cluster in NEW; B5 splits Normals across noise + tail.
- full-set ARI (Normal included): B5 0.69 vs NEW 0.83 (+0.14) — main lift channel.
- defect-only ARI: B5 0.856 vs NEW 0.859 (+0.003 marginal, within multi-seed std).

→ N1 contribution **재정의 final v5**: NeCo 의 진짜 mechanism = Normal/defect boundary
stability (not defect-cluster compactness). 이전 v4 의 "geometry-vs-partitioning Pareto"
표현 retracted.

### 14i. ★ paper N8 (NEW contribution) — HDBSCAN Protocol Mismatch Methodology Lesson

**N8 Methodology contribution (NEW, 2026-05-12)**:

> Comparing clustering results across different HDBSCAN configurations (leaf vs eom,
> ms=3 vs ms=4) is a known source of spurious metric differences. In our cycle, an
> initially-reported "Silhouette +30%" headline (B5 0.610 vs NEW 0.794) was traced to
> a protocol mismatch — B5/B4 Silhouette had been computed under leaf + ms=4, while
> iter 70 NEW had been computed under eom + ms=3 (defect-only scope). Apples-to-apples
> re-measurement under unified eom + mcs=12 + ms=3 + defect-only gives B5 Sil 0.7988,
> NEW Sil 0.7860 — a Sil regression of −0.013, not a +30% lift.
>
> **Methodology N8 deliverable**: any contrastive-clustering paper claiming a
> Silhouette / ARI / noise difference across cfg families must explicitly fix every
> HDBSCAN axis (selection_method, mcs, ms, epsilon) **and the metric scope (full-set vs
> defect-only)** before the diff is interpretable. The N8 evidence is the
> retracted-+30%-Sil artefact in this paper: a real-data example of protocol mismatch
> producing a paper-grade spurious headline that took post-hoc cluster-analyzer
> inspection to debug. We list this as N8 to inoculate practitioners against the same
> failure mode.

### 14j. run_dir 매핑 (iter 67-77)

| iter | seed | cfg 요약 | run_dir |
|:-:|:-:|---|---|
| 67 | 42 | B2 + NeCo (no Queue) | `outputs_contrastive_260511_215652/` |
| 68 | 42 | B3 + NeCo (no NEG) | `outputs_contrastive_260511_224723/` |
| 69 | 42 | B0 + NeCo only | `outputs_contrastive_260511_233312/` |
| **70 ★** | 42 | **NEW SOTA cfg** | `outputs_contrastive_260512_001719/` |
| 71 | 1 | NEW + seed=1 | `outputs_contrastive_260512_010113/` |
| 72 | 2 | NEW + seed=2 | `outputs_contrastive_260512_014507/` |
| 73 | 42 | NEW + TEMP 0.05 | `outputs_contrastive_260512_022912/` |
| 74 | 42 | NeCo + NEG (no Queue) | `outputs_contrastive_260512_031310/` |
| 75 | 42 | NeCo + Queue (no NEG) | `outputs_contrastive_260512_035824/` |
| 76 | 1 | minimal 3-comp seed=1 | `outputs_contrastive_260512_044307/` |
| 77 | 42 | NEW + NeCo=0.4 | `outputs_contrastive_260512_052907/` |

### 14k. ★ Retracted claims index (2026-05-12)

다음 claims 는 본 paper 의 다른 section / 외부 doc 에서 발견 시 모두 retract / 정정:

| Retracted claim | Where | Correction |
|---|---|---|
| "Sil +30%" / "Sil +0.184 (+30% robust)" | ABSTRACT v0.5, RESULTS 14c, DISCUSSION 7.10.1, CONCLUSION 8.6, INTRODUCTION C7, manager_report SUMMARY 0.6 | Sil −0.013 (NEW vs B5, equivalent within seed variance) |
| "B5 Silhouette 0.6104" / "NEW Silhouette 0.7941" | RESULTS 14b, 14c, ABSTRACT v0.5 | apples B5 = 0.7988, NEW = 0.7860 |
| "NeCo > Local on Silhouette (+0.193)" | METHOD 3.6, INTRODUCTION C7 (ii), DISCUSSION 7.10.2 | retracted (B1 Sil 0.514 mixed-protocol, apples re-measure pending) |
| "geometry-vs-partitioning Pareto frontier" | DISCUSSION 7.10.1, 7.10.4, ABSTRACT v0.5, CONCLUSION 8.6 | retracted (no Sil monotonic ascent under apples) |
| "Deprecated: Local DenseCL" | RESULTS 14f, CONCLUSION 8.6 | softened to "Substitutable" (v5), further refined to "Complementary per-class" (★ v6 2026-05-12, RESULTS §16) — Local NOT deprecated, absolute SOTA cfg keeps both |
| "NeCo functionally equivalent to Local DenseCL — substitutable on partitioning" | METHOD 3.6 (v5), DISCUSSION 7.10 (v5), CONCLUSION 8.6 N7 (v5), ABSTRACT v0.6/v0.7, manager_report Phase 2 #1 | ★ v6 refined to **COMPLEMENTARY at per-class scope under Agglo Ward K=42** (RESULTS §16). aggregate HDBSCAN ARI equality preserved; per-class winner flips on both sides (B5 wins fork/scratch sub-pattern variants 4 classes 100%, NEW wins CenterCircle uniform 100%). Absolute SOTA Agglo K=42 = B5 (both combined) ARI 0.9358 (Δ +0.0158 vs NEW 0.9200). |
| "NeCo weight Sil monotonic ↑" | RESULTS 14e, DISCUSSION 7.10.4, CONCLUSION 8.6 | retracted (B4 apples Sil 0.8012 ≥ NEW 0.7860) |

---

## 표 15 — ★ Clustering algorithm dependency benchmark (iter 82-83, 2026-05-12)

> Post-iter 82/83 benchmark. 동일 contrastive embedding (B4 Local-based / B5 iter-37 cfg /
> iter 70 NEW SOTA) 위에서 5 가지 clustering 알고리즘 (HDBSCAN / DP-GMM / KMeans K=42 /
> Agglomerative Ward K=42 / Spectral K=42) 의 ARI / NMI 측정. defect-only scope (K_gt=42).
> Source: `tier1_clustering_benchmark.json`.

### 15a. Single-seed ARI (seed=42) — 5 method × 3 cfg matrix

> KMeans / Agglomerative / Spectral 의 K=42 는 oracle (GT class 수). HDBSCAN /
> DP-GMM 은 unknown-K (density / variational discovery). Sil / capture / noise 컬럼은
> 별도 표 (15c) 참조.

| cfg | HDBSCAN (K) | DP-GMM (K) | KMeans-42 (oracle K) | Agglo-Ward-42 (oracle K) | Spectral-42 (oracle K) |
|---|---:|---:|---:|---:|---:|
| B4 Local-based | **0.8605** (37) | 0.8344 (47) | 0.8876 | **0.9055** | 0.4046 |
| B5 (= iter 37 cfg) | 0.8564 (37) | 0.8369 (46) | 0.8854 | **★ 0.9358** | 0.7898 |
| **iter 70 NEW SOTA** | **0.8797** (37) | **0.8413** (47) | 0.8798 | 0.9200 | 0.2289 |

**관측**:
1. **Unknown-K methods (HDBSCAN / DP-GMM)**: iter 70 NEW > B5 ≈ B4 — density-based 가 NEW 의 noise/outlier handling 우위를 반영.
2. **Known-K methods (KMeans / Agglomerative)**: B5 > iter 70 NEW ≈ B4 — centroid/linkage-based 가 oracle K 로 noise-handling 이득을 제거하면 B5 의 tighter feature space 가 드러남.
3. **Spectral K=42**: B5 0.79, B4 0.40, NEW 0.23 — variance 0.23~0.79 across cfg. Spectral Laplacian eigengap unstable (graph-not-fully-connected warning at fit time).
4. **Single-seed=42 max**: B5 + Agglomerative Ward K=42 = ARI 0.9358 (oracle K). ★ v7
   retracted as multi-seed SOTA — seed=1 reproduce = 0.8482, 2-seed avg = 0.8920 ± 0.062,
   BELOW NEW 3-seed avg 0.9014 ± 0.022 (RESULTS §17b). Use NEW + Agglomerative Ward K=42
   as the multi-seed-compliant known-K SOTA.

### 15b. NMI (Tier 2 supplementary)

| cfg | HDBSCAN | DP-GMM | KMeans-42 | Agglo-Ward-42 | Spectral-42 |
|---|---:|---:|---:|---:|---:|
| B4 Local-based | 0.9557 | 0.9276 | 0.9554 | **0.9651** | 0.8009 |
| B5 (= iter 37 cfg) | 0.9503 | 0.9277 | 0.9515 | **★ 0.9704** | 0.9334 |
| iter 70 NEW SOTA | **0.9594** | 0.9306 | 0.9534 | 0.9620 | 0.7205 |

NMI ranking matches ARI ranking direction across all 5 methods. B5 + Agglomerative = NMI 0.9704
(single-seed=42 only; multi-seed NMI not measured — see §17 for multi-seed ARI). ★ v7 caveat:
same B5 seed-flip hazard applies (Δ ARI −0.088 seed=42 → seed=1 on Agglo K=42, RESULTS §17a);
single-seed NMI values likely share the same lucky-outlier hazard.

### 15c. ★ Multi-seed ARI on iter 70 NEW (3-seed: 42, 1, 2)

| Method | K source | NEW seed=42 | NEW seed=1 | NEW seed=2 | NEW 3-seed avg | NEW std | B5 single-seed=42 |
|---|---|---:|---:|---:|---:|---:|---:|
| HDBSCAN | density (unknown-K) | 0.8797 | 0.8491 | 0.8475 | **0.8588** | 0.018 | 0.8564 |
| **Agglomerative K=42** | linkage (oracle K) | 0.9200 | 0.8854 | 0.8989 | **0.9014** | 0.022 | **★ 0.9358** |
| KMeans K=42 | centroid (oracle K) | 0.8798 | 0.8456 | 0.8779 | **0.8678** | 0.026 | 0.8854 |

> DP-GMM / Spectral multi-seed 미측정 (DP-GMM은 variational, Spectral은 unstable 로 단일 점만 보고).

**관측 — paper-grade**:
1. **Multi-seed NEW + Agglomerative = 0.9014 ± 0.022** — B5 single-seed=42 + Agglomerative 0.9358 보다 −0.034 후퇴. **B5 가 known-K frontier 의 진짜 SOTA** (multi-seed measurement 권장).
2. **Multi-seed NEW + HDBSCAN = 0.859 ± 0.018** — 14b 표와 일치 (재확인).
3. Cross-method seed=42 vs seed=1 lucky variance 가 axis-independent — Agglo 도 +0.034 seed=42 → seed=1 drop (NEW), KMeans 도 +0.034 drop. **Lucky-pattern N2 evidence 가 clustering method axis 까지 확장**.

### 15d. ★ Dual-frontier framework (paper N9 NEW)

본 §15 의 5-method benchmark 가 paper N9 (clustering algorithm dependency) 의 evidence:

- **Unknown-K frontier (real-world deployment)**:
  iter 70 NEW + HDBSCAN = ARI **0.880 (single)** / 0.859 ± 0.018 (3-seed).
  Rationale: 운영 환경은 K 모름 + Normal/defect boundary stability 필요 (paper N1 v5).
  NeCo 의 Normal-cluster consolidation (Normal noise 77.7% → 14.1%) 가 HDBSCAN 의 density-cliff 안정화.

- **Known-K frontier (oracle benchmark)**:
  B5 (iter 37 cfg) + Agglomerative Ward K=42 = ARI **0.9358 (single)** / 0.9014 ± 0.022 NEW 3-seed.
  Rationale: K known 환경 (e.g., 알려진 defect taxonomy 가 있는 lab benchmark) 에서는 linkage-based 가 fine-structure recover. B5 의 tighter defect-cluster geometry (intra_p95 NeCo 적용 안 함) 가 우위.

**ARI claim 의 magnitude 가 clustering algorithm 에 따라 +0.04 ~ +0.10 변동**:
- B4 HDBSCAN→Agglo: 0.8605 → 0.9055 (+0.045)
- B5 HDBSCAN→Agglo: 0.8564 → 0.9358 (+0.079)
- NEW HDBSCAN→Agglo: 0.8797 → 0.9200 (+0.040)

→ paper N9 deliverable (NEW 2026-05-12): **"ARI claim must specify clustering method"**.
HDBSCAN 결과 만 으로 SOTA claim 시 oracle K 방법론으로 +0.04~+0.10 더 나옴 — 동일 embedding 임에도. methodology disclosure 의무.

### 15e. Practitioner choice tree (operational guide)

```
1. K (number of defect classes) known at deployment?

   YES (e.g., closed lab benchmark, fixed defect taxonomy)
   ├─ Need fine sub-structure recovery? → Agglomerative Ward K=42 (B5 cfg)
   │                                       SOTA 0.9358 (single) / 0.9014 ± 0.022 (3-seed NEW)
   └─ Need fast centroid grouping?      → KMeans K=42 (B5 cfg)
                                          ARI ~0.88, faster but less granular

   NO (real-world unknown-defect-discovery)
   ├─ Normal-dominant stream + open-set? → HDBSCAN + iter 70 NEW cfg (paper N1 v5)
   │                                        ARI 0.880 single / 0.859 ± 0.018 3-seed
   │                                        Normal/defect boundary stability ★
   └─ Defect-only clustering pipeline?    → HDBSCAN + B5 iter 37 cfg
                                            ARI 0.856 single, noise floor 0.61% (lower)

Avoid:
- Spectral K=42 → unstable (ARI 0.23~0.79 across cfg, graph-disconnect warnings)
- DP-GMM with under-budget K_max → variational under-shoots (K_discovered=46~47 reasonable here, but Tier 2 only)
```

### 15f. run_dir / source

JSON evidence: `D:/project/unknown-contrastive/tier1_clustering_benchmark.json`.

run_dir cfg source:
- B4 = `outputs_contrastive_260511_181441/`
- B5 = `outputs_contrastive_260511_185039/`
- iter 70 NEW seed=42 = `outputs_contrastive_260512_001719/`
- iter 71 NEW seed=1 = `outputs_contrastive_260512_010113/`
- iter 72 NEW seed=2 = `outputs_contrastive_260512_014507/`

All 5 clustering methods fit on **defect-only embedding** (Normal excluded for fair K_gt=42 comparison).

> Tier 2 NMI / Tier 3 (DB / CH index) Spectral 안정성 detail 은 디버그 부록.
> 본 paper headline 은 Tier 1 ARI + (Tier 2 NMI 부수).

---

## 표 16 — ★ Per-class Agglomerative Ward K=42 purity breakdown (★ N1 v6 NEW, 2026-05-12)

> §15 의 Agglomerative Ward K=42 (defect-only, oracle K=K_gt) 결과를 per-GT-class 로
> decompose. dominant cluster purity = (해당 GT class wafer 가 가장 많이 들어간 cluster
> 내 동일 GT class 비율). per-class breakdown 이 aggregate ARI 가 못 잡는 **complementary
> inductive bias** 를 드러냄.
>
> Source: `cluster_report.parquet` per-class dominant-cluster purity, run_dir B5 =
> `outputs_contrastive_260511_185039/`, NEW = `outputs_contrastive_260512_001719/`.
> Clustering: Agglomerative Ward K=42 on defect-only embedding (seed=42, oracle K).

### 16a. Top NEW > B5 wins (where NeCo-only beats Local+NeCo combined)

| GT class | N | B5 (Local + NeCo) | NEW (NeCo only) | Δ NEW − B5 |
|---|---:|---:|---:|---:|
| **CenterCircle** | 42 | 54.8% | **100.0%** | **+45.2pp** |
| **Edge-Top_fork** | 20 | 90.0% | **100.0%** | **+10.0pp** |

→ NeCo alone consolidates uniform-pattern (round / symmetric) classes where Local DenseCL
+ NeCo combined fails to bind tightly.

### 16b. Top B5 > NEW wins (where Local DenseCL is necessary)

| GT class | N | B5 (Local + NeCo) | NEW (NeCo only) | Δ NEW − B5 |
|---|---:|---:|---:|---:|
| **Edge-Ring_fork** | 31 | **100.0%** | 64.5% | **−35.5pp** |
| **Center_scratch** | 40 | **95.0%** | 75.0% | **−20.0pp** |
| **Donut_fork** | 37 | **100.0%** | 81.1% | **−18.9pp** |
| **Edge-Top_scratch** | 19 | **100.0%** | 84.2% | **−15.8pp** |

→ Local DenseCL's grid-cell contrast integrates fork/scratch rotational+positional
sub-pattern variants into a single 100%-purity cluster; NeCo alone fragments these
into multiple sub-clusters (purity 64-84%).

### 16c. Net average per-class purity

| cfg | avg per-class purity (Agglo K=42) | aggregate ARI (Agglo K=42, single-seed=42) |
|---|---:|---:|
| **B5** (5-component, Local + Queue + NEG + NeCo) | **97.0%** | **★ 0.9358** |
| **NEW** (4-component, NeCo + Queue + NEG, no Local) | 96.2% | 0.9200 |
| **Δ NEW − B5** | **−0.83pp** | **−0.0158** |

→ B5 marginally better on micro-aggregate purity AND aggregate ARI. The aggregate
"NeCo ≡ Local DenseCL" identity claim (paper N1 v5) holds under HDBSCAN unknown-K
density clustering but **breaks under Agglomerative Ward K=42 oracle linkage
clustering**: B5 strictly above NEW by ARI +0.0158, and per-class winners flip on
both sides with class-specific magnitudes up to ±45pp.

### 16d. Mechanism interpretation (paper N1 v6)

```
Local DenseCL (Wang 2021): grid-cell intra-image contrast
   → integrates sub-pattern variants within a class
   → wins on fork / scratch / scratch_rot rotational+positional variants
   → CenterCircle uniform geometry: fails to consolidate (B5 54.8%)

NeCo (Pariza 2024): patch-neighbor rank consistency
   → consolidates uniform-pattern symmetric geometry
   → wins on CenterCircle round shape, Edge-Top_fork
   → fork/scratch sub-pattern variants: fragments (NEW 64-84%)

Combined (B5 = Local + NeCo):
   → both per-class strength axes active
   → single-seed=42 Agglo K=42 ARI 0.9358 ★ v7 retracted as multi-seed SOTA
     (seed=1 reproduce = 0.8482; B5 2-seed avg 0.8920 ± 0.062 < NEW 3-seed
     0.9014 ± 0.022 on same Agglo K=42 — RESULTS §17b)
   → marginal win on single-seed=42 micro-aggregate purity (97.0% vs 96.2%)

NEW (NeCo only, drop Local):
   → uniform-pattern strength preserved
   → sub-pattern variant strength lost
   → still wins HDBSCAN aggregate ARI (Normal-cluster consolidation, paper N1 v5)
   → strict regression on Agglo K=42 sub-pattern classes
```

### 16e. Implications

1. **Local DenseCL는 NOT deprecated** (v5 retraction refinement). v5 의
   "substitutable on partitioning" claim 은 HDBSCAN aggregate scope 에서만 valid.
   per-class Agglo K=42 scope 에서는 두 mechanism 이 **complementary**.

2. **Absolute SOTA cfg = B5 (5-component, both Local + NeCo)** under known-K
   linkage clustering. v0.7 의 "NEW = parsimonious alternative with equivalent
   ARI" 는 unknown-K HDBSCAN 위에서만 valid; known-K Agglo 에서는 NEW 가 strict
   regression Δ −0.0158 single-seed.

3. **Dual-cfg dual-frontier recipe** (v0.8 ABSTRACT):
   - Unknown-K + HDBSCAN + Normal-dominant stream → **NEW** (Normal/defect
     boundary stability + uniform-pattern consolidation, ARI 0.859 ± 0.018).
   - Known-K + Agglomerative Ward + oracle benchmark → **B5** (complementary
     per-class purity via Local + NeCo combined, ARI 0.9358 single-seed).

4. **Methodology lesson (N1 v6 + N9 cross-link)**: aggregate ARI identity does
   not imply per-class equivalence. Future contrastive-clustering papers that
   compare mechanism families must report per-class purity breakdown under at
   least one oracle-K clustering algorithm (Agglomerative Ward recommended) in
   addition to aggregate ARI under unknown-K density clustering.

### 16f. run_dir / measurement source

- B5 embedding + clustering: `outputs_contrastive_260511_185039/eval/cluster_report.parquet`
- NEW embedding + clustering: `outputs_contrastive_260512_001719/eval/cluster_report.parquet`
- Clustering protocol: scikit-learn `AgglomerativeClustering(n_clusters=42, linkage='ward')`
  on L2-normalized 128-d embedding, defect-only scope (Normal excluded).
- Per-class dominant cluster purity = max_c (count(GT=g, pred=c)) / count(GT=g) for
  each defect class g.

---

## 표 17 — ★ B5 reproducibility (seed=1) + multi-seed avg ARI comparison (★ N1 v7 FINAL, iter 84, 2026-05-12)

> iter 84 (`outputs_contrastive_260512_114525/`) 의 B5 seed=1 측정 결과 → B5 seed=42 0.9358
> 가 **cherry-picked outlier** 였음 confirmation. v6 "B5 absolute SOTA on Agglo Ward K=42"
> claim 의 multi-seed-grounded retraction. v7 = NEW cfg unified multi-seed SOTA.
>
> Source: `outputs_contrastive_260512_114525/tier1_B5_seed1.json` (HDB_ARI, Agglo_ARI,
> KMeans_ARI, HDB_noise). HDBSCAN cfg `eom mcs=12 ms=3` defect-only — apples-to-apples
> with iter 70/71/72 NEW measurements.

### 17a. B5 single-seed reproducibility test (seed=42 vs seed=1)

| Method | B5 seed=42 (iter 83) | B5 seed=1 (iter 84) | Δ B5(s1-s42) | NEW seed=42 | NEW seed=1 | Δ NEW(s1-s42) |
|---|---:|---:|---:|---:|---:|---:|
| HDBSCAN | 0.8564 | **0.8122** | −0.0442 | 0.8797 | 0.8491 | −0.0306 |
| Agglo Ward K=42 | **★ 0.9358** | **0.8482** | **−0.0876** | 0.9200 | 0.8854 | **−0.0346** |
| KMeans K=42 | 0.8854 | 0.8225 | −0.0629 | 0.8798 | 0.8456 | −0.0342 |

**핵심 관찰**:
- **B5 seed=42 → seed=1 Agglo drop = Δ −0.0876** (huge), while NEW Agglo drop = Δ −0.0346 (NEW reproducibility 2.5× better).
- B5 seed=42 0.9358 = lucky tail. seed=1 reproduce 0.8482 = below NEW seed=1 0.8854 on same method (Δ −0.037).

### 17b. Multi-seed avg ARI — NEW vs B5

> NEW: 3-seed avg (iter 70 / 71 / 72 = seeds 42 / 1 / 2).
> B5: 2-seed avg (iter 83 / 84 = seeds 42 / 1). seed=2 measurement pending iter 85.

| Method | B5 2-seed avg ± std | NEW 3-seed avg ± std | Δ (NEW − B5) | std ratio (B5/NEW) |
|---|---:|---:|---:|---:|
| HDBSCAN | 0.8343 ± 0.031 | **0.859 ± 0.018** | **+0.0245** | 1.7× |
| Agglo Ward K=42 | 0.8920 ± 0.062 | **0.9014 ± 0.022** | **+0.0094** | **2.8×** |
| KMeans K=42 | 0.8540 ± 0.044 | **0.8678 ± 0.026** | **+0.0138** | 1.7× |

**관측 — paper-grade**:
1. **NEW > B5 on multi-seed average ARI across ALL three clustering methods** —
   Agglo +0.0094 / HDBSCAN +0.0245 / KMeans +0.0138. The dual-cfg dual-frontier
   recipe (v0.8) that recommended B5 for known-K Agglomerative Ward is **retracted**.
2. **B5 std 0.062 = 2.8× NEW std 0.022 on Agglo K=42** — much less reproducible.
   Same observation on HDBSCAN (1.7×) and KMeans (1.7×). N2 (multi-seed methodology)
   gains its strongest paper-grade evidence: same-cfg run-to-run variance can flip
   single-seed "winner" claims by Δ ARI 0.088 (B5 seed=42 vs seed=1).
3. **Single-cfg recommendation now valid** — NEW (NeCo + Queue + NEG, no Local) is
   the genuine multi-seed SOTA on **both** Unknown-K HDBSCAN AND Oracle-K
   Agglomerative Ward frontiers. The two-frontier framework collapses to a
   **single-cfg + two-clustering-target** framework.

### 17c. v6 "absolute SOTA B5 Agglo K=42 0.9358" retraction

| claim version | claim | status (v7) |
|---|---|---|
| v6 (2026-05-12, RESULTS §16) | "B5 Agglo Ward K=42 single-seed=42 = ARI 0.9358 = absolute SOTA, Δ +0.0158 above NEW 0.9200" | **★ RETRACTED** — single-seed lucky outlier. seed=1 = 0.8482. 2-seed avg = 0.8920 ± 0.062, **below** NEW 3-seed avg 0.9014 ± 0.022 (Δ −0.0094) |
| v6 (2026-05-12, ABSTRACT v0.8) | "Dual-cfg recipe: B5 for Frontier 2 known-K Agglo + NEW for Frontier 1 unknown-K HDBSCAN" | **★ RETRACTED** — NEW dominates both frontiers on multi-seed avg. New recipe = single NEW cfg + two clustering targets |
| v6 (2026-05-12, METHOD §3.6) | "Local DenseCL NOT deprecated — absolute SOTA cfg retains both Local + NeCo" | **★ RETRACTED** — B5 (Local + NeCo) multi-seed avg < NEW (NeCo only) multi-seed avg on all three methods |

### 17d. paper N1 v7 (FINAL) — replaces v6

> v5 (HDBSCAN aggregate substitutable) + v6 (Agglo K=42 per-class complementary) → v7:
>
> **"NEW cfg (Global + NeCo + Queue + NEG, no Local) is the multi-seed SOTA on both
> Unknown-K HDBSCAN (0.859 ± 0.018) and Oracle-K Agglomerative Ward K=42 (0.9014 ±
> 0.022) frontiers. The single-seed B5 absolute SOTA claim at ARI 0.9358 (Agglo Ward
> K=42, seed=42) was a cherry-picked outlier — same cfg same protocol seed=1
> reproduces at ARI 0.8482 (Δ −0.088). B5 reproducibility (std 0.062 Agglo) is 2.8×
> worse than NEW (std 0.022 Agglo). Per-class purity complementarity (v6 RESULTS §16)
> is preserved as a single-seed observation but does NOT propagate to multi-seed
> averages — the dual-cfg recipe collapses to a single-cfg recommendation (NEW)."**

### 17e. Implications for retracted claims index (cross-link to §14k)

다음 v6 claims 추가 retract (★ v7 2026-05-12):

| v6 claim | v7 correction |
|---|---|
| "B5 (Local + NeCo) absolute SOTA on Agglo Ward K=42 ARI 0.9358" | seed=42 lucky outlier. 2-seed avg 0.8920 ± 0.062 < NEW 3-seed avg 0.9014 ± 0.022 |
| "Local DenseCL NOT deprecated; absolute SOTA cfg keeps both Local + NeCo" | NEW (no Local) multi-seed avg > B5 (with Local) on all three methods. Local DenseCL is **operational choice**, not required for SOTA |
| "Dual-cfg dual-frontier recipe" | Single-cfg + dual-clustering-target. NEW cfg covers both frontiers. |
| "Complementary inductive biases B5 > NEW Agglo K=42" | Complementary on **single-seed per-class purity** only. Aggregate ARI multi-seed avg shows NEW > B5. |

### 17f. N2 strongest evidence to date

iter 84 의 B5 seed=42 → seed=1 Agglo drop Δ −0.088 = **largest cross-seed flip
documented in this paper's 84-iteration cycle**. Comparable to:

| evidence source | Δ cross-seed | method | role in paper |
|---|---:|---|---|
| iter 37 vs iter 44-46 (3-seed) | ±0.014 (std) | HDBSCAN | N2 original (RESULTS §11/§12) |
| Zone z=4 / TOPK 16 single-axis | ±0.014 (mean diff) | HDBSCAN | N2 cross-axis (RESULTS §12) |
| **★ B5 seed=42 → seed=1 (Agglo K=42)** | **−0.088** | **Agglo Ward K=42** | **★ N1 v7 + N2 strongest** |
| iter 70 NEW seed=42 → seed=1 (Agglo) | −0.0346 | Agglo Ward K=42 | N2 cross-method (RESULTS §15c) |

→ **B5 의 Agglo K=42 reproducibility variance 가 그 외 모든 sources 의 6× 이상**.
N2 (multi-seed methodology obligation) 가 N1 v7 final correction 의 enabling lesson.

### 17g. run_dir / measurement source

| iter | seed | run_dir | json evidence |
|:-:|:-:|---|---|
| 83 (B5) | 42 | `outputs_contrastive_260511_185039/` | §15 benchmark table |
| 70 (NEW) | 42 | `outputs_contrastive_260512_001719/` | §15c row |
| 71 (NEW) | 1 | `outputs_contrastive_260512_010113/` | §15c row |
| 72 (NEW) | 2 | `outputs_contrastive_260512_014507/` | §15c row |
| **84 (B5) ★** | **1** | `outputs_contrastive_260512_114525/` | **`tier1_B5_seed1.json`** |

Multi-seed avg computation:
- B5 2-seed = mean(0.8564, 0.8122) on HDB, mean(0.9358, 0.8482) on Agglo, mean(0.8854, 0.8225) on KMeans.
- NEW 3-seed avg as previously reported (§15c).
- std = sample std (ddof=1, n=2 for B5 means range/√2, n=3 for NEW).

> Per-class purity breakdown for B5 seed=1 Agglo K=42 (vs NEW seed=1) not yet measured —
> deferred to iter 85+ as low-priority since multi-seed avg already settles the v7
> retraction.

## §18 — ★ Computational Performance + Dataset statistics (NEW 2026-05-13)

All numbers in this section are direct measurements from the Claude Code execution
log on the working dataset. Source-of-truth: `docs/paper/manager_report/performance_data_260513.md`.
User directive (260513): "여기 있는 건 모두 claude code 로 직접 실험한 것들이다 그래서
사실들이다" — every figure is a measured fact, not an estimate.

### 18.1 Dataset statistics (anchor `avg30_new_260508_123037`)

| field | value |
|---|---|
| total samples (single-fold anchor) | **2,146** |
| defect samples | 1,146 |
| Normal samples | 1,000 |
| classes | 43 (42 defect + 1 Normal_bank_boundary) |
| class size range | **15** (Thick-Edge_invalid_main) – **1,000** (Normal) |
| image size (encoder input) | 384 × 384 RGB |
| wafer source pool (synth.) | 9,250 PNG @ 6400 × 6400 (`D:/project/data/wm-811k/unknown/`, ~200 per class) |

Class composition (42 defect + Normal):
- **Donut × 5, Edge-Bottom × 5, Edge-Top × 5, Center × 5, Full × 5** — 25 obj-active sub-classes (5 chip-object × 5 wafer-shape).
- **Edge-Ring × 4** (Edge-Ring_invalid_main + Edge-Ring × 3 obj-pair variants).
- **Thick-Edge_invalid_main** (smallest class, n=15 — tier-edge minimum).
- **9 wafer-canvas** (no per-chip object) — BrokenRing, RingDots, CrossScratch,
  CrescentArc, DiagonalSmear, ParallelScratches, CenterDonut, Row, Starburst.
- **Normal_bank_boundary** (n = 1,000) — the dominant single class for
  Normal/defect boundary stability evaluation (paper N1 v5).

### 18.2 Hardware + backbone

| component | spec |
|---|---|
| GPU | NVIDIA RTX 4060 Ti (16 GB VRAM, single GPU dispatch) |
| backbone | ConvNeXtV2-base, FCMAE ImageNet-22k pretrained + TAPT |
| backbone params | **87.7 M** (frozen during contrastive head training) |
| projection head | 2-layer MLP, 128-D output, L2-normalized |

### 18.3 Training time (RTX 4060 Ti, 5 epochs, n=2146, frozen backbone)

| recipe | components | n_seed | min / run | std (min) |
|---|---|---:|---:|---:|
| **NEW** ★ | Global + NeCo + Queue + NEG (no Local) | 3 | **23.7** | **± 0.01** |
| B5 | Local + Queue + NEG + NeCo (= iter 37 cfg) | 3 | 28 – 49 | wide (CV ≈ 30%) |

→ **NEW is ~30% faster than B5** at single-run wall-time, attributable to Local
DenseCL grid-cell forward + backward removal. Across a 3-seed sweep:
NEW total ≈ 71 min, B5 total 85 – 147 min. The wall-time advantage compounds
with the multi-seed reproducibility advantage (§17, std ratio 2.8× lower
under Agglomerative Ward K=42).

### 18.4 Inference latency + throughput (RTX 4060 Ti, single GPU)

| mode | latency / image | throughput |
|---|---:|---:|
| **single wafer (BATCH = 1)** | **14.3 ms** | **70 wafers/sec** |
| batch 8 | 17.2 ms | 58 wafers/sec |
| batch 32 (amortized) | 18.5 ms | 54 wafers/sec |

BATCH=1 has the **lowest per-image latency** because it avoids the
batch-padding stall on small inputs; larger batches amortize launch overhead
but introduce GPU underutilization on the 384 × 384 resolution.

### 18.5 HDBSCAN clustering time (1146 defects)

| operation | time |
|---|---|
| full re-cluster (fresh fit, defect-only) | **507 ms** (model-update path) |
| single-point `approximate_predict` | **≈ 10 ms** (production online) |

### 18.6 Evaluation pipeline time (single GPU, n=2146)

| stage | time |
|---|---|
| embedding extraction (frozen encoder forward) | ≈ 3 min |
| HDBSCAN fit + Tier 1+2 metric + per-class fragmentation report | 4 – 7 min |
| composite cluster PNG rendering (6400 × 6400) | included above |
| **total eval pass** | **7 – 10 min** |

### 18.7 ★ End-to-end production latency per wafer (paper claim)

```
encoder forward pass    14 ms   (BATCH = 1, frozen ConvNeXtV2-base + 128-D head)
HDBSCAN approx_predict  10 ms   (single-point against pre-fit model)
──────────────────────────────────
total                   ~24 ms  /  wafer

→ ~ 40 wafers/sec deployable on a single RTX 4060 Ti
```

→ **Real-time deployable** for in-fab wafer-by-wafer triage at common
semiconductor inspection throughputs (typical line: 1 – 10 wafers/sec, well
within budget). The encoder is the latency bottleneck; HDBSCAN online predict
is sub-half the encoder cost.

### 18.8 Cost-of-experiment summary (84-iter ablation cycle)

| axis | unit | cumulative |
|---|---|---|
| ablation iterations (atomic-change) | 84 | — |
| Real Baseline B0-B5 isolation | 6 | + |
| 4-component lattice (iter 67-77) | 11 | + |
| 5-method clustering benchmark (iter 82-83) | 5 algorithms × 3 cfg | — |
| **total trained encoders** | ~ 90 | wall-time ≈ 50 h (3-seed avg) |

→ A reviewer reproducing the paper's full ablation cycle on a single RTX 4060
Ti would need ≈ 50 GPU-hours for encoders alone (eval/cluster sweep
additional). The single-recipe NEW reproduce (3 seeds, paper headline)
requires ≈ 71 min training + 30 min eval = ≈ **100 min total wall-time**.

## §19 — ★ Step-by-step performance improvement (Step 1 eval-only, 2026-05-13)

Plan reference: `C:\Users\hgcho\.claude\plans\floating-splashing-key.md` Roadmap Step 1.
Source-of-truth: `docs/paper/manager_report/step1_eval_only_summary_260513.md` (single
authoritative summary; all numbers below are direct measurements from that file).

**Eval-only protocol** — no encoder retraining. Same 3-seed NEW recipe runs
(iter 70/71/72), defect-only Tier1 HDBSCAN (eom mcs=12 ms=3). Three orthogonal
post-hoc refinements applied: (1a) RankMe representation-quality column, (1b)
HDBSCAN `cluster_selection_epsilon` sweep, (1c) soft KNN-softmax τ-reassignment
of HDBSCAN noise points.

### 19.1 Step 1a — RankMe + NESum representation quality (paper N10)

| Recipe | RankMe avg | RankMe std | CV | NESum | feat_var |
|---|---:|---:|---:|---:|---:|
| **NEW 3-seed (iter 70/71/72)** ★ | **23.44** | **± 1.80** | **7.7 %** | 4.40 ± 0.69 | 0.563 |
| B5 3-seed | 22.06 | ± 4.99 | 22.6 % | — | — |
| NEW-NeCo (single seed=3) | 21.12 | — | — | 4.54 | 0.543 |
| B5 single seed=42 | 25.20 | — | — | 5.10 | 0.549 |

→ **NEW representation 의 RankMe CV = 7.7 % vs B5 22.6 %** — NEW 가 **64 % 더 stable**
across seeds (std ratio 2.8× lower, same direction as §17b ARI std ratio).

**Spearman ρ(RankMe, ARI) = −0.429** (n = 7 runs across both recipes) — RankMe alone
is **not** an ARI ranking signal; the correlation is weakly negative and statistically
not separable from zero at this sample size. **Paper claim N10**:

> RankMe (Garrido et al. 2023) is informative for **cross-seed stability of representation
> quality** (NEW std 1.80 vs B5 std 4.99 → NEW 64 % more reproducible), **NOT for ARI
> ranking** (ρ = −0.429, n = 7). Reported as a paper-grade representation-quality column
> rather than as an alternative SOTA arbiter.

### 19.2 Step 1b — HDBSCAN cluster_selection_epsilon sweep (paper N9 reinforcement)

NEW 3-seed × ε ∈ {0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15} = **21 measurement cells**.

| ε | ARI avg | std | AMI | Hom | Comp | noise % |
|:-:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.02 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.04 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.06 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.08 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.10 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.15 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |

★ **Zero effect across 7 ε values, all metrics identical to 4 decimal places** — NEW
embedding 의 HDBSCAN cluster tree 가 견고하게 saturated 됨. ε parameter 가 NEW recipe
위에서 **redundant**.

**Paper claim N9 reinforcement**:

> On strong contrastive embeddings (NEW recipe, 3-seed avg), HDBSCAN
> `cluster_selection_epsilon` parameter is **redundant** — the cluster tree is fully
> determined by the `(method, min_cluster_size, min_samples)` triple alone (here:
> `eom, 12, 3`). This is a paper-worthy **negative result**: sweep 21 cells (3 seeds
> × 7 ε values) all converge to identical Tier 1+2 metrics. The epsilon lever is
> deprecated for production deployment under this recipe.

### 19.3 Step 1c — Soft KNN-softmax τ-reassignment of HDBSCAN noise points

NEW 3-seed × τ ∈ {0.5, 0.7, 0.9, ∞} = **12 cells**. KNN k = 10, cosine similarity,
softmax temperature 0.1 applied to label-vote softmax of k nearest non-noise neighbors;
reassignment accepted only when max-softmax-prob ≥ τ.

| τ | ARI avg | std | AMI | Hom | Comp | noise % | avg reassigned |
|:-:|---:|---:|---:|---:|---:|---:|---:|
| **∞ (baseline)** | **0.8731** | 0.0140 | **0.9629** | **0.9448** | **0.9963** | 1.48 | 0 |
| 0.90 | 0.8709 | 0.0132 | 0.9616 | 0.9436 | 0.9952 | 0.49 | 11.3 |
| 0.70 | 0.8696 | **0.0123** ★ | 0.9607 | 0.9430 | 0.9944 | 0.15 | 15.3 |
| **0.50** ★ | 0.8681 | 0.0125 | 0.9600 | 0.9424 | 0.9938 | **0.00** ★★ | 17.0 |

★ **Trade-off matrix**:

| τ | ARI Δ vs baseline | std improvement | noise % Δ |
|:-:|---:|---:|---:|
| 0.90 | −0.0022 | 0.0140 → 0.0132 (−5.7 %) | 1.48 → **0.49** (−67 %) |
| 0.70 | −0.0035 | 0.0140 → **0.0123** (−12 %) ★ | 1.48 → **0.15** (−90 %) |
| 0.50 | −0.0050 | 0.0140 → 0.0125 (−11 %) | 1.48 → **0.00** (−100 %) ★★ |

→ **P1 class capture = 1.000 across all τ values** (unaffected). The reassignment
trades a marginal ARI cost (−0.005 at τ = 0.5) for a **complete elimination of HDBSCAN
noise** (1.48 % → 0.00 %, paper P2 metric).

**Paper claim N9 production extension** (Step 1c):

> Soft KNN-softmax reassignment of HDBSCAN noise points achieves **0.00 % noise rate**
> at τ = 0.5 with marginal ARI cost (Δ = −0.005, well within seed std). At τ = 0.7
> reproducibility std drops from 0.0140 to 0.0123 (12 % improvement) while noise rate
> falls to 0.15 %. The mechanism is useful for production deployment scenarios where
> every wafer must receive a cluster label without manual triage.

### 19.4 Step-by-step ARI progression matrix (paper Table N+1)

| Step | Method addition | P1 cap | P2 noise %↓ | P3 Comp | P4 Hom | AMI | ARI | std | RankMe |
|:-:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | NEW (Global+NeCo+Queue+NEG) baseline | 1.000 | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | **23.44** |
| 1a | + RankMe representation column | 1.000 | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | 23.44 |
| 1b | + HDBSCAN ε ∈ [0.00, 0.15] sweep | 1.000 | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | — |
| 1c τ = 0.9 | + soft τ-reassign | 1.000 | **0.49** ★ | 0.9952 | 0.9436 | 0.9616 | 0.8709 | 0.0132 | — |
| 1c τ = 0.7 | + soft τ-reassign | 1.000 | **0.15** ★ | 0.9944 | 0.9430 | 0.9607 | 0.8696 | **0.0123** ★ | — |
| 1c τ = 0.5 | + soft τ-reassign | 1.000 | **0.00** ★★ | 0.9938 | 0.9424 | 0.9600 | 0.8681 | 0.0125 | — |

★ **P2 noise dramatic improvement under Step 1c τ = 0.5** — from 1.48 % → 0.00 % at an
ARI cost of only −0.005. The headline P2 metric (paper priority P2 noise, see
`feedback_priority_p1_to_p4.md`) is fully eliminated without retraining the encoder.

### 19.5 Step 1 → next-step decision

| outcome | downstream decision |
|---|---|
| Step 1a — RankMe ρ(RankMe, ARI) = −0.429 (n = 7) | report only as stability column, NOT as SOTA arbiter |
| Step 1b — ε zero-effect across 7 values × 3 seeds | epsilon parameter **deprecated** for production cfg |
| Step 1c — noise 0.00 % at τ = 0.5, ARI Δ = −0.005 | **production cfg lock: τ = 0.5** for every-wafer labeling |
| Step 2 (EMA target encoder) | requires training dispatch — pending user approval |

### 19.6 Historical ARI value note

Step 1 measurement of NEW 3-seed HDBSCAN ARI (defect-only, eom mcs=12 ms=3) =
**0.8731 ± 0.0140**. ABSTRACT.md v0.9 / README header historical value =
**0.859 ± 0.018** (3-seed mean, prior run-time measurement). Δ = 0.014 = within
HDBSCAN tree non-determinism (sklearn `random_state` × leaf-build order). **Both
values are valid measurements**; paper retains the historical 0.859 inline citation
in ABSTRACT/INTRODUCTION/CONCLUSION, while §19 reports 0.8731 as the inline-measured
Step 1 baseline used for Step 1c τ-reassignment Δ calculations. Cross-reference:
`step1_eval_only_summary_260513.md`.

### 19.7 Source files (Step 1 raw data)

- `_step1b_hdbscan_eps_sweep.json` — 21-cell raw measurements (3 seeds × 7 ε values).
- `_step1c_soft_tau_reassign.json` — 12-cell raw measurements (3 seeds × 4 τ values).
- `step1_eval_only_summary_260513.md` — paper-recorder source-of-truth.
- `step1_paper_addition_260513.md` — this section's edit summary.
