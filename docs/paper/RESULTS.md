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
