# Experimental setup

## Hardware

- GPU: NVIDIA (16 GB VRAM)
- RAM: 64 GB
- OS: Windows 11
- CUDA: 12.x

## Software

- Python 3.13
- PyTorch 2.x
- timm (ConvNeXtV2 backbone)
- scikit-learn (clustering metrics)
- hdbscan (clustering)
- numpy, pandas, PIL

## Hyperparameter table (per run)

| Run | tag | EPOCHS | BATCH | IMAGE | TEMP | LR_HEAD | QUEUE_SIZE | USE_LOCAL | LOCAL_WEIGHT | IGNORE_NEG_SIM | n_train | sampling |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Iter 0 (historical) | normal1000_n50_b16_global_e10_resize_reuse | 10 | 16 | 384 | 0.07 | 1e-3 | 4096 | False | — | — | 8,357 | uniform 200/class + Normal 1000 |
| Iter A0 | 260506_093847 | 5 | 8 | 384 | 0.07 | 1e-3 | 4096 | True | 0.5 | 0.72 | 2,146 | avg30_260505_203615 (D-15 anchor) |
| Iter 1 ★ best | 260506_103302 | 5 | 8 | 384 | 0.07 | 1e-3 | 4096 | True | **1.0** | 0.72 | 2,146 | same as Iter A0 (anchor reuse) |
| Iter 2 REJECT | 260506_112604 | 5 | 8 | 384 | 0.07 | 1e-3 | 4096 | True | 1.0 | **0.65** | 2,146 | same as Iter A0 (anchor reuse) |
| Iter 3 NULL | 260506_120656 | 5 | 8 | 384 | 0.07 | 1e-3 | 4096 | True | **0.7** | 0.72 | 2,146 | same as Iter A0 (anchor reuse) |

★ Atomic 변경 (each from Iter 1 base, except A0 which is from itself):
- A0 → 1: `LOCAL_WEIGHT 0.5 → 1.0`
- 1 → 2: `IGNORE_NEG_SIM 0.72 → 0.65` (REJECT — dead branch)
- 1 → 3: `LOCAL_WEIGHT 1.0 → 0.7` (NULL — A0 와 통계적 동일, LW 0.5~0.7 plateau)

Per ITERATION 의 detail change history → `ITERATIONS.md`.

## Data anchor spec (D-15)

`avg30_260505_203615` (★ method-track Iter A0+ 의 fixed anchor)
- defect 42 class — per-class avg 30, random 분포 (uniform 아님)
- `Normal_bank_boundary` — source 전체 1,000
- 합계 **2,146 wafer**
- file_list.parquet (subset hardlink builder 자동 저장) → 재현 보장
- 모든 method ablation 이 같은 anchor 디렉터리 reuse — data 측 변경 금지 (D-13/D-15)

Iter 0 (historical) 의 8,357 wafer / 38 defect class anchor 와 다름. 두 track 의
metric 직접 비교 의미 X — anchor 별 baseline 따로 두고 method 효과만 atomic 비교.

## GPU dispatch operational notes (Iter A0 운영 issue)

Iter A0 dispatch 7 attempt 끝에 success. 운영 issue:

| attempt | 결과 | 원인 / 대처 |
|---|---|---|
| 1-3 | fail | GPU 사용률 93% — 다른 process 점유 또는 단발 spike |
| 4-5 | fail | GPU throttle 도입 시도 (interval 부족, 여전히 driver 부담) |
| 6 | fail | driver TDR (timeout detection & recovery) crash — windows display driver reset |
| 7 | success | **BATCH=8** (16 → 8) + **EPOCHS=5** (10 → 5) + **throttle 5000ms** (per-step pause) 도입 |

★ **Iter A0+ 의 method-track lock-in 운영 조건** (data anchor 와 별개로 dispatch 안정성 spec):
- `BATCH=8` (D-9 의 BATCH=16 보다 더 작게, 사용자 GPU 보호)
- `EPOCHS=5`
- GPU per-step throttle **5000ms** (driver TDR 회피)
- 같은 조건이 Iter 1 에도 그대로 적용 (atomic 변경 LOCAL_WEIGHT 만).

이 조건 변경은 method ablation 비교 대상 아님 — 모든 Iter 가 동일 dispatch 조건이면
method 효과 비교 가능. 향후 dispatch 조건 변경 시 새 baseline 재설정 필요.

## Resource policy

- **GPU**: BATCH=16, IMAGE_SIZE=384 — 사용자 명시 "GPU 작게 써라" (`docs/contrastive-eval/DECISIONS.md` D-9).
- **monitoring (별도)**: alignment + uniformity (label 무관) + (옵션) k-NN top-1.
- **rogue process 보호**: canvas gen / obj_id_maps 자동 spawn 감지 + kill watchdog.

## 코드 위치 (root scripts)

| 파일 | 역할 |
|---|---|
| `contrastive.py` | InfoNCE 학습 engine (수정 금지, wrapper CFG override) |
| `run_contrastive.py` | env wrapper, sister repo backbone state_dict 추출 |
| `_contrastive_n50.py` | small-budget 학습 wrapper (subset hardlink builder) |
| `_eval_contrastive_unknown_n50.py` | 학습 후 평가 — Tier 1+2 + class_fragmentation_summary |
| `eval_align_uniform.py` | post-hoc alignment + uniformity helper |
| `compose_clusters.py` | per-cluster K=20 medoid composite (binary + grademean) |
| `predict_contrastive_daily.py` | production daily inference |

## Output 구조

```
outputs/logs_contrastive/<tag>_<TS>/
├── _init_backbone.pth        # backbone state_dict 사본
├── _wrapper_manifest.json    # wrapper 환경
├── run.log                   # 학습 log (epoch loss / metric)
├── run_info.json             # CFG dump
├── checkpoints/
│   ├── final_infer.pt        # final encoder + head
│   └── last_training.pt      # last training state
├── eval/
│   ├── eval_summary.json     # Tier 1+2 + class_fragmentation_summary
│   ├── align_uniform.json    # Wang & Isola metric (post-hoc)
│   ├── embeddings/           # embedding.npy + files.txt + classes.txt
│   ├── cluster_report.parquet
│   ├── class_fragmentation.parquet
│   ├── retrieval_report.parquet
│   └── plots/                # heatmap / histogram / silhouette
├── clusters/hdbscan/cluster_XXX_size_YYY/  # cluster member 이미지
└── cluster_summary/                          # medoid + composite
```

## Reproducibility

- random seed: 42 (CFG default).
- Augmentation: deterministic with seed.
- HDBSCAN: deterministic (no randomness).

## 검증 protocol

각 run 평가 시 자동:
1. `_eval_contrastive_unknown_n50.py` 실행 — Tier 1+2 + class_fragmentation_summary
2. `eval_align_uniform.py --run <run_dir>` — alignment + uniformity
3. (옵션) `compose_clusters.py --run <run_dir>` — composite map
4. 콘솔 1-2 줄 보고 자동 출력

## ★ 추가 metric 정책 (Iter 1 lesson, 2026-05-06)

Iter 2 reject + Iter 3 null 분석 과정에서 Tier 1+2 외 두 가지 cluster 구조 metric 의
diagnostic 가치 확인. 이후 모든 iteration 평가에 추가 보고:

### M1. per-cluster purity (mega-cluster 진단)
- 정의: cluster 별 dominant class fraction. `purity < 0.5 ∧ size > 50` 인 cluster 를 **mega-cluster** 로 분류 — 여러 class 가 한 cluster 에 합쳐진 collapse.
- 활용: Iter 1 (mega=2), Iter 2 REJECT (mega=3, +1 collapse), Iter 3 NULL (mega=2 = A0).
  Tier 1 metric 변화가 모호할 때 mega-cluster count 가 선명한 collapse signal.
- Iter 2 의 c034 (Edge-Bottom), c009 (Center), c023 (Edge-Top) 추가 collapse 가
  IGNORE_NEG_SIM 강화 reject 의 결정적 evidence.

### M2. sister-pair centroid distance (atomic 변경 효과 검증)
- 정의: same class 내 split-2 cluster 쌍의 centroid 간 cosine distance. atomic
  변경 후 sister-pair centroid 가 모두 baseline 과 소수점 셋째 자리까지 동일하면
  변경의 cluster 구조 영향이 noise level 이라고 판정.
- 활용: Iter 3 (LW=0.7) sister-pair centroid 모두 A0 (LW=0.5) 와 셋째 자리 동일 →
  LW 0.5~0.7 plateau 확정 (Tier 1 metric 만으로는 noise 인지 plateau 인지 모호).
- Tier 1 의 random fluctuation < 0.01 영역에서 **null effect** vs **micro-improvement**
  구분에 결정적.

두 metric 모두 표준 sklearn / numpy 로 계산 가능, 커스텀 metric 정책 (Tier 1+2 only)
유지 — diagnostic 부록으로 보고.

---

## ★ New anchor track (iter 34+) — 추가 hyperparameter rows

> Data anchor 변경: `avg30_260505_203615` → `avg30_new_260508_123037` (43 class, 2,146 wafer,
> v19o chip 합성 + canvas 9). Iter 34+ row 분리 — old anchor row 와 직접 비교 의미 X.
> 새 lever 추가: `NECO_WEIGHT` (NeCo patch-neighbor consistency, arXiv:2408.11054).

| Run | tag (run_dir) | EPOCHS | BATCH | TEMP | LR_HEAD | NEG_SIM | LW | NECO_WEIGHT | UNFREEZE / LR_SCALE | HDBSCAN cfg |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| iter 34 | 260508_123101 | 5 | 8 | 0.05 | 5e-4 | 0.65 | 1.0 | 0 | — / — | eom mcs=12 ms=3 |
| **iter 35** | 260508_162812 | 5 | 8 | **0.07** | **1e-3** | **0.72** | 1.0 | 0 | — / — | eom mcs=12 ms=3 |
| iter 36 ✗ | 260509_062741 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0 | **last_N=1 / 0.02** | eom mcs=12 ms=4 |
| **iter 37 ★** | 260509_072137 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | **0.2** | — / — | eom mcs=12 ms=3 |
| iter 38 ✗ | 260509_085046 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | **0.1** | — / — | eom mcs=12 ms=3 |
| iter 39 ✗ | 260509_125153 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | **0.3** | — / — | eom mcs=12 ms=3 |
| iter 40 ✗ | 260509_151225 | 5 | 8 | **0.05** | **5e-4** | **0.65** | 1.0 | 0.2 | — / — | eom mcs=12 ms=3 |
| iter 41 ✗ | (iter 37 reuse) | — | — | — | — | — | — | — | encoder X | HDBSCAN forcing (P1 violation) |
| iter 42 ✗ | 260509_172703 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | **last_N=1 / 0.005** | eom mcs=12 ms=3 |
| iter 43 (z=3, single) | 260509_TBD | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | — / — + NECO_ZONE_VERTICAL=3 | eom mcs=12 ms=3 |
| iter 44-46 (3-seed mean) | 260510_seed_x | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | — / — | eom mcs=12 ms=3 |
| **iter 50** | 260510_002649 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | + NECO_HIER_POOLS="1,2,4" | eom mcs=12 ms=3 |
| iter 51 | 260510_011836 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | + Hier; **seed=1** | eom mcs=12 ms=3 |
| iter 52 | 260510_020431 | 5 | 8 | 0.07 | 1e-3 | 0.72 | **1.2** | 0.2 | — | eom mcs=12 ms=3 |
| iter 53 | 260510_025007 | 5 | 8 | 0.07 | **7e-4** | 0.72 | 1.0 | 0.2 | — | eom mcs=12 ms=3 |
| iter 54 | 260510_035823 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | LOCAL_POS_TOPK=**16** | eom mcs=12 ms=3 |
| iter 55 | 260510_072458 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | TOPK=16; **seed=1** | eom mcs=12 ms=3 |
| iter 56 | 260510_082121 | 5 | 8 | 0.07 | 1e-3 | 0.72 | 1.0 | 0.2 | QUEUE_SIZE=**8192** | eom mcs=12 ms=3 |
| iter 57 | 260510_102747 | 5 | 8 | **0.06** | 1e-3 | 0.72 | 1.0 | 0.2 | — | eom mcs=12 ms=3 |
| iter 58 | 260510_111451 | 5 | 8 | 0.07 | 1e-3 | **0.65** | 1.0 | 0.2 | — | eom mcs=12 ms=3 |

**iter 50-58 atomic 변경 path (saturation sweep)**:
- 50 = iter 37 + NECO_HIER_POOLS="1,2,4" (Hierarchical NeCo, code-level)
- 51 = iter 50 + seed=1 (Hierarchical variance test)
- 52 = iter 37 + LW 1.0 → 1.2 (LW saturate probe)
- 53 = iter 37 + LR 1e-3 → 7e-4 (LR saturate probe)
- 54 = iter 37 + LOCAL_POS_TOPK 12 → 16 (TOPK saturate probe; single-seed lucky 0.880)
- 55 = iter 54 + seed=1 (TOPK 16 variance test; ★ Zone z=4 lucky pattern 정확 재현)
- 56 = iter 37 + QUEUE 4096 → 8192 (QUEUE saturate probe)
- 57 = iter 37 + TEMP 0.07 → 0.06 (TEMP saturate probe)
- 58 = iter 37 + NEG 0.72 → 0.65 (NEG saturate probe; NeCo 추가 후 lever 죽음)

**판정 종합**: 모든 9 iter (50-58) 의 atomic 변경이 iter 37 multi-seed std (0.014 ARI)
안으로 tied. **iter 37 cfg = 6-axis multi-axis saturation point** 확정. 이 결과가
**N5 contribution** (paper finalize 의 마지막 contribution) 의 evidence.

**Atomic 변경 path (new anchor track)**:
- iter 34 = anchor switch + Iter 14 cfg
- 34 → 35: 3-axis switch (LR + NEG + TEMP) → Iter 1 P2 cfg (★ new anchor base)
- 35 → 36: + UNFREEZE last_N=1, LR_SCALE=0.02 (✗ P1 violation)
- 35 → 37: + NECO_WEIGHT 0 → 0.2 (★★★★★ SOTA, lever 5)
- 37 → 38: NeCo 0.2 → 0.1 (✗ under-signal)
- 37 → 39: NeCo 0.2 → 0.3 (✗ over-signal, sweet spot lock)
- 34 → 40: Quality King + NeCo 0.2 (✗ cross-cfg incompat)
- 37 → 41: HDBSCAN forcing on iter 37 embedding (✗ P1)
- 37 → 42: + UNFREEZE 0.005 (✗ axis 영구 reject)
- 37 → 43: + NECO_ZONE_VERTICAL=3 (in progress, novelty A)

**dispatch / GPU note**: 새 anchor 도 BATCH=8 + EPOCHS=5 + per-step throttle 5000ms 동일.
Iter 41 의 HDBSCAN forcing 은 encoder 학습 X — iter 37 embedding 위 cluster cfg 만 변경.

**Code-level lever 추가 (Iter 30 deferred → 활성화)**:
- `NECO_WEIGHT` (env-override 기반 dispatch flag, contrastive.py 의 NeCo loss hook)
- `BACKBONE_UNFREEZE_LAST_N` + `LR_SCALE` (env-override, partial unfreeze) — **axis 영구 reject** (iter 36, 42)
- `NECO_ZONE_VERTICAL` (env-override, zone-aware NeCo variant — iter 43 novelty A)
