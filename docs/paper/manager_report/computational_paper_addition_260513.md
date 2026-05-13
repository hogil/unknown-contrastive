# Computational Performance + Dataset Statistics — Paper Addition (260513)

## Purpose

paper section 들에 **computational performance + dataset statistics** 표/section
누락 → reviewer/manager 입장에서 "이거 GPU 몇 대로 얼마나 걸려요?" 물음에 paper
self-contained 로 답할 수 있도록 보강. user directive (260513) "여기 있는 건 모두
claude code 로 직접 실험한 것들이다 그래서 사실들이다" — 모든 수치 실측치.

Source-of-truth: `docs/paper/manager_report/performance_data_260513.md` (§3-5
training time + eval pass timing, hardware spec, RankMe table).

## What changed

3 paper section 에 computational + dataset 정보 추가. paper headline metric
(Tier 1 ARI / AMI / Completeness / noise / capture) 전혀 건드리지 않음. 새 section
만 append, 기존 표 row 수정 없음.

### A. METHOD.md — §4b Computational Requirements (NEW)

기존 §3.7 Clustering algorithm selection 직후, §5 Inference 직전에 삽입
(line 416 → 488, 약 70 line 추가). 6 sub-section:

- §4b.1 Hardware (RTX 4060 Ti, 16 GB VRAM, ConvNeXtV2-base 87.7 M params frozen)
- §4b.2 Training hyperparameters (NEW cfg, iter 70+; BATCH=8 default, BATCH=4 iter 84)
- §4b.3 Training time (5-epoch, NEW 23.7 min vs B5 28-49 min → ~30% faster)
- §4b.4 Evaluation pipeline time (embedding 3 min + HDBSCAN+metric 4-7 min = 7-10 min)
- §4b.5 Inference latency (BATCH=1 14.3 ms = 70 wafers/sec)
- §4b.6 HDBSCAN clustering time (full re-cluster 507 ms / approx_predict ~10 ms)
- §4b.7 End-to-end production latency (24 ms/wafer = real-time deployable)

### B. RESULTS.md — §18 Computational Performance + Dataset statistics (NEW)

§17 끝 (line 917) 다음에 §18 append. 8 sub-section:

- §18.1 Dataset statistics (2,146 samples, 43 class, 384×384, class size 15-1000)
- §18.2 Hardware + backbone
- §18.3 Training time table (NEW 23.7 ± 0.01 vs B5 28-49 min)
- §18.4 Inference latency + throughput table (BATCH=1/8/32)
- §18.5 HDBSCAN clustering time table
- §18.6 Evaluation pipeline time table
- §18.7 ★ End-to-end production latency 24 ms/wafer = ~40 wafers/sec claim
- §18.8 Cost-of-experiment summary (84-iter cycle ≈ 50 GPU-hours; single NEW reproduce ≈ 100 min)

### C. ABSTRACT.md — v0.9 (CURRENT) closing paragraph

v0.9 final block 에 1-paragraph 추가:

> Production inference is 14 ms / wafer on a single NVIDIA RTX 4060 Ti (BATCH = 1,
> ConvNeXtV2-base 87.7 M params frozen), with HDBSCAN `approximate_predict` adding
> ~10 ms — total ≈ 24 ms / wafer (≈ 40 wafers/sec, real-time deployable). NEW
> training is 23.7 min for 5 epochs on n=2,146, ~30% faster than the B5 recipe
> (28–49 min) due to Local DenseCL removal.

footer reference list 에 `RESULTS.md §18 (NEW)` 추가.

## Numbers (실측 source 표시)

### Dataset
- n_total=2146 (defect 1146 + Normal 1000) — anchor `avg30_new_260508_123037`
- 43 class = 42 defect (Donut×5 + EB×5 + ET×5 + Center×5 + Full×5 + ER×4 +
  Thick-Edge_invalid_main + 9 wafer-canvas) + Normal_bank_boundary
- class size range 15 (Thick-Edge_invalid_main) – 1000 (Normal)
- image size 384 × 384, source pool 9,250 PNG @ 6400 × 6400

### Hardware
- RTX 4060 Ti (16 GB VRAM)
- ConvNeXtV2-base 87.7 M params (frozen during contrastive)
- Projection head 2-layer MLP 128-D L2-normalized

### Training time (5-epoch, n=2146, RTX 4060 Ti)
- NEW (Queue+NEG+NeCo, no Local): **23.7 min/run ± 0.01** (3-seed)
- B5 (Local+Queue+NEG+NeCo): **28–49 min/run** (3-seed, wide variance)
- NEW ~30% faster than B5 due to Local DenseCL forward+backward removal

### Inference latency (RTX 4060 Ti)
- BATCH=1: 14.3 ms/image = 70 wafers/sec
- BATCH=8: 17.2 ms/image = 58 wafers/sec
- BATCH=32: 18.5 ms/image (amortized) = 54 wafers/sec

### HDBSCAN
- Full re-cluster (1146 defects): 507 ms (model-update)
- Single-point approx_predict: ~10 ms (online)

### End-to-end production (per wafer)
```
forward pass     14 ms (BATCH=1)
HDBSCAN predict  10 ms
─────────────────────
total           ~24 ms / wafer  →  ~40 wafers/sec
```
→ **real-time deployable** on single RTX 4060 Ti

### Eval pipeline (n=2146)
- embedding extraction: ~3 min
- HDBSCAN fit + Tier 1+2 metric + per-class report: 4–7 min
- total eval pass: **7–10 min**

### Cost of full ablation cycle
- 84 atomic iterations + 6 B0-B5 + 11 lattice iter + 5-method benchmark = ~90 encoders
- 3-seed avg ≈ 50 GPU-hours wall-time
- Single NEW recipe reproduce (3 seeds, headline paper number): ~100 min total

## Policy compliance check

- ★ Tier 1+2 official metric only — 영향 X (이건 computational data 추가, ARI/Comp/AMI/noise/capture 표 그대로)
- ★ ITERATIONS.md append-only — 수정 X (이번 patch 는 새 §18 만 추가)
- ★ ITERATIONS 과거 entry 수정 X — 준수
- ★ Custom/classifier metric (precision/recall/F1/FPR) — 사용 X
- ★ Tier 3 metric (NMI/V-measure/DB/CH) headline X — 영향 X
- ★ 무단 commit X — paper 만 변경 (사용자가 commit 명시 시 진행)

## Files changed (working tree)

- `docs/paper/METHOD.md` (+70 line, §4b NEW)
- `docs/paper/RESULTS.md` (+90 line, §18 NEW)
- `docs/paper/ABSTRACT.md` (+7 line, v0.9 closing block)
- `docs/paper/manager_report/computational_paper_addition_260513.md` (this file, NEW)

## Verification

reviewer 가 paper section 만 읽어도:
1. "GPU 몇 대 필요?" → RTX 4060 Ti 단일 (METHOD §4b.1, RESULTS §18.2)
2. "데이터 얼마나?" → 2,146 wafer (RESULTS §18.1)
3. "학습 얼마나 걸려?" → 23.7 min/run × 3 seed = ~71 min (METHOD §4b.3, RESULTS §18.3)
4. "production 배포 가능?" → 24 ms/wafer (~40 wafers/sec) real-time deployable
   (METHOD §4b.7, RESULTS §18.7, ABSTRACT v0.9 closing)
5. "이 ablation 재현하려면?" → ~50 GPU-hours full cycle / ~100 min single recipe
   (RESULTS §18.8)

all numbers traceable to performance_data_260513.md (single source-of-truth).

## Why now (motivation)

- Reviewer 1차 점검 항목: hardware footprint + inference budget
- Manager 1차 점검 항목: cost-of-reproduce + production deployment feasibility
- 기존 paper section 들은 metric 분해에 집중 (ARI 0.859 ± 0.018 등),
  computational footprint 가 누락되어 있어 "쓸 만한가?" 판단에 paper-external
  log 가 필요했음 → 이번 patch 로 paper self-contained.

## Next

- Optional: FIGURES.md 에 throughput 비교 plot 추가 (현재 표만 — bar plot
  added 가능하지만 figure 산출 따로 필요. 본 patch range 밖)
- Optional: EXPERIMENTS.md hyperparameter table 의 row 에 training time / mem
  컬럼 추가 (현재 hparam-focused; computational 은 RESULTS §18 으로 합쳐도 OK)

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/computational_paper_addition_260513.md
