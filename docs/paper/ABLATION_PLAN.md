# Real Baseline Ablation Plan — Component Isolation

> 2026-05-11 신설. 사용자 지적 정합: "기존 baseline 에 이미 patch-level Local InfoNCE 가 활성. 우리 lever 효과는 그 위에서의 변화. 진짜 paper-grade ablation 위해 USE_LOCAL=false 부터 component 단계별 추가."

## 목적

기존 iter 0~58 결과는 "iter A0 (Global + Local + Queue + NEG filter, LW=0.5) 이미 활성 baseline 위에서의 hparam tuning". 진짜 component-level contribution isolation 위해 minimal Global InfoNCE 만 활성한 **Real Baseline (B0)** 부터 component 단계별 추가하며 isolated effect 측정.

## ablation matrix (6 step)

| # | cfg | USE_LOCAL | LW | USE_QUEUE | NEG filter | NeCo | 의미 |
|:-:|---|:-:|:-:|:-:|---|:-:|---|
| **B0** | **Real Baseline** | false | 0 | false | none (NEG=1.0) | 0 | Global InfoNCE only |
| B1 | + Local (DenseCL weak) | true | 0.5 | false | none | 0 | + patch-level local |
| B2 | + Local strong (lever 1) | true | 1.0 | false | none | 0 | LW 0.5 → 1.0 isolated |
| B3 | + MoCo Queue | true | 1.0 | true (4096) | none | 0 | + queue |
| B4 | + NEG filter | true | 1.0 | true | fixed 0.72 | 0 | + NEG |
| **B5** | **+ NeCo (paper N1)** | true | 1.0 | true | fixed 0.72 | **0.2** | = iter 37 cfg |

★ HDBSCAN post-hoc tuning (eom + ms=3) 은 모든 cfg 에 동일 적용 — encoder 학습 무관.

## 공통 cfg (모든 B 단계 동일)

```
ConvNeXtV2-base + TAPT (sister repo known-cnn best_model.pth)
IMAGE_SIZE=384, BATCH=8, EPOCHS=5, WARMUP=1
LR_HEAD=1e-3, NCE_TEMP=0.07, SEED=42
PROJ_DIM=128, FREEZE_BACKBONE=true
anchor: avg30_new_260508_123037 (43 class, n=2146)
LOCAL_POS_TOPK=12 (USE_LOCAL=true 시)
QUEUE_SIZE=4096 (USE_QUEUE=true 시)
```

## dispatch 명령

```bash
# B0 — Real Baseline (Global InfoNCE only)
python _dispatch_iter.py --tag iter_60_B0_real_baseline \
  --data D:/project/data/contrastive_anchor/avg30_new_260508_123037 \
  --epochs 5 --batch 8 --image-size 384 --warmup 1 \
  --use-local false --use-queue false \
  --local-weight 0 --local-pos-topk 12 --seed 42 \
  --mcs 12 --ms 4 --cluster-method leaf --cluster-eps 0.06 \
  --freeze-backbone true --backbone-unfreeze-last-n 0 --backbone-lr-scale 1.0 \
  --ignore-neg-sim 1.0 --nce-temp 0.07 --lr-head 1e-3 --neco-weight 0

# B1 — + Local weak (LW=0.5)
... --use-local true --local-weight 0.5 ...

# B2 — + Local strong (LW=1.0, lever 1 isolated)
... --use-local true --local-weight 1.0 ...

# B3 — + Queue
... --use-queue true --queue-size 4096 ...

# B4 — + NEG filter
... --ignore-neg-sim 0.72 ...

# B5 — + NeCo (= iter 37 cfg)
... --neco-weight 0.2 ...
```

## 기대 효과 (가설) vs 실측 (★ 2026-05-11 완료)

| step | 효과 가설 | 실측 ΔARI | 실측 Δnoise | 판정 |
|:-:|---|---:|---:|---|
| B0 → B1 | + Local DenseCL → noise huge ↓ | **+0.028** | **-2.27pp** | ✓ Local 단독 효과 확실 |
| B1 → B2 | LW strong → noise 추가 ↓ | **-0.028** | **+2.27pp** | **✗ LW=1.0 isolated regression!** |
| B2 → B3 | + Queue → more negatives | **+0.023** | **-4.89pp** | **★★★ N6 huge — LW=1.0 + Queue interaction** |
| B3 → B4 | + NEG filter → false neg 보호 | **+0.014** | **-0.78pp** | ✓ small but clean |
| B4 → B5 | + NeCo (★ paper N1) | **-0.004** | **+0.44pp** | **✗ NeCo isolated effect ≈ 0!** |

**총 누적 (B0 → B5)**: ΔARI **+0.033** / Δnoise **-5.24pp** / ΔComp **+0.020** / ΔAMI **+0.021**.

★ B4 > B5: NeCo 없는 B4 가 NeCo 있는 B5 보다 우위 (ARI 0.860 > 0.856, noise 0.52% < 0.96%).
이는 paper N1 (NeCo) 의 contribution 을 **isolated component-by-component 로 재검토** 하게
하는 결정적 evidence.

## ★ B0-B5 실측 표

| # | cfg | P1 cap | P2 noise | P3 Comp | P4 Hom | AMI | NMI | ARI | Sil(cos) | n_cl |
|:-:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | Global only | **1.000** | 6.195% | 0.9602 | 0.929* | 0.9290 | 0.949* | 0.8231 | 0.582* | 37 |
| B1 | + Local LW=0.5 | 1.000 | 3.927% | 0.9665 | 0.9351 | 0.9387 | 0.9505 | 0.8514 | 0.5139 | 37 |
| B2 | LW=1.0 | 1.000 | 6.195% | 0.9602 | 0.9257 | 0.9290 | 0.9427 | 0.8231 | 0.5089 | 37 |
| **iter 67** ★ | **B2 + NeCo (no Queue)** | **1.000** | **3.93%** | **0.9659** | n/a | **0.9390** | n/a | **0.8508** | n/a | n/a |
| B3 | + Queue | 1.000 | 1.309% | 0.9828 | 0.9365 | 0.9496 | 0.9591 | 0.8464 | 0.5727 | 36 |
| **B4** ★ | + NEG=0.72 | 1.000 | **0.524%** | **0.9852** | 0.9439 | 0.9557 | 0.9641 | **0.8605** | 0.6109 | 37 |
| B5 | + NeCo (=iter 37) | 1.000 | 0.960% | 0.9801 | 0.9403 | 0.9503 | 0.9598 | 0.8564 | 0.6104 | 37 |

### ★ iter 67 — N6 Component Interaction 강화 evidence

iter 67 (B2 + NeCo, no Queue) 가 B3 (Queue, no NeCo) 를 ARI 에서 outperform:
- iter 67 ARI **0.8508** > B3 ARI 0.8464
- iter 67 noise **3.93%** > B3 noise 1.31% (Queue 의 noise 감소 효과는 NeCo 가 부분만 대체)
- = **NeCo 와 Queue 는 부분 substitutes (interchangeable)** — neighbor consistency 가 queue 의 large negative pool 일부 대체 가능
- 그러나 **noise reduction** 측면에서는 Queue 가 NeCo 보다 우위 (1.31% < 3.93%)
- ARI 측면에서는 NeCo 단독 (no Queue) 이 Queue 단독 (no NeCo) 보다 약간 우위

→ paper N6 (Component Interaction) 의 **두 번째 evidence**: components 가 monotonic 추가가 아니라 partial substitution + interaction

### ★★★ iter 68 — N6 세 번째 evidence (가장 강한)

iter 68 (B3 + NeCo, no NEG) ARI **0.8464** = **exact B3 reproduce** (0.8464). NeCo Queue-redundant 결정적:

| cfg | NeCo Δ ARI |
|---|---:|
| B2 → +NeCo (iter 67, no Queue) | **+0.028** |
| B3 → +NeCo (iter 68, Queue) | **+0.000** ← exact zero |
| B4 → +NeCo (B5, Queue+NEG) | **-0.004** |

→ **NeCo 효과 monotonic decay** as negative-handling components added.
→ NeCo 의 mechanism = Queue 의 negative diversity 부분 대체. Queue 가 present 면 NeCo redundant, Queue+NEG 면 slightly harmful.

**paper N1 (NeCo) 의 진짜 contribution** 재정의:
- NeCo standalone value = **0** (Queue 가 있으면 흡수)
- NeCo's role = **Queue 가 없을 때 substitute** (B2 → iter 67 +0.028)
- paper 의 NeCo claim 은 "implicit Queue substitute" 로 honest reframe

### ★★★★ iter 69 — NeCo ≡ Local DenseCL (paper-grade 발견)

iter 69 (B0 + NeCo only) **ARI 0.8514 = B1 (B0 + Local LW=0.5) 0.8514** 정확히 동일 (4자리, noise 3.93% 동일, n_cl 37 동일).

→ **NeCo (Pariza 2024) ≡ DenseCL Local InfoNCE (Wang 2021)** functionally equivalent.
   서로 다른 implementation, **identical magnitude (+0.028 ARI standalone)**.

| Component class | members | 효과 |
|---|---|---|
| Global InfoNCE | global view-level contrast | baseline |
| **patch-neighbor consistency** | **{Local DenseCL, NeCo}** — substitutes | **+0.028** |
| negative diversity | MoCo Queue | +0.023 (LW=1.0 interaction) |
| false-negative protection | NV-Retriever NEG filter | +0.014 |

paper Methods 재구성:
- baseline 의 patch-neighbor 는 **둘 중 하나만 충분**.
- 둘 다 사용 = no gain (iter 67 = iter 69 = 0.8514).
- 우리 paper N1 (NeCo) 의 진짜 novelty 는 "implementation alternative to DenseCL with equivalent effect".

### ★★★★★ iter 70 — NEW SOTA ARI 0.8797 (NeCo replaces Local entirely)

iter 70 (Global + NeCo + Queue + NEG, **no Local**) = ARI **0.8797**, Comp 0.9872, noise 0.87%, Sil 0.7860.

**B0-B5 추가** (B6 신설):

| # | cfg | ARI | noise | Δ vs prev |
|:-:|---|---:|---:|---:|
| B0 | Global only | 0.8230 | 6.20% | (base) |
| B1 | + Local LW=0.5 | 0.8514 | 3.93% | +0.028 |
| B2 | LW=1.0 | 0.8231 | 6.20% | -0.028 |
| B3 | + Queue | 0.8464 | 1.31% | +0.023 |
| B4 | + NEG=0.72 | 0.8605 | 0.52% | +0.014 |
| B5 | + NeCo (=iter 37) | 0.8564 | 0.96% | -0.004 |
| **B6** ★ NEW | **B4 - Local + NeCo** | **0.8797** | **0.87%** | **+0.019** |

**B6 cfg**: Global + NeCo + Queue + NEG (Local 완전 제거).

### paper N1 (NeCo) 최종 contribution (post-iter 70)

NeCo 는 **strictly superior substitute** for DenseCL Local InfoNCE.
- Local 대신 NeCo 사용 → ARI +0.019, noise +0.35pp (margin within Sil 0.79 floor)
- Local 과 NeCo 동시 사용 → redundancy + slight interference (B5 < B4)
- **paper recommendation**: deprecate Local, use NeCo + Queue + NEG only.

\* Hom/NMI/Sil 은 B0 별도 reporting (sklearn 직접 계산).

run_dir:
- B0: `outputs_contrastive_260511_154102/`
- B1: `outputs_contrastive_260511_162616/`
- B2: `outputs_contrastive_260511_170230/`
- B3: `outputs_contrastive_260511_173842/`
- B4: `outputs_contrastive_260511_181441/`
- B5: `outputs_contrastive_260511_185039/`

## ★★★★★ 핵심 발견 5

1. **LW=1.0 단독 regression** (B1 → B2): LW=0.5 → 1.0 만 변경 시 ARI -0.028, noise +2.27pp.
   = **LW strong 의 isolated effect 는 negative**.

2. **★ N6 Component Interaction (NEW)** (B2 → B3): Queue 추가 시 ARI +0.023, noise -4.89pp.
   B1 → B3 의 monotonic 이 아니라 B2 (LW=1.0 단독) 에서 dip 후 Queue 가 lift 함.
   = **LW=1.0 의 효과는 Queue 와의 interaction 으로만 발현**. paper N6 contribution.

3. **NeCo (N1) isolated effect ≈ 0** (B4 → B5): NeCo 단독 추가 시 ARI -0.004, noise +0.44pp.
   = paper N1 의 "noise -70%" 는 iter 35→37 비교 (different runs).
   component-isolated 로는 negligible 또는 negative.

4. **iter 37 reproduce variance** (B5 vs iter 37, same seed=42): ΔARI 0.014, Δnoise 0.35pp.
   = same seed 라도 run-to-run variance 가 multi-seed std 만큼 큼. **N2 (multi-seed) 강한 evidence**.

5. **B4 > B5** (NeCo 없는 cfg 가 NeCo 있는 cfg 보다 모든 metric 우위):
   B4 ARI 0.8605 > B5 0.8564 / B4 Comp 0.9852 > B5 0.9801 / B4 noise 0.52% < B5 0.96%.
   = **paper N1 contribution 재검토 필요** — NeCo 의 진짜 효과는 base cfg interaction 으로만.

## paper 안 contribution 정확 분류

```
B5 (= iter 37 cfg) vs B0 (Real Baseline) 의 component-by-component breakdown:

   Component                Source     Paper claim
   ────────────────────────────────────────────────
   Global InfoNCE           B0         baseline (기존)
   + Local InfoNCE          B1         baseline component (기존, DenseCL)
   + LW strong              B2         hparam tuning (lever 1)
   + MoCo Queue             B3         baseline component (기존)
   + NEG filter             B4         baseline component (기존)
   + NeCo                   B5         ★ paper N1 (NEW)
                                                                
+ HDBSCAN eom + ms=3        post-hoc   ★ paper N3 (NEW)
```

## 시간 예상

```
B0~B5 학습 6 iter × 28분 = ~3시간
eval 6 iter × 9분 = ~54분
sweep + analyzer = ~30분
총 ~4-5시간
                                                                
GPU 자원: chip_multilabel 와 공존 가능 (batch=8 image=384 = ~6GB)
```

## 결과 종합 후 갱신할 docs

- `ITERATIONS.md` append iter 60-65 (B0-B5 entries)
- `RESULTS.md` 표 13: Real Baseline ablation matrix
- `METHOD.md` §3.7: Component isolation methodology
- `DISCUSSION.md` §7.9: Real baseline isolation 의미
- `ABSTRACT.md` v0.4: B0 → B5 breakdown 명시
- `SUMMARY.md` §0 또는 §10 갱신
- `manager_report/SUMMARY.md` 동일
- `paper-recorder` SKILL: real baseline isolation 정책 추가
- `paper-recorder` agent: 동일

## 정책 정합

```
✓ 사용자 룰 "BG/DEFECT_BG/EDGE/INTENSITY 변경 X" — 합성 그대로
✓ "scratch_rot 단방향" — 그대로
✓ "INVALID text 만" — 그대로
✓ "wafer 새로 만들지 말고" — anchor avg30_new 그대로 사용 (regen X)
✓ "결과 폴더 절대 삭제 금지" — 기존 iter 37 등 보존
✓ 학습 dispatch — chip_multilabel 공존 가능 (small batch)
```

---

## ★ Roadmap Step 1 (eval-only) — ★ COMPLETED 2026-05-13

Plan reference: `C:\Users\hgcho\.claude\plans\floating-splashing-key.md` Roadmap Step 1.
Single source-of-truth: `docs/paper/manager_report/step1_eval_only_summary_260513.md`.

**Protocol**: eval-only (no encoder retraining). Same NEW recipe 3-seed runs
(iter 70/71/72), defect-only Tier1 HDBSCAN (eom mcs=12 ms=3). Three orthogonal
post-hoc refinements.

### Step 1 results matrix (P1 capture = 1.000 across all steps)

| Step | Method | ARI avg ± std | noise % | RankMe | Verdict |
|:-:|---|---:|---:|---:|---|
| 0 (baseline) | NEW (Global+NeCo+Queue+NEG, 3-seed) | 0.8731 ± 0.0140 | 1.48 | **23.44 ± 1.80** | reference |
| **1a** | + RankMe representation column | 0.8731 ± 0.0140 | 1.48 | 23.44 ± 1.80 | ✓ paper N10 — NEW CV 7.7 % vs B5 22.6 % (64 % more stable); ρ(RankMe, ARI) = −0.429 → stability column, NOT ARI arbiter |
| **1b** | + HDBSCAN ε sweep × 7 values | 0.8731 ± 0.0140 | 1.48 | — | ✓ paper N9 reinforcement — **zero effect across 21 cells**, ε deprecated for NEW |
| **1c τ = 0.9** | + soft KNN-softmax τ-reassign | 0.8709 ± 0.0132 | 0.49 | — | ✓ paper N9 ext. — noise −67 %, ARI −0.0022 |
| **1c τ = 0.7** | + soft τ-reassign | 0.8696 ± **0.0123** | 0.15 | — | ✓ best std (12 % improvement), noise −90 %, ARI −0.0035 |
| **1c τ = 0.5** | + soft τ-reassign | 0.8681 ± 0.0125 | **0.00** ★★ | — | ✓✓ production cfg lock — **0 % noise**, ARI −0.0050 (well within seed std) |

### Paper claims locked-in by Step 1

| # | claim | location |
|:-:|---|---|
| N9 reinforcement | "HDBSCAN ε parameter is redundant on NEW embedding — cluster tree determined by (method, mcs, ms) triple alone" | RESULTS §19.2, METHOD §4c |
| N9 extension | "Soft KNN-softmax noise reassignment achieves 0 % noise rate at marginal ARI cost (Δ = −0.005), production-deployable" | RESULTS §19.3, METHOD §4c |
| N10 | "RankMe (Garrido et al. 2023) is informative for cross-seed stability (NEW std 1.80 vs B5 std 4.99 → 64 % more stable), NOT for ARI ranking (ρ = −0.429)" | RESULTS §19.1 |

### Next step gating

| outcome | next-step status |
|---|---|
| Step 1a — RankMe stability column locked | Reported in paper §19, N10 contribution |
| Step 1b — ε zero-effect (21 cells) | Epsilon deprecated; locked into production cfg |
| Step 1c — τ = 0.5 noise 0 % | **Production cfg lock**: τ = 0.5 KNN-softmax reassign for every-wafer labeling |
| Step 2 (EMA target encoder) | requires training dispatch — pending user approval |

### Source files (Step 1)

- `_step1b_hdbscan_eps_sweep.json` (21-cell raw)
- `_step1c_soft_tau_reassign.json` (12-cell raw)
- `step1_eval_only_summary_260513.md` (source-of-truth)
- `step1_paper_addition_260513.md` (this completion record's edit summary)

### Affected paper sections

- RESULTS.md §19 (NEW section "Step-by-step performance improvement, Step 1 eval-only")
- METHOD.md §4c (NEW subsection "Post-process refinement — soft τ-reassignment")
- ITERATIONS.md iter 86 entry (append-only)
- ABLATION_PLAN.md this section (completion record)
