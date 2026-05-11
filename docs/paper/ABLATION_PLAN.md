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
| B3 | + Queue | 1.000 | 1.309% | 0.9828 | 0.9365 | 0.9496 | 0.9591 | 0.8464 | 0.5727 | 36 |
| **B4** ★ | + NEG=0.72 | 1.000 | **0.524%** | **0.9852** | 0.9439 | 0.9557 | 0.9641 | **0.8605** | 0.6109 | 37 |
| B5 | + NeCo (=iter 37) | 1.000 | 0.960% | 0.9801 | 0.9403 | 0.9503 | 0.9598 | 0.8564 | 0.6104 | 37 |

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
