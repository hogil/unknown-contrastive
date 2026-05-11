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

## 기대 효과 (가설)

| step | 효과 가설 | 실측 |
|:-:|---|---|
| B0 → B1 | + Local DenseCL → noise huge ↓ | TBD |
| B1 → B2 | LW strong → noise 추가 ↓ | -50% (iter A0→Iter 1 패턴) |
| B2 → B3 | + Queue → more negatives | small ↑ |
| B3 → B4 | + NEG filter → false neg 보호 | ↑ |
| B4 → B5 | + NeCo (★ paper N1) | noise -70% (iter 35→37 패턴) |

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
