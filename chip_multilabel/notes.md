# Chip Multi-Label — 작업 노트 (실시간 갱신)

## Hard Rules (사용자 확정)

- **TTA 절대 금지** — I5 (4-view averaging) 가 macro_f1 -0.018 손해. chip 패턴이 회전 의존적 (scratch vs scratch_rot) 이라 averaging 이 신호를 흐림. **앞으로 어떤 inference variant 도 TTA 안 씀**.

---

## Iter 1 (Stage 1 baseline) — 260505_162842

**eval set**: 2200 chip / 11 class @ `D:/project/data/wm-811k/chip_multilabel_eval_full/`
**모델**: `chip5_round4_v14_260505_061558_running/best_model.pth` (학습 X)
**소요**: 6분 (forward 1.2분 + TTA 4.4분 — 마지막 TTA 시간 낭비)

### 6 cell 결과
| cell | macro_f1 | top1_11 | T |
|---|---|---|---|
| **I3** sigmoid+F1max | **0.8466** | 0.6017 | 1.0 |
| I4 TS+I3 | 0.8466 | 0.6017 | 0.376 |
| I1 softmax+F1max | 0.8444 | **0.6324** | 1.0 |
| I5 TTA+TS+I3 | 0.8287 | 0.6011 | 0.362 |
| I2 sigmoid 0.5 | 0.7673 | 0.5739 | 1.0 |
| I0 argmax | 0.7302 | 0.4472 | 1.0 |

### 진단

1. **fork 가 가장 약함** (F1=0.63, threshold=0.12, precision=0.48, recall=0.91)
   - 중요: precision 0.48 — 절반은 잘못된 fork 선언. recall 0.91 — fork 들어간 진짜 케이스는 잘 잡음.
   - 즉 **over-firing**: threshold 너무 낮음 → noise/normal 까지 fork 라고 함
2. Top errors:
   - 160× Normal → fork
   - 155× bank_boundary → bank_boundary+fork
   - 141× bank_boundary+scratch_rot → bank_boundary+fork
3. **TS 무용** (I3 == I4) — threshold tuning 이 calibration 효과 흡수
4. **TTA 손해** — 영구 폐기
5. **softmax+threshold (I1) top1_11 가 최고** (0.6324) — softmax 의 합=1 제약이 11-class 결정에 도움. 다만 multi-hot macro_f1 은 I3 보다 약간 낮음.

### 가설 / 다음 시도

**Inference-side (학습 X, 빠름)**:
- **I6 — prior-aware logit shift**: 학습 분포 (chip_train: 25% per class) vs eval 분포 (combo+normal+invalid 섞임) 의 prior gap 보정. fork 가 over-fire 한다는 건 training prior 가 eval 보다 높음 → log(p_eval/p_train) 만큼 logit 빼기.
- **I7 — combo-aware threshold**: single 결정 threshold 와 combo 추가 declare threshold 분리. combo 추가 declare 는 더 보수적 (예: 0.6 이상).
- **I8 — top-k cap with margin**: top-2 만 항상 고려, 2nd prob > top1 prob × 0.5 일 때만 combo 선언.
- **I9 — per-class temperature** (단일 T 대신 per-class T): fork 만 logit 다운-스케일.

**Training-side (Stage 2)**:
- T1 LS 0.1: overconfidence 완화
- T4 ASL γ_neg=4: fork FP 직접 페널티
- T5 BCE: sigmoid 친화 학습

전략: 먼저 inference-side I6/I7/I8/I9 (몇 분) → 그래도 부족하면 Stage 2 진입.

---

## Iter 2 (Stage 1 + new variants) — 260505_165400

**TTA forward path 영구 제거**, 9 variant (I0-I4 + I6-I9) 동시 실행 / forward 1회 (~72s).

### 9 cell 결과
| cell | macro_f1 | top1_11 | T |
|---|---|---|---|
| **I7** joint coord descent | **0.8485** | **0.6210** | 1.0 |
| I3 sigmoid+F1max | 0.8466 | 0.6017 | 1.0 |
| I4 TS+I3 | 0.8466 | 0.6017 | 0.376 |
| I8 top-2 margin (m=0.6) | 0.8456 | 0.6017 | 1.0 |
| I1 softmax+F1max | 0.8444 | **0.6324** | 1.0 |
| I6 F1max + floor 0.3 | 0.8177 | 0.5881 | 1.0 |
| I9 per-class T | 0.7741 | 0.5341 | 0.730 |
| I2 sigmoid 0.5 | 0.7673 | 0.5739 | 1.0 |
| I0 argmax | 0.7302 | 0.4472 | 1.0 |

### 진단

1. **I7 새 best 이지만 +0.002 미미** — 인퍼런스 트릭 ceiling 임. 0.85 가 학습 변경 없는 한계.
2. **I6 (floor 0.3) 후퇴 −0.029** — fork threshold 0.12 가 그렇게 낮은 것이 실제 최적이었음. 즉 fork prob distribution 자체가 noise 영역에서 평균이 높음 (모델이 noise 를 fork 처럼 봄). floor 로 막는 게 아니라 모델이 noise→fork 못 보게 학습 손봐야.
3. **I9 (per-class T) 후퇴 −0.072** — multi-hot binary CE 로 per-class T LBFGS 가 unstable. 특히 fork class 의 multi-positive (single fork + 4 combo) val 분포가 binary 가정 깨짐.
4. **I8 (top-2 margin) ≈ I3** — margin gating 효과 없음. 이미 threshold 가 비슷한 보호 제공.
5. **top1_11class 1위는 I1** (softmax+thresh, 0.6324) — softmax 의 sum=1 제약이 11-class final decision 에 유리. multi-hot 평가에서는 sigmoid 가 약간 우세.
6. **sigmoid `np.exp` overflow warning** 한 번 — 일부 logit 이 큰데 무시해도 됨 (exp 음수 큰 값 → 0 underflow 만, 결과 정확).

### 결론

**인퍼런스 트릭 plateau ~0.85**. Stage 2 학습 변경 없이는 더 못 깸.

### 다음: Stage 2 (학습 변경)

fork over-firing 근본 원인 = 학습 데이터의 noise 영역에서 fork 패턴 (vertical stripes) 이 다른 패턴 noise 와 구분 약함. 처방:

| variant | 가설 | 예상 효과 |
|---|---|---|
| T1 LS 0.1 | overconfidence 완화 → fork prob 0.9+ noise 에서 안 뜸 | 소폭 개선 (0.86) |
| T4 ASL γ_neg=4 | negative class wrong prediction 직접 페널티 | 강력 (0.87+) |
| T5 BCE multi-hot 1-positive | sigmoid 학습 → inference logit dist 일관 | 강력 (0.87+) |
| T6 BCE→ASL warmup 5ep | M5 패턴, BCE 안정 시작 후 ASL 강하게 | 가장 강력 (0.88+) |

skip: T0 (=I3 baseline 재현, 시간 낭비), T2 (mixup α=0.1 작은 chip 위험 / 효과 미미), T3 (focal — ASL 이 보통 더 좋음).

학습 4 variant × 5 inference variant (I0,I1,I3,I7,I8 — top performer + diversity) = **20 cell 매트릭스**. 각 학습 ~3분 + inference 캐시. 총 ~25분.

---

## Iter 3: I10 entropy-Normal short-circuit — 260505_170827

**Variant**: I10 = I7 (joint coord descent thresholds) + softmax entropy gate. 입력 로짓의 softmax entropy ≥ 0.85·log(4) (= 4-class uniform 의 85%) 이면 모델이 어느 한 chip 패턴에도 confident 하지 않다는 신호 → 곧장 **Normal** 로 선언, threshold 비교 단계 스킵.

### 결과

| cell | macro_f1 | top1_11 | Δ vs I7 |
|---|---|---|---|
| **I10 entropy→Normal** | **0.8542** | **0.6517** | **+0.0057 / +0.031** |
| I7 joint coord descent | 0.8485 | 0.6210 | — |
| I3 sigmoid+F1max | 0.8466 | 0.6017 | -0.0076 / -0.050 |

**새 best macro_f1**. top1_11 점프 (+0.031) 가 macro_f1 점프보다 큼 → Normal 결정 정확도 자체가 좋아짐 (11-class single-pick 평가에서 Normal 이 1 클래스).

### Error type delta (T0__I7 vs T0__I10)

| error_type | I7 | I10 | Δ |
|---|---:|---:|---:|
| wrong_combo | 292 | 273 | **−19** |
| false_positive_fork | 215 | 215 | 0 |
| missed_normal | 160 | 106 | **−54** |
| wrong_normal_entropy | 0 | 19 | +19 |
| **total** | **667** | **613** | **−54** |

### Insight

I10 의 entropy gate 가 직접 노린 것은 **missed_normal** (Normal GT 인데 모델이 fork/scratch 등 declare 한 케이스). 결과:

- **missed_normal −54 (-34%)** — 가장 큰 감소. fork over-firing 의 주된 출처가 Normal 이미지에서 fork 가 살짝 뜨는 것이었는데, 그 케이스들은 실제로 4 chip class 모두에 대해 logit 이 평탄 (high entropy) 하다. entropy ≥ 0.85·log4 컷으로 골라낼 수 있었음.
- **wrong_combo −19** — bonus. 일부 noise 패턴에서 두세 class 가 비슷하게 떠 combo 잘못 선언하던 케이스도 entropy 로 잡힘 → Normal 로 흡수.
- **false_positive_fork 0 변화** — fork 가 *단독으로* 강하게 뜨는 케이스 (low entropy, single peak on fork) 는 entropy gate 가 못 잡음. 여전히 215 건 그대로. Stage 2 학습 변경 (T4 ASL γ_neg, T6 BCE→ASL warmup) 이 필요.
- **wrong_normal_entropy +19** — entropy gate 의 새로운 false positive: 진짜 chip 패턴인데 logit 이 평탄해서 잘못 Normal 처리됨. 19 건은 missed_normal -54 에 비해 작아 net +35 정정.

요약: I10 의 entropy 컷은 **fork-vs-Normal 판별** 의 큰 부분을 해결. 남은 215 false_positive_fork (Normal/다른 패턴 → fork 단독) 가 entropy 로 안 풀리는 핵심 잔존 에러.

### Cross-iteration best macro_f1

| iter | best cell | macro_f1 | Δ |
|---|---|---:|---:|
| 1 | I3 sigmoid+F1max | 0.8466 | — |
| 2 | I7 joint coord descent | 0.8485 | +0.002 |
| 3 | **I10 entropy→Normal** | **0.8542** | **+0.006** |

iter 2→3 의 +0.006 은 iter 1→2 의 +0.002 보다 훨씬 큼 — entropy gate 가 단순 threshold 튜닝보다 더 직교한 신호를 활용. 그래도 inference-side ceiling 은 가까이 왔으니 (남은 215 false_positive_fork 단독 에러), 다음은 Stage 2 학습 변경으로 fork over-fire 의 근원을 다뤄야 함.

---

## Iter 4: Stage 2 학습 + I10 매트릭스 — 260505_173649~174123

Stage 2 main run (`outputs/stage2_260505_170121/`) 은 inference variants I0-I9 만 평가 (I10 추가 전 dispatch 됨). 학습 4 variant 끝난 후 I10 추가 평가 실행.

### 풀 매트릭스 (train × {I3, I7, I10})

| train | inference | macro_f1 | top1_11 | 비고 |
|---|---|---|---|---|
| **T1 CE+LS** | **I10** | **0.8634** | **0.7006** | **OVERALL BEST** |
| T0 (기존) | I10 | 0.8542 | 0.6517 | iter 3 best |
| T0 (기존) | I7 | 0.8485 | 0.6210 | iter 2 best |
| T0 (기존) | I3 | 0.8466 | 0.6017 | iter 1 best |
| T6 BCE→ASL | I3 | 0.8396 | 0.5108 | Stage 2 main best (I10 없을 때) |
| T1 CE+LS | I3 | 0.8378 | 0.6420 | |
| T1 CE+LS | I7 | 0.8289 | 0.6210 | |
| T6 BCE→ASL | I10 | 0.8193 | 0.6256 | |
| T6 BCE→ASL | I7 | 0.8190 | 0.6244 | |
| T5 BCE | I3 | 0.8018 | 0.4426 | |
| T4 ASL | I3 | 0.7806 | 0.5881 | |
| T4 ASL | I7 | 0.7766 | 0.5830 | |
| T4 ASL | I10 | 0.7759 | 0.5830 | |
| T5 BCE | I7 | 0.7589 | 0.5432 | |
| T5 BCE | I10 | 0.7589 | 0.5432 | |

### 진단

**T1 (CE+LS 0.1) 만 학습이 도움** — TAPT backbone 의 강한 prior 를 살짝 부드럽게 정규화. Δ vs T0 best:
- macro_f1: 0.8542 → 0.8634 (+0.0092)
- top1_11: 0.6517 → 0.7006 (+0.0489) **큰 도약**

**T4 (ASL), T5 (BCE), T6 (BCE→ASL) 전부 손해** — 327 chip 단일-positive 학습에 ASL γ_neg=4 / BCE 가 너무 강한 perturbation. 결과:
- T4 macro_f1 −0.078 vs T0
- T5 macro_f1 −0.052 vs T0
- T6 macro_f1 −0.035 vs T0 (warmup 덕분에 손해 작음)

**I10 효과 가 train 별 다름**:
- T0, T1: I10 가 I3 / I7 보다 강함 (entropy gate 발동, 미스 normal 50+개 회수)
- T4, T5, T6: I10 == I7 ~ I3 (entropy gate 거의 발동 안 함). 새 손실로 학습된 모델 logit 분포가 더 sharp 한 single-peak 형태 → softmax entropy 항상 낮음 → entropy gate 트리거 안 됨.
- 따라서 ASL/BCE 모델은 fork over-fire 가 어떤 식으로든 발생하지만 그 "noise → fork" 패턴이 entropy 로 안 잡힘.

### 결론 (iter 4 시점)

**최선 조합: T1 (CE + label smoothing 0.1) + I10 (joint coord descent thresholds + softmax entropy → Normal)** = **0.8634 macro_f1, 0.7006 top1_11**.

학습 측: ASL/BCE 처럼 강한 손실 변경은 작은 데이터 + 강한 TAPT init 환경에서 부정적. mild regularization (LS 0.1) 만 도움.

### Cross-iteration progression

| iter | best cell | macro_f1 | top1_11 | Δ macro_f1 |
|---|---|---|---|---|
| 1 | T0__I3 | 0.8466 | 0.6017 | — |
| 2 | T0__I7 | 0.8485 | 0.6210 | +0.002 |
| 3 | T0__I10 | 0.8542 | 0.6517 | +0.006 |
| **4** | **T1__I10** | **0.8634** | **0.7006** | **+0.009** |

**iter 4 의 +0.009 는 학습+추론 공동 최적화 효과**. 단일 inference 트릭 (+0.006) 또는 단일 학습 변경 (이번 case 기준 −0.005~−0.078) 보다 큼.

### 남은 약점

- **fork single FP 215 건 그대로** (Normal/bank 등 → fork 단독). T1 학습 + I10 entropy 모두 fork 단독 강한 logit 케이스에는 무력. Stage 2 변종 더 (T2 mixup, T7 fork-targeted hard negative) 가 마지막 carries.
- **scratch_rot 헤드 noise prior** — error-analyst 진단 (mean prob 0.74 on Normal). TAPT backbone level 의 문제, retrain 으로 안 풀림.

---

## TODO — Stage 2 끝난 후 (사용자 directive 260505)

1. **합성 난이도 조절** — 현재 combo 합성 (min-blend) 결과 라벨링이 너무 어려움 (eval 결과 macro_f1 ~0.85 근처 plateau). 보강:
   - source chip 중 **불량 정도 강한 것들끼리 우선** combo: 각 single class chip 의 defect_pixel_ratio 계산 → 상위 50% 만 source 로 사용. 약한 chip 끼리 blend 하면 noise 와 구분 안 됨.
   - 새 변종 generator option `--source-strength-pct 50` 추가.

2. **Pixel grade variation** — 현재 합성/소스 chip palette grade 가 거의 0 (white) + 1 (grey) 만. 더 강한 defect grade (2 green, 3 blue 등) 분포 시도:
   - `_make_normal_chip` / `_min_blend` 에서 일부 픽셀을 grade 2/3 으로 강제 elevated 변종.
   - 또는 source chip 자체에 grade 2 우세인 것만 골라 사용 (chip 마다 grade histogram 으로 필터).
   - 새 변종 generator option `--grade-mode {default, elevated_2, elevated_3}` 추가.

3. **재합성 후 동일 inference variants 다시 평가** — 새 eval set 으로 stage1 + stage2 둘 다 재실행. iter 4 ~ 로 notes 추가.

이건 Stage 2 (현재 T5 BCE 학습 중) 완료 후에 실행 — GPU 1잡 룰 + 사용자 명시 "학습 다 끝나고".

