---
name: cnn-analyze
description: log/<run>/ 학습 결과 진단 — per-class weakness, val/test gap, confusion off-diagonal pair, threshold sweep. 다중 run 비교 표 출력.
---

# cnn-analyze skill

이 스킬은 CNN 학습 산출물을 읽고 진단·비교 보고를 작성한다.

## 가장 먼저 읽기

| 문서 | 용도 |
|---|---|
| `.claude/skills/cnn-training/SKILL.md` | 출력 폴더 구조 spec |
| `cnn_train.py::save_per_class_report` | TXT 형식 정의 |
| 분석 대상 `log/<run>/` 폴더 | history.json, eval_summary.json, val/test_per_class_report.txt, hparams.yaml |

## 입력 → 산출

| 입력 | 산출 |
|---|---|
| 1개 `log/<run>/` 경로 | 단일 run 진단 보고 |
| 2+개 `log/<run>/` 경로 | 다중 run 비교 표 + 차이 분석 |

## 단일 run 분석 항목

### 1. 학습 stability
- `history.json` 읽어 train_loss 변동 (std), val_loss 발산 여부
- val_macro_f1 over epochs — best_epoch 위치 (early/late)
- val 가이드 reject 횟수 (run.log grep)
- gradient norm 이상값 (있다면)

### 2. Overfitting 진단
- val vs test gap: weighted F1 차이 > 5%p → overfitting 의심
- `eval_summary.json`의 val/test 비교

### 3. Per-class weakness
- `test_per_class_report.txt` 읽어 F1 < 0.85 class 목록
- 같은 class들의 FP/FN 비율 (precision-bound or recall-bound 진단)
- `support` 작은 minor class는 별도 표시

### 4. Confusion top pairs
- best_confusion_matrix_test.png 생성용 raw cm 데이터 (eval_summary.json의 labels/preds 또는 재계산)
- off-diagonal top 5: "A를 B로 자주 오인" 빈도 순

### 5. Threshold 추천
- 만약 threshold sweep 결과 있으면 (현재 미구현, 권고)
- 또는 confidence histogram 기반:
  - val에서 max_prob 분포 보고 5/95/99 percentile
  - 운영 threshold 예: 0.7 (95%ile of val correct conf)

### 6. Hyperparameter 진단
- `hparams.yaml` 읽어 setting 요약
- 의심 패턴: lr 너무 높음 (loss spike), label_smoothing 과도, EMA off였는데 noisy 등

## 다중 run 비교

| 비교 항목 | 방식 |
|---|---|
| Weighted F1 (test) | 표로 정렬, 색 강조 |
| Macro F1 (val/test) | 표 |
| 특정 class F1 | minor class 후보 자동 선택 (subset 명시 class) |
| Best epoch | 수렴 빠른 run 식별 |
| Total time | 시간 효율성 |

비교 표 예:
```
run                              wF1 test  mF1 test  best_ep  time(min)
baseline_20260501_F0.93_R0.91    0.927    0.910      18       145
imbal_v1_eff_cw_F0.89_R0.87      0.890    0.870      22       148
imbal_v1_no_cw_F0.83_R0.81       0.830    0.810      25       147
loss_focal_F0.91_R0.89           0.910    0.890      19       150
```

## 추가 저장 권고 (현재 미구현)

다음 학습부터 추가하면 분석 풍부해짐 (사용자 승인 필요):
- per-class F1 evolution plot (per_class_f1_curve.png)
- max_prob histogram (confidence_hist.png)
- top-N misclass JSON (top_misclass.json)

## 출력 형식

stdout 또는 사용자 요청 시 `log/<run>/analysis_report.md` 작성:

```markdown
# Run analysis: log/<run>/

## Stability
- train_loss std (last 10 ep): 0.012  ← 안정
- val_loss spike count: 0
- best_epoch: 18 / 30 (ample headroom 있음 → 더 학습 시도 가능)

## Val/Test gap
- val weighted F1: 0.945 / test: 0.927  → gap 1.8%p (정상 범위)

## Weak classes (test F1 < 0.85)
- Donut_scratch: F1 0.78 (recall-bound, FN=15)
- Edge-Bottom_invalid_main: F1 0.82 (precision-bound, FP=12)

## Top confusion
- Donut_scratch → Donut_bank_boundary: 8회
- Edge-Bottom_invalid_main → Edge-Bottom_particle_blast: 5회
...
```

## 금지

- `log/<run>/` 폴더 수정·삭제 금지 (read-only 분석)
- `cnn_train.py` 직접 실행 금지

## 반환

- 분석 보고 (stdout 또는 markdown 파일)
- 주요 weak class list (다음 실험 candidate)
- 다음 step 추천 (어떤 hparam 조정·재학습)
