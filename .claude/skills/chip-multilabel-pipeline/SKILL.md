---
name: chip-multilabel-pipeline
description: chip 200x200 single-label train -> multi-label predict 평가 파이프라인. classification_chips/ 4 class (bank_boundary, fork, scratch, scratch_rot) 학습 + 합성 11-class eval set (4 single + 5 combo + Normal + Invalid) 평가 매트릭스. Stage 1 = 기존 모델 + inference variants (I0-I4, I6-I9; I5 TTA 영구 금지), Stage 2 = T1/T4/T5/T6 학습 후 inference 매트릭스. 결과는 outputs/stage1_<TS>/ 또는 outputs/stage2_<TS>/ 의 results_matrix.parquet + report.md. 오답은 errors/<cell>/<error_type>/ 에 cap 200/type.
---

## 사용 시나리오

1. **합성 eval set 만들기** (한 번만):
   ```
   python -m chip_multilabel.gen_eval_set --out-root D:/project/data/wm-811k/chip_multilabel_eval_<tag> --per-class 200
   ```
   → 2200 chip, 11 폴더 + manifest.csv + _preview/ + _rejected/ (5% 미만 통과 못한 거)

2. **Stage 1: 기존 backbone + inference variants**:
   ```
   python -m chip_multilabel.run_stage1 --eval-set <eval_root> --out-root outputs --batch-size 32
   ```
   → 9 cell 매트릭스 (~6분, GPU 일부)

3. **Stage 2: 학습 + inference 매트릭스**:
   ```
   python -m chip_multilabel.run_stage2 --eval-set <eval_root> --epochs 8 --batch 8 --accum 4
   ```
   → 4 train (T1/T4/T5/T6) × 9 inference = 36 cell (~30분)

## Hard rules (위반 금지)

- **TTA (4-view averaging) 절대 금지** — chip 회전 의존적 (scratch vs scratch_rot). iter 1 실측 -0.018 macro_f1 손해. `forward_all_logits(... tta=False)` 만 사용.
- **outputs/ 결과 폴더 무단 삭제 금지** — CLAUDE.md 절대 금기 룰 적용.
- **`D:/project/known-cnn/` 코드 수정 금지** — read-only backbone 공급원.

## 결과 해석 가이드

| 메트릭 | 의미 | 기준선 |
|---|---|---|
| macro_f1 | 4-dim multi-hot per-class F1 평균 | iter 1 baseline I3=0.8466 |
| top1_11class | 11-class single-equivalent exact match | iter 1 best I1=0.6324 |
| ECE pre/post | calibration error before/after T scaling | < 0.05 desirable |
| temperature | scalar T (I4/I9 가 제공) | T<1 sharpens, T>1 flattens |

## 오답 분석

`outputs/<run>/errors/<cell_id>/<error_type>/<chip>.png` 와 sidecar `.json`
- false_positive_<class>: 해당 class GT 에 없는데 declare
- false_negative_<class>: GT 에 있는데 못 잡음
- wrong_combo: combo 인데 다른 combo 또는 single 로
- missed_invalid: invalid heuristic 실패
- missed_normal: normal 인데 defect 로

`image-analyzer` agent 호출해 cluster outlier / pixel pattern 자동 review:
```
Agent(subagent_type='image-analyzer', prompt='--run outputs/<stage>_<TS>')
```

## 자료 + 의견 누적

진행 중 발견 / hparam 실험 의견은 `chip_multilabel/notes.md` 에 iter 단위 append.
