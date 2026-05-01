---
name: cnn-pipeline
description: cnn-training → cnn-inference 순차 chain coordinator. 학습 후 best_model.pth로 Normal pool에 predict, max_prob 분포 분석해 운영 threshold 추천.
---

# cnn-pipeline skill

CNN 학습 → 검증 → threshold 추천 1-shot pipeline.

## 가장 먼저 읽기

| 문서 | 용도 |
|---|---|
| `.claude/skills/cnn-training/SKILL.md` | 학습 wrapper |
| `.claude/skills/cnn-inference/SKILL.md` | predict wrapper |
| `.claude/skills/cnn-analyze/SKILL.md` | 학습 결과 진단 |

## 단계

1. **학습** (cnn-training agent)
   - 사용자 옵션 그대로 (`--epochs`, `--subset-config`, `--class-weight` 등)
   - 결과: `log/<run>_F<f1>_R<recall>/`

2. **Test eval 검증** (이미 cnn_train.py에서 자동 수행)
   - `eval_summary.json` 의 `test.weighted_f1`, `test.macro_f1` 확인
   - threshold 합리적이면 다음 단계, 아니면 stop + 사용자 보고

3. **Normal pool inference** (cnn-inference agent)
   - 입력: `D:/project/data/wm-811k/unknown/Normal/` (5000장 무라벨 풀)
   - 명령: `python cnn_predict.py --model log/<run>/best_model.pth --input D:/project/data/wm-811k/unknown/Normal --output normal_preds.json`
   - max_prob 분포 (히스토그램) 추출

4. **Threshold 추천**
   - 95% Normal 잡는 threshold = max_prob 분포의 5-percentile 이상
   - 99% Normal 잡는 threshold = 1-percentile 이상
   - **권고 threshold**: 95% target 기준 (운영 false-defect rate 5%)
   - 보고: 권고값 + 분포 히스토그램 numeric summary (min, mean, p5, p25, p50, p75, p95, max)

5. **Optional cnn-analyze**: per-class weakness, val/test gap, off-diagonal pair 진단

## 출력

```
log/<run>_F<f1>_R<recall>/
  ... (학습 결과)
  normal_pool_preds.json        # Normal 5000장 예측
  threshold_recommendation.json # {target: 0.95, threshold: 0.7, distribution: {...}}
```

## 주의

- Normal pool 경로 hardcoded: `D:/project/data/wm-811k/unknown/Normal/`
- 사용자 다른 경로 원하면 `--normal-pool PATH` 옵션 처리 (slash command에서 받음)
- Normal pool은 inference만 — 학습엔 절대 들어가면 안 됨
