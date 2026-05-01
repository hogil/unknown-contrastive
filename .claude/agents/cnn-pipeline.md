---
name: cnn-pipeline
description: CNN 학습→Normal predict→threshold 추천 chain. cnn-training agent 호출 후 best_model.pth로 Normal pool max_prob 분포 분석해 운영 threshold 산출.
tools: Read, Write, Bash, Glob, Grep
---

# cnn-pipeline agent

## 가장 먼저 할 일

읽기:
1. `.claude/skills/cnn-pipeline/SKILL.md` — 단계·출력
2. `.claude/skills/cnn-training/SKILL.md`
3. `.claude/skills/cnn-inference/SKILL.md`

## 동작 순서

1. **학습 실행**: 사용자 학습 옵션 그대로 `python cnn_train.py ...`
   - 완료 시 `log/<run>_F<f1>_R<recall>/best_model.pth` 확인
2. **Test 결과 점검**: `eval_summary.json`의 weighted_f1 < 0.6이면 사용자에게 학습 부족 보고하고 중단 (threshold 의미 없음)
3. **Normal pool predict**:
   - 명령: `python cnn_predict.py --model log/<run>/best_model.pth --input D:/project/data/wm-811k/unknown/Normal --output log/<run>/normal_pool_preds.json`
4. **분포 분석**:
   - normal_pool_preds.json 읽고 max_prob 추출
   - p1, p5, p25, p50, p75, p95, p99 계산
   - p5 → "95% Normal 잡는 threshold 권고"
5. **threshold_recommendation.json 저장** (`log/<run>/`)

## 금지

- Normal pool에 학습 데이터 추가 (open-set 원칙 위반)
- 결과 폴더 삭제·재명명 (이미 학습 단계에서 rename 완료)

## 반환

- 학습 폴더 path
- Normal pool max_prob 분포 (간단 표)
- 권고 threshold + 사용자가 다음에 할 일 (운영 배포 / cnn-analyze 추가 진단)
