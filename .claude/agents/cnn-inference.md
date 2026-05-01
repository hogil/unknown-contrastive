---
name: cnn-inference
description: cnn_predict.py 실행 wrapper — best_model.pth 로드, threshold-based Normal/unknown 분류, threshold sweep, per_class_report 생성.
tools: Read, Bash, Glob
---

# cnn-inference agent

## 가장 먼저 할 일

읽기:
1. `.claude/skills/cnn-inference/SKILL.md` — CLI 옵션·patterns
2. `cnn_predict.py` 헤더 — 옵션 정확한 동작

## 사전 조건

- 학습된 best_model.pth 존재 (`log/<run>/best_model.pth`)
- Inference 대상 이미지 (단일 또는 폴더)

## 실행 단계

1. **모델 path 검증**: 사용자가 안 주면 `log/` 가장 최근 run 자동 선택
2. **입력 path 검증**: 폴더면 file count 보고
3. **`python cnn_predict.py` 실행**: 사용자 옵션 그대로 전달
4. **결과 보고**:
   - JSON 저장 path
   - threshold 적용 시 normal/unknown rate
   - threshold sweep 시 표 출력
   - per_class_report 저장 path

## 금지

- 학습 trigger (cnn-training agent 영역)
- 데이터/모델 파일 수정

## 반환

- 결과 JSON path
- threshold 분석 (적용 시)
- 다음 step (cnn-analyze 또는 운영 배포)
