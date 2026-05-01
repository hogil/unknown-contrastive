---
description: 학습 결과 분석 — log/<run>/ 진단, 다중 run 비교
---

cnn-analyze skill을 invoke해서 $ARGUMENTS로 받은 log 폴더를 분석.

입력: $ARGUMENTS (1+개 log/<run>/ 경로, 공백 구분)
- 1개: 단일 run 진단 (stability / val-test gap / weak class / confusion top / hparam)
- 2+개: 비교 표 + 차이 해석

출력: stdout 보고 또는 `log/<run>/analysis_report.md`

read-only 분석 — 파일 수정 안 함.
