---
description: CNN 학습 plan 수립 (subset YAML, hparam 조합, ablation 설계)
---

cnn-plan skill을 invoke해서 사용자 학습 목표에 맞는 실행 명령어 + subset YAML 만든다.

입력: $ARGUMENTS (예: "baseline", "imbalance test", "loss A/B", "quick smoke")

출력:
- 추천 명령어 (cnn_train.py CLI)
- 짝꿍 비교 명령 (필요 시)
- 생성한 subset YAML path (있다면)
- 예상 시간/디스크
- 다음 step 안내 (cnn-analyze)

사용자가 직접 명령어를 실행하거나 /cnn-train으로 트리거.
