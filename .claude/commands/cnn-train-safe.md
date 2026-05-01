---
description: 자원 가드 포함 안전 CNN 학습 — cnn-master + resource-monitor team으로 dispatch (RAM 80% / GPU 90% 한계 자동 polling)
---

cnn-master agent를 invoke. team_name=`cnn-team`으로 spawn해서 resource-monitor와 협조.

워크플로우:
1. resource-monitor agent에 mode=check 질의 → 시작 가능 판단
2. 차단 시 mode=wait_until_ok로 polling (60s 주기, max 30분)
3. ok 떨어지면 `python cnn_train.py $ARGUMENTS` 백그라운드 실행
4. 학습 중 30s 주기 watch — RAM>=80% 시 process kill + log 폴더 `_PAUSED` rename + 자원 회복 대기 + 재시작 (새 model-tag)
5. 정상 종료 시 최종 폴더 path 보고

인자: `$ARGUMENTS` — cnn_train.py 인자 그대로 전달
예: `/cnn-train-safe --epochs 30 --batch 16 --model-tag baseline`

코드 수정 없이 agent layer만으로 가드 운영. cnn_train.py 인라인 `_resource_guard`는 이중 안전망(redundant).
