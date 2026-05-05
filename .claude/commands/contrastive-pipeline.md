---
description: contrastive 학습 → 평가 → composite → cluster 분석 → image 분석 (옵션 --research) 풀체인. resource-monitor 가드 포함.
---

contrastive-master agent 를 invoke. team_name=`contrastive-team` 으로 spawn 해서 resource-monitor 와 협조.

워크플로우:
1. resource-monitor agent 에 `mode=check` 질의 → 시작 가능 판단
2. 차단 시 `mode=wait_until_ok` 로 polling (60s 주기, max 30 분)
3. ok 떨어지면 `python run_contrastive.py $ARGUMENTS` (또는 preset 매칭 wrapper) 백그라운드 실행
4. 학습 중 30s 주기 watch — RAM≥80% 시 process kill + outputs 폴더 `_PAUSED_<TS>` rename + 자원 회복 대기 + 새 tag `_resumed_<n>` 재시작
5. 학습 종료 후 evaluation → composite-map → cluster-analyzer → image-analyzer 순차 dispatch
6. `--research` flag 있을 때만 performance-research 도 chain
7. run_dir + analyze_*.md path 보고

인자: `$ARGUMENTS` — `--preset <name>` 또는 `_contrastive_n50.py` 인자, `--research` flag 포함 가능

예:
- `/contrastive-pipeline --preset normal1000_n50_b16_global_e10`
- `/contrastive-pipeline --preset normal1000_n50_b16_global_e10 --research`
- `/contrastive-pipeline --epochs 20 --batch 16 --per-class 50 --normal 200 --backbone D:/project/known-cnn/outputs/logs_wafer/overall/best_model.pth`

코드 수정 없이 agent layer 만으로 가드 운영. PAUSED / FAILED 폴더 삭제 절대 금지.
