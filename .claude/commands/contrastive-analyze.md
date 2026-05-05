---
description: 기존 contrastive run 에 cluster-analyzer + image-analyzer 만 실행 (학습 안 함, 자원 가드 skip)
---

contrastive-master agent 를 `mode=analyze` 로 invoke. team_name=`contrastive-team`.

입력: `$ARGUMENTS` — `--run <run_dir>` (필수)

예:
- `/contrastive-analyze --run outputs/logs_contrastive/overall`
- `/contrastive-analyze --run outputs/logs_contrastive/normal1000_n50_b16_global_e10_resize_reuse_260505_110513`

워크플로우:
1. run_dir 검증 (`<run_dir>/eval/eval_summary.json` 존재 필수)
2. cluster-analyzer agent 호출 → `<run_dir>/analyze_clusters.{md,json}`
3. image-analyzer agent 호출 → `<run_dir>/analyze_images.{md,json}`
4. 두 보고서 path + 핵심 카운트 요약 반환

read-only — 폴더 수정 / 학습 trigger 안 함. 새 산출 파일만 추가.
