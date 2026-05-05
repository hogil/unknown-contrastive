---
description: contrastive learning + HDBSCAN 성능 향상 외부 리서치 (arxiv + GitHub). run_dir 옵션 시 eval_summary.json 의 weak point 자동 query.
---

contrastive-master agent 를 `mode=research` 로 invoke. team_name=`contrastive-team`.

입력: `$ARGUMENTS`
- `--run <run_dir>` (옵션) — weak point 자동 추출
- `--topic <a,b,c>` (옵션) — 명시 주제
- `--max-papers <N>` / `--max-repos <N>` (옵션, default 8/5)

예:
- `/contrastive-research --run outputs/logs_contrastive/overall --max-papers 10`
- `/contrastive-research --topic supcon_loss,hdbscan_noise --max-papers 5`
- `/contrastive-research --run outputs/logs_contrastive/overall --max-papers 6 --max-repos 4`

워크플로우:
1. (run_dir 있을 때) `eval_summary.json` 읽고 weak point → topic 매핑
2. performance-research agent 에 위임 → arxiv (WebSearch + WebFetch) + GitHub (WebSearch + README WebFetch)
3. `research_<TS>.md` / `.json` 산출 (run_dir 있으면 그 안, 없으면 `research/research_<TS>.{md,json}`)
4. 상위 3 paper / 3 repo 요약 + action_items 반환

코드 수정 / weights 다운로드 / 학습 trigger 안 함 — 권고만. WebFetch 실패 시 추정 답변 금지 (`fetch_failed` 명시).
