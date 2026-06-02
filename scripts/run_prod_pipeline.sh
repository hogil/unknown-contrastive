#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Production one-shot pipeline
#   raw labeled synthetic/source images -> split -> CNN
#   -> production classless contrastive train -> production grouping
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_prod_pipeline.sh
# ============================================================

# 폴더 옵션 사용법:
# - 상대경로: 프로젝트 루트 기준. 예: data/images/unknown
# - 절대경로: 그대로 사용. 예: /mnt/data/unknown
# - 다중 폴더: 콤마로 연결. 예: /mnt/a,/mnt/b
#
# SOURCE_ROOT 는 반드시 <class>/*.png 구조.
SOURCE_ROOT="data/images/unknown"
PROD_TRAIN_DIRS="data/images/contrastive_train"
PROD_PRED_DIRS="data/images/contrastive_eval"

cd "$(dirname "$0")/.."

# python -u: print/log 를 버퍼링하지 않고 바로 출력. 학습 동작에는 영향 없음.
cmd=(
  python -u scripts/train_pipeline_ddp.py
  --source-root "$SOURCE_ROOT"
  --prod-train-dirs "$PROD_TRAIN_DIRS"
  --prod-pred-dirs "$PROD_PRED_DIRS"
)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
