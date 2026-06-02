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

# 비워두면 train_pipeline_ddp.py CONFIG/default 자동 탐색 사용.
# 필요할 때만 지정. 반드시 <class>/*.png 구조.
SOURCE_ROOT="${SOURCE_ROOT:-}"

# 비워두면 기본 contrastive split/eval 흐름. 현업 classless 학습을 할 때만 지정.
# 콤마로 여러 개 가능.
PROD_TRAIN_DIRS="${PROD_TRAIN_DIRS:-}"

# 비워두면 grouping 생략. 현업 grouping 할 때만 지정. 콤마로 여러 개 가능.
PROD_PRED_DIRS="${PROD_PRED_DIRS:-}"

# Split/CNN/contrastive/grouping options.
CLEAN_SPLIT=1
PROD_EPOCHS=5
CNN_BATCH=32
CL_BATCH=64
GROUPING_BATCH=128
GROUPING_WORKERS=16
GROUPING_REPS_PER_CLUSTER=5

cd "$(dirname "$0")/.."

cmd=(
  python -u scripts/train_pipeline_ddp.py
  --cnn-batch "$CNN_BATCH"
  --cl-batch "$CL_BATCH"
)

if [[ -n "$SOURCE_ROOT" ]]; then
  cmd+=(--source-root "$SOURCE_ROOT")
fi
if [[ "$CLEAN_SPLIT" == "1" ]]; then
  cmd+=(--clean-split)
fi
if [[ -n "$PROD_TRAIN_DIRS" ]]; then
  cmd+=(--prod-train-dirs "$PROD_TRAIN_DIRS" --prod-epochs "$PROD_EPOCHS")
fi
if [[ -n "$PROD_PRED_DIRS" ]]; then
  cmd+=(
    --prod-pred-dirs "$PROD_PRED_DIRS"
    --grouping-batch "$GROUPING_BATCH"
    --grouping-workers "$GROUPING_WORKERS"
    --grouping-reps-per-cluster "$GROUPING_REPS_PER_CLUSTER"
  )
fi

echo "[run] ${cmd[*]}"
"${cmd[@]}"
