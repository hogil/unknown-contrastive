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

# 비워두면 CUDA_VISIBLE_DEVICES 를 건드리지 않음.
# 필요할 때만 쉘 앞에서 지정:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_prod_pipeline.sh
CUDA_DEVICES="${CUDA_DEVICES:-}"

# CNN supervised source. 반드시 <class>/*.png 구조.
SOURCE_ROOT="data/images/unknown"

# 현업 classless contrastive 학습 폴더. 콤마로 여러 개 가능.
PROD_TRAIN_DIRS="data/images/prod_train"

# grouping/pred 할 다른 현업 폴더. 콤마로 여러 개 가능.
PROD_PRED_DIRS="data/images/prod_pred"

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
  --source-root "$SOURCE_ROOT"
  --prod-train-dirs "$PROD_TRAIN_DIRS"
  --prod-pred-dirs "$PROD_PRED_DIRS"
  --prod-epochs "$PROD_EPOCHS"
  --cnn-batch "$CNN_BATCH"
  --cl-batch "$CL_BATCH"
  --grouping-batch "$GROUPING_BATCH"
  --grouping-workers "$GROUPING_WORKERS"
  --grouping-reps-per-cluster "$GROUPING_REPS_PER_CLUSTER"
)

if [[ "$CLEAN_SPLIT" == "1" ]]; then
  cmd+=(--clean-split)
fi

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'(scheduler/default)'} ${cmd[*]}"
"${cmd[@]}"
