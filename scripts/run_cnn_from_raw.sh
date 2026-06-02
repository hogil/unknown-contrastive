#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CNN only
#   raw labeled images -> auto split -> CNN DDP train
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_cnn_from_raw.sh
# ============================================================

# 비워두면 CUDA_VISIBLE_DEVICES 를 건드리지 않음.
# 필요할 때만 쉘 앞에서 지정:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_cnn_from_raw.sh
CUDA_DEVICES="${CUDA_DEVICES:-}"

# 반드시 <class>/*.png 구조.
SOURCE_ROOT="data/images/unknown"

CLEAN_SPLIT=1
CNN_BATCH=32
CNN_EPOCHS=30
CNN_WORKERS_PER_GPU=""     # 빈 값이면 script auto
CNN_PREFETCH=""            # 빈 값이면 script default

cd "$(dirname "$0")/.."

cmd=(
  python -u scripts/train_cnn_ddp.py
  --source-root "$SOURCE_ROOT"
  --batch "$CNN_BATCH"
  --epochs "$CNN_EPOCHS"
)

if [[ "$CLEAN_SPLIT" == "1" ]]; then
  cmd+=(--clean-split)
fi
if [[ -n "$CNN_WORKERS_PER_GPU" ]]; then
  cmd+=(--workers "$CNN_WORKERS_PER_GPU")
fi
if [[ -n "$CNN_PREFETCH" ]]; then
  cmd+=(--prefetch "$CNN_PREFETCH")
fi

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'(scheduler/default)'} ${cmd[*]}"
"${cmd[@]}"
