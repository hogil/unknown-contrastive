#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Production classless contrastive only
#   CNN best_model.pth -> classless production contrastive train
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_contrastive_prod.sh
# ============================================================

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"

# 비워두면 최신 runs/*_cnn_ddp/cnn/best_model.pth 자동 사용.
CNN_BEST=""

# 현업 classless contrastive 학습 폴더. 콤마로 여러 개 가능.
PROD_TRAIN_DIRS="data/images/prod_train"

PROD_EPOCHS=5
CL_BATCH=64

cd "$(dirname "$0")/.."

if [[ -z "$CNN_BEST" ]]; then
  CNN_BEST="$(ls -dt runs/*_cnn_ddp/cnn/best_model.pth 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$CNN_BEST" ]]; then
  echo "[ERR] CNN_BEST is empty and no runs/*_cnn_ddp/cnn/best_model.pth found" >&2
  exit 1
fi

cmd=(
  python -u scripts/train_contrastive_ddp.py
  --backbone "$CNN_BEST"
  --train-dirs "$PROD_TRAIN_DIRS"
  --no-eval
  --epochs "$PROD_EPOCHS"
  --batch "$CL_BATCH"
)

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ${cmd[*]}"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${cmd[@]}"
