#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Production classless contrastive only
#   CNN best_model.pth -> classless production contrastive train
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_contrastive_prod.sh
# ============================================================

# CNN best model. 필요하면 직접 수정.
CNN_BEST="runs/<CNN_RUN>/cnn/best_model.pth"

# 현업 classless contrastive 학습 폴더. 콤마로 여러 개 가능.
PROD_TRAIN_DIRS="data/images/prod_train"

PROD_EPOCHS=5
CL_BATCH=64

cd "$(dirname "$0")/.."

cmd=(
  python -u scripts/train_contrastive_ddp.py
  --backbone "$CNN_BEST"
  --train-dirs "$PROD_TRAIN_DIRS"
  --no-eval
  --epochs "$PROD_EPOCHS"
  --batch "$CL_BATCH"
)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
