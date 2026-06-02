#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Production classless contrastive only
#   CNN best_model.pth -> classless production contrastive train
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_contrastive_prod.sh
# ============================================================

# 폴더/모델 옵션 사용법:
# - CNN_BEST 는 CNN 학습 결과 best_model.pth 경로.
# - PROD_TRAIN_DIRS 는 class 없는 현업 이미지 폴더. 콤마 다중 가능.
# - 상대경로는 프로젝트 루트 기준, 절대경로도 가능.
CNN_BEST="runs/<CNN_RUN>/cnn/best_model.pth"
PROD_TRAIN_DIRS="data/images/prod_train"

cd "$(dirname "$0")/.."

# python -u: print/log 를 버퍼링하지 않고 바로 출력. 학습 동작에는 영향 없음.
cmd=(
  python -u scripts/train_contrastive_ddp.py
  --backbone "$CNN_BEST"
  --train-dirs "$PROD_TRAIN_DIRS"
  --no-eval
)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
