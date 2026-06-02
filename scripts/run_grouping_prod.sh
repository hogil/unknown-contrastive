#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Production grouping only
#   contrastive best_model.pt -> embed + HDBSCAN grouping
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_grouping_prod.sh
# ============================================================

# 폴더/모델 옵션 사용법:
# - MODEL 은 runs/<contrastive_run> 또는 contrastive/best_model.pt 경로.
# - PROD_PRED_DIRS 는 grouping 할 현업 이미지 폴더. 콤마 다중 가능.
# - 상대경로는 프로젝트 루트 기준, 절대경로도 가능.
MODEL="runs/<CONTRASTIVE_RUN>"
PROD_PRED_DIRS="data/images/prod_pred"

cd "$(dirname "$0")/.."

# python -u: print/log 를 버퍼링하지 않고 바로 출력. 학습 동작에는 영향 없음.
cmd=(
  python -u scripts/predict_grouping_prod.py
  --model "$MODEL"
  --image-roots "$PROD_PRED_DIRS"
)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
