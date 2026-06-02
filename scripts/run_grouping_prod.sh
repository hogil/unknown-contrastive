#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Production grouping only
#   contrastive best_model.pt -> embed + HDBSCAN grouping
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_grouping_prod.sh
# ============================================================

# 비워두면 최신 contrastive best_model.pt 자동 사용.
# runs/<run_dir> 또는 runs/<run_dir>/contrastive/best_model.pt 둘 다 가능.
MODEL="${MODEL:-}"

# grouping/pred 할 현업 폴더. 콤마로 여러 개 가능.
PROD_PRED_DIRS="${PROD_PRED_DIRS:-data/images/prod_pred}"

GROUPING_BATCH=128
GROUPING_WORKERS=16
GROUPING_REPS_PER_CLUSTER=5
COPY_PNG=0
POOL=0
POOL_NAME="pooled"

cd "$(dirname "$0")/.."

if [[ "$PROD_PRED_DIRS" == "data/images/prod_pred" ]]; then
  cat >&2 <<'EOF'
[ERR] scripts/run_grouping_prod.sh 상단 CONFIG 를 실제 grouping 폴더로 수정하세요.
  PROD_PRED_DIRS="data/images/<현업_grouping_대상_폴더>"

또는 환경변수로 한 번만 지정:
  PROD_PRED_DIRS=data/images/B bash scripts/run_grouping_prod.sh
EOF
  exit 1
fi

if [[ -z "$MODEL" ]]; then
  MODEL="$(ls -dt runs/*_contrastive*_ddp*/contrastive/best_model.pt 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$MODEL" ]]; then
  echo "[ERR] MODEL is empty and no contrastive best_model.pt found" >&2
  exit 1
fi

cmd=(
  python -u scripts/predict_grouping_prod.py
  --model "$MODEL"
  --image-roots "$PROD_PRED_DIRS"
  --batch "$GROUPING_BATCH"
  --workers "$GROUPING_WORKERS"
  --reps-per-cluster "$GROUPING_REPS_PER_CLUSTER"
)

if [[ "$COPY_PNG" == "1" ]]; then
  cmd+=(--copy-png)
fi
if [[ "$POOL" == "1" ]]; then
  cmd+=(--pool --pool-name "$POOL_NAME")
fi

echo "[run] ${cmd[*]}"
"${cmd[@]}"
