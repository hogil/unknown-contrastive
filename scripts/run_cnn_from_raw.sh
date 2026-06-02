#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CNN only
#   raw labeled images -> auto split -> CNN DDP train
#
# 서버에서 이 파일 상단 CONFIG 만 수정하고 실행:
#   bash scripts/run_cnn_from_raw.sh
# ============================================================

# 폴더 옵션 사용법:
# - SOURCE_ROOT 는 반드시 <class>/*.png 구조.
# - 상대경로는 프로젝트 루트 기준, 절대경로도 가능.
SOURCE_ROOT="data/images/unknown"
CNN_DATA_DIR="data/images/cnn_train"

cd "$(dirname "$0")/.."

# python -u: print/log 를 버퍼링하지 않고 바로 출력. 학습 동작에는 영향 없음.
cmd=(
  python -u scripts/train_cnn_ddp.py
  --source-root "$SOURCE_ROOT"
  --data-dir "$CNN_DATA_DIR"
)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
