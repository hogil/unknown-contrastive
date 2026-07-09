#!/usr/bin/env bash
# 값 스윕: NeCo / NV-filter / MoCo 를 여러 값으로 (사용자 "여러 값으로 더 해봐라").
# q4k(queue4096) baseline 위 additive. mixed29 트랙. 1ep (ep1 정점 패턴), per-config skip-existing.
# GPU 비면 자동 시작 (현 lg 스윕 뒤). kill 당해도 완료 config 는 .npy 로 생존 + 재개.
set -u
cd "$(dirname "$0")"
EMB=result_grouping/_field_mixed29/embeddings
LOG=_value_sweep.log
RES=docs/paper/VALUE_SWEEP_NECO_NV_MOCO.md
TR=data/images/wm811k_train_pool_v1
EV=data/images/mixedwm38_pool_mixed29
export PYTHONIOENCODING=utf-8 TF_ENABLE_ONEDNN_OPTS=0 HF_HUB_OFFLINE=1

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# tag|extra-flags|batch
CONFIGS=(
  "vn_neco001|--neco 0.01|4"
  "vn_neco002|--neco 0.02|4"
  "vn_neco003|--neco 0.03|4"
  "vn_neco010|--neco 0.10|4"
  "vn_neco050|--neco 0.50|4"
  "vn_nv085|--nv-filter 0.85|8"
  "vn_nv090|--nv-filter 0.90|8"
  "vn_nv093|--nv-filter 0.93|8"
  "vn_nv096|--nv-filter 0.96|8"
  "vn_moco95|--method moco --m 0.95|8"
  "vn_moco97|--method moco --m 0.97|8"
  "vn_moco999|--method moco --m 0.999|8"
)

# ── GPU 비길 대기: 25분간 새 .npy 안 나오면 idle 로 판단 ──
wait_free(){
  while true; do
    recent=$(find "$EMB" -name '*.npy' -mmin -25 2>/dev/null | head -1)
    [ -z "$recent" ] && break
    say "GPU busy (최근 .npy: $(basename "$recent")) — 3분 대기"
    sleep 180
  done
  say "GPU idle — 값 스윕 시작"
}

wait_free
say "=== 값 스윕 시작 (${#CONFIGS[@]} configs) ==="

for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag flags batch <<< "$c"
  if [ -f "$EMB/${tag}_ep1.npy" ]; then say "skip $tag (이미 있음)"; continue; fi
  # moco 는 method swap, 나머지는 simclr+queue 위 additive
  case "$flags" in
    *"--method moco"*) base="";;
    *) base="--method simclr";;
  esac
  say ">>> $tag : $flags (batch=$batch)"
  python _ssl_methods.py $base $flags \
      --epochs 1 --batch "$batch" --temp 0.05 --use-queue --queue-size 4096 \
      --train-dir "$TR" --eval-dir "$EV" --out-dir "$EMB" --tag "$tag" \
      >> "$LOG" 2>&1
  if [ -f "$EMB/${tag}_ep1.npy" ]; then
    say "<<< $tag done — 채점"
    python _score_umapfree.py "$EMB/${tag}_ep1.npy" --skip-umap --pool "$EV" \
        >> "$RES" 2>>"$LOG"
    echo "" >> "$RES"
  else
    say "!!! $tag FAILED (no ep1.npy) — 다음으로"
  fi
done
say "=== 값 스윕 완료 ==="
echo "[OUT] $EMB" | tee -a "$LOG"
