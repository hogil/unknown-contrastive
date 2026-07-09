#!/usr/bin/env bash
# 누적 ablation (사용자: "계속 옵션을 더하고 추가해가면서" + "10 까지는해야지").
# frag 챔피언 local-grid 0.15 위에 부품 누적. ★ epochs=10 (수렴), 각 epoch 저장 → 최적 epoch 선택.
# kill 나도 per-epoch .npy 생존 + skip-existing(ep10) 재개. batch 4 (local=patch).
set -u
cd "$(dirname "$0")"
EMB=result_grouping/_field_mixed29/embeddings
LOG=_cumulative_sweep.log
RES=docs/paper/CUMULATIVE_FROZEN_BASE.md
TR=data/images/wm811k_train_pool_v1
EV=data/images/mixedwm38_pool_mixed29
EPOCHS=10
export PYTHONIOENCODING=utf-8 TF_ENABLE_ONEDNN_OPTS=0 HF_HUB_OFFLINE=1

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# tag|cumulative-flags|batch  (각 행 = 직전 + 1 부품, local 0.15 고정 base)
CONFIGS=(
  "cum10_lg|--local 0.15|4"
  "cum10_lg_nv|--local 0.15 --nv-filter 0.90|4"
  "cum10_lg_sn|--local 0.15 --ls 0.02 --ls-topk 20|4"
  "cum10_lg_ls|--local 0.15 --ls 0.02|4"
)

wait_free(){
  for i in $(seq 1 15); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ "${mem:-99999}" -lt 11000 ]; then say "GPU free (${mem}MiB) — 10ep 누적 시작"; return; fi
    say "GPU busy (${mem}MiB) — 2분 대기"; sleep 120
  done
  say "max-wait — 그래도 시작"
}

wait_free
say "=== 10-epoch 누적 스윕 시작 (${#CONFIGS[@]} configs × ${EPOCHS}ep) ==="
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag flags batch <<< "$c"
  if [ -f "$EMB/${tag}_ep${EPOCHS}.npy" ]; then say "skip $tag (ep${EPOCHS} 있음)"; continue; fi
  say ">>> $tag : $flags (${EPOCHS}ep, batch=$batch)"
  python _ssl_methods.py --method simclr $flags \
      --epochs "$EPOCHS" --batch "$batch" --temp 0.05 --use-queue --queue-size 4096 \
      --ckpt-every 50 \
      --train-dir "$TR" --eval-dir "$EV" --out-dir "$EMB" --tag "$tag" >> "$LOG" 2>&1
  say "<<< $tag 학습종료 — epoch 궤적 채점"
  echo "### $tag ($flags) — 10ep 궤적" >> "$RES"
  for ep in 1 3 5 7 10; do
    [ -f "$EMB/${tag}_ep${ep}.npy" ] && \
      python _score_umapfree.py "$EMB/${tag}_ep${ep}.npy" --skip-umap --pool "$EV" 2>>"$LOG" \
        | grep -E "finch_p1" | sed "s/^/ep${ep}: /" >> "$RES"
  done
  echo "" >> "$RES"
done
say "=== 10-epoch 누적 스윕 완료 ==="
echo "[OUT] $EMB" | tee -a "$LOG"
