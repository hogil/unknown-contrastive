#!/usr/bin/env bash
# NV-Retriever first-pass sweep on the currently available hard unknown split.
# Train: Normal only. Eval: unknown_eval100. Backbone: default DINOv3 no-CNN.
set -u
cd "$(dirname "$0")"

EMB="result_grouping/_unknown_hard_nv260710/embeddings"
LOG="_unknown_hard_nv_sweep_260710.log"
RES="docs/paper/UNKNOWN_HARD_NV_SWEEP_260710.md"
TR="data/images/unknown_train_normal"
EV="data/images/unknown_eval100"
EPOCHS=10

mkdir -p "$EMB" "$(dirname "$RES")"
export PYTHONIOENCODING=utf-8 TF_ENABLE_ONEDNN_OPTS=0 HF_HUB_OFFLINE=1

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

CONFIGS=(
  "unk_nv085|--method simclr --nv-filter 0.85|8"
  "unk_nv090|--method simclr --nv-filter 0.90|8"
  "unk_nv095|--method simclr --nv-filter 0.95|8"
  "unk_nv097|--method simclr --nv-filter 0.97|8"
  "unk_nv099|--method simclr --nv-filter 0.99|8"
)

score_tag(){
  local tag="$1"
  for ep in 1 2 3 4 6 8 10; do
    local p="$EMB/${tag}_ep${ep}.npy"
    if [ -f "$p" ]; then
      echo "#### ${tag} ep${ep} backbone" >> "$RES"
      CUDA_VISIBLE_DEVICES="" python _score_umapfree.py "$p" --skip-umap --pool "$EV" 2>>"$LOG" \
        | grep -E "finch_p1|louvain_res6|hdbscan_raw" \
        | sed "s/^/${tag} ep${ep} F /" >> "$RES"
    fi

    local pz="$EMB/${tag}_ep${ep}_proj.npy"
    if [ -f "$pz" ] && { [ "$ep" = 3 ] || [ "$ep" = 6 ] || [ "$ep" = 8 ] || [ "$ep" = 10 ]; }; then
      echo "#### ${tag} ep${ep} projection" >> "$RES"
      CUDA_VISIBLE_DEVICES="" python _score_umapfree.py "$pz" --skip-umap --pool "$EV" 2>>"$LOG" \
        | grep -E "finch_p1|louvain_res6|hdbscan_raw" \
        | sed "s/^/${tag} ep${ep} Z /" >> "$RES"
    fi
  done
  echo "" >> "$RES"
}

if [ ! -f "$RES" ]; then
  {
    echo "# Unknown Hard NV-Retriever Sweep 260710"
    echo ""
    echo "- train: D:\\project\\unknown-contrastive\\data\\images\\unknown_train_normal"
    echo "- eval: D:\\project\\unknown-contrastive\\data\\images\\unknown_eval100"
    echo "- embeddings: D:\\project\\unknown-contrastive\\result_grouping\\_unknown_hard_nv260710\\embeddings"
    echo "- log: D:\\project\\unknown-contrastive\\_unknown_hard_nv_sweep_260710.log"
    echo "- protocol: DINOv3 no-CNN, SimCLR + one NV-Retriever threshold, 10 epochs, score backbone f and projection z."
    echo ""
  } > "$RES"
fi

say "=== unknown hard NV sweep start (${#CONFIGS[@]} configs x ${EPOCHS} epochs) ==="
say "train=$TR eval=$EV out=$EMB"

for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag flags batch <<< "$c"
  if [ -f "$EMB/${tag}_ep${EPOCHS}.npy" ]; then
    say "skip $tag (ep${EPOCHS} exists) — scoring"
    score_tag "$tag"
    continue
  fi

  say ">>> $tag : $flags (batch=$batch)"
  echo "### ${tag} (${flags}, DINOv3 no-CNN, unknown hard)" >> "$RES"
  python -u _ssl_methods.py $flags --epochs "$EPOCHS" --batch "$batch" --temp 0.05 \
    --ckpt-every 100 --train-dir "$TR" --eval-dir "$EV" --out-dir "$EMB" --tag "$tag" \
    >> "$LOG" 2>&1
  rc=$?
  say "<<< $tag exit=$rc — scoring available epochs"
  score_tag "$tag"
  if [ "$rc" -ne 0 ]; then
    say "STOP: $tag failed; leaving remaining configs untouched"
    exit "$rc"
  fi
done

say "=== unknown hard NV sweep done ==="
echo "[OUT] D:\\project\\unknown-contrastive\\result_grouping\\_unknown_hard_nv260710\\embeddings" | tee -a "$LOG"
