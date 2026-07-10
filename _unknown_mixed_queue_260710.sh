#!/usr/bin/env bash
# Unknown hard follow-up queue:
# 1) defect-aware strict-novel adaptation: Normal + known defects, score unseen defects only.
# 2) production mixed adaptation: all unlabeled classes, score field-mixed pool.
set -u
cd "$(dirname "$0")"

EMB="result_grouping/_unknown_mixed260710/embeddings"
LOG="_unknown_mixed_queue_260710.log"
RES="docs/paper/UNKNOWN_MIXED_QUEUE_260710.md"
EV="data/images/unknown_eval100"
DA_TR="data/images/unknown_train_defectaware_260710"
ALL_TR="data/images/unknown_train_all"
EPOCHS=10

DA_EXCL="Normal,Random,R,Center_bank_boundary,Center_scratch,Donut_bank_boundary,Donut_fork,Edge-Ring_bank_boundary,Edge-Ring_scratch,Edge-Top_fork,Full_scratch,ParallelScratches,RingDots"
FIELD_EXCL="Normal,Random,R"

mkdir -p "$EMB" "$(dirname "$RES")"
export PYTHONIOENCODING=utf-8 TF_ENABLE_ONEDNN_OPTS=0 HF_HUB_OFFLINE=1

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

CONFIGS=(
  "unkda_base|--method simclr|$DA_TR|$DA_EXCL|defect-aware strict novel|8"
  "unkda_nv090|--method simclr --nv-filter 0.90|$DA_TR|$DA_EXCL|defect-aware strict novel + NV 0.90|8"
  "unkda_nv095|--method simclr --nv-filter 0.95|$DA_TR|$DA_EXCL|defect-aware strict novel + NV|8"
  "unkda_nv098|--method simclr --nv-filter 0.98|$DA_TR|$DA_EXCL|defect-aware strict novel + NV 0.98|8"
  "unkda_q4k|--method simclr --use-queue --queue-size 4096|$DA_TR|$DA_EXCL|defect-aware strict novel + queue4096|8"
  "unkda_adapter_frozen|--method simclr --head adapter --lr-bb 0|$DA_TR|$DA_EXCL|defect-aware strict novel + adapter-only|8"
  "unkda_fcmae|--method simclr --timm convnextv2_base.fcmae_ft_in22k_in1k_384|$DA_TR|$DA_EXCL|defect-aware strict novel, FCMAE init|8"
  "unkall_base|--method simclr|$ALL_TR|$FIELD_EXCL|field mixed all-unlabeled|8"
)

score_tag(){
  local tag="$1"
  local excl="$2"
  for ep in 1 2 3 4 6 8 10; do
    local p="$EMB/${tag}_ep${ep}.npy"
    if [ -f "$p" ]; then
      echo "#### ${tag} ep${ep} backbone" >> "$RES"
      CUDA_VISIBLE_DEVICES="" python _score_umapfree.py "$p" --skip-umap --pool "$EV" --exclude-classes "$excl" 2>>"$LOG" \
        | grep -E "finch_p1|finch_p2|louvain_res6|hdbscan_raw" \
        | sed "s/^/${tag} ep${ep} F /" >> "$RES"
    fi
    local pz="$EMB/${tag}_ep${ep}_proj.npy"
    if [ -f "$pz" ] && { [ "$ep" = 3 ] || [ "$ep" = 6 ] || [ "$ep" = 8 ] || [ "$ep" = 10 ]; }; then
      echo "#### ${tag} ep${ep} projection" >> "$RES"
      CUDA_VISIBLE_DEVICES="" python _score_umapfree.py "$pz" --skip-umap --pool "$EV" --exclude-classes "$excl" 2>>"$LOG" \
        | grep -E "finch_p1|finch_p2|louvain_res6|hdbscan_raw" \
        | sed "s/^/${tag} ep${ep} Z /" >> "$RES"
    fi
  done
  echo "" >> "$RES"
}

group_ep(){
  local tag="$1"
  local ep="$2"
  local p="$EMB/${tag}_ep${ep}.npy"
  [ -f "$p" ] || return 0
  local out="result_grouping/_production_review_260710/${tag}_ep${ep}_finch_p2"
  python scripts/group_saved_embeddings.py \
    --embedding "$p" \
    --pool "$EV" \
    --out-dir "$out" \
    --method finch_p2 \
    --reps 10 \
    --background-classes "Normal,Random,R" \
    --exclude-background-min-count 20 \
    --exclude-background-min-ratio 0.5 \
    >> "$LOG" 2>&1 || true
}

if [ ! -f "$RES" ]; then
  {
    echo "# Unknown Mixed / Defect-Aware Queue 260710"
    echo ""
    echo "- eval: D:\\project\\unknown-contrastive\\data\\images\\unknown_eval100"
    echo "- defect-aware train: D:\\project\\unknown-contrastive\\data\\images\\unknown_train_defectaware_260710"
    echo "- all-unlabeled train: D:\\project\\unknown-contrastive\\data\\images\\unknown_train_all"
    echo "- embeddings: D:\\project\\unknown-contrastive\\result_grouping\\_unknown_mixed260710\\embeddings"
    echo "- log: D:\\project\\unknown-contrastive\\_unknown_mixed_queue_260710.log"
    echo ""
  } > "$RES"
fi

say "=== unknown mixed queue start (${#CONFIGS[@]} configs x ${EPOCHS} epochs) ==="
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag flags tr excl note batch <<< "$c"
  if [ -f "$EMB/${tag}_ep${EPOCHS}.npy" ]; then
    if grep -q "^#### ${tag} ep${EPOCHS} backbone" "$RES" 2>/dev/null; then
      say "skip $tag (ep${EPOCHS} and score rows exist)"
    else
      say "skip training $tag (ep${EPOCHS} exists) — scoring/grouping only"
      score_tag "$tag" "$excl"
      group_ep "$tag" 10
    fi
    continue
  fi
  say ">>> $tag : $flags train=$tr ($note, batch=$batch)"
  echo "### ${tag} (${note})" >> "$RES"
  python -u _ssl_methods.py $flags --epochs "$EPOCHS" --batch "$batch" --temp 0.05 \
    --ckpt-every 100 --train-dir "$tr" --eval-dir "$EV" --out-dir "$EMB" --tag "$tag" \
    >> "$LOG" 2>&1
  rc=$?
  say "<<< $tag exit=$rc — scoring available epochs"
  score_tag "$tag" "$excl"
  group_ep "$tag" 3
  group_ep "$tag" 10
  if [ "$rc" -ne 0 ]; then
    say "STOP: $tag failed; leaving remaining configs untouched"
    exit "$rc"
  fi
done

say "=== unknown mixed queue done ==="
echo "[OUT] D:\\project\\unknown-contrastive\\result_grouping\\_unknown_mixed260710\\embeddings" | tee -a "$LOG"
