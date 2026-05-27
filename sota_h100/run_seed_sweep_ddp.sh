#!/usr/bin/env bash
# Multi-GPU (true DDP) seed sweep of the iter116J single-SOTA recipe (H100).
#
# Each seed trains ONE model across G GPUs via torchrun (torch.distributed env://).
# The trainer's DDP path: per-rank batch = --batch, DistributedSampler, DDP wrap;
# effective global batch = BATCH * GPUS * ACCUM. Keep that = 16 to match the SOTA
# recipe (default GPUS=4 BATCH=4 ACCUM=1 -> 16). Seeds run sequentially; each seed
# uses all G GPUs. Eval (run_stage1) runs single-GPU after each train.
#
# Usage:
#   bash sota_h100/run_seed_sweep_ddp.sh
#   GPUS=8 BATCH=2 ACCUM=1 SEEDS="1 2 3 4" EPOCHS=24 bash sota_h100/run_seed_sweep_ddp.sh
#   IMAGES_ROOT=/data/images bash sota_h100/run_seed_sweep_ddp.sh
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- config (override via env) ----
# GPUs: derive from CUDA_VISIBLE_DEVICES if set (standard: `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_..._ddp.sh`),
# else GPUS env, else 4. torchrun --nproc_per_node = that count; each rank maps LOCAL_RANK -> visible device.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPUS=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
elif [ -n "${GPUS:-}" ]; then
    :
else
    GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')   # auto: all visible GPUs
    [ -z "$GPUS" ] || [ "$GPUS" -lt 1 ] && GPUS=1
fi
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
EPOCHS="${EPOCHS:-24}"
ACCUM="${ACCUM:-1}"
# per-rank batch: default keeps effective (BATCH*GPUS*ACCUM) = 16 to match SOTA recipe.
if [ -z "${BATCH:-}" ]; then
    BATCH=$(( 16 / (GPUS * ACCUM) )); [ "$BATCH" -lt 1 ] && BATCH=1
fi
BACKBONE="${BACKBONE:-convnextv2_base.fcmae_ft_in22k_in1k_384}"
IMG_SIZE="${IMG_SIZE:-384}"
WEIGHTS="${WEIGHTS:-models/${BACKBONE}.pth}"
IMAGES_ROOT="${IMAGES_ROOT:-data/images/sota_h100}"
TRAIN_ROOT="${TRAIN_ROOT:-$IMAGES_ROOT/classification_chips}"
EVAL_SET="${EVAL_SET:-$IMAGES_ROOT/eval_set}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-200}"
EVAL_PER_CLASS="${EVAL_PER_CLASS:-2000}"
EVAL_N_USE="${EVAL_N_USE:-2000}"
GEN_SEED="${GEN_SEED:-20260527}"
MASTER_PORT="${MASTER_PORT:-29577}"
EFF=$((BATCH * GPUS * ACCUM))
TS=$(date +%Y%m%d_%H%M%S)
SWEEP_ROOT="${SWEEP_ROOT:-outputs/sota_h100_seedsweep_ddp_$TS}"
mkdir -p "$SWEEP_ROOT"
LOG="$SWEEP_ROOT/sweep.log"

log() { echo "$(date '+%H:%M:%S') [ddp] $*" | tee -a "$LOG"; }

log "SWEEP_ROOT=$SWEEP_ROOT seeds=[$SEEDS] epochs=$EPOCHS GPUS=$GPUS per-rank-batch=$BATCH accum=$ACCUM eff_batch=$EFF"
[ "$EFF" -ne 16 ] && log "WARN effective batch $EFF != 16 (SOTA recipe = 16); results may differ"

# ---- 1. data (single process, before DDP) ----
if [ ! -d "$TRAIN_ROOT/bank_boundary" ]; then
    log "gen train pool -> $TRAIN_ROOT (per-class $TRAIN_PER_CLASS)"
    python -u -m sota_h100.gen_data --mode train --per-class "$TRAIN_PER_CLASS" \
        --out "$TRAIN_ROOT" --seed "$GEN_SEED" 2>&1 | tee -a "$LOG"
else
    log "train pool exists: $TRAIN_ROOT (skip gen)"
fi
if [ ! -d "$EVAL_SET/Normal" ]; then
    log "gen eval set -> $EVAL_SET (per-class $EVAL_PER_CLASS)"
    python -u -m sota_h100.gen_data --mode eval --per-class "$EVAL_PER_CLASS" \
        --out "$EVAL_SET" --seed "$GEN_SEED" 2>&1 | tee -a "$LOG"
else
    log "eval set exists: $EVAL_SET (skip gen)"
fi
if [ ! -f "$WEIGHTS" ]; then
    log "ERROR missing backbone weights: $WEIGHTS"; exit 2
fi

# ---- 2-3. seed loop: DDP train (all GPUs) + single-GPU eval ----
for seed in $SEEDS; do
    OUT="$SWEEP_ROOT/seed_$seed"
    mkdir -p "$OUT"
    SLOG="$OUT/run.log"
    log "=== seed $seed: DDP TRAIN on $GPUS GPUs ==="
    SEED=$seed torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" \
        --master_port="$MASTER_PORT" \
        -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs "$EPOCHS" --batch "$BATCH" --accum "$ACCUM" --seed "$seed" \
        --num-workers 4 --lr 1e-4 --no-normal --val-criterion margin_max \
        --multi-val-set "$EVAL_SET" --multi-val-n-per-class 50 \
        --data-root "$TRAIN_ROOT" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size "$IMG_SIZE" --backbone-timm-weights "$WEIGHTS" \
        --out-root "$OUT" --tag "seed${seed}_ddp${GPUS}" > "$SLOG" 2>&1
    if ! grep -q "DONE" "$SLOG"; then
        log "seed $seed DDP TRAIN FAIL (tail):"; tail -8 "$SLOG" | tee -a "$LOG"; continue
    fi
    MODEL=$(find "$OUT" -name "best_model.pth" 2>/dev/null | head -1)
    [ -z "$MODEL" ] && { log "seed $seed: no best_model.pth"; continue; }
    log "=== seed $seed: EVAL I10,I13 (single GPU) ==="
    CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel.run_stage1 \
        --model "$MODEL" --eval-set "$EVAL_SET" --out-root "$OUT/eval" \
        --variants I10,I13 --n-per-class "$EVAL_N_USE" \
        --batch-size 32 --num-workers 0 --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "$SLOG" 2>&1
    RES=$(grep "eval_bit_F1" "$SLOG" | tail -1)
    log "seed $seed: $RES"
done

# ---- 4. report ----
log "=== make report ==="
python -u -X utf8 -m sota_h100.make_report --sweep-root "$SWEEP_ROOT" 2>&1 | tee -a "$LOG"
log "DONE -> $SWEEP_ROOT/RESULTS.md"
echo "[OUT] $(cd "$SWEEP_ROOT" && pwd)"
