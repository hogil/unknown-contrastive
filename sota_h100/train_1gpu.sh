#!/usr/bin/env bash
# Single-GPU SOTA seed-sweep training. No options.
#   bash sota_h100/train_1gpu.sh                    # GPU 0
#   CUDA_VISIBLE_DEVICES=2 bash sota_h100/train_1gpu.sh
# Data auto-generated if missing. Result -> outputs/sota_h100_seedsweep_1gpu_<TS>/RESULTS.md
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash sota_h100/run_seed_sweep_1gpu.sh
