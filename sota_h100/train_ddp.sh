#!/usr/bin/env bash
# Multi-GPU (true DDP) SOTA seed-sweep training. No options.
# Just set the GPUs you want (or none = all GPUs):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash sota_h100/train_ddp.sh
#   bash sota_h100/train_ddp.sh                     # uses every GPU
# Data auto-generated if missing. Result -> outputs/sota_h100_seedsweep_ddp_<TS>/RESULTS.md
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash sota_h100/run_seed_sweep_ddp.sh
