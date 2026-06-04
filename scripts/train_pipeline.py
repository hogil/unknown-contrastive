#!/usr/bin/env python3
"""한방에 학습: CNN backbone (Split A) → Contrastive (Split B) → tier1 metric.

사용법:
    python scripts/train_pipeline.py
        → runs/<TS>_pipeline/cnn/ + contrastive/ + metrics.json

핵심: CNN backbone 학습 끝나면 그 best_model.pth 를 contrastive 의 backbone init 으로 사용.
"""
from __future__ import annotations

# ===================================================================
# === CONFIG (실행 시 이 부분만 수정) ===
# ===================================================================
# ★ Data 폴더 분리 (사용자 명시 260527 — 이미지 생성때부터 따로) ★
#   CNN_DATA_DIR  : ImageFolder (Split A class subdir, supervised)
#   CL_TRAIN_DIR  : flat (Split B wafer 80%, class label hidden)
#   CL_EVAL_DIR   : ImageFolder (Split B wafer 20%, class for metric)
# 먼저 scripts/_split_data.py 실행해 폴더 생성 필요.
CNN_DATA_DIR          = "E:/data/images/cnn_train"        # ★ 절대규: 모든 이미지 E:/data/images/
CL_TRAIN_DIR          = "E:/data/images/contrastive_train"  # ★ 절대규
CL_EVAL_DIR           = "E:/data/images/contrastive_eval"   # ★ 절대규

CNN_ACTIVE_YAML       = None        # CNN_DATA_DIR 이 이미 split 됐으면 None
CONTRASTIVE_ACTIVE_YAML = None
EXCLUDE_CLASSES       = {"classification", "classification_chips"}

WEIGHTS_DIR           = "weights"
OUTPUT_ROOT           = "runs"
TAG                   = "pipeline"

BACKBONE              = "convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG_SIZE              = 384

# ===== CNN stage =====
CNN_EPOCHS            = 30
CNN_BATCH             = 16
CNN_NUM_WORKERS       = 4
CNN_LR_BACKBONE       = 2e-5
CNN_LR_HEAD           = 2e-4
CNN_WEIGHT_DECAY      = 0.01
CNN_GRAD_CLIP         = 1.0
CNN_WARMUP_EPOCHS     = 5
CNN_LABEL_SMOOTHING   = 0.02
CNN_EARLY_STOP_PATIENCE = 7
CNN_USE_EMA           = False
CNN_USE_AMP           = False
CNN_STOCHASTIC_DEPTH  = 0.0
CNN_SPLIT_RATIOS      = (0.8, 0.1, 0.1)

# ===== Contrastive stage =====
CL_PROJ_DIM           = 128
CL_BATCH              = 8
CL_NUM_WORKERS        = 4
CL_EPOCHS             = 5
CL_WARMUP_EPOCHS      = 1
CL_TRAIN_SAMPLING_RATIO = 0.25
CL_LR_HEAD            = 1e-3
CL_WEIGHT_DECAY       = 1e-6
CL_NCE_TEMP           = 0.07
CL_GRAD_CLIP          = 1.0
CL_LABEL_SMOOTHING    = 0.02
CL_FREEZE_BACKBONE    = True
CL_USE_QUEUE          = True
CL_QUEUE_SIZE         = 4096
CL_IGNORE_NEG_SIM     = 0.90
CL_USE_LOCAL          = False
CL_NECO_WEIGHT        = 0.2
CL_NECO_TAU           = 0.1
CL_USE_EMA            = False
CL_USE_AMP            = False
CL_PER_CLASS_CAP      = 500
CL_NORMAL_CAP         = 2000

# HDBSCAN eval
MIN_CLUSTER_SIZE      = 12
MIN_SAMPLES           = 15
CLUSTER_SELECTION_METHOD = "eom"
CLUSTER_SELECTION_EPSILON = 0.0

SEED                  = 42
# ===================================================================

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    ensure_backbone_weights,
    log_stage_metric,
    make_run_dir,
    snapshot_config,
    system_info,
)


def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    seed_all(SEED)
    run_dir = make_run_dir(OUTPUT_ROOT, TAG)
    print(f"[run_dir] {run_dir.resolve()}")

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
    cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
    snapshot_config(run_dir, cfg)
    system_info(run_dir)

    backbone_path = ensure_backbone_weights(WEIGHTS_DIR, BACKBONE)
    print(f"[backbone] {backbone_path}")

    # =========================================================
    # Stage 1: CNN backbone (Split A) — import train_cnn machinery
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 1: CNN backbone supervised (Split A)")
    print("=" * 60)
    import train_cnn as tc

    # config injection
    tc.DATA_DIR             = CNN_DATA_DIR
    tc.ACTIVE_CLASSES_YAML  = CNN_ACTIVE_YAML
    tc.EXCLUDE_CLASSES      = EXCLUDE_CLASSES
    tc.WEIGHTS_DIR          = WEIGHTS_DIR
    tc.OUTPUT_ROOT          = str(run_dir.parent)
    tc.TAG                  = f"{run_dir.name}_cnn"  # nested
    tc.BACKBONE             = BACKBONE
    tc.IMG_SIZE             = IMG_SIZE
    tc.BATCH                = CNN_BATCH
    tc.NUM_WORKERS          = CNN_NUM_WORKERS
    tc.EPOCHS               = CNN_EPOCHS
    tc.WARMUP_EPOCHS        = CNN_WARMUP_EPOCHS
    tc.LR_BACKBONE          = CNN_LR_BACKBONE
    tc.LR_HEAD              = CNN_LR_HEAD
    tc.WEIGHT_DECAY         = CNN_WEIGHT_DECAY
    tc.GRAD_CLIP            = CNN_GRAD_CLIP
    tc.LABEL_SMOOTHING      = CNN_LABEL_SMOOTHING
    tc.EARLY_STOP_PATIENCE  = CNN_EARLY_STOP_PATIENCE
    tc.USE_EMA              = CNN_USE_EMA
    tc.USE_AMP              = CNN_USE_AMP
    tc.STOCHASTIC_DEPTH     = CNN_STOCHASTIC_DEPTH
    tc.SPLIT_RATIOS         = CNN_SPLIT_RATIOS
    tc.SEED                 = SEED

    cnn_run_dir = tc.main()
    cnn_best = cnn_run_dir / "cnn" / "best_model.pth"
    if not cnn_best.exists():
        print(f"[FATAL] CNN best_model.pth not found at {cnn_best}")
        return run_dir

    # move CNN artifacts under our pipeline run_dir
    import shutil
    target_cnn = run_dir / "cnn"
    if cnn_run_dir != run_dir:
        if target_cnn.exists():
            shutil.rmtree(target_cnn)
        shutil.copytree(cnn_run_dir / "cnn", target_cnn)
        # Also bring stage metrics
        if (cnn_run_dir / "metrics.json").exists():
            cnn_metrics = json.loads((cnn_run_dir / "metrics.json").read_text(encoding="utf-8"))
            our = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            our["stages"].extend(cnn_metrics["stages"])
            (run_dir / "metrics.json").write_text(json.dumps(our, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    cnn_best_in_pipeline = run_dir / "cnn" / "best_model.pth"
    print(f"[stage 1] CNN best → {cnn_best_in_pipeline}")

    log_stage_metric(run_dir, "stage_1_cnn_done", {
        "best_path": str(cnn_best_in_pipeline.resolve()),
    })

    # =========================================================
    # Stage 2: Contrastive (Split B) using CNN backbone
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Contrastive head (Split B) on top of CNN backbone")
    print("=" * 60)
    import train_contrastive as tcl

    tcl.TRAIN_DATA_DIR        = CL_TRAIN_DIR
    tcl.EVAL_DATA_DIR         = CL_EVAL_DIR
    tcl.ACTIVE_CLASSES_YAML   = CONTRASTIVE_ACTIVE_YAML
    tcl.EXCLUDE_CLASSES       = EXCLUDE_CLASSES
    tcl.CNN_RUN_DIR           = None
    tcl.BACKBONE_CKPT         = str(cnn_best_in_pipeline)
    tcl.WEIGHTS_DIR           = WEIGHTS_DIR
    tcl.BACKBONE              = BACKBONE
    tcl.FREEZE_BACKBONE       = CL_FREEZE_BACKBONE
    tcl.OUTPUT_ROOT           = str(run_dir.parent)
    tcl.TAG                   = f"{run_dir.name}_contrastive"
    tcl.IMG_SIZE              = IMG_SIZE
    tcl.PROJ_DIM              = CL_PROJ_DIM
    tcl.BATCH                 = CL_BATCH
    tcl.NUM_WORKERS           = CL_NUM_WORKERS
    tcl.EPOCHS                = CL_EPOCHS
    tcl.WARMUP_EPOCHS         = CL_WARMUP_EPOCHS
    tcl.TRAIN_SAMPLING_RATIO  = CL_TRAIN_SAMPLING_RATIO
    tcl.LR_HEAD               = CL_LR_HEAD
    tcl.WEIGHT_DECAY          = CL_WEIGHT_DECAY
    tcl.NCE_TEMP              = CL_NCE_TEMP
    tcl.GRAD_CLIP             = CL_GRAD_CLIP
    tcl.LABEL_SMOOTHING       = CL_LABEL_SMOOTHING
    tcl.USE_QUEUE             = CL_USE_QUEUE
    tcl.QUEUE_SIZE            = CL_QUEUE_SIZE
    tcl.IGNORE_NEG_SIM        = CL_IGNORE_NEG_SIM
    tcl.USE_LOCAL             = CL_USE_LOCAL
    tcl.NECO_WEIGHT           = CL_NECO_WEIGHT
    tcl.NECO_TAU              = CL_NECO_TAU
    tcl.USE_EMA               = CL_USE_EMA
    tcl.USE_AMP               = CL_USE_AMP
    tcl.MIN_CLUSTER_SIZE      = MIN_CLUSTER_SIZE
    tcl.MIN_SAMPLES           = MIN_SAMPLES
    tcl.CLUSTER_SELECTION_METHOD = CLUSTER_SELECTION_METHOD
    tcl.CLUSTER_SELECTION_EPSILON = CLUSTER_SELECTION_EPSILON
    tcl.PER_CLASS_CAP         = CL_PER_CLASS_CAP
    tcl.NORMAL_CAP            = CL_NORMAL_CAP
    tcl.SEED                  = SEED

    cl_run_dir = tcl.main()

    # move contrastive artifacts
    target_cl = run_dir / "contrastive"
    if cl_run_dir != run_dir:
        if target_cl.exists():
            shutil.rmtree(target_cl)
        shutil.copytree(cl_run_dir / "contrastive", target_cl)
        if (cl_run_dir / "metrics.json").exists():
            cl_metrics = json.loads((cl_run_dir / "metrics.json").read_text(encoding="utf-8"))
            our = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            our["stages"].extend(cl_metrics["stages"])
            (run_dir / "metrics.json").write_text(json.dumps(our, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(f"[stage 2] Contrastive → {run_dir / 'contrastive'}")

    log_stage_metric(run_dir, "pipeline_done", {
        "cnn_best": str(cnn_best_in_pipeline.resolve()),
        "contrastive_best": str((run_dir / "contrastive" / "best_model.pt").resolve()),
    }, notes="CNN backbone (Split A) → Contrastive (Split B) sequential, disjoint class sets")

    print(f"\n[OUT] {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    main()
