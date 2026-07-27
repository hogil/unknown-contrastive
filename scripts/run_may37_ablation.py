#!/usr/bin/env python3
"""Run one historical-equivalent May B0-B5 contrastive ablation cell.

The original May anchor and trainer artifact are unavailable.  This runner
locks the documented B0-B5 settings on the reconstructed 2,260-image anchor,
then delegates per-epoch scoring to eval_may37_checkpoints.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
ANCHOR = ROOT / "data" / "pools" / "anchor_avg30_repro.json"
TAPT_CKPT = Path(r"D:\project\known-cnn\models\iter116J_frozen\best_model.pth")
FCMAE_CKPT = ROOT / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"

CELLS = {
    "FROZEN": {"local": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B0": {"local": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B1": {"local": 0.5, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B2": {"local": 1.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B3": {"local": 1.0, "queue": True, "ignore": 1.0, "neco": 0.0},
    "B4": {"local": 1.0, "queue": True, "ignore": 0.72, "neco": 0.0},
    "B5": {"local": 1.0, "queue": True, "ignore": 0.72, "neco": 0.2},
    # ★ 260721 플랜: Local OFF May-best + leave-one-out (additive 누적 대신 최종조합 주변 기여 확인)
    "MAYBEST":    {"local": 0.0, "queue": True,  "ignore": 0.72, "neco": 0.2},   # 플랜 May-best (Global+NeCo0.2+Queue4096+NEG0.72, Local OFF)
    "MB_NONECO":  {"local": 0.0, "queue": True,  "ignore": 0.72, "neco": 0.0},   # -NeCo
    "MB_NOQUEUE": {"local": 0.0, "queue": False, "ignore": 0.72, "neco": 0.2},   # -Queue
    "MB_NONEG":   {"local": 0.0, "queue": True,  "ignore": 1.0,  "neco": 0.2},   # -NEG(ignore off)
}


def load_trainer():
    path = SCRIPT_DIR / "train_contrastive_ddp.py"
    spec = importlib.util.spec_from_file_location("may37_ablation_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure(module, backbone: str, cell: str, output_root: Path) -> None:
    if not ANCHOR.exists():
        raise FileNotFoundError(f"reconstructed anchor is unavailable: {ANCHOR}")
    if backbone == "cnn_tapt":
        if not TAPT_CKPT.exists():
            raise FileNotFoundError(f"TAPT checkpoint is unavailable: {TAPT_CKPT}")
        backbone_ckpt = TAPT_CKPT
    else:
        if not FCMAE_CKPT.exists():
            raise FileNotFoundError(f"FCMAE checkpoint is unavailable: {FCMAE_CKPT}")
        backbone_ckpt = FCMAE_CKPT
    values = CELLS[cell]
    module.TRAIN_DATA_DIR = str(ANCHOR)
    module.EVAL_DATA_DIR = str(ANCHOR)
    # The external evaluator scores every checkpoint under one fixed protocol.
    # Skip the trainer's final-only evaluator to avoid duplicate extraction.
    module.NO_EVAL = True
    module.CNN_RUN_DIR = None
    module.BACKBONE_CKPT = str(backbone_ckpt)
    module.BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    module.FREEZE_BACKBONE = True
    module.BACKBONE_TRAIN_MODE = "frozen"
    module.OUTPUT_ROOT = str(output_root)
    module.TAG = f"may37_{backbone}_{cell.lower()}"
    module.IMG_SIZE = 384
    module.PROJ_DIM = 128
    module.BATCH_PER_GPU = 8
    module.NUM_WORKERS_PER_GPU = 0
    module.EPOCHS = 0 if cell == "FROZEN" else 5
    module.WARMUP_EPOCHS = 1
    module.TRAIN_SAMPLING_RATIO = 1.0
    module.LR_HEAD = 1e-3
    module.LR_BACKBONE = 0.0
    module.LR_MIN = 0.0
    module.LR_BACKBONE_MIN = 0.0
    module.WEIGHT_DECAY = 1e-6
    module.NCE_TEMP = 0.07
    module.LOSS_MODE = "nce"
    module.GRAD_CLIP = 0.0
    module.LABEL_SMOOTHING = 0.0
    module.USE_QUEUE = values["queue"]
    module.QUEUE_SIZE = 4096
    module.USE_MOMENTUM_ENCODER = False
    module.MOMENTUM_ENCODER_M = 0.99
    module.IGNORE_NEG_SIM = values["ignore"]
    module.PSEUDO_POS_WEIGHT = 0.0
    module.PSEUDO_NEG_REMOVE = False
    module.LOCAL_WEIGHT = values["local"]
    module.LOCAL_TAU = 0.07
    module.LOCAL_GRID = 6
    module.LOCAL_WINDOW = 4
    module.NECO_WEIGHT = values["neco"]
    module.NECO_TAU = 0.1
    module.NECO_GRID = 0
    module.INFER_EMBED_MODE = "backbone" if cell == "FROZEN" else "projection"
    module.MIN_CLUSTER_SIZE = 12
    module.MIN_SAMPLES = 3
    module.CLUSTER_SELECTION_METHOD = "eom"
    module.CLUSTER_SELECTION_EPSILON = 0.0
    module.PER_CLASS_CAP = 500
    module.NORMAL_CAP = 2000
    module.EVAL_IGNORE_CLASSES = {"Normal", "Random", "R"}
    module.SEED = 42
    module.SAVE_EPOCH_CKPTS = True
    module.SAVE_EPOCH_EVERY = 1
    module.SAVE_WRONG_IMAGES = False
    module.SAVE_REPRESENTATIVES = False


def run_single_process(module) -> None:
    """Use the DDP trainer's loss loop without Gloo on one Windows GPU."""
    class SingleProcessDDP(torch.nn.Module):
        def __init__(self, wrapped, *args, **kwargs) -> None:
            super().__init__()
            self.module = wrapped

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    class SingleProcessDist:
        @staticmethod
        def broadcast_object_list(*args, **kwargs) -> None:
            return None

        @staticmethod
        def barrier() -> None:
            return None

        @staticmethod
        def is_initialized() -> bool:
            return False

    module.DDP = SingleProcessDDP
    module.dist = SingleProcessDist()
    module.setup_ddp = lambda rank, world_size: None
    module.cleanup_ddp = lambda: None
    module.is_main = lambda rank: rank == 0
    module.all_reduce_avg = lambda metrics, device: metrics
    module.all_gather_concat = lambda tensor, device: tensor
    module.train_worker(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn_tapt", "nocnn"], required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "may37_manifest_reproduction")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=None)          # ★ 260722: seed 오버라이드 (3-seed 확인)
    parser.add_argument("--lr-head", type=float, default=None)     # ★ 260722: head LR 오버라이드 (과적합 안정화 스윕)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # This PyTorch build lacks libuv support; setup_ddp's setdefault is
        # insufficient when a parent shell exports USE_LIBUV=1.
        os.environ["USE_LIBUV"] = "0"
    module = load_trainer()
    configure(module, args.backbone, args.cell, output_root)
    if args.seed is not None:
        module.SEED = args.seed
    if args.lr_head is not None:
        module.LR_HEAD = args.lr_head
    module.TAG = f"{module.TAG}_s{module.SEED}_lr{module.LR_HEAD:g}"   # ★ 태그에 seed/lr 반영 (덮어쓰기 방지)
    made: list[Path] = []
    original_make_run_dir = module.make_run_dir

    def tracked_make_run_dir(root, tag):
        run_dir = original_make_run_dir(root, tag)
        made.append(Path(run_dir))
        return run_dir

    module.make_run_dir = tracked_make_run_dir
    print(f"[START] backbone={args.backbone} cell={args.cell} anchor={ANCHOR}", flush=True)
    run_single_process(module)
    if len(made) != 1:
        raise RuntimeError(f"expected one run directory, got {made}")
    run_dir = made[0].resolve()
    if args.cell == "FROZEN":
        checkpoint_dir = run_dir / "contrastive" / "epoch_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "contrastive" / "best_model.pt", checkpoint_dir / "epoch_000.pt")
    print(f"[RUN_DIR] {run_dir}", flush=True)
    if not args.skip_eval:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "eval_may37_checkpoints.py"),
            "--run-dir", str(run_dir),
            "--eval-root", str(ANCHOR),
            "--batch", "64",
            "--workers", "4",
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
