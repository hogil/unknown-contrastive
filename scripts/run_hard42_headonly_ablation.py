#!/usr/bin/env python3
"""Run one hard-42 frozen-backbone head-only contrastive ablation cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from hard42_headonly_common import (  # noqa: E402
    DEV_ROOT,
    HOLDOUT_ROOT,
    STRICT_EXCLUDED,
    TRAIN_ROOT,
    audit_backbone_checkpoint,
    checkpoint_for,
    install_head_variant,
    inventory,
    load_trainer,
    sha256_file,
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def recipe_from_args(args: argparse.Namespace) -> dict:
    return {
        "head": args.head,
        "adapter_dim": args.adapter_dim,
        "projection_dim": args.projection_dim,
        "temperature": args.temperature,
        "queue_size": args.queue_size,
        "ignore_neg_sim": args.ignore_neg_sim,
        "local_weight": args.local_weight,
        "neco_weight": args.neco_weight,
        "lr_head": args.lr_head,
        "epochs": 0 if args.frozen else args.epochs,
        "seed": args.seed,
        "batch": args.batch,
    }


def configure(module, args: argparse.Namespace, checkpoint: Path, output_root: Path) -> None:
    recipe = recipe_from_args(args)
    module.TRAIN_DATA_DIR = str(TRAIN_ROOT)
    module.EVAL_DATA_DIR = str(DEV_ROOT)
    module.NO_EVAL = True
    module.CNN_RUN_DIR = None
    module.BACKBONE_CKPT = str(checkpoint)
    module.BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
    module.FREEZE_BACKBONE = True
    module.BACKBONE_TRAIN_MODE = "frozen"
    module.OUTPUT_ROOT = str(output_root)
    module.TAG = safe_name(f"hard42_headonly_{args.backbone}_{args.cell}_s{args.seed}")
    module.EXPERIMENT_BACKBONE = args.backbone
    module.EXPERIMENT_CELL = args.cell
    module.HEAD_KIND = args.head
    module.IMG_SIZE = 384
    module.PROJ_DIM = args.projection_dim
    module.ADAPTER_DIM = args.adapter_dim if args.head.lower().startswith("adapter") else 0
    module.ADAPTER_SCALE = 1.0
    module.SPATIAL_ADAPTER = "none"
    module.BATCH_PER_GPU = args.batch
    module.NUM_WORKERS_PER_GPU = args.workers
    module.EPOCHS = 0 if args.frozen else args.epochs
    module.WARMUP_EPOCHS = min(1, module.EPOCHS)
    module.TRAIN_SAMPLING_RATIO = 1.0
    module.LR_HEAD = args.lr_head
    module.LR_BACKBONE = 0.0
    module.LR_MIN = 0.0
    module.LR_BACKBONE_MIN = 0.0
    module.WEIGHT_DECAY = 1e-6
    module.NCE_TEMP = args.temperature
    module.LOSS_MODE = "nce"
    module.GRAD_CLIP = 0.0
    module.LABEL_SMOOTHING = 0.0
    module.USE_QUEUE = args.queue_size > 0
    module.QUEUE_SIZE = max(1, args.queue_size)
    module.USE_MOMENTUM_ENCODER = False
    module.MOMENTUM_ENCODER_M = 0.99
    module.IGNORE_NEG_SIM = args.ignore_neg_sim
    module.PSEUDO_POS_WEIGHT = 0.0
    module.PSEUDO_NEG_REMOVE = False
    module.LOCAL_WEIGHT = args.local_weight
    module.LOCAL_TAU = 0.07
    module.LOCAL_GRID = 6
    module.LOCAL_WINDOW = 4
    module.NECO_WEIGHT = args.neco_weight
    module.NECO_TAU = 0.1
    module.NECO_GRID = 0
    module.INFER_EMBED_MODE = "backbone" if args.frozen else "projection"
    module.INFER_BACKBONE_WEIGHT = 1.0
    module.INFER_PROJ_WEIGHT = 0.2
    module.PER_CLASS_CAP = 100000
    module.NORMAL_CAP = 100000
    module.EVAL_IGNORE_CLASSES = set(STRICT_EXCLUDED)
    module.SEED = args.seed
    module.SAVE_EPOCH_CKPTS = True
    module.SAVE_EPOCH_EVERY = 1
    module.SAVE_WRONG_IMAGES = False
    module.SAVE_REPRESENTATIVES = False
    module.EXPERIMENT_RECIPE_SHA256 = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode("utf-8")
    ).hexdigest()


def run_single_process(module) -> None:
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


def evaluate(
    run_dir: Path,
    eval_root: Path,
    scope: str,
    epochs: str,
    modes: str,
    concat_weights: str,
    batch: int,
    workers: int,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "eval_hard42_headonly_checkpoints.py"),
        "--run-dir", str(run_dir),
        "--eval-root", str(eval_root),
        "--scope", scope,
        "--epochs", epochs,
        "--modes", modes,
        "--concat-proj-weights", concat_weights,
        "--exclude-classes", ",".join(sorted(STRICT_EXCLUDED)),
        "--batch", str(batch),
        "--workers", str(workers),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn_tapt", "nocnn"], required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--head", choices=["linear", "mlp", "ad", "adapter", "adapterN2", "adapterN3"], default="mlp")
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--queue-size", type=int, default=0)
    parser.add_argument("--ignore-neg-sim", type=float, default=1.0)
    parser.add_argument("--local-weight", type=float, default=0.0)
    parser.add_argument("--neco-weight", type=float, default=0.0)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-epochs", default="1,2,3,5,8,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--concat-proj-weights", default="")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "hard42_headonly_ablation")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--eval-holdout", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for required in (TRAIN_ROOT, DEV_ROOT):
        if not required.exists():
            raise FileNotFoundError(required)
    checkpoint = checkpoint_for(args.backbone)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit = audit_backbone_checkpoint(checkpoint)
    provenance = {
        "recipe": recipe_from_args(args),
        "backbone": args.backbone,
        "cell": args.cell,
        "backbone_audit": audit,
        "trainer": str((SCRIPT_DIR / "train_contrastive_ddp.py").resolve()),
        "trainer_sha256": sha256_file(SCRIPT_DIR / "train_contrastive_ddp.py"),
        "train_inventory": inventory(TRAIN_ROOT),
        "dev_inventory": inventory(DEV_ROOT),
        "holdout_inventory": inventory(HOLDOUT_ROOT) if HOLDOUT_ROOT.exists() else None,
        "strict_excluded": sorted(STRICT_EXCLUDED),
        "protocol": "frozen backbone; head/adapter only; grade_only; strict-novel dev and image-disjoint holdout",
    }
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if os.name == "nt":
        os.environ["USE_LIBUV"] = "0"
    os.environ["UC_PALETTE_MODE"] = "grade_only"
    module = load_trainer(f"hard42_headonly_trainer_{os.getpid()}")
    configure(module, args, checkpoint, output_root)
    install_head_variant(module, args.head, args.adapter_dim)
    made: list[Path] = []
    original_make_run_dir = module.make_run_dir

    def tracked_make_run_dir(root, tag):
        run_dir = original_make_run_dir(root, tag)
        made.append(Path(run_dir))
        return run_dir

    module.make_run_dir = tracked_make_run_dir
    run_single_process(module)
    if len(made) != 1:
        raise RuntimeError(f"expected one run directory, got {made}")
    run_dir = made[0].resolve()
    checkpoint_dir = run_dir / "contrastive" / "epoch_checkpoints"
    if args.frozen:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "contrastive" / "best_model.pt", checkpoint_dir / "epoch_000.pt")
    (run_dir / "headonly_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[RUN_DIR] {run_dir}", flush=True)
    if not args.skip_eval:
        epochs = "0" if args.frozen else args.eval_epochs
        modes = "backbone" if args.frozen else (
            "projection,adapter" if args.head.lower().startswith("adapter") else "projection"
        )
        evaluate(
            run_dir,
            DEV_ROOT,
            "dev_strict_novel",
            epochs,
            modes,
            args.concat_proj_weights,
            args.eval_batch,
            args.eval_workers,
        )
        if args.eval_holdout:
            if not HOLDOUT_ROOT.exists():
                raise FileNotFoundError(HOLDOUT_ROOT)
            evaluate(
                run_dir,
                HOLDOUT_ROOT,
                "holdout_strict_novel",
                epochs,
                modes,
                args.concat_proj_weights,
                args.eval_batch,
                args.eval_workers,
            )
    completion = {
        "run_dir": str(run_dir),
        "backbone": args.backbone,
        "cell": args.cell,
        "recipe": recipe_from_args(args),
        "dev_evaluated": args.skip_eval is False,
        "holdout_evaluated": bool(args.eval_holdout and not args.skip_eval),
    }
    (run_dir / "headonly_completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
