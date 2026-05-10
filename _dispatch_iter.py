#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detached dispatch for contrastive training that survives Bash subprocess death.

Pattern: subprocess.Popen with DETACHED_PROCESS + new console (Windows-only)
→ python child fully detaches → Bash 60min timeout 무관.

Output dir 은 contrastive.py 가 자체 RUN_TS 로 자동 생성 (outputs_contrastive_<TS>/).
dispatcher 는 launch 직후 newest 폴더를 발견해 보고만 함.

Usage:
    python _dispatch_iter.py --tag iter_a0 --epochs 10 --batch 8 \
        --data D:/project/data/contrastive_anchor/avg30_260505_203615
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="iter_a0")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--ignore-neg-sim", type=float, default=0.72)
    p.add_argument("--nce-temp", type=float, default=0.07)
    p.add_argument("--use-local", default="true")
    p.add_argument("--use-queue", default="true")
    p.add_argument("--queue-size", type=int, default=4096)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mcs", type=int, default=12)
    p.add_argument("--ms", type=int, default=4)
    p.add_argument("--cluster-method", default="leaf")
    p.add_argument("--cluster-eps", type=float, default=0.06)
    p.add_argument("--gpu-throttle-ms", type=float, default=0.0,
                   help="sleep ms after each optimizer.step (GPU compute throttle)")
    p.add_argument("--local-weight", type=float, default=0.5,
                   help="LOCAL_WEIGHT: USE_LOCAL grid contrast loss weight (default 0.5)")
    p.add_argument("--local-pos-topk", type=int, default=12,
                   help="LOCAL_POS_TOPK: per-anchor local positive count (default 12)")
    p.add_argument("--neg-filter", default="fixed", choices=["fixed", "perc_pos"],
                   help="negative filter mode: fixed (IGNORE_NEG_SIM) or perc_pos (NV-Retriever TopK-PercPos)")
    p.add_argument("--neg-filter-alpha", type=float, default=0.95,
                   help="NV-Retriever PercPos α (only used when --neg-filter=perc_pos, default 0.95)")
    p.add_argument("--freeze-backbone", default="true", choices=["true", "false"],
                   help="FREEZE_BACKBONE: true=head only, false=joint training")
    p.add_argument("--backbone-unfreeze-last-n", type=int, default=0,
                   help="Unfreeze last N stages of ConvNeXtV2 backbone (default 0). "
                        "Combined with --freeze-backbone=true, enables partial fine-tuning: "
                        "head + last N stages trainable, earlier stages frozen.")
    p.add_argument("--backbone-lr-scale", type=float, default=1.0,
                   help="Per-grad multiplier on unfrozen backbone params (default 1.0 = "
                        "same LR as head). Use e.g. 0.02 to make effective backbone LR = "
                        "LR_HEAD × 0.02. Implemented via register_hook on backbone params.")
    p.add_argument("--neco-weight", type=float, default=0.0,
                   help="NeCo (arXiv:2408.11054) patch ordering loss weight (default 0 = off)")
    p.add_argument("--neco-zone-vertical", type=int, default=0,
                   help="★ Novelty A — Zone-Aware NeCo. >0 partitions the 12×12 patch "
                        "grid into N horizontal zones (default 0 = standard NeCo, 3 = "
                        "Edge-Top/Center/Edge-Bottom 4-row bands). Within-zone NeCo only.")
    p.add_argument("--neco-hier-pools", type=str, default="",
                   help="★ Novelty B — Hierarchical Multi-Scale NeCo. Comma-separated "
                        "pool factors (e.g. '1,2,4' = 12×12 + 6×6 + 3×3 grids). "
                        "Empty = standard NeCo. Mutually exclusive with zone-vertical.")
    args = p.parse_args()

    if not Path(args.data).exists():
        print(f"data dir not found: {args.data}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).parent.resolve()
    dispatch_log_dir = repo_root / "_dispatch_logs"
    dispatch_log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    boot_log = dispatch_log_dir / f"{args.tag}_{ts}_boot.log"
    cfg_path = dispatch_log_dir / f"{args.tag}_{ts}_cfg.json"

    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(Path(args.data).resolve()),
        "EPOCHS": str(args.epochs),
        "BATCH": str(args.batch),
        "IMAGE_SIZE": str(args.image_size),
        "WARMUP_EPOCHS": str(args.warmup),
        "IGNORE_NEG_SIM": str(args.ignore_neg_sim),
        "NCE_TEMP": str(args.nce_temp),
        "USE_LOCAL": args.use_local,
        "USE_QUEUE": args.use_queue,
        "QUEUE_SIZE": str(args.queue_size),
        "LR_HEAD": str(args.lr_head),
        "SEED": str(args.seed),
        "MIN_CLUSTER_SIZE": str(args.mcs),
        "MIN_SAMPLES": str(args.ms),
        "CLUSTER_SELECTION_METHOD": args.cluster_method,
        "CLUSTER_SELECTION_EPSILON": str(args.cluster_eps),
        "GPU_THROTTLE_MS": str(args.gpu_throttle_ms),
        "LOCAL_WEIGHT": str(args.local_weight),
        "LOCAL_POS_TOPK": str(args.local_pos_topk),
        "NEG_FILTER": args.neg_filter,
        "NEG_FILTER_ALPHA": str(args.neg_filter_alpha),
        "FREEZE_BACKBONE": args.freeze_backbone,
        "BACKBONE_UNFREEZE_LAST_N": str(args.backbone_unfreeze_last_n),
        "BACKBONE_LR_SCALE": str(args.backbone_lr_scale),
        "NECO_WEIGHT": str(args.neco_weight),
        "NECO_ZONE_VERTICAL": str(args.neco_zone_vertical),
        "NECO_HIER_POOLS": args.neco_hier_pools,
        "PYTHONUNBUFFERED": "1",
    })

    cfg_path.write_text(
        json.dumps({"tag": args.tag, "dispatch_ts": ts,
                    **{k: v for k, v in vars(args).items()}},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    cmd = [sys.executable, "-u", str(repo_root / "run_contrastive.py")]
    log_fp = open(boot_log, "wb")

    # Snapshot existing outputs_contrastive_*/ before launch — to detect new dir
    existing = set(p.name for p in repo_root.glob("outputs_contrastive_*"))

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        close_fds=True,
    )

    # Wait briefly + verify start, find new outputs dir
    new_dir = None
    for _ in range(40):  # 40 × 0.5s = 20s
        time.sleep(0.5)
        rc = proc.poll()
        current = set(p.name for p in repo_root.glob("outputs_contrastive_*"))
        added = sorted(current - existing)
        if added:
            new_dir = repo_root / added[-1]
            break
        if rc is not None:
            print(f"[dispatch] FAIL — child exited rc={rc} before creating output dir",
                  file=sys.stderr)
            print(f"  boot_log: {boot_log}", file=sys.stderr)
            return 1

    if new_dir is None:
        print(f"[dispatch] WARN — output dir not detected in 20s. PID alive={proc.poll() is None}",
              file=sys.stderr)
        print(f"  boot_log: {boot_log}", file=sys.stderr)
    else:
        print(f"[dispatch] OK")
        print(f"  pid: {proc.pid}")
        print(f"  out_dir: {new_dir}")
        print(f"  out_log: {new_dir / 'run.log'}")
        print(f"  boot_log: {boot_log}")
        print(f"  cfg: {cfg_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
