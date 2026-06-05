#!/usr/bin/env python3
"""Recommended production run: CNN backbone -> prod contrastive train -> prod grouping.

CUDA_VISIBLE_DEVICES 는 여기서 설정하지 않는다. 스케줄러/쉘에서 필요하면 바깥에서 지정.

사용 예:
  python scripts/run_prod_recommended.py \
    --backbone runs/<CNN_RUN>/cnn/best_model.pth \
    --prod-train-dirs data/images/prod_train_A,data/images/prod_train_B \
    --prod-pred-dirs data/images/prod_pred_A,data/images/prod_pred_B

폴더를 기본값으로 맞춰두면 아래 한 줄만:
  python scripts/run_prod_recommended.py --backbone runs/<CNN_RUN>/cnn/best_model.pth
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_PROD_TRAIN_DIRS = "data/images/prod_train"
DEFAULT_PROD_PRED_DIRS = "data/images/prod_pred"


def _resolve_cnn_best(repo: Path, raw: str | None) -> Path:
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (repo / p).resolve()
        candidates = [p]
        if p.is_dir():
            candidates = [p / "cnn" / "best_model.pth", p / "best_model.pth"]
        for c in candidates:
            if c.exists() and c.is_file():
                return c
        raise SystemExit(f"CNN best_model.pth not found: {p}")

    bests = sorted((repo / "runs").glob("*/cnn/best_model.pth"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not bests:
        raise SystemExit(
            "No CNN best_model.pth found under runs/*/cnn/.\n"
            "Pass --backbone runs/<CNN_RUN>/cnn/best_model.pth")
    return bests[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=None,
                   help="CNN best_model.pth 또는 CNN run 폴더. 생략 시 latest runs/*/cnn/best_model.pth")
    p.add_argument("--prod-train-dirs", type=str, default=DEFAULT_PROD_TRAIN_DIRS,
                   help="현업 contrastive 학습 폴더, 콤마구분.")
    p.add_argument("--prod-pred-dirs", type=str, default=DEFAULT_PROD_PRED_DIRS,
                   help="현업 grouping 대상 폴더, 콤마구분.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    backbone = _resolve_cnn_best(repo, args.backbone)

    cmd = [
        sys.executable, str(repo / "scripts" / "train_pipeline_ddp.py"),
        "--backbone", str(backbone),
        "--prod-train-dirs", args.prod_train_dirs,
        "--prod-pred-dirs", args.prod_pred_dirs,
        "--prod-epochs", "20",
        "--ignore-neg-sim", "1.01",
        "--nce-temp", "0.05",
        "--cl-lr-head", "5e-4",
        "--neco-weight", "0.2",
        "--grouping-cluster-selection-method", "eom",
        "--grouping-cluster-selection-epsilon", "0.0",
        "--grouping-min-cluster-size", "12",
        "--grouping-min-samples", "3",
    ]

    print("[recommended prod run]")
    print(f"  backbone:        {backbone}")
    print(f"  prod_train_dirs: {args.prod_train_dirs}")
    print(f"  prod_pred_dirs:  {args.prod_pred_dirs}")
    print("$ " + " ".join(cmd), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.run(cmd, cwd=str(repo)).returncode)


if __name__ == "__main__":
    main()
