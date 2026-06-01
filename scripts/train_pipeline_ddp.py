#!/usr/bin/env python3
"""DDP 한방에 — CNN_DDP → Contrastive_DDP sequential 실행.

사용:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_pipeline_ddp.py
        → 4 GPU 으로 CNN 학습 → backbone 으로 Contrastive 학습

내부적으로 subprocess.run 으로 두 스크립트 순차 실행 (각자 mp.spawn 사용).
"""
from __future__ import annotations

# ===================================================================
# === CONFIG ===
# ===================================================================
SOURCE_ROOT           = "E:/data/images/unknown"          # generate_data.py 출력 (split 의 source)
CNN_DATA_DIR          = "E:/data/images/cnn_train"        # ↓ 아래 3개는 _split_data.py 가 SOURCE_ROOT 에서 생성
CL_TRAIN_DIR          = "E:/data/images/contrastive_train"
CL_EVAL_DIR           = "E:/data/images/contrastive_eval"
AUTO_SPLIT            = True                                # split 폴더 없으면 _split_data.py 자동 실행

# ★ H100 batch per GPU (80GB). 0 = 각 _ddp.py CONFIG default (로컬 16GB: CNN16/CL8) 그대로.
#   total batch = BATCH_PER_GPU × GPU 수. CNN 은 full fine-tune(무거움), CL 은 frozen(가벼움).
CNN_BATCH_PER_GPU     = 32             # convnextv2_base 384 full-ft + AMP → H100 80GB 32 안전
CL_BATCH_PER_GPU      = 64             # frozen backbone → 가벼움, contrastive 는 batch 클수록 negative↑ 유리

# 두 stage 의 hyperparam 은 각 _ddp.py 의 CONFIG block 을 직접 수정
#   scripts/train_cnn_ddp.py        ← CNN
#   scripts/train_contrastive_ddp.py ← Contrastive
#
# 이 pipeline 은 CNN best_model.pth path 를 자동 추출 후 Contrastive 에 inject
PIPELINE_TAG          = "pipeline_ddp"
# ===================================================================

import json
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-data-dir", type=str, default=None,
                   help="Stage1 CNN ImageFolder train 폴더.")
    p.add_argument("--cl-train-dir", type=str, default=None,
                   help="Stage2 contrastive train flat 폴더.")
    p.add_argument("--cl-eval-dir", type=str, default=None,
                   help="Stage2 contrastive eval ImageFolder 폴더.")
    p.add_argument("--cnn-batch", type=int, default=None,
                   help="CNN batch per GPU override.")
    p.add_argument("--cl-batch", type=int, default=None,
                   help="Contrastive batch per GPU override.")
    p.add_argument("--source-root", type=str, default=None,
                   help="split source (generate_data 출력 unknown/). split 폴더 없을 때 사용.")
    p.add_argument("--no-auto-split", action="store_true",
                   help="split 폴더 없어도 _split_data.py 자동 실행 안 함 (에러).")
    return p.parse_args()


def main():
    global CNN_DATA_DIR, CL_TRAIN_DIR, CL_EVAL_DIR, CNN_BATCH_PER_GPU, CL_BATCH_PER_GPU, SOURCE_ROOT
    args = parse_args()
    if args.cnn_data_dir:
        CNN_DATA_DIR = args.cnn_data_dir
    if args.cl_train_dir:
        CL_TRAIN_DIR = args.cl_train_dir
    if args.cl_eval_dir:
        CL_EVAL_DIR = args.cl_eval_dir
    if args.cnn_batch is not None:
        CNN_BATCH_PER_GPU = args.cnn_batch
    if args.cl_batch is not None:
        CL_BATCH_PER_GPU = args.cl_batch
    if args.source_root:
        SOURCE_ROOT = args.source_root

    repo = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    from _common import resolve_path

    # split 폴더(cnn_train/contrastive_train/contrastive_eval) 중 하나라도 없으면
    # _split_data.py 를 SOURCE_ROOT 기준으로 자동 실행 (generate_data 만 한 상태 대응).
    missing = [resolve_path(p) for p in (CNN_DATA_DIR, CL_TRAIN_DIR, CL_EVAL_DIR)
               if not resolve_path(p).exists()]
    if missing and not args.no_auto_split:
        src = resolve_path(SOURCE_ROOT)
        if not src.exists():
            raise SystemExit(
                f"split 출력 폴더가 없고 SOURCE_ROOT 도 없음: {src}\n"
                f"  먼저 합성: python scripts/generate_data.py\n"
                f"  (또는 --source-root 로 unknown/ 경로 지정)")
        print(f"[auto-split] split 폴더 없음 → _split_data.py 실행 (source={src})")
        rc = subprocess.run(
            [sys.executable, "-u", str(repo / "scripts" / "_split_data.py"),
             "--source-root", str(src)],
            cwd=str(repo),
        ).returncode
        if rc != 0:
            raise SystemExit(f"_split_data.py 실패 rc={rc}")

    resolved_dirs = {}
    for name, path in [
        ("CNN_DATA_DIR", CNN_DATA_DIR),
        ("CL_TRAIN_DIR", CL_TRAIN_DIR),
        ("CL_EVAL_DIR", CL_EVAL_DIR),
    ]:
        resolved = resolve_path(path)
        if not resolved.exists():
            raise SystemExit(
                f"{name} not found: {resolved}\n"
                f"  split 미실행 상태. 다음 중 하나:\n"
                f"  1) python scripts/generate_data.py  (이미지 합성)\n"
                f"  2) python scripts/_split_data.py    (cnn_train/contrastive_* 생성)\n"
                f"  pipeline 은 split 폴더가 있어야 함 (AUTO_SPLIT 로 자동 시도하지만 SOURCE_ROOT 필요).")
        resolved_dirs[name] = resolved

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = repo / "runs" / f"{ts}_{PIPELINE_TAG}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "pipeline.log"
    print(f"[pipeline_ddp] {run_dir}")
    print("[data dirs]")
    for name, resolved in resolved_dirs.items():
        print(f"  {name}: {resolved}")
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"[pipeline_ddp] {datetime.now().isoformat()}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}\n\n")
        f.write(f"CNN_DATA_DIR={CNN_DATA_DIR}\n")
        f.write(f"CL_TRAIN_DIR={CL_TRAIN_DIR}\n")
        f.write(f"CL_EVAL_DIR={CL_EVAL_DIR}\n\n")
        f.write(f"RESOLVED_CNN_DATA_DIR={resolved_dirs['CNN_DATA_DIR']}\n")
        f.write(f"RESOLVED_CL_TRAIN_DIR={resolved_dirs['CL_TRAIN_DIR']}\n")
        f.write(f"RESOLVED_CL_EVAL_DIR={resolved_dirs['CL_EVAL_DIR']}\n\n")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["CNN_DATA_DIR"] = CNN_DATA_DIR
    env["CL_TRAIN_DIRS"] = CL_TRAIN_DIR
    env["CL_EVAL_DIRS"] = CL_EVAL_DIR
    # H100 batch 주입 (0 이면 각 스크립트 CONFIG default 유지)
    if CNN_BATCH_PER_GPU:
        env["CNN_BATCH_PER_GPU"] = str(CNN_BATCH_PER_GPU)
    if CL_BATCH_PER_GPU:
        env["CL_BATCH_PER_GPU"] = str(CL_BATCH_PER_GPU)

    # ============ Stage 1: CNN DDP ============
    print("\n" + "=" * 60)
    print("STAGE 1 — CNN DDP")
    print("=" * 60)
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, "-u", str(repo / "scripts" / "train_cnn_ddp.py")],
        env=env, cwd=str(repo),
    ).returncode
    print(f"[stage1 done] rc={rc}, elapsed={(time.time()-t0)/60:.1f} min")
    if rc != 0:
        raise SystemExit(f"CNN DDP failed rc={rc}")

    # latest CNN run dir from runs/
    cnn_runs = sorted((repo / "runs").glob("*_cnn_ddp"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not cnn_runs:
        raise SystemExit("no *_cnn_ddp run dir found after CNN stage")
    cnn_run = cnn_runs[0]
    cnn_best = cnn_run / "cnn" / "best_model.pth"
    print(f"[stage1 result] {cnn_run}  best={cnn_best.exists()}")
    if not cnn_best.exists():
        raise SystemExit(f"no best_model.pth in {cnn_run}")

    # ============ Stage 2: Contrastive DDP ============
    print("\n" + "=" * 60)
    print("STAGE 2 — Contrastive DDP")
    print("=" * 60)
    # CNN backbone + tag 를 env 로 주입 (mp.spawn 자식이 module-level 에서 읽음)
    env["CL_BACKBONE_CKPT"] = str(cnn_best)
    env["CL_TAG"] = "contrastive_ddp_pipe"

    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, "-u", str(repo / "scripts" / "train_contrastive_ddp.py")],
        env=env, cwd=str(repo),
    ).returncode
    print(f"[stage2 done] rc={rc}, elapsed={(time.time()-t0)/60:.1f} min")
    if rc != 0:
        raise SystemExit(f"Contrastive DDP failed rc={rc}")

    cl_runs = sorted((repo / "runs").glob("*_contrastive_ddp_pipe"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if cl_runs:
        cl_run = cl_runs[0]
        print(f"[stage2 result] {cl_run}")

    # ============ Pipeline summary ============
    summary = {
        "pipeline_run": str(run_dir),
        "cnn_run": str(cnn_run),
        "cnn_best": str(cnn_best),
        "contrastive_run": str(cl_run) if cl_runs else None,
        "contrastive_best": str(cl_run / "contrastive" / "best_model.pt") if cl_runs else None,
        "cnn_data_dir": CNN_DATA_DIR,
        "cl_train_dir": CL_TRAIN_DIR,
        "cl_eval_dir": CL_EVAL_DIR,
        "resolved_cnn_data_dir": str(resolved_dirs["CNN_DATA_DIR"]),
        "resolved_cl_train_dir": str(resolved_dirs["CL_TRAIN_DIR"]),
        "resolved_cl_eval_dir": str(resolved_dirs["CL_EVAL_DIR"]),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
        "finished_at": datetime.now().isoformat(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {run_dir}")
    print(f"  CNN best:         {cnn_best}")
    if cl_runs:
        print(f"  Contrastive best: {cl_run / 'contrastive' / 'best_model.pt'}")


if __name__ == "__main__":
    main()
