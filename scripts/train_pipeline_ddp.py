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
AUTO_GENERATE         = True                                # SOURCE_ROOT 없으면 generate_data.py 자동 실행
AUTO_SPLIT            = True                                # split 폴더 없으면 _split_data.py 자동 실행

# ★ H100 batch per GPU (80GB). 0 = 각 _ddp.py CONFIG default (로컬 16GB: CNN16/CL8) 그대로.
#   total batch = BATCH_PER_GPU × GPU 수. CNN 은 full fine-tune(무거움), CL 은 frozen(가벼움).
CNN_BATCH_PER_GPU     = 32             # convnextv2_base 384 full-ft + AMP → H100 80GB 32 안전
CL_BATCH_PER_GPU      = 64             # frozen backbone → 가벼움, contrastive 는 batch 클수록 negative↑ 유리
GROUPING_BATCH        = 128
GROUPING_WORKERS      = 16
GROUPING_REPS_PER_CLUSTER = 5

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
    p.add_argument("--generate-workers", type=int, default=None,
                   help="generate_data.py worker 수. 기본 None=os.cpu_count() 전체 사용.")
    p.add_argument("--clean-split", action="store_true",
                   help="학습 전 _split_data.py --clean-output 실행. 기존 split stale 파일 제거.")
    p.add_argument("--no-auto-generate", action="store_true",
                   help="SOURCE_ROOT 없거나 비었을 때 generate_data.py 자동 실행 안 함.")
    p.add_argument("--no-auto-split", action="store_true",
                   help="split 폴더 없어도 _split_data.py 자동 실행 안 함 (에러).")
    p.add_argument("--prod-train-dirs", type=str, default=None,
                   help="현업 classless contrastive 학습 폴더(콤마구분). 지정 시 --no-eval 로 stage2 실행.")
    p.add_argument("--prod-pred-dirs", type=str, default=None,
                   help="stage2 결과로 바로 grouping 할 현업 폴더(콤마구분).")
    p.add_argument("--prod-epochs", type=int, default=None,
                   help="현업 contrastive epoch override.")
    p.add_argument("--grouping-batch", type=int, default=None,
                   help="predict_grouping_prod.py batch override.")
    p.add_argument("--grouping-workers", type=int, default=None,
                   help="predict_grouping_prod.py workers override.")
    p.add_argument("--grouping-reps-per-cluster", type=int, default=None,
                   help="grouping cluster별 대표 이미지 개수.")
    return p.parse_args()


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _has_any_images(root: Path) -> bool:
    return root.exists() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in root.rglob("*")
    )


def _has_class_images(root: Path) -> bool:
    if not root.exists():
        return False
    for cls_dir in root.iterdir():
        if cls_dir.is_dir() and _has_any_images(cls_dir):
            return True
    return False


def _find_source_root(repo: Path, resolve_path, source_root: str | None) -> Path | None:
    candidates = [source_root] if source_root else [
        SOURCE_ROOT,
        "data/images/unknown",
        "data/unknown",
    ]
    for c in candidates:
        if not c:
            continue
        p = resolve_path(c)
        if _has_class_images(p):
            return p
    return None


def _generate_source(repo: Path, resolve_path, source_root: str, workers: int | None) -> Path:
    out = resolve_path(source_root)
    print(f"[auto-generate] SOURCE_ROOT 없음/비어있음 → generate_data.py 실행 (output={out})")
    cmd = [
        sys.executable, "-u", str(repo / "scripts" / "generate_data.py"),
        "--output-dir", str(out),
    ]
    if workers is not None:
        cmd += ["--workers", str(workers)]
    rc = subprocess.run(cmd, cwd=str(repo)).returncode
    if rc != 0:
        raise SystemExit(f"generate_data.py 실패 rc={rc}")
    if not _has_class_images(out):
        raise SystemExit(f"generate_data.py 완료 후에도 <class>/*.png 구조가 없음: {out}")
    return out


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
    grouping_batch = args.grouping_batch if args.grouping_batch is not None else GROUPING_BATCH
    grouping_workers = args.grouping_workers if args.grouping_workers is not None else GROUPING_WORKERS
    grouping_reps = (args.grouping_reps_per_cluster
                     if args.grouping_reps_per_cluster is not None
                     else GROUPING_REPS_PER_CLUSTER)
    if args.clean_split and args.no_auto_split:
        raise SystemExit("--clean-split and --no-auto-split cannot be used together")

    repo = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    from _common import resolve_path

    src = _find_source_root(repo, resolve_path, args.source_root)
    if src is None:
        if args.no_auto_generate or not AUTO_GENERATE:
            checked = args.source_root or f"{SOURCE_ROOT}, data/images/unknown, data/unknown"
            raise SystemExit(
                f"SOURCE_ROOT 이미지 없음: {checked}\n"
                f"  서버 기준 후보: {resolve_path(SOURCE_ROOT)}, {repo / 'data' / 'images' / 'unknown'}, {repo / 'data' / 'unknown'}")
        src = _generate_source(repo, resolve_path, SOURCE_ROOT, args.generate_workers)
    SOURCE_ROOT = str(src)

    resolved_cnn = resolve_path(CNN_DATA_DIR)
    resolved_cl_train = resolve_path(CL_TRAIN_DIR)
    resolved_cl_eval = resolve_path(CL_EVAL_DIR)
    split_ready = (
        _has_class_images(resolved_cnn)
        and _has_any_images(resolved_cl_train)
        and _has_class_images(resolved_cl_eval)
    )

    if (not split_ready or args.clean_split) and (args.no_auto_split or not AUTO_SPLIT):
        raise SystemExit(
            "split 출력이 준비되지 않음.\n"
            f"  source_root: {src}\n"
            f"  CNN_DATA_DIR ready={_has_class_images(resolved_cnn)} path={resolved_cnn}\n"
            f"  CL_TRAIN_DIR ready={_has_any_images(resolved_cl_train)} path={resolved_cl_train}\n"
            f"  CL_EVAL_DIR  ready={_has_class_images(resolved_cl_eval)} path={resolved_cl_eval}\n"
            f"  --no-auto-split 제거하면 pipeline 이 _split_data.py 를 자동 실행함")

    # split 폴더가 없거나 비어 있거나 clean 요청이면 SOURCE_ROOT 기준으로 자동 생성.
    if (not split_ready or args.clean_split) and not args.no_auto_split and AUTO_SPLIT:
        reason = "clean-split 요청" if args.clean_split else "split 출력 없음/비어있음"
        print(f"[auto-split] {reason} → _split_data.py 실행 (source={src})")
        split_cmd = [
            sys.executable, "-u", str(repo / "scripts" / "_split_data.py"),
            "--source-root", str(src),
            "--cnn-train-dir", CNN_DATA_DIR,
            "--cl-train-dir", CL_TRAIN_DIR,
            "--cl-eval-dir", CL_EVAL_DIR,
        ]
        if args.clean_split or not split_ready:
            split_cmd.append("--clean-output")
        rc = subprocess.run(split_cmd, cwd=str(repo)).returncode
        if rc != 0:
            raise SystemExit(f"_split_data.py 실패 rc={rc}")

    resolved_dirs = {}
    for name, path in [
        ("CNN_DATA_DIR", CNN_DATA_DIR),
        ("CL_TRAIN_DIR", CL_TRAIN_DIR),
        ("CL_EVAL_DIR", CL_EVAL_DIR),
    ]:
        resolved = resolve_path(path)
        ready = _has_any_images(resolved)
        if name in ("CNN_DATA_DIR", "CL_EVAL_DIR"):
            ready = _has_class_images(resolved)
        if not ready:
            raise SystemExit(
                f"{name} not ready: {resolved}\n"
                f"  source_root: {src}\n"
                f"  pipeline 은 source 생성/감지 → split → CNN → contrastive 순서로 자동 처리함.\n"
                f"  직접 source 를 쓰려면 --source-root data/unknown 처럼 지정.")
        resolved_dirs[name] = resolved

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = repo / "runs" / f"{ts}_{PIPELINE_TAG}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "pipeline.log"
    print(f"[pipeline_ddp] {run_dir}")
    print(f"[source_root] {src}")
    print("[data dirs]")
    for name, resolved in resolved_dirs.items():
        print(f"  {name}: {resolved}")
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"[pipeline_ddp] {datetime.now().isoformat()}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}\n\n")
        f.write(f"CNN_DATA_DIR={CNN_DATA_DIR}\n")
        f.write(f"CL_TRAIN_DIR={CL_TRAIN_DIR}\n")
        f.write(f"CL_EVAL_DIR={CL_EVAL_DIR}\n\n")
        f.write(f"SOURCE_ROOT={SOURCE_ROOT}\n")
        f.write(f"RESOLVED_SOURCE_ROOT={src}\n")
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
    print("STAGE 2 — Contrastive DDP" + (" (production classless)" if args.prod_train_dirs else ""))
    print("=" * 60)
    # CNN backbone + tag 를 env 로 주입 (mp.spawn 자식이 module-level 에서 읽음)
    env["CL_BACKBONE_CKPT"] = str(cnn_best)
    if args.prod_train_dirs:
        env["CL_TAG"] = "contrastive_prod_ddp_pipe"
        env["CL_TRAIN_DIRS"] = args.prod_train_dirs
        env["CL_NO_EVAL"] = "1"
        if args.prod_epochs is not None:
            env["CL_EPOCHS"] = str(args.prod_epochs)
        print(f"[prod train dirs] {args.prod_train_dirs}")
        print("[prod eval] skipped (--no-eval, classless production data)")
    else:
        env["CL_TAG"] = "contrastive_ddp_pipe"
        env.pop("CL_NO_EVAL", None)

    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, "-u", str(repo / "scripts" / "train_contrastive_ddp.py")],
        env=env, cwd=str(repo),
    ).returncode
    print(f"[stage2 done] rc={rc}, elapsed={(time.time()-t0)/60:.1f} min")
    if rc != 0:
        raise SystemExit(f"Contrastive DDP failed rc={rc}")

    cl_tag = "contrastive_prod_ddp_pipe" if args.prod_train_dirs else "contrastive_ddp_pipe"
    cl_runs = sorted((repo / "runs").glob(f"*_{cl_tag}"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    cl_run = None
    if cl_runs:
        cl_run = cl_runs[0]
        print(f"[stage2 result] {cl_run}")

    grouping_run = None
    if args.prod_pred_dirs:
        if cl_run is None:
            raise SystemExit("grouping requested but no contrastive run found")
        print("\n" + "=" * 60)
        print("STAGE 3 — Production Grouping")
        print("=" * 60)
        group_cmd = [
            sys.executable, "-u", str(repo / "scripts" / "predict_grouping_prod.py"),
            "--model", str(cl_run),
            "--image-roots", args.prod_pred_dirs,
            "--batch", str(grouping_batch),
            "--workers", str(grouping_workers),
            "--reps-per-cluster", str(grouping_reps),
        ]
        print(f"[prod pred dirs] {args.prod_pred_dirs}")
        t0 = time.time()
        rc = subprocess.run(group_cmd, env=env, cwd=str(repo)).returncode
        print(f"[stage3 done] rc={rc}, elapsed={(time.time()-t0)/60:.1f} min")
        if rc != 0:
            raise SystemExit(f"Production grouping failed rc={rc}")
        group_runs = sorted((repo / "result_grouping").glob("*_grouping"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if group_runs:
            grouping_run = group_runs[0]
            print(f"[stage3 result] {grouping_run}")

    # ============ Pipeline summary ============
    summary = {
        "pipeline_run": str(run_dir),
        "cnn_run": str(cnn_run),
        "cnn_best": str(cnn_best),
        "contrastive_run": str(cl_run) if cl_runs else None,
        "contrastive_best": str(cl_run / "contrastive" / "best_model.pt") if cl_runs else None,
        "production_mode": bool(args.prod_train_dirs),
        "prod_train_dirs": args.prod_train_dirs,
        "prod_pred_dirs": args.prod_pred_dirs,
        "grouping_run": str(grouping_run) if grouping_run else None,
        "cnn_data_dir": CNN_DATA_DIR,
        "cl_train_dir": CL_TRAIN_DIR,
        "cl_eval_dir": CL_EVAL_DIR,
        "source_root": SOURCE_ROOT,
        "resolved_source_root": str(src),
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
    if grouping_run:
        print(f"  Grouping result:   {grouping_run}")


if __name__ == "__main__":
    main()
