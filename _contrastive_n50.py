#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive n50 small-budget wrapper.

contrastive.py 무수정 — CFG override + subset hardlink builder + backbone unwrap.

Output: outputs/logs_contrastive/<tag>_<TS>/
  (CLAUDE.md 의 logs_<kind>/ 컨벤션 — 결과 파일들은 outputs/ 아래)

Usage:
    python _contrastive_n50.py [--epochs 20] [--batch 16] [--per-class 50]
                               [--normal 200] [--workers 0]
                               [--backbone logs_wafer/overall/best_model.pth]
"""
import argparse, os, sys, shutil, json
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16,
                    help="default 16 (small data + GPU mem safe)")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--normal", type=int, default=200)
    ap.add_argument("--normal-class", default="Normal_bank_boundary")
    ap.add_argument("--workers", type=int, default=0,
                    help="default 0 (Windows pickle safe — contrastive.py tfm has lambda)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--queue-size", type=int, default=4096)
    ap.add_argument("--backbone",
                    default="logs_wafer/overall/best_model.pth",
                    help="Backbone checkpoint (TAPT). 'none' = no pretrained.")
    ap.add_argument("--output-root", default="outputs/logs_contrastive",
                    help="Parent dir for run_dir (CLAUDE.md outputs/logs_<kind>/ 컨벤션)")
    ap.add_argument("--model-tag", default="n50_norm200",
                    help="run_dir prefix (model_tag like cnn_train_chip --model-tag)")
    ap.add_argument("--unknown-source",
                    default="D:/project/data/wm-811k/unknown")
    ap.add_argument("--subset-root",
                    default="D:/project/data/contrastive_n50_subsets",
                    help="Where to materialize hardlink subsets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-backbone", action="store_true", default=True)
    ap.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                    action="store_false")
    ap.add_argument("--use-queue", action="store_true", default=True)
    ap.add_argument("--no-use-queue", dest="use_queue", action="store_false")
    ap.add_argument("--use-local", action="store_true", default=True)
    ap.add_argument("--no-use-local", dest="use_local", action="store_false")
    return ap.parse_args()


def list_class_dirs(root: Path):
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        pngs = list(d.glob("*.png"))
        if not pngs:
            continue  # skip empty (e.g., classification, classification_chips)
        out.append((d, sorted(pngs)))
    return out


def hardlink_subset(class_dirs, dst_root: Path, per_class: int,
                    normal_class: str, normal_n: int, seed: int):
    """Materialize subset via hardlink (same volume — instant, no copy)."""
    import random
    rng = random.Random(seed)
    dst_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cdir, pngs in class_dirs:
        cls = cdir.name
        n_take = normal_n if cls == normal_class else per_class
        picks = pngs[:n_take]  # sorted-pick first N (deterministic)
        cls_dst = dst_root / cls
        cls_dst.mkdir(exist_ok=True)
        for p in cls_dst.glob("*.png"):
            p.unlink()
        for src in picks:
            dst = cls_dst / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        summary[cls] = len(picks)
    return summary


def hardlink_full(class_dirs, dst_root: Path):
    """Mirror unknown/ filtered (skip empty dirs) — for cluster target."""
    dst_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cdir, pngs in class_dirs:
        cls = cdir.name
        cls_dst = dst_root / cls
        cls_dst.mkdir(exist_ok=True)
        for p in cls_dst.glob("*.png"):
            p.unlink()
        for src in pngs:
            dst = cls_dst / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        summary[cls] = len(pngs)
    return summary


def unwrap_backbone(ckpt_path: Path, out_path: Path):
    """cnn_train.py best_model.pth → contrastive-friendly state_dict.

    cnn_train.py saves dict with 'model' key holding raw timm state_dict
    (no 'model.' / 'backbone.' prefix). contrastive.py looks for 'state_dict'
    key and prefix-strips. Re-save as {'state_dict': ...} so loader works.
    """
    import torch
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        inner = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        inner = sd["state_dict"]
    else:
        inner = sd
    torch.save({"state_dict": inner}, out_path)


def main():
    args = parse_args()

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    output_root = REPO / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    # run_dir = outputs/logs_contrastive/<tag>_<TS>/
    # eval script 가 끝나면 _ari{:.2f}_nmi{:.2f} suffix rename + overall/ mirror
    run_dir = output_root / f"{args.model_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.unknown_source)
    if not src.exists():
        sys.exit(f"[err] unknown_source not found: {src}")

    print(f"[wrapper] scan {src}", flush=True)
    class_dirs = list_class_dirs(src)
    print(f"[wrapper] classes (non-empty): {len(class_dirs)}", flush=True)

    subset_root = Path(args.subset_root) / ts
    train_dir = subset_root / "train"
    unknown_dir = subset_root / "unknown"

    print(f"[wrapper] build train subset → {train_dir}", flush=True)
    train_summary = hardlink_subset(
        class_dirs, train_dir, args.per_class,
        args.normal_class, args.normal, args.seed)
    print(f"[wrapper] train classes: {len(train_summary)} | total: "
          f"{sum(train_summary.values())}", flush=True)

    print(f"[wrapper] build unknown mirror → {unknown_dir}", flush=True)
    unknown_summary = hardlink_full(class_dirs, unknown_dir)
    print(f"[wrapper] unknown classes: {len(unknown_summary)} | total: "
          f"{sum(unknown_summary.values())}", flush=True)

    backbone_path = None
    if args.backbone and args.backbone.lower() != "none":
        bp = Path(args.backbone)
        if not bp.is_absolute():
            bp = REPO / bp
        if not bp.exists():
            sys.exit(f"[err] backbone not found: {bp}")
        backbone_path = run_dir / "_init_backbone.pth"
        print(f"[wrapper] unwrap backbone → {backbone_path}", flush=True)
        unwrap_backbone(bp, backbone_path)

    # Manifest
    manifest = {
        "ts": ts,
        "args": vars(args),
        "src": str(src),
        "train_dir": str(train_dir),
        "unknown_dir": str(unknown_dir),
        "subset_summary_train": train_summary,
        "subset_summary_unknown_total": sum(unknown_summary.values()),
        "n_classes": len(class_dirs),
        "backbone_unwrapped": str(backbone_path) if backbone_path else None,
        "run_dir": str(run_dir),
    }
    (run_dir / "_wrapper_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # === CFG override ===
    import contrastive
    contrastive.CFG.update({
        "TRAIN_DIR":   str(train_dir),
        "UNKNOWN_DIR": str(unknown_dir),
        "OVERLAY_DIR": str(unknown_dir),
        # OUTPUT_DIR + RUN_TS 가 합쳐져 run_dir 이 됨 (contrastive.py L819).
        # 우리는 이미 run_dir 만들었으니 OUTPUT_DIR_RUN_TS 가 그것과 일치하도록 셋업.
        "OUTPUT_DIR":  str(run_dir.parent / args.model_tag),  # contrastive.py 가 _<TS> append
        "EPOCHS":      args.epochs,
        "BATCH":       args.batch,
        "LR_HEAD":     args.lr,
        "TEMP":        args.temp,
        "NUM_WORKERS": args.workers,
        "QUEUE_SIZE":  args.queue_size,
        "SEED":        args.seed,
        "FREEZE_BACKBONE": args.freeze_backbone,
        "USE_QUEUE":   args.use_queue,
        "USE_LOCAL":   args.use_local,
        "TRAIN_SAMPLING_RATIO": 1.0,
    })
    if backbone_path is not None:
        contrastive.CFG["LOCAL_BACKBONE_WEIGHTS"] = str(backbone_path)

    # contrastive.RUN_TS overwrite — run_dir 일관 명명
    contrastive.RUN_TS = ts

    print("[wrapper] CFG (overridden):", flush=True)
    print(json.dumps({k: contrastive.CFG[k] for k in [
        "TRAIN_DIR", "UNKNOWN_DIR", "OUTPUT_DIR", "BACKBONE_NAME",
        "LOCAL_BACKBONE_WEIGHTS", "EPOCHS", "BATCH", "LR_HEAD",
        "NUM_WORKERS", "FREEZE_BACKBONE", "USE_QUEUE", "USE_LOCAL",
        "QUEUE_SIZE", "TRAIN_SAMPLING_RATIO",
    ]}, indent=2, ensure_ascii=False), flush=True)
    print(f"[wrapper] run_dir = {run_dir}", flush=True)

    print("[wrapper] >>> calling contrastive.main()", flush=True)
    contrastive.main()
    print("[wrapper] <<< contrastive.main() done", flush=True)
    print(f"[wrapper] run_dir = {run_dir}  (eval script: "
          f"python _eval_contrastive_n50.py --run-dir {run_dir})", flush=True)


if __name__ == "__main__":
    main()
