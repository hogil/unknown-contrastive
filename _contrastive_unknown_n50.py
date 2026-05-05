#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive unknown n50 small-budget wrapper.

contrastive_unknown_n50.py 기반 — CFG override + subset hardlink builder + backbone unwrap.

Output: outputs/logs_contrastive/<tag>_<TS>/
  (CLAUDE.md 의 logs_<kind>/ 컨벤션 — 결과 파일들은 outputs/ 아래)

Usage:
    python _contrastive_unknown_n50.py [--epochs 20] [--batch 16] [--per-class 50]
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
                    help="default 0 (Windows pickle safe — contrastive module tfm has lambda)")
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
    ap.add_argument("--prepared-subset", default=None,
                    help="reuse an existing subset root containing train/ and unknown/")
    ap.add_argument("--resize-cache", action="store_true",
                    help="materialize subset images resized to CFG image size instead of hardlinking originals")
    ap.add_argument("--resize-size", type=int, default=384,
                    help="square resize size used with --resize-cache")
    ap.add_argument("--cache-workers", type=int, default=4,
                    help="parallel workers for --resize-cache materialization")
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


def materialize_image(src: Path, dst: Path, resize_size: int | None):
    if resize_size is None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB").resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        im.save(dst, optimize=False, compress_level=1)


def materialize_many(pairs, resize_size: int | None, workers: int):
    if resize_size is None or workers <= 1:
        for src, dst in pairs:
            materialize_image(src, dst, resize_size)
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(materialize_image, src, dst, resize_size) for src, dst in pairs]
        for fut in as_completed(futures):
            fut.result()


def hardlink_subset(class_dirs, dst_root: Path, per_class: int,
                    normal_class: str, normal_n: int, seed: int,
                    resize_size: int | None = None, workers: int = 1):
    """Materialize subset via hardlink/copy, or resized cache when requested."""
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
        materialize_many([(src, cls_dst / src.name) for src in picks], resize_size, workers)
        summary[cls] = len(picks)
    return summary


def hardlink_full(class_dirs, dst_root: Path, resize_size: int | None = None,
                  workers: int = 1):
    """Mirror unknown/ filtered (skip empty dirs) — for cluster target."""
    dst_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cdir, pngs in class_dirs:
        cls = cdir.name
        cls_dst = dst_root / cls
        cls_dst.mkdir(exist_ok=True)
        for p in cls_dst.glob("*.png"):
            p.unlink()
        materialize_many([(src, cls_dst / src.name) for src in pngs], resize_size, workers)
        summary[cls] = len(pngs)
    return summary


def hardlink_train_unknown_split(class_dirs, train_root: Path, unknown_root: Path,
                                 per_class: int, normal_class: str,
                                 normal_n: int, resize_size: int | None = None,
                                 workers: int = 1):
    """Materialize disjoint train and heldout unknown subsets."""
    train_root.mkdir(parents=True, exist_ok=True)
    unknown_root.mkdir(parents=True, exist_ok=True)
    train_summary = {}
    unknown_summary = {}
    for cdir, pngs in class_dirs:
        cls = cdir.name
        n_take = normal_n if cls == normal_class else per_class
        train_picks = pngs[:n_take]
        unknown_picks = pngs[n_take:]

        train_dst = train_root / cls
        unknown_dst = unknown_root / cls
        train_dst.mkdir(exist_ok=True)
        unknown_dst.mkdir(exist_ok=True)
        for p in train_dst.glob("*.png"):
            p.unlink()
        for p in unknown_dst.glob("*.png"):
            p.unlink()

        materialize_many([(src, train_dst / src.name) for src in train_picks],
                         resize_size, workers)
        materialize_many([(src, unknown_dst / src.name) for src in unknown_picks],
                         resize_size, workers)
        train_summary[cls] = len(train_picks)
        if unknown_picks:
            unknown_summary[cls] = len(unknown_picks)
    return train_summary, unknown_summary


def count_subset(root: Path):
    summary = {}
    for d in sorted(root.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.png")))
            if n:
                summary[d.name] = n
    return summary


def unwrap_backbone(ckpt_path: Path, out_path: Path):
    """cnn_train.py best_model.pth → contrastive-friendly state_dict.

    cnn_train.py saves dict with 'model' key holding raw timm state_dict
    (no 'model.' / 'backbone.' prefix). contrastive module looks for 'state_dict'
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

    if args.prepared_subset:
        subset_root = Path(args.prepared_subset)
        train_dir = subset_root / "train"
        unknown_dir = subset_root / "unknown"
        if not train_dir.exists() or not unknown_dir.exists():
            sys.exit(f"[err] prepared subset must contain train/ and unknown/: {subset_root}")
        print(f"[wrapper] reuse prepared subset → {subset_root}", flush=True)
        train_summary = count_subset(train_dir)
        unknown_summary = count_subset(unknown_dir)
        print(f"[wrapper] train classes: {len(train_summary)} | total: "
              f"{sum(train_summary.values())}", flush=True)
        print(f"[wrapper] unknown classes: {len(unknown_summary)} | total: "
              f"{sum(unknown_summary.values())}", flush=True)
        source_label = str(subset_root)
        n_classes = len(train_summary)
    else:
        src = Path(args.unknown_source)
        if not src.exists():
            sys.exit(f"[err] unknown_source not found: {src}")

        print(f"[wrapper] scan {src}", flush=True)
        class_dirs = list_class_dirs(src)
        print(f"[wrapper] classes (non-empty): {len(class_dirs)}", flush=True)

        subset_root = Path(args.subset_root) / ts
        train_dir = subset_root / "train"
        unknown_dir = subset_root / "unknown"
        resize_size = args.resize_size if args.resize_cache else None

        cache_workers = max(1, args.cache_workers)
        cache_mode = f"resize={resize_size}, workers={cache_workers}" if resize_size else "hardlink"
        print(f"[wrapper] build train/unknown split ({cache_mode}) → {subset_root}", flush=True)
        train_summary, unknown_summary = hardlink_train_unknown_split(
            class_dirs, train_dir, unknown_dir, args.per_class,
            args.normal_class, args.normal, resize_size, cache_workers)
        print(f"[wrapper] train classes: {len(train_summary)} | total: "
              f"{sum(train_summary.values())}", flush=True)
        print(f"[wrapper] unknown classes: {len(unknown_summary)} | total: "
              f"{sum(unknown_summary.values())}", flush=True)
        source_label = str(src)
        n_classes = len(class_dirs)

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
        "src": source_label,
        "train_dir": str(train_dir),
        "unknown_dir": str(unknown_dir),
        "subset_summary_train": train_summary,
        "subset_summary_unknown_total": sum(unknown_summary.values()),
        "n_classes": n_classes,
        "backbone_unwrapped": str(backbone_path) if backbone_path else None,
        "run_dir": str(run_dir),
    }
    (run_dir / "_wrapper_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # === CFG override ===
    import contrastive_unknown_n50 as contrastive
    contrastive.CFG.update({
        "TRAIN_DIR":   str(train_dir),
        "UNKNOWN_DIR": str(unknown_dir),
        "OVERLAY_DIR": str(unknown_dir),
        # OUTPUT_DIR + RUN_TS 가 합쳐져 run_dir 이 됨.
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
          f"python _eval_contrastive_unknown_n50.py --run-dir {run_dir})", flush=True)


if __name__ == "__main__":
    main()
