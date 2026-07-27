#!/usr/bin/env python3
"""Create an ImageFolder train/eval split for rendered MixedWM38 PNGs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_ROOT = Path("E:/data/images")
SRC_ALL = IMAGE_ROOT / "mixedwm38" / "rendered" / "all"
OUT_ROOT = IMAGE_ROOT / "mixedwm38_train_eval"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def class_images(root: Path, cls: str) -> list[Path]:
    return sorted(p for p in (root / cls).rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-all", default=str(SRC_ALL), help="MixedWM38 all ImageFolder root.")
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="Output root.")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    src_all = Path(args.src_all).resolve()
    out_root = Path(args.out_root).resolve()
    if not src_all.is_dir():
        raise SystemExit(f"src-all not found: {src_all}")
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    classes = sorted(d.name for d in src_all.iterdir() if d.is_dir())
    counts: dict[str, dict[str, int]] = {}
    for cls in classes:
        imgs = class_images(src_all, cls)
        shuffled = list(imgs)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * args.train_ratio))
        train_imgs = sorted(shuffled[:n_train])
        eval_imgs = sorted(shuffled[n_train:])
        for src in train_imgs:
            copy_file(src, out_root / "train" / cls / src.name)
        for src in eval_imgs:
            copy_file(src, out_root / "eval" / cls / src.name)
        counts[cls] = {"train": len(train_imgs), "eval": len(eval_imgs), "total": len(imgs)}

    summary = {
        "src_all": str(src_all),
        "out_root": str(out_root),
        "train": str((out_root / "train").resolve()),
        "eval": str((out_root / "eval").resolve()),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "classes": classes,
        "counts": counts,
        "totals": {
            "train": sum(v["train"] for v in counts.values()),
            "eval": sum(v["eval"] for v in counts.values()),
            "total": sum(v["total"] for v in counts.values()),
        },
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
