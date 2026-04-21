"""Create leave-one/two/three-out training subdirectories for OOD experiments.

Uses symlinks where possible (Windows requires dev mode or admin), falls back to
physical copy. val/test are unchanged — only TRAIN_DIR is pruned.

Usage
-----
    python scripts/make_ood_train.py
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

TRAIN_SRC = Path("data/wm811k_train")
# Named OOD experiments — class names match data/wm811k_train/ sub-dirs exactly.
HOLDOUT_SETS: dict[str, set[str]] = {
    "ood_A": {"Scratch"},
    "ood_B": {"Scratch", "Random"},
    "ood_C": {"Scratch", "Random", "Donut"},
}


def link_or_copy_dir(src: Path, dst: Path) -> str:
    """Return 'symlink' or 'copy' depending on which method succeeded."""
    if dst.exists():
        return "skipped (already exists)"
    try:
        os.symlink(src.absolute(), dst.absolute(), target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst)
        return "copy"


def main() -> int:
    if not TRAIN_SRC.is_dir():
        print(f"[err] source not found: {TRAIN_SRC}", file=sys.stderr)
        return 1

    # Verify classes exist
    existing_classes = {d.name for d in TRAIN_SRC.iterdir() if d.is_dir()}
    for name, holdouts in HOLDOUT_SETS.items():
        missing = holdouts - existing_classes
        if missing:
            print(
                f"[err] {name}: holdout classes not present in {TRAIN_SRC}: {missing}",
                file=sys.stderr,
            )
            return 2

    total_report = []
    for name, holdouts in HOLDOUT_SETS.items():
        dst_root = Path(f"data/wm811k_train_{name}")
        dst_root.mkdir(parents=True, exist_ok=True)
        kept = sorted(existing_classes - holdouts)
        print(f"\n[{name}] holdout={sorted(holdouts)}  kept={kept}")
        methods = set()
        for cls in kept:
            src = TRAIN_SRC / cls
            dst = dst_root / cls
            method = link_or_copy_dir(src, dst)
            methods.add(method)
            n_png = len(list(dst.glob("*.png"))) if dst.exists() else 0
            print(f"   {cls}: {method}  {n_png} PNG")
        total_report.append(
            {
                "name": name,
                "holdouts": sorted(holdouts),
                "kept_classes": kept,
                "method": "/".join(sorted(methods)),
                "out_dir": str(dst_root),
            }
        )

    print("\n[summary]")
    for r in total_report:
        print(f"  {r['name']:6s}  holdout={r['holdouts']}  kept={len(r['kept_classes'])} classes  via {r['method']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
