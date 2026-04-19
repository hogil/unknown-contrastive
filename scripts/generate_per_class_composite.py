#!/usr/bin/env python3
"""
Per-class composite map smoke/eval tool.

학습 전에 "같은 GT 클래스의 N장으로 만든 average composite"이 class 특성을
얼마나 잘 재현하는지 육안 평가용. 실제 학습 후 cluster-level composite이
올바르게 돌아갈지에 대한 sanity check.

Usage
-----
    python scripts/generate_per_class_composite.py \\
        --data-root data/wm811k \\
        --out-dir   outputs_smoke/composite_per_class_96 \\
        --target    96 96 \\
        --n         10

각 클래스 하위 폴더에서 앞 N장을 뽑아(=랜덤 아님, 재현성용) composite를
생성한다. WM-811K처럼 wafer 크기가 가변이면 --target으로 고정 리사이즈 필수.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (공용 모듈 import용)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.composite import render_composite_png
from common.palette_io import stack_and_normalize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/wm811k",
                    help="ImageFolder-style root (<data-root>/<class>/*.png)")
    ap.add_argument("--out-dir", default="outputs_smoke/composite_per_class_96",
                    help="composite PNGs go here")
    ap.add_argument("--target", nargs=2, type=int, default=[96, 96],
                    metavar=("H", "W"),
                    help="force-resize target before stacking (0 0 = use first image size)")
    ap.add_argument("--n", type=int, default=10,
                    help="samples per class (앞 N장, 결정적)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    target_size = tuple(args.target) if args.target[0] > 0 else None

    classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if not classes:
        print(f"[err] no class directories under {data_root}", file=sys.stderr)
        return 1

    print(f"classes ({len(classes)}): {classes}")
    print(f"target_size: {target_size}, n_per_class: {args.n}")
    print()

    for cls in classes:
        pngs = sorted(data_root.joinpath(cls).glob("*.png"))[: args.n]
        if not pngs:
            print(f"[{cls}] no png files, skip")
            continue
        stacked, _invalid = stack_and_normalize([str(p) for p in pngs],
                                                target_size=target_size)
        res = render_composite_png(
            stacked,
            out_root / f"{cls}_composite.png",
            title=cls,
        )
        p95 = res["square_mean_stats"]["p95"]
        print(f"[{cls:12s}] N={res['image_count']:2d} "
              f"HxW={res['height']}x{res['width']} p95={p95:.2f}")

    print()
    print(f"[done] outputs -> {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
