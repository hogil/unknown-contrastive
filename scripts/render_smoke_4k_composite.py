#!/usr/bin/env python3
"""Re-render the 4 smoke-run KEPT cluster composites at 4000x4000.

Inputs:
    outputs_cpu_smoke/baseline_260420_181745/clusters/hdbscan/cluster_0{00..03}_*/

Outputs:
    outputs_cpu_smoke/baseline_260420_181745/cluster_summary/composite_4k/cluster_XXX_composite.png

Note: the original smoke run did not persist embeddings, so true medoid ranking
is not available. This script stacks ALL PNG members of each cluster folder
(size 41 / 20 / 17 / 100) at target_size=(4000, 4000), which is a strictly
more representative composite than the stale n=10 medoid subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.composite import render_composite_png
from common.palette_io import stack_and_normalize


RUN_DIR = ROOT / "outputs_cpu_smoke" / "baseline_260420_181745"
CLUSTERS_DIR = RUN_DIR / "clusters" / "hdbscan"
OUT_DIR = RUN_DIR / "cluster_summary" / "composite_4k"
TARGET = (4000, 4000)


def main() -> int:
    if not CLUSTERS_DIR.is_dir():
        print(f"[err] missing clusters dir: {CLUSTERS_DIR}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cluster_dirs = sorted(
        d for d in CLUSTERS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("cluster_")
    )
    if not cluster_dirs:
        print(f"[err] no cluster_* subdirs under {CLUSTERS_DIR}", file=sys.stderr)
        return 2

    results = []
    for cdir in cluster_dirs:
        cid = cdir.name.split("_")[1]  # "000"
        if cid == "-1":
            print(f"[skip] {cdir.name} (noise)")
            continue

        pngs = sorted(cdir.glob("*.png"))
        if not pngs:
            print(f"[warn] {cdir.name}: no PNG members; skip")
            continue

        print(f"[{cdir.name}] stacking n={len(pngs)} at {TARGET}...", flush=True)
        stacked, _invalid = stack_and_normalize([str(p) for p in pngs], target_size=TARGET)

        out_path = OUT_DIR / f"cluster_{cid}_composite.png"
        info = render_composite_png(
            stacked,
            out_path,
            title=f"cluster_{cid} (n={stacked.shape[0]}, 4k)",
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(
            f"  -> {out_path.name}  "
            f"{info['width']}x{info['height']}  n={info['image_count']}  "
            f"{size_mb:.2f} MB  p95={info['square_mean_stats']['p95']:.2f}"
        )
        results.append((cid, int(info["image_count"]), size_mb))

    print()
    print(f"[done] {len(results)} composites -> {OUT_DIR}")
    for cid, n, mb in results:
        print(f"  cluster_{cid}: n={n}, {mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
