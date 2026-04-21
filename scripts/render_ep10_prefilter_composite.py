#!/usr/bin/env python3
"""Cluster composite fallback for the 10-ep CPU run.

The 10-ep baseline ended with KEEP=0 (kept_labels.txt empty). Per task #6
fallback spec, render composites for all PRE-FILTER clusters instead. Uses
true top-K medoid selection (cosine distance to cluster centroid) since
`emb.npy` + `cluster_labels.npy` + `files.txt` are persisted (task #7).

This script intentionally inlines the render loop (instead of calling
`cluster_composite.generate_cluster_composites`) so we can capture p95 per
cluster for reporting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster_composite import _select_medoid_indices  # noqa: E402
from common.composite import render_composite_png  # noqa: E402
from common.palette_io import stack_and_normalize  # noqa: E402


RUN_DIR = ROOT / "outputs_cpu_ep10" / "baseline_260420_201121"
DEFAULT_OUT = RUN_DIR / "cluster_summary" / "composite"
TARGET_SIZE = (800, 800)
N_PER_CLUSTER = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="destination dir (default: cluster_summary/composite)")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = RUN_DIR / "emb.npy"
    labels_path = RUN_DIR / "cluster_labels.npy"
    files_path = RUN_DIR / "files.txt"

    missing = [p for p in (emb_path, labels_path, files_path) if not p.exists()]
    if missing:
        print(f"[err] missing artifacts: {missing}", file=sys.stderr)
        return 2

    emb = np.load(emb_path)
    cluster_labels = np.load(labels_path)
    files = files_path.read_text(encoding="utf-8").splitlines()

    prefilter_ids = sorted(int(x) for x in np.unique(cluster_labels) if int(x) != -1)
    print(f"[info] pre-filter clusters: {prefilter_ids}")
    print(f"[info] emb={emb.shape}, files={len(files)}, labels={cluster_labels.shape}")
    print(f"[info] target_size={TARGET_SIZE}, n_per_cluster={N_PER_CLUSTER}")
    print(f"[info] out_dir={out_dir}")
    print()

    rows = []
    for lab in prefilter_ids:
        member_idxs = np.where(cluster_labels == lab)[0]
        top_idxs = _select_medoid_indices(emb, member_idxs, N_PER_CLUSTER)
        paths = [files[int(i)] for i in top_idxs]
        stacked, _invalid = stack_and_normalize(paths, target_size=TARGET_SIZE)
        out_path = out_dir / f"cluster_{lab:03d}_composite.png"
        info = render_composite_png(
            stacked,
            out_path,
            title=f"cluster_{lab:03d} (n={stacked.shape[0]})",
        )
        mb = out_path.stat().st_size / (1024 * 1024)
        stats = info["square_mean_stats"]
        rows.append((lab, member_idxs.size, stacked.shape[0], mb, stats["p95"], stats["max"]))
        print(
            f"  cluster_{lab:03d}  size={member_idxs.size:4d}  n={stacked.shape[0]}  "
            f"{mb:.2f} MB  p95={stats['p95']:.3f}  max={stats['max']:.3f}"
        )

    print()
    print(f"[done] {len(rows)} composites under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
