# -*- coding: utf-8 -*-
"""Compare results_matrix.parquet across multiple stage1/stage2 runs.

Usage:
    python -m chip_multilabel.compare_runs <run_dir> [<run_dir> ...]

Emits a unified table sorted by macro_f1, with provenance tag (stage1_<TS> / stage2_<TS>).
Helps see whether new ideas actually broke previous best.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
from pathlib import Path

import pandas as pd


def load_run(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "results_matrix.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing results_matrix.parquet in {run_dir}")
    df = pd.read_parquet(p)
    df = df.copy()
    df["run_tag"] = run_dir.name
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run_dir(s)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None, help="optional: write merged parquet to this path")
    args = ap.parse_args()

    parts = [load_run(Path(r)) for r in args.runs]
    df = pd.concat(parts, ignore_index=True)
    df_sorted = df.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    cols = ["run_tag", "cell_id", "train_id", "inference_id",
            "macro_f1", "micro_f1", "top1_11class", "temperature", "elapsed_sec"]
    print(f"\n=== Top {args.top} cells across {len(args.runs)} run(s) ===\n")
    print(df_sorted[cols].head(args.top).to_string(index=False))

    print(f"\n=== Per (train_id, inference_id) best macro_f1 across runs ===\n")
    grouped = df.groupby(["train_id", "inference_id"])["macro_f1"].max().reset_index()
    print(grouped.sort_values("macro_f1", ascending=False).head(args.top).to_string(index=False))

    if args.out:
        df.to_parquet(args.out, index=False)
        print(f"\nmerged parquet -> {args.out}")


if __name__ == "__main__":
    main()
