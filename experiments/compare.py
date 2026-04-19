#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate eval_summary.json files from a sweep directory into a CSV table.

Usage
-----
    python -m experiments.compare /path/to/outputs_exp/ > comparison.csv
    python -m experiments.compare /path/to/outputs_exp/ --format tsv

Columns
-------
    preset, n_samples, n_clusters_found, noise_ratio,
    ari, nmi, cluster_purity, epochs, temp, wall_time_sec, run_dir
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


COLUMNS = [
    "preset",
    "n_samples",
    "n_clusters_found",
    "noise_ratio",
    "ari",
    "nmi",
    "cluster_purity",
    "epochs",
    "temp",
    "wall_time_sec",
    "run_dir",
]


def _find_summaries(root: Path) -> List[Path]:
    """Recursively find eval_summary.json files."""
    return sorted(root.rglob("eval_summary.json"))


def _row_from_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception as e:
        print(f"[compare] WARN: failed to read {path}: {e}", file=sys.stderr)
        return None

    m = s.get("metrics", {})
    cfg = s.get("cfg_snapshot", {})
    return {
        "preset": s.get("preset", ""),
        "n_samples": m.get("n_samples", ""),
        "n_clusters_found": m.get("n_clusters_found", ""),
        "noise_ratio": s.get(
            "derived_noise_ratio",
            m.get("noise_ratio", ""),
        ),
        "ari": m.get("ari", ""),
        "nmi": m.get("nmi", ""),
        "cluster_purity": m.get("cluster_purity", ""),
        "epochs": cfg.get("EPOCHS", ""),
        "temp": cfg.get("TEMP", ""),
        "wall_time_sec": s.get("wall_time_sec", ""),
        "run_dir": s.get("run_dir", str(path.parent)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="experiments.compare",
        description=(
            "Aggregate eval_summary.json files produced by "
            "experiments.run_experiment into a CSV/TSV comparison table."
        ),
    )
    p.add_argument(
        "root",
        help="Directory to recursively search for eval_summary.json files.",
    )
    p.add_argument(
        "--format",
        choices=("csv", "tsv"),
        default="csv",
        help="Output format (default: csv).",
    )
    p.add_argument(
        "--sort-by",
        default="ari",
        choices=COLUMNS,
        help="Column to sort rows by (descending). Default: ari.",
    )
    p.add_argument(
        "--output",
        "-o",
        default="-",
        help="Output file path, or '-' for stdout (default).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[compare] root does not exist: {root}", file=sys.stderr)
        return 2

    summaries = _find_summaries(root)
    if not summaries:
        print(
            f"[compare] no eval_summary.json found under {root}",
            file=sys.stderr,
        )
        return 1

    rows: List[Dict[str, Any]] = []
    for p in summaries:
        r = _row_from_summary(p)
        if r is not None:
            rows.append(r)

    # Sort descending by the chosen numeric column where possible.
    def _sort_key(r: Dict[str, Any]):
        v = r.get(args.sort_by, "")
        try:
            return -float(v)
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=_sort_key)

    delim = "," if args.format == "csv" else "\t"

    if args.output == "-":
        fh = sys.stdout
        close = False
    else:
        fh = open(args.output, "w", encoding="utf-8", newline="")
        close = True

    try:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=delim)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    finally:
        if close:
            fh.close()

    print(f"[compare] wrote {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
