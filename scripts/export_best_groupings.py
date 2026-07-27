#!/usr/bin/env python3
"""Export image groupings for the best scored embeddings in one condition."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


LOWER_IS_BETTER = {"fragment_ratio", "P2_noise_pct", "k_total", "k_noise"}


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key, "")
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-root", required=True)
    ap.add_argument("--pool", default="", help="Override eval pool. Defaults to condition.json split.eval.")
    ap.add_argument("--scores", default="", help="Override scores CSV. Defaults to <condition-root>/scores_all.csv.")
    ap.add_argument("--cluster-method", default="finch_p2",
                    choices=["finch_p0", "finch_p1", "finch_p2", "finch_p3", "louvain", "hdbscan"])
    ap.add_argument("--metric", default="ARI")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--min-capture", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--copy-groups", action="store_true")
    ap.add_argument("--background-classes", default="Normal,Random,R",
                    help="comma-separated labels treated as background for review-group exclusion")
    ap.add_argument("--exclude-background-min-count", type=int, default=20)
    ap.add_argument("--exclude-background-min-ratio", type=float, default=0.5)
    args = ap.parse_args()

    cond = Path(args.condition_root)
    if not cond.is_absolute():
        cond = REPO / cond
    scores = Path(args.scores) if args.scores else cond / "scores_all.csv"
    if not scores.is_absolute():
        scores = REPO / scores
    meta_path = cond / "condition.json"
    if args.pool:
        pool = Path(args.pool)
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pool = Path(meta["split"]["eval"])
    if not pool.is_absolute():
        pool = REPO / pool

    rows: list[dict[str, str]] = []
    with scores.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("method", "").startswith(args.cluster_method):
                continue
            cap = as_float(row, "P1_capture")
            metric = as_float(row, args.metric)
            if cap is None or metric is None or cap < args.min_capture:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"no rows matched method={args.cluster_method}, metric={args.metric}, min_capture={args.min_capture}")

    reverse = args.metric not in LOWER_IS_BETTER
    rows.sort(key=lambda r: as_float(r, args.metric) or float("-inf"), reverse=reverse)
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        emb = row["embedding"]
        if emb in seen:
            continue
        selected.append(row)
        seen.add(emb)
        if len(selected) >= args.top_n:
            break

    out_root = cond / "groupings"
    out_root.mkdir(parents=True, exist_ok=True)
    exported = []
    for row in selected:
        emb = Path(row["embedding"])
        name = row.get("embedding_name") or emb.stem
        out = out_root / f"{name}_{args.cluster_method}_{args.metric}"
        cmd = [
            sys.executable, str(REPO / "scripts" / "group_saved_embeddings.py"),
            "--embedding", str(emb),
            "--pool", str(pool),
            "--out-dir", str(out),
            "--method", args.cluster_method,
            "--reps", str(args.reps),
            "--background-classes", args.background_classes,
            "--exclude-background-min-count", str(args.exclude_background_min_count),
            "--exclude-background-min-ratio", str(args.exclude_background_min_ratio),
        ]
        if args.copy_groups:
            cmd.append("--copy-groups")
        subprocess.check_call(cmd, cwd=REPO)
        exported.append({
            "embedding": str(emb.resolve()),
            "out_dir": str(out.resolve()),
            "method": row["method"],
            "metric": args.metric,
            "metric_value": row[args.metric],
            "P1_capture": row.get("P1_capture", ""),
            "ARI": row.get("ARI", ""),
            "Sil": row.get("Sil", ""),
            "fragment_ratio": row.get("fragment_ratio", ""),
            "k_total": row.get("k_total", ""),
        })

    summary = out_root / f"selected_{args.cluster_method}_{args.metric}.json"
    summary.write_text(json.dumps(exported, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OUT] {summary.resolve()}")


if __name__ == "__main__":
    main()
