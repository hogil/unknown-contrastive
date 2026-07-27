#!/usr/bin/env python3
"""Focused sweep around the best ARI/noise balance from the local follow-up."""
from __future__ import annotations

import argparse
import csv
from typing import Any

from build_simclr_component_report import (
    BASELINES,
    EVAL_IGNORE_CLASSES,
    P_COLS,
    backfill_p_metrics,
    baseline_row,
    list_labels,
)
from run_simclr_component_ablation import EVAL_DIR, REPO, TRAIN_DIR, train_condition


DOC_DIR = REPO / "docs" / "contrastive-eval"
OUT_CSV = DOC_DIR / "SIMCLR_BALANCED_FOLLOWUP_WITH_BASELINES.csv"
OUT_MD = DOC_DIR / "SIMCLR_BALANCED_FOLLOWUP_WITH_BASELINES.md"


def local_args(queue_size: str, ignore: str, weight: str, grid: str = "6", window: str = "1") -> list[str]:
    return [
        "--queue-size", queue_size,
        "--ignore-neg-sim", ignore,
        "--local-weight", weight,
        "--local-grid", grid,
        "--local-window", window,
        "--local-tau", "0.1",
    ]


CONDITIONS: list[dict[str, Any]] = [
    {
        "stage": "F0",
        "name": "queue3072_lrb3e6_local_w0p20",
        "method": "queue3072+lr3e-6+ignore0.70+local_w0.20",
        "lr_backbone": 3e-6,
        "args": local_args("3072", "0.70", "0.20"),
    },
    {
        "stage": "F1",
        "name": "queue3072_lrb3e6_local_w0p15",
        "method": "queue3072+lr3e-6+ignore0.70+local_w0.15",
        "lr_backbone": 3e-6,
        "args": local_args("3072", "0.70", "0.15"),
    },
    {
        "stage": "F2",
        "name": "queue3072_lrb3e6_local_w0p18",
        "method": "queue3072+lr3e-6+ignore0.70+local_w0.18",
        "lr_backbone": 3e-6,
        "args": local_args("3072", "0.70", "0.18"),
    },
    {
        "stage": "F3",
        "name": "queue3072_lrb3e6_ignore0p80_local_w0p20",
        "method": "queue3072+lr3e-6+ignore0.80+local_w0.20",
        "lr_backbone": 3e-6,
        "args": local_args("3072", "0.80", "0.20"),
    },
    {
        "stage": "F4",
        "name": "queue3072_lrb3e6_ignore0p60_local_w0p20",
        "method": "queue3072+lr3e-6+ignore0.60+local_w0.20",
        "lr_backbone": 3e-6,
        "args": local_args("3072", "0.60", "0.20"),
    },
    {
        "stage": "F5",
        "name": "queue2048_lrb3e6_local_w0p15",
        "method": "queue2048+lr3e-6+ignore0.70+local_w0.15",
        "lr_backbone": 3e-6,
        "args": local_args("2048", "0.70", "0.15"),
    },
    {
        "stage": "F6",
        "name": "queue3072_lrb3e6_local_w0p25",
        "method": "queue3072+lr3e-6+ignore0.70+local_w0.25",
        "lr_backbone": 3e-6,
        "args": local_args("3072", "0.70", "0.25"),
    },
    {
        "stage": "F7",
        "name": "queue4096_lrb3e6_local_w0p20",
        "method": "queue4096+lr3e-6+ignore0.70+local_w0.20",
        "lr_backbone": 3e-6,
        "args": local_args("4096", "0.70", "0.20"),
    },
    {
        "stage": "F8",
        "name": "queue3072_lrb3e6_temp0p04_local_w0p20",
        "method": "queue3072+lr3e-6+temp0.04+ignore0.70+local_w0.20",
        "lr_backbone": 3e-6,
        "nce_temp": 0.04,
        "args": local_args("3072", "0.70", "0.20"),
    },
]


def _parse_float(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_existing_rows() -> list[dict[str, Any]]:
    if not OUT_CSV.exists():
        return []
    rows = []
    with OUT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("stage", "").startswith("F"):
                rows.append(backfill_p_metrics({k: _parse_float(v) for k, v in row.items()}))
    return rows


def write_outputs(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], classes: list[str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = [backfill_p_metrics(dict(r)) for r in baseline_rows + rows]
    cols = [
        "stage", "method", "kmeans_ari", "kmeans_nmi", "kmeans_ami",
        *P_COLS,
        "top1", "k3", "k5", "k7", "k9", "dist_ratio",
        "hdbscan_ari", "hdbscan_ami", "hdbscan_noise_pct", "hdbscan_clusters",
        "loss", "nce", "local", "neco", "run_dir", "model_path",
        "embedding_path", "log_path",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in cols})

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# SimCLR Balanced Follow-up With Baselines\n\n")
        f.write(f"- train folder: `{TRAIN_DIR.resolve()}`\n")
        f.write(f"- eval folder: `{EVAL_DIR.resolve()}`\n")
        f.write(f"- eval classes: `{', '.join(classes)}`\n")
        f.write(f"- ignored eval classes: `{', '.join(sorted(EVAL_IGNORE_CLASSES))}`\n")
        f.write("- data: WM-811K class-disjoint v1, not generated synthetic wafer\n")
        f.write("- eval embedding: full 1024-dim backbone feature, L2-normalized\n")
        f.write("- primary metric: k-means(k=#eval classes) ARI on held-out shape classes\n")
        f.write("- P1 capture: fraction of eval classes that own at least one dominant-class cluster\n")
        f.write("- image cap: auxiliary dominant-class image coverage\n")
        f.write("- purpose: combine L4 queue3072 ARI strength with L8 lr3e-6 P1/P2 balance\n\n")
        f.write("| Stage | Method | ARI | NMI | AMI | P1 capture | image cap | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in all_rows:
            f.write(
                f"| {row['stage']} | {row['method']} | {float(row.get('kmeans_ari') or 0):.4f} | "
                f"{float(row.get('kmeans_nmi') or 0):.4f} | {float(row.get('kmeans_ami') or 0):.4f} | "
                f"{float(row.get('p1_capture_rate') or 0):.4f} | "
                f"{float(row.get('p1_image_capture_rate') or 0):.4f} | "
                f"{float(row.get('p2_noise_pct') or 0):.2f}% | "
                f"{float(row.get('p3_completeness') or 0):.4f} | "
                f"{float(row.get('p4_homogeneity') or 0):.4f} | "
                f"{float(row.get('top1') or 0) * 100:.2f}% | {float(row.get('k5') or 0) * 100:.2f}% | "
                f"{float(row.get('k9') or 0) * 100:.2f}% | {float(row.get('dist_ratio') or 0):.4f} | "
                f"{float(row.get('hdbscan_ari') or 0):.4f} |\n"
            )
        f.write("\n## Artifacts\n\n")
        for row in all_rows:
            f.write(f"- {row['stage']} {row['method']}\n")
            f.write(f"  - run: `{row.get('run_dir', '')}`\n")
            f.write(f"  - model: `{row.get('model_path', '')}`\n")
            f.write(f"  - embedding: `{row.get('embedding_path', '')}`\n")
            if row.get("log_path"):
                f.write(f"  - log: `{row['log_path']}`\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--lr-backbone", type=float, default=2e-6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--start-at", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    y, classes = list_labels(EVAL_DIR)
    baseline_rows = [baseline_row(spec, y) for spec in BASELINES]
    rows = read_existing_rows()
    existing_stages = {r.get("stage") for r in rows}
    started = args.start_at is None

    for cond in CONDITIONS:
        if args.only and args.only not in {cond["stage"], cond["name"]}:
            continue
        if not started:
            started = args.start_at in {cond["stage"], cond["name"]}
        if not started or cond["stage"] in existing_stages:
            continue
        row = train_condition(cond, args.epochs, args.batch, args.img_size, args.lr_backbone, args.lr_head)
        rows.append(row)
        write_outputs(rows, baseline_rows, classes)
        print(f"[sweep] updated {OUT_MD.resolve()}", flush=True)

    write_outputs(rows, baseline_rows, classes)
    print(f"[OUT] {OUT_MD.resolve()}")
    print(f"[OUT] {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
