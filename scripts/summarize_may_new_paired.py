#!/usr/bin/env python3
"""Summarize completed May NEW paired-control runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "runs" / "may_new_tapt_removed_paired_2260"
METRICS = (
    "P1_cap",
    "P2_noise_pct",
    "P3_completeness",
    "P4_homogeneity",
    "ARI",
    "Sil_cos",
    "k",
    "fragment_ratio",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(label: str, row: dict) -> str:
    return (
        f"| {label} | {row['P1_capture']} | {float(row['P2_noise_pct']):.2f} | "
        f"{float(row['P3_completeness']):.3f} | {float(row['P4_homogeneity']):.3f} | "
        f"{float(row['ARI']):.3f} | {float(row['Sil_cos']):.3f} | {int(row['k'])} | "
        f"{float(row['fragment_ratio']):.2f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    records = []
    for completion_path in results_root.glob("*/completion.json"):
        completion = load(completion_path)
        if completion.get("cell") != "NEW_FIXED":
            continue
        run_dir = completion_path.parent
        final = load(run_dir / "canonical_eval" / "metrics_projection.json")
        frozen = load(run_dir / "canonical_eval" / "metrics_backbone.json")
        tier1 = load(run_dir / "canonical_eval" / "historical_tier1_defect_only_projection.json")
        trajectory = load(run_dir / "canonical_eval" / "epoch_trajectory_projection.json")
        records.append(
            {
                "backbone": completion["backbone"],
                "seed": int(completion["seed"]),
                "run_dir": run_dir,
                "frozen": frozen,
                "z0": trajectory[0],
                "final": final,
                "tier1": tier1,
                "trajectory": trajectory,
            }
        )
    records.sort(key=lambda row: (row["backbone"], row["seed"]))
    if not records:
        print(f"No completed NEW_FIXED runs under {results_root}")
        return

    lines = [
        "# May NEW TAPT-Removed Paired Control",
        "",
        "Semantic recipe: `Global InfoNCE + NeCo(0.2) + Queue(4096) + NEG(0.72); Local OFF`.",
        "Only the backbone checkpoint changes between paired arms.",
        "Decision priority: P1 capture first, then P2/P3/P4 and k/fragmentation; ARI and Silhouette remain recorded as supporting metrics.",
        "",
        "## Full-Pool Canonical HDBSCAN",
        "",
        "| Backbone/seed/space | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        prefix = f"{record['backbone']} s{record['seed']}"
        lines.append(metric_row(f"{prefix} frozen-f", record["frozen"]))
        lines.append(metric_row(f"{prefix} z0", record["z0"]))
        for row in record["trajectory"][1:]:
            lines.append(metric_row(f"{prefix} z ep{row['epoch']}", row))

    lines.extend(
        [
            "",
            "## May-Style Defect-Only HDBSCAN (Final ep5)",
            "",
            "| Backbone/seed | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        lines.append(metric_row(f"{record['backbone']} s{record['seed']}", record["tier1"]))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["backbone"]].append(record["final"])
    lines.extend(["", "## Final ep5 Seed Aggregate", ""])
    for backbone, rows in sorted(grouped.items()):
        lines.append(f"### {backbone} ({len(rows)} seed)")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|---|---:|---:|")
        for key in METRICS:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            lines.append(f"| {key} | {values.mean():.4f} | {values.std(ddof=1) if len(values) > 1 else 0.0:.4f} |")
        lines.append("")

    report = results_root / "may_new_paired_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OUT] {report}")


if __name__ == "__main__":
    main()
