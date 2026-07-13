#!/usr/bin/env python3
"""Summarize completed source-faithful May B0-B5 cells in canonical metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "runs" / "may37_original_reproduction"
CELL_ORDER = {"FROZEN": 0, "B0": 1, "B1": 2, "B2": 3, "B3": 4, "B4": 5, "B5": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    rows: list[dict] = []
    for metrics_path in results_root.glob("*_mayexact_*_*/canonical_eval/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parents[1]
        metrics["run_dir"] = str(run_dir)
        metrics["metrics_path"] = str(metrics_path)
        rows.append(metrics)
    if not rows:
        raise SystemExit(f"No completed canonical May cells under {results_root}")
    rows.sort(key=lambda row: (str(row["backbone"]), CELL_ORDER.get(str(row["cell"]), 99)))
    fieldnames = [
        "backbone", "cell", "embedding", "P1_capture", "P1_cap", "P2_noise_pct",
        "P3_completeness", "P4_homogeneity", "ARI", "AMI", "Sil_cos", "k",
        "fragment_ratio", "dominant_cluster_count", "legacy_presence_capture",
        "legacy_presence_rate", "class_coverage", "run_dir", "metrics_path",
    ]
    csv_path = results_root / "canonical_may37_original_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Source-Faithful May B0-B5 Reproduction",
        "",
        "Archived loss/wrapper source: `b796ecbe5f70c9b88944480292e12706b64db83b`. "
        "P1 is current canonical unique dominant/main-class capture computed after full-pool HDBSCAN.",
        "",
        "`FROZEN` is evaluated in backbone space. B0–B5 are evaluated in the source-faithful projection space. "
        "The mode is shown explicitly because the historical trainer used that contract.",
        "",
        "| Backbone | Cell | Embedding | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Dominant clusters | Legacy presence (audit) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['cell']} | {row['embedding']} | {row['P1_capture']} ({float(row['P1_cap']):.3f}) | "
            f"{float(row['P2_noise_pct']):.2f} | {float(row['P3_completeness']):.3f} | {float(row['P4_homogeneity']):.3f} | "
            f"{float(row['ARI']):.3f} | {float(row['Sil_cos']):.3f} | {int(row['k'])} | {float(row['fragment_ratio']):.2f} | "
            f"{int(row['dominant_cluster_count'])} | {row['legacy_presence_capture']} |"
        )
    lines.append("")
    for backbone in sorted({str(row["backbone"]) for row in rows}):
        subset = {str(row["cell"]): row for row in rows if str(row["backbone"]) == backbone}
        if "B0" in subset and "B5" in subset:
            b0, b5 = subset["B0"], subset["B5"]
            lines.append(
                f"- {backbone} B0 -> B5: ARI {float(b0['ARI']):.3f} -> {float(b5['ARI']):.3f} "
                f"(delta {float(b5['ARI']) - float(b0['ARI']):+.3f}); P1 {b0['P1_capture']} -> {b5['P1_capture']}."
            )
    lines.extend(
        [
            "",
            "`legacy_presence_capture` is retained only to compare historical tables. It is not P1 and must not be used as an acceptance metric.",
            "",
        ]
    )
    md_path = results_root / "canonical_may37_original_ablation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {md_path}")


if __name__ == "__main__":
    main()
