#!/usr/bin/env python3
"""Summarize the paired May-source protocol control without mixing scopes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "runs" / "may37_protocol_control_current2260"
CELL_ORDER = {
    "FROZEN": 0,
    "PCA128": 1,
    "RANDOM128": 2,
    "B0": 3,
    "B1": 4,
    "B2": 5,
    "B3": 6,
    "B4": 7,
    "B5": 8,
    "B6": 9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    rows: list[dict] = []
    tier1_rows: list[dict] = []
    multicluster_rows: list[dict] = []
    for metrics_path in results_root.glob("*/canonical_eval/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parents[1]
        backbone_metrics_path = metrics_path.with_name("metrics_backbone.json")
        if backbone_metrics_path.exists():
            backbone_metrics = json.loads(backbone_metrics_path.read_text(encoding="utf-8"))
            metrics["backbone_embedding_sha256"] = backbone_metrics.get("embedding_sha256", "")
            metrics["backbone_ARI"] = backbone_metrics.get("ARI", "")
        metrics["run_dir"] = str(run_dir)
        metrics["metrics_path"] = str(metrics_path)
        rows.append(metrics)
        tier1_path = metrics_path.with_name("historical_tier1_defect_only.json")
        if tier1_path.exists():
            tier1 = json.loads(tier1_path.read_text(encoding="utf-8"))
            tier1["run_dir"] = str(run_dir)
            tier1["metrics_path"] = str(tier1_path)
            tier1_rows.append(tier1)
        multicluster_path = metrics_path.with_name(f"multicluster_{metrics['embedding']}.csv")
        if multicluster_path.exists():
            with multicluster_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    method = str(row.get("method", ""))
                    if method.startswith("finch_p2(") or method == "louvain_res6":
                        row.update(
                            {
                                "backbone": metrics["backbone"],
                                "cell": metrics["cell"],
                                "embedding_space": metrics["embedding"],
                                "run_dir": str(run_dir),
                            }
                        )
                        multicluster_rows.append(row)
    if not rows:
        raise SystemExit(f"No completed canonical May cells under {results_root}")
    rows.sort(key=lambda row: (str(row["backbone"]), CELL_ORDER.get(str(row["cell"]), 99)))
    fieldnames = [
        "backbone", "cell", "embedding", "P1_capture", "P1_cap", "P2_noise_pct",
        "P3_completeness", "P4_homogeneity", "ARI", "AMI", "Sil_cos", "k",
        "fragment_ratio", "dominant_cluster_count", "legacy_presence_capture",
        "legacy_presence_rate", "class_coverage", "embedding_sha256",
        "backbone_embedding_sha256", "backbone_ARI", "run_dir", "metrics_path",
    ]
    csv_path = results_root / "operational_full_pool_canonical.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tier1_rows.sort(key=lambda row: (str(row["backbone"]), CELL_ORDER.get(str(row["cell"]), 99)))
    tier1_csv_path = results_root / "historical_tier1_defect_only.csv"
    if tier1_rows:
        with tier1_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(tier1_rows)
    multicluster_rows.sort(
        key=lambda row: (
            str(row["backbone"]),
            CELL_ORDER.get(str(row["cell"]), 99),
            str(row["method"]),
        )
    )
    multicluster_csv_path = results_root / "operational_finch_louvain_sanity.csv"
    if multicluster_rows:
        with multicluster_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(multicluster_rows[0].keys()))
            writer.writeheader()
            writer.writerows(multicluster_rows)
    lines = [
        "# May Source-Protocol Paired Control",
        "",
        "Archived loss/wrapper source: `b796ecbe5f70c9b88944480292e12706b64db83b`. "
        "The original May `file_list.parquet` is unavailable, so this is a manifest-locked paired control, not an exact historical-score reproduction.",
        "",
        "`FROZEN`, `PCA128`, and `RANDOM128` are no-training controls; B0–B6 use trained projection `z`. "
        "P1 is current canonical unique dominant/main-class capture, never the legacy presence capture.",
        "",
        "## Operational Full-Pool Canonical",
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
        frozen_sha = str(subset.get("FROZEN", {}).get("embedding_sha256", ""))
        if frozen_sha:
            mismatched = [
                cell for cell, row in subset.items()
                if cell.startswith("B")
                and str(row.get("backbone_embedding_sha256", "")) != frozen_sha
            ]
            lines.append(
                f"- {backbone} frozen-f checksum: `{frozen_sha}`; "
                f"trained backbone mismatch cells: {mismatched or 'none'} (see mode-specific metrics)."
            )
        if "B0" in subset and "B6" in subset:
            b0, b6 = subset["B0"], subset["B6"]
            lines.append(
                f"- {backbone} B0 -> B6: ARI {float(b0['ARI']):.3f} -> {float(b6['ARI']):.3f} "
                f"(delta {float(b6['ARI']) - float(b0['ARI']):+.3f}); P1 {b0['P1_capture']} -> {b6['P1_capture']}."
            )
    if multicluster_rows:
        lines.extend(
            [
                "",
                "## Operational FINCH/Louvain Sanity",
                "",
                "FINCH and Louvain are clusterer sanity checks, not datasets. P2 is zero unless a method emits explicit noise.",
                "",
                "| Backbone | Cell | Embedding | Clusterer | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |",
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in multicluster_rows:
            p1 = f"{row['P1_capture_count']}/{row['P1_target_class_count']} ({float(row['P1_capture']):.3f})"
            lines.append(
                f"| {row['backbone']} | {row['cell']} | {row['embedding_space']} | {row['method']} | {p1} | "
                f"{float(row['P2_noise_pct']):.2f} | {float(row['P3_completeness']):.3f} | "
                f"{float(row['P4_homogeneity']):.3f} | {float(row['ARI']):.3f} | "
                f"{float(row['Sil']) if row['Sil'] not in ('', None) else 0.0:.3f} | "
                f"{int(row['k_total'])} | {float(row['fragment_ratio']):.2f} |"
            )
    if tier1_rows:
        lines.extend(
            [
                "",
                "## May-Style Tier1 Defect-Only",
                "",
                "This table is HDBSCAN on defects only (`eom`, `mcs=12`, `ms=3`, `eps=0`). It is reported separately because it is not the operational full-pool protocol.",
                "",
                "| Backbone | Cell | Embedding | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Dominant clusters | Legacy presence (audit) |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in tier1_rows:
            lines.append(
                f"| {row['backbone']} | {row['cell']} | {row['embedding']} | {row['P1_capture']} ({float(row['P1_cap']):.3f}) | "
                f"{float(row['P2_noise_pct']):.2f} | {float(row['P3_completeness']):.3f} | {float(row['P4_homogeneity']):.3f} | "
                f"{float(row['ARI']):.3f} | {float(row['Sil_cos']):.3f} | {int(row['k'])} | {float(row['fragment_ratio']):.2f} | "
                f"{int(row['dominant_cluster_count'])} | {row['legacy_presence_capture']} |"
            )
    lines.extend(
        [
            "",
            "`legacy_presence_capture` is retained only to compare historical tables. It is not P1 and must not be used as an acceptance metric.",
            "",
        ]
    )
    md_path = results_root / "may37_protocol_control_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] {csv_path}")
    if tier1_rows:
        print(f"[OUT] {tier1_csv_path}")
    if multicluster_rows:
        print(f"[OUT] {multicluster_csv_path}")
    print(f"[OUT] {md_path}")


if __name__ == "__main__":
    main()
