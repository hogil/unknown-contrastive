#!/usr/bin/env python3
"""Re-score retained cross-dataset embeddings with canonical full-pool P1.

This does not train a model.  It makes the frozen and selected contrastive
embeddings comparable after the P1 definition was tightened to require a
cluster's dominant/main class.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "result_grouping" / "_field_robust" / "embeddings"
SCORE = ROOT / "_score_umapfree.py"

ROWS = (
    {
        "dataset": "WM-811K",
        "protocol": "strict-novel / Normal-only train",
        "primary": "finch_p2(",
        "pool": ROOT / "data" / "images" / "wm811k_eval500_512" / "eval",
        "label": "frozen",
        "recipe": "DINOv3 frozen",
        "embedding": EMB / "frozen_wm.npy",
    },
    {
        "dataset": "WM-811K",
        "protocol": "strict-novel / Normal-only train",
        "primary": "finch_p2(",
        "pool": ROOT / "data" / "images" / "wm811k_eval500_512" / "eval",
        "label": "candidate",
        "recipe": "SimCLR + queue4096 + local0.3 + ignore0.75, seed4 ep8",
        "embedding": EMB / "wm_l03_ig75_s4_ep8.npy",
    },
    {
        "dataset": "RESISC45",
        "protocol": "transductive self-adaptation",
        "primary": "finch_p1(",
        "pool": ROOT / "data" / "images" / "hf_resisc45",
        "label": "frozen",
        "recipe": "DINOv3 frozen",
        "embedding": EMB / "frozen_rs.npy",
    },
    {
        "dataset": "RESISC45",
        "protocol": "transductive self-adaptation",
        "primary": "finch_p1(",
        "pool": ROOT / "data" / "images" / "hf_resisc45",
        "label": "candidate",
        "recipe": "SimCLR + queue4096 + ignore0.75, seed3 ep6",
        "embedding": EMB / "rs75_s3_ep6.npy",
    },
    {
        "dataset": "DTD",
        "protocol": "transductive self-adaptation",
        "primary": "finch_p1(",
        "pool": ROOT / "data" / "images" / "hf_dtd",
        "label": "frozen",
        "recipe": "DINOv3 frozen",
        "embedding": EMB / "frozen_dtd.npy",
    },
    {
        "dataset": "DTD",
        "protocol": "transductive self-adaptation",
        "primary": "finch_p1(",
        "pool": ROOT / "data" / "images" / "hf_dtd",
        "label": "candidate",
        "recipe": "SimCLR + queue4096 + ignore0.75, seed3 ep8",
        "embedding": EMB / "dtd75_s3_ep8.npy",
    },
)


def score_one(spec: dict) -> list[dict]:
    embedding = Path(spec["embedding"])
    pool = Path(spec["pool"])
    if not embedding.exists():
        raise FileNotFoundError(f"embedding is unavailable: {embedding}")
    if not pool.exists():
        raise FileNotFoundError(f"pool is unavailable: {pool}")
    with tempfile.TemporaryDirectory(prefix="canonical_p1_") as temp_dir:
        output = Path(temp_dir) / "scores.csv"
        command = [
            sys.executable,
            str(SCORE),
            str(embedding),
            "--pool",
            str(pool),
            "--skip-umap",
            "--out-csv",
            str(output),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            {
                "dataset": spec["dataset"],
                "protocol": spec["protocol"],
                "label": spec["label"],
                "recipe": spec["recipe"],
                "primary_prefix": spec["primary"],
            }
        )
    return rows


def as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {None, ""} else float("nan")


def primary_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if str(row["method"]).startswith(str(row["primary_prefix"]))]


def write_markdown(rows: list[dict], output: Path) -> None:
    selected = primary_rows(rows)
    by_dataset: dict[str, dict[str, dict]] = {}
    for row in selected:
        by_dataset.setdefault(row["dataset"], {})[row["label"]] = row
    lines = [
        "# Canonical Cross-Dataset Rescore",
        "",
        "P1 is `cluster dominant/main class count / target class count`, computed after clustering the full pool. ",
        "`legacy_presence_*` remains in the CSV only for historical audit and is not P1.",
        "",
        "| Dataset | Protocol | Recipe | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        p1 = f"{row['P1_capture_count']}/{row['P1_target_class_count']} ({as_float(row, 'P1_capture'):.3f})"
        lines.append(
            "| {dataset} | {protocol} | {recipe} | {p1} | {P2_noise_pct} | {P3_completeness} | "
            "{P4_homogeneity} | {ARI} | {Sil} | {k_total} | {fragment_ratio} |".format(p1=p1, **row)
        )
    lines.extend(["", "## Primary Deltas", ""])
    for dataset, pair in by_dataset.items():
        if {"frozen", "candidate"} <= set(pair):
            frozen, candidate = pair["frozen"], pair["candidate"]
            lines.append(
                f"- {dataset}: ARI {as_float(frozen, 'ARI'):.4f} -> {as_float(candidate, 'ARI'):.4f} "
                f"(delta {as_float(candidate, 'ARI') - as_float(frozen, 'ARI'):+.4f}); "
                f"P1 {frozen['P1_capture_count']}/{frozen['P1_target_class_count']} -> "
                f"{candidate['P1_capture_count']}/{candidate['P1_target_class_count']}."
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "docs" / "paper" / "canonical_rescore_260713"
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for spec in ROWS:
        print(f"[score] {spec['dataset']} {spec['label']} {Path(spec['embedding']).name}", flush=True)
        rows.extend(score_one(spec))
    csv_path = output_dir / "cross_dataset_canonical_scores.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "cross_dataset_canonical_summary.md"
    write_markdown(rows, markdown_path)
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {markdown_path}")


if __name__ == "__main__":
    main()
