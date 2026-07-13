#!/usr/bin/env python3
"""Explain strict-novel grouping changes against the frozen hard-unknown baseline."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score

from cluster_metrics import capture_metrics
from cluster_scoring import l2, tier1


ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "data" / "images" / "unknown_eval100"
FROZEN = ROOT / "result_grouping" / "_field_robust" / "embeddings" / "frozen_unknown_dinov3_grade_only_260709.npy"
CANDIDATE = ROOT / "result_grouping" / "_unknown_mixed260710" / "embeddings" / "unkda_nv050_ep6.npy"
EXCLUDED = (
    "Normal,Random,R,Center_bank_boundary,Center_scratch,Donut_bank_boundary,"
    "Donut_fork,Edge-Ring_bank_boundary,Edge-Ring_scratch,Edge-Top_fork,"
    "Full_scratch,ParallelScratches,RingDots"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "paper" / "canonical_rescore_260713" / "unknown_strict_novel",
    )
    return parser.parse_args()


def labels_pool(pool: Path) -> list[str]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [path.parent.name for path in sorted(pool.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes]


def finch_p2(z: np.ndarray) -> np.ndarray:
    from finch import FINCH

    clusters, _, _ = FINCH(z, verbose=False)
    if clusters.shape[1] < 3:
        raise RuntimeError("FINCH did not produce p2")
    return clusters[:, 2]


def dominant_by_cluster(pred: np.ndarray, labels: np.ndarray) -> dict[int, str | None]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for cluster_id, label in zip(pred, labels):
        if int(cluster_id) >= 0:
            counts[int(cluster_id)][str(label)] += 1
    result: dict[int, str | None] = {}
    for cluster_id, values in counts.items():
        largest = max(values.values())
        winners = [label for label, count in values.items() if count == largest]
        result[cluster_id] = winners[0] if len(winners) == 1 else None
    return result


def summary(z: np.ndarray, pred: np.ndarray, labels: np.ndarray, excluded: set[str]) -> dict[str, object]:
    target = ~np.isin(labels, list(excluded))
    classes = sorted(set(labels[target].tolist()), key=str)
    true_idx = np.asarray([classes.index(label) if label in classes else -1 for label in labels])
    result = tier1(pred, true_idx, labels, classes, excluded)
    ids = set(pred.tolist()) - {-1}
    result["k_total"] = len(ids)
    result["fragment_ratio"] = len(ids) / max(1, len(classes))
    target_pred = pred[target]
    non_noise = target_pred != -1
    result["sil"] = (
        float(silhouette_score(z[target][non_noise], target_pred[non_noise], metric="cosine"))
        if non_noise.sum() > 10 and len(set(target_pred[non_noise].tolist())) > 1
        else float("nan")
    )
    return result


def class_rows(pred: np.ndarray, labels: np.ndarray, excluded: set[str]) -> list[dict[str, object]]:
    capture = capture_metrics(pred, labels, excluded)
    dominant = dominant_by_cluster(pred, labels)
    captured = set(capture["captured_classes"])
    rows = []
    for label in sorted(set(labels.tolist()) - excluded, key=str):
        indices = np.where(labels == label)[0]
        clustered = indices[pred[indices] >= 0]
        cluster_counts = Counter(pred[clustered].tolist())
        if cluster_counts:
            cluster_id, count = cluster_counts.most_common(1)[0]
            main = dominant.get(int(cluster_id))
        else:
            cluster_id, count, main = -1, 0, None
        rows.append(
            {
                "class": label,
                "captured": label in captured,
                "best_cluster": int(cluster_id),
                "best_cluster_class_count": int(count),
                "class_total": int(len(indices)),
                "best_cluster_coverage": float(count / max(1, len(indices))),
                "best_cluster_main": "" if main is None else main,
                "best_cluster_is_main": main == label,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    candidate_key = args.candidate.stem.removeprefix("unkda_")
    labels = np.asarray(labels_pool(args.pool.resolve()))
    excluded = {value.strip() for value in EXCLUDED.split(",") if value.strip()}
    frozen_z = l2(np.load(args.frozen.resolve()).astype(np.float32))
    candidate_z = l2(np.load(args.candidate.resolve()).astype(np.float32))
    if not (len(labels) == frozen_z.shape[0] == candidate_z.shape[0]):
        raise SystemExit("label and embedding row counts differ")
    frozen_pred = finch_p2(frozen_z)
    candidate_pred = finch_p2(candidate_z)
    frozen_summary = summary(frozen_z, frozen_pred, labels, excluded)
    candidate_summary = summary(candidate_z, candidate_pred, labels, excluded)
    frozen_rows = {row["class"]: row for row in class_rows(frozen_pred, labels, excluded)}
    candidate_rows = {row["class"]: row for row in class_rows(candidate_pred, labels, excluded)}
    deltas = []
    for label in sorted(frozen_rows, key=str):
        before, after = frozen_rows[label], candidate_rows[label]
        deltas.append(
            {
                "class": label,
                "frozen_captured": before["captured"],
                "candidate_captured": after["captured"],
                "capture_change": int(after["captured"]) - int(before["captured"]),
                "frozen_best_coverage": before["best_cluster_coverage"],
                "candidate_best_coverage": after["best_cluster_coverage"],
                "coverage_delta": after["best_cluster_coverage"] - before["best_cluster_coverage"],
                "frozen_main": before["best_cluster_main"],
                "candidate_main": after["best_cluster_main"],
                "frozen_best_cluster": before["best_cluster"],
                "candidate_best_cluster": after["best_cluster"],
            }
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{candidate_key}_vs_frozen_class_delta.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0].keys()))
        writer.writeheader()
        writer.writerows(deltas)
    metrics = [
        ("P1", "capture"), ("P2", "noise_pct"), ("P3", "completeness"),
        ("P4", "homogeneity"), ("ARI", "ari"), ("Sil", "sil"),
        ("k", "k_total"), ("fragment", "fragment_ratio"),
    ]
    regained = [row["class"] for row in deltas if row["capture_change"] > 0]
    lost = [row["class"] for row in deltas if row["capture_change"] < 0]
    markdown = [
        "# Hard-Unknown FINCH-p2 Delta Diagnosis",
        "",
        "Both rows cluster the full pool and score the same 32 strict-novel defect classes.",
        "",
        f"| Metric | DINOv3 frozen | {candidate_key} | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in metrics:
        markdown.append(
            f"| {label} | {float(frozen_summary[key]):.4f} | {float(candidate_summary[key]):.4f} | "
            f"{float(candidate_summary[key]) - float(frozen_summary[key]):+.4f} |"
        )
    markdown.extend(
        [
            "",
            f"- Regained unique-dominant captures: {', '.join(regained) if regained else 'none'}.",
            f"- Lost unique-dominant captures: {', '.join(lost) if lost else 'none'}.",
            "- The detailed CSV reports the target class's largest-cluster coverage and the main class of that cluster; it distinguishes boundary reassignment from simple cluster-count change.",
            "",
        ]
    )
    md_path = output_dir / f"{candidate_key}_vs_frozen_diagnosis.md"
    md_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"[OUT] {csv_path}")
    print(f"[OUT] {md_path}")


if __name__ == "__main__":
    main()
