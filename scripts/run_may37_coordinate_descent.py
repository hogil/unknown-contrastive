#!/usr/bin/env python3
"""Coordinate-descent reproduction of the May FCMAE contrastive recipe.

Each stage sweeps one axis from the previously accepted recipe. A candidate is
accepted only when canonical P1/P2/P3/P4/ARI all do not regress. This avoids
carrying a losing component into the next stage, unlike the historical B0-B5
component-isolation matrix.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_may37_original_ablation.py"
DEFAULT_ROOT = ROOT / "runs" / "may37_coordinate_descent"


def load_runner():
    spec = importlib.util.spec_from_file_location("may37_coordinate_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_recipe() -> dict[str, float | int | bool]:
    return {
        "local": False,
        "local_weight": 0.0,
        "queue": False,
        "queue_size": 4096,
        "ignore": 1.0,
        "neco": 0.0,
        "temp": 0.07,
        "lr_head": 1e-3,
    }


def stage_candidates(stage: str, recipe: dict[str, float | int | bool]):
    values: list[tuple[str, float | int]] = {
        "local": [("local030", 0.3), ("local050", 0.5), ("local070", 0.7), ("local100", 1.0)],
        "queue": [("queue1024", 1024), ("queue4096", 4096), ("queue8192", 8192)],
        "ignore": [("ignore080", 0.80), ("ignore075", 0.75), ("ignore072", 0.72), ("ignore065", 0.65)],
        "neco": [("neco005", 0.05), ("neco010", 0.10), ("neco020", 0.20), ("neco030", 0.30)],
        "temp": [("temp040", 0.04), ("temp050", 0.05), ("temp060", 0.06)],
        "lr": [("lr0003", 3e-4), ("lr0005", 5e-4)],
    }[stage]
    result = []
    for suffix, value in values:
        candidate = deepcopy(recipe)
        if stage == "local":
            candidate["local"] = True
            candidate["local_weight"] = value
        elif stage == "queue":
            candidate["queue"] = True
            candidate["queue_size"] = value
        elif stage == "ignore":
            candidate["ignore"] = value
        elif stage == "neco":
            candidate["neco"] = value
        elif stage == "temp":
            candidate["temp"] = value
        elif stage == "lr":
            candidate["lr_head"] = value
        result.append((f"cd_{suffix}", candidate))
    return result


def metric_value(row: dict, key: str) -> float:
    return float(row["metrics"][key])


def passes_core_gate(candidate: dict, incumbent: dict) -> bool:
    return (
        metric_value(candidate, "P1_cap") >= metric_value(incumbent, "P1_cap")
        and metric_value(candidate, "P2_noise_pct") <= metric_value(incumbent, "P2_noise_pct")
        and metric_value(candidate, "P3_completeness") >= metric_value(incumbent, "P3_completeness")
        and metric_value(candidate, "P4_homogeneity") >= metric_value(incumbent, "P4_homogeneity")
        and metric_value(candidate, "ARI") >= metric_value(incumbent, "ARI")
    )


def rank(row: dict) -> tuple[float, ...]:
    metrics = row["metrics"]
    return (
        float(metrics["P1_cap"]),
        -float(metrics["P2_noise_pct"]),
        float(metrics["P3_completeness"]),
        float(metrics["P4_homogeneity"]),
        float(metrics["ARI"]),
        float(metrics["Sil_cos"]),
        -float(metrics["fragment_ratio"]),
    )


def existing_record(output_root: Path, backbone: str, tag: str, recipe: dict | None) -> dict | None:
    for meta_path in output_root.glob("*/coordinate_meta.json"):
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            record.get("backbone") == backbone
            and record.get("tag") == tag
            and record.get("recipe") == recipe
        ):
            metrics_path = meta_path.parent / "canonical_eval" / "metrics.json"
            if metrics_path.exists():
                record["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
                record["run_dir"] = str(meta_path.parent)
                return record
    return None


def execute(runner, output_root: Path, backbone: str, tag: str, recipe: dict | None) -> dict:
    expected_recipe = recipe if tag != "cd_frozen" else {"embedding": "backbone_f", "training": False}
    prior = existing_record(output_root, backbone, tag, expected_recipe)
    if prior is not None:
        print(f"[REUSE] {tag}: {prior['run_dir']}", flush=True)
        return prior

    source_dir = runner.materialize_source(output_root)
    if tag == "cd_frozen":
        run_dir = runner.make_frozen_run(source_dir, output_root, backbone)
        metrics = runner.evaluate_run(source_dir, run_dir, "backbone")
        recipe = {"embedding": "backbone_f", "training": False}
    else:
        if recipe is None:
            raise ValueError(f"recipe is required for {tag}")
        runner.CELLS[tag] = dict(recipe)
        run_dir = runner.run_archived_training(source_dir, output_root, backbone, tag)
        metrics = runner.evaluate_run(source_dir, run_dir, "projection")

    record = {
        "backbone": backbone,
        "tag": tag,
        "recipe": recipe,
        "run_dir": str(run_dir),
        "metrics": metrics,
    }
    (run_dir / "coordinate_meta.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DONE] {tag} P1={metrics['P1_capture']} P2={metrics['P2_noise_pct']} "
        f"P3={metrics['P3_completeness']} P4={metrics['P4_homogeneity']} "
        f"ARI={metrics['ARI']}",
        flush=True,
    )
    return record


def write_report(output_root: Path, backbone: str, frozen: dict, history: list[dict], best: dict) -> None:
    payload = {"backbone": backbone, "frozen": frozen, "history": history, "best": best}
    (output_root / "coordinate_descent_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for event in history:
        for row in event["rows"]:
            metrics = row["metrics"]
            rows.append(
                {
                    "stage": event["stage"],
                    "tag": row["tag"],
                    "selected": row["tag"] == event["winner"],
                    "core_gate_vs_incumbent": row["tag"] == event["incumbent"] or passes_core_gate(row, event["incumbent_row"]),
                    "P1_capture": metrics["P1_capture"],
                    "P1_cap": metrics["P1_cap"],
                    "P2_noise_pct": metrics["P2_noise_pct"],
                    "P3_completeness": metrics["P3_completeness"],
                    "P4_homogeneity": metrics["P4_homogeneity"],
                    "ARI": metrics["ARI"],
                    "Sil_cos": metrics["Sil_cos"],
                    "k": metrics["k"],
                    "fragment_ratio": metrics["fragment_ratio"],
                    "recipe": json.dumps(row["recipe"], sort_keys=True),
                    "run_dir": row["run_dir"],
                }
            )
    csv_path = output_root / "coordinate_descent_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# May Coordinate-Descent Reproduction",
        "",
        "Only a candidate that preserves canonical P1/P2/P3/P4/ARI replaces the incumbent. "
        "All trained rows use historical projection z; frozen is a separate backbone-f diagnostic.",
        "",
        "| Stage | Recipe | Selected | P1 | P2 | P3 | P4 | ARI | Sil | k | Fragment |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['tag']} | {'yes' if row['selected'] else 'no'} | "
            f"{row['P1_capture']} | {float(row['P2_noise_pct']):.2f} | "
            f"{float(row['P3_completeness']):.3f} | {float(row['P4_homogeneity']):.3f} | "
            f"{float(row['ARI']):.3f} | {float(row['Sil_cos']):.3f} | {int(row['k'])} | "
            f"{float(row['fragment_ratio']):.2f} |"
        )
    lines.extend(
        [
            "",
            f"Best trained recipe: `{best['tag']}` at `{best['run_dir']}`.",
            "",
        ]
    )
    (output_root / "coordinate_descent_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["nocnn", "cnn_tapt"], required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    output_root = (args.output_root / args.backbone).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    frozen = execute(runner, output_root, args.backbone, "cd_frozen", None)
    incumbent = execute(runner, output_root, args.backbone, "cd_base", base_recipe())
    history: list[dict] = []

    for stage in ("local", "queue", "ignore", "neco", "temp", "lr"):
        candidates = [incumbent]
        for tag, recipe in stage_candidates(stage, incumbent["recipe"]):
            candidates.append(execute(runner, output_root, args.backbone, tag, recipe))
        passing = [row for row in candidates if row["tag"] == incumbent["tag"] or passes_core_gate(row, incumbent)]
        winner = max(passing, key=rank)
        history.append(
            {
                "stage": stage,
                "incumbent": incumbent["tag"],
                "incumbent_row": incumbent,
                "winner": winner["tag"],
                "rows": candidates,
            }
        )
        incumbent = winner
        write_report(output_root, args.backbone, frozen, history, incumbent)
        print(f"[SELECT] stage={stage} winner={incumbent['tag']}", flush=True)

    print(f"[OUT] {output_root / 'coordinate_descent_results.md'}", flush=True)


if __name__ == "__main__":
    main()
