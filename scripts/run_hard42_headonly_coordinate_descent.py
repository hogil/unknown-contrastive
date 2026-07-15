#!/usr/bin/env python3
"""Run resumable one-axis hard-42 head-only ablations for TAPT and no-TAPT."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_hard42_headonly_ablation.py"
EVALUATOR = ROOT / "scripts" / "eval_hard42_headonly_checkpoints.py"
RESULT_AGENT = ROOT / "scripts" / "run_result_analysis_agent.py"
SOURCE_RESULTS = ROOT / "runs" / "may37_protocol_control_current2260"
DEFAULT_OUTPUT = ROOT / "runs" / "hard42_headonly_coordinate_descent"
EXCLUDED = (
    "Center_bank_boundary,Center_scratch,Donut_bank_boundary,Donut_fork,"
    "Edge-Ring_bank_boundary,Edge-Ring_scratch,Edge-Top_fork,Full_scratch,"
    "Normal,ParallelScratches,R,Random,RingDots"
)
DEV_ROOT = ROOT / "data" / "images" / "unknown_eval100"
HOLDOUT_ROOT = ROOT / "data" / "images" / "unknown_holdout_100_260713"

MAY_RECIPES = {
    "B0": {"local_weight": 0.0, "queue_size": 0, "ignore_neg_sim": 1.0, "neco_weight": 0.0},
    "B1": {"local_weight": 0.5, "queue_size": 0, "ignore_neg_sim": 1.0, "neco_weight": 0.0},
    "B2": {"local_weight": 1.0, "queue_size": 0, "ignore_neg_sim": 1.0, "neco_weight": 0.0},
    "B3": {"local_weight": 1.0, "queue_size": 4096, "ignore_neg_sim": 1.0, "neco_weight": 0.0},
    "B4": {"local_weight": 1.0, "queue_size": 4096, "ignore_neg_sim": 0.72, "neco_weight": 0.0},
    "B5": {"local_weight": 1.0, "queue_size": 4096, "ignore_neg_sim": 0.72, "neco_weight": 0.2},
    "B6": {"local_weight": 0.0, "queue_size": 4096, "ignore_neg_sim": 0.72, "neco_weight": 0.2},
}


def base_recipe(seed: int, source_cell: str) -> dict:
    return {
        "head": "mlp",
        "adapter_dim": 128,
        "projection_dim": 128,
        "temperature": 0.07,
        **MAY_RECIPES[source_cell],
        "lr_head": 1e-3,
        "epochs": 10,
        "seed": seed,
        "batch": 4,
    }


def pair_summary(rows: list[dict], epoch: int, mode: str) -> dict | None:
    selected = [
        row for row in rows
        if int(row["epoch"]) == epoch
        and row["embedding_mode"] == mode
        and (row["method"].startswith("finch_p2(") or row["method"] == "louvain_res6")
    ]
    if len(selected) != 2:
        return None
    by_method = {"finch" if row["method"].startswith("finch_p2(") else "louvain": row for row in selected}
    if set(by_method) != {"finch", "louvain"}:
        return None
    values = list(by_method.values())
    p1_min = min(float(row["P1_capture"]) for row in values)
    noise_max = max(float(row["P2_noise_pct"]) for row in values)
    p3_min = min(float(row["P3_completeness"]) for row in values)
    p4_min = min(float(row["P4_homogeneity"]) for row in values)
    ari_min = min(float(row["ARI"]) for row in values)
    ari_mean = sum(float(row["ARI"]) for row in values) / 2.0
    sil_values = [float(row["Sil"]) for row in values if row.get("Sil") not in (None, "")]
    sil_min = min(sil_values) if sil_values else -1.0
    fragment_mean = sum(float(row["fragment_ratio"]) for row in values) / 2.0
    score = (
        0.40 * ari_min
        + 0.15 * ari_mean
        + 0.15 * p3_min
        + 0.15 * p4_min
        + 0.10 * p1_min
        + 0.05 * sil_min
        - 0.01 * abs(math.log(max(fragment_mean, 1e-6)))
        - 0.001 * noise_max
    )
    return {
        "epoch": epoch,
        "mode": mode,
        "finch": by_method["finch"],
        "louvain": by_method["louvain"],
        "p1_min": p1_min,
        "noise_max": noise_max,
        "p3_min": p3_min,
        "p4_min": p4_min,
        "ari_min": ari_min,
        "ari_mean": ari_mean,
        "sil_min": sil_min,
        "fragment_mean": fragment_mean,
        "score": score,
    }


def best_point(metrics_path: Path, allowed_modes: set[str] | None = None) -> dict:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if allowed_modes is not None:
        rows = [row for row in rows if row["embedding_mode"] in allowed_modes]
    keys = sorted({(int(row["epoch"]), row["embedding_mode"]) for row in rows})
    candidates = [pair_summary(rows, epoch, mode) for epoch, mode in keys]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise RuntimeError(f"FINCH-p2/Louvain pair is unavailable: {metrics_path}")
    return max(candidates, key=lambda row: row["score"])


def passes_core_gate(candidate: dict, incumbent: dict, tolerance: float = 1e-6) -> bool:
    for method in ("finch", "louvain"):
        current = candidate[method]
        reference = incumbent[method]
        if not (
            float(current["P1_capture"]) + tolerance >= float(reference["P1_capture"])
            and float(current["P2_noise_pct"]) <= float(reference["P2_noise_pct"]) + tolerance
            and float(current["P3_completeness"]) + tolerance >= float(reference["P3_completeness"])
            and float(current["P4_homogeneity"]) + tolerance >= float(reference["P4_homogeneity"])
            and float(current["ARI"]) + tolerance >= float(reference["ARI"])
        ):
            return False
    return True


def passes_final_gate(candidate: dict, frozen: dict, tolerance: float = 1e-4) -> bool:
    if not passes_core_gate(candidate, frozen, tolerance=1e-6):
        return False
    for method in ("finch", "louvain"):
        current = candidate[method]
        reference = frozen[method]
        if not (
            float(current["P3_completeness"]) > float(reference["P3_completeness"]) + tolerance
            and float(current["P4_homogeneity"]) > float(reference["P4_homogeneity"]) + tolerance
            and float(current["ARI"]) > float(reference["ARI"]) + tolerance
        ):
            return False
    return True


def source_winner(backbone: str) -> str:
    path = SOURCE_RESULTS / "operational_finch_louvain_sanity.csv"
    if not path.exists():
        raise FileNotFoundError(f"May source-protocol summary is unavailable: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["backbone"] == backbone and row["cell"] in MAY_RECIPES]
    candidates = []
    for cell in MAY_RECIPES:
        current = [row for row in rows if row["cell"] == cell]
        if len(current) < 2:
            continue
        mapped = []
        for row in current:
            mapped.append(
                {
                    "epoch": 5,
                    "embedding_mode": row["embedding_space"],
                    **row,
                }
            )
        point = pair_summary(mapped, 5, mapped[0]["embedding_mode"])
        if point is not None:
            candidates.append((cell, point))
    if not candidates:
        raise RuntimeError(f"no paired source cell for {backbone}")
    return max(candidates, key=lambda item: item[1]["score"])[0]


def gpu_available() -> bool:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = [line.strip() for line in apps.stdout.splitlines() if line.strip() and line.strip() != "[N/A]"]
    status = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        util, memory = [int(value.strip()) for value in status.stdout.splitlines()[0].split(",")[:2]]
    except (IndexError, ValueError):
        return False
    return not pids and util <= 30 and memory <= 1200


def wait_for_gpu() -> None:
    while not gpu_available():
        print("[WAIT] GPU is occupied; retrying in 60 seconds", flush=True)
        time.sleep(60)


def analyze_result(event_file: Path) -> None:
    if not event_file.exists():
        raise FileNotFoundError(f"result event is unavailable: {event_file}")
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(RESULT_AGENT),
            "--event-file",
            str(event_file),
            "--context",
            "hard42",
        ],
        cwd=ROOT,
        check=True,
    )


def existing_run(output_root: Path, backbone: str, recipe: dict) -> Path | None:
    for provenance_path in output_root.glob("*/headonly_provenance.json"):
        completion = provenance_path.parent / "headonly_completion.json"
        if not completion.exists():
            continue
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
        if record.get("backbone") == backbone and record.get("recipe") == recipe:
            metrics = provenance_path.parent / "contrastive" / "evaluation" / "dev_strict_novel" / "hard42_dev_strict_novel_metrics.csv"
            if metrics.exists():
                return provenance_path.parent
    return None


def runner_command(backbone: str, cell: str, recipe: dict, output_root: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--backbone", backbone,
        "--cell", cell,
        "--head", str(recipe["head"]),
        "--adapter-dim", str(recipe["adapter_dim"]),
        "--projection-dim", str(recipe["projection_dim"]),
        "--temperature", str(recipe["temperature"]),
        "--queue-size", str(recipe["queue_size"]),
        "--ignore-neg-sim", str(recipe["ignore_neg_sim"]),
        "--local-weight", str(recipe["local_weight"]),
        "--neco-weight", str(recipe["neco_weight"]),
        "--lr-head", str(recipe["lr_head"]),
        "--epochs", str(recipe["epochs"]),
        "--seed", str(recipe["seed"]),
        "--batch", str(recipe["batch"]),
        "--output-root", str(output_root),
    ]
    return command


def execute(output_root: Path, backbone: str, cell: str, recipe: dict) -> dict:
    prior = existing_run(output_root, backbone, recipe)
    if prior is None:
        wait_for_gpu()
        print(f"[START] backbone={backbone} cell={cell} recipe={json.dumps(recipe, sort_keys=True)}", flush=True)
        subprocess.run(runner_command(backbone, cell, recipe, output_root), cwd=ROOT, check=True)
        prior = existing_run(output_root, backbone, recipe)
        if prior is None:
            raise RuntimeError(f"completed run was not discoverable: {backbone} {cell}")
    else:
        print(f"[REUSE] backbone={backbone} cell={cell} run={prior}", flush=True)
    metrics = prior / "contrastive" / "evaluation" / "dev_strict_novel" / "hard42_dev_strict_novel_metrics.csv"
    analyze_result(prior / "headonly_completion.json")
    return {"cell": cell, "recipe": deepcopy(recipe), "run_dir": str(prior), "point": best_point(metrics)}


def run_frozen(output_root: Path, backbone: str) -> dict:
    recipe = base_recipe(42, "B0")
    recipe.update({"epochs": 0, "head": "mlp"})
    prior = existing_run(output_root, backbone, recipe)
    if prior is None:
        wait_for_gpu()
        command = runner_command(backbone, "frozen", recipe, output_root)
        command.extend(["--frozen", "--eval-holdout"])
        subprocess.run(command, cwd=ROOT, check=True)
        prior = existing_run(output_root, backbone, recipe)
    if prior is None:
        raise RuntimeError(f"frozen run failed: {backbone}")
    dev = prior / "contrastive" / "evaluation" / "dev_strict_novel" / "hard42_dev_strict_novel_metrics.csv"
    holdout = prior / "contrastive" / "evaluation" / "holdout_strict_novel" / "hard42_holdout_strict_novel_metrics.csv"
    analyze_result(prior / "headonly_completion.json")
    return {"run_dir": str(prior), "dev": best_point(dev), "holdout": best_point(holdout)}


def stage_variants(stage: str, recipe: dict) -> list[tuple[str, dict]]:
    variants: list[tuple[str, object]] = {
        "head": [("linear", "linear"), ("mlp", "mlp"), ("ad", "ad"), ("adapter", "adapter"), ("adapterN2", "adapterN2"), ("adapterN3", "adapterN3")],
        "temperature": [("t005", 0.05), ("t007", 0.07), ("t010", 0.10)],
        "queue_size": [("q0", 0), ("q1024", 1024), ("q4096", 4096), ("q16384", 16384)],
        "ignore_neg_sim": [("ig050", 0.50), ("ig060", 0.60), ("ig070", 0.70), ("ig072", 0.72), ("ig075", 0.75), ("ig080", 0.80), ("ig090", 0.90), ("igoff", 1.0)],
        "local_weight": [("l0", 0.0), ("l015", 0.15), ("l030", 0.30), ("l050", 0.50), ("l100", 1.0)],
        "neco_weight": [("n0", 0.0), ("n010", 0.10), ("n020", 0.20), ("n030", 0.30)],
        "lr_head": [("lr3e4", 3e-4), ("lr5e4", 5e-4), ("lr1e3", 1e-3)],
    }[stage]
    result = []
    seen = set()
    for suffix, value in variants:
        candidate = deepcopy(recipe)
        candidate[stage] = value
        encoded = json.dumps(candidate, sort_keys=True)
        if encoded in seen:
            continue
        seen.add(encoded)
        result.append((suffix, candidate))
    return result


def write_state(root: Path, backbone: str, frozen: dict, source_cell: str, history: list[dict], incumbent: dict) -> None:
    payload = {
        "backbone": backbone,
        "source_cell": source_cell,
        "frozen": frozen,
        "history": history,
        "incumbent": incumbent,
    }
    path = root / backbone / "coordinate_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Hard-42 Head-Only Coordinate Descent: {backbone}",
        "",
        f"May source-protocol initializer: `{source_cell}`.",
        "",
        "A stage winner must preserve P1/P2/P3/P4/ARI in both FINCH-p2 and Louvain.",
        "",
        "| Stage | Cell | Selected | Epoch | Mode | P1(min) | P2(max) | P3(min) | P4(min) | ARI(min/mean) |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for event in history:
        for row in event["rows"]:
            point = row["point"]
            lines.append(
                f"| {event['stage']} | {row['cell']} | {'yes' if row['cell'] == event['winner'] else 'no'} | "
                f"{point['epoch']} | {point['mode']} | {point['p1_min']:.3f} | {point['noise_max']:.2f} | "
                f"{point['p3_min']:.3f} | {point['p4_min']:.3f} | {point['ari_min']:.3f}/{point['ari_mean']:.3f} |"
            )
    (root / backbone / "coordinate_report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_scope(run_dir: Path, scope: str, eval_root: Path, epoch: int, mode: str) -> Path:
    output_dir = run_dir / "contrastive" / "evaluation" / scope
    metrics_path = output_dir / f"hard42_{scope}_metrics.csv"
    if metrics_path.exists():
        return metrics_path
    if mode.startswith("weighted_concat_z") or mode.startswith("adapter_concat_z"):
        weight = int(mode.rsplit("z", 1)[1]) / 100.0
        modes = "backbone,projection" if mode.startswith("weighted") else "adapter,projection"
        concat = str(weight)
    else:
        modes = mode
        concat = ""
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--run-dir", str(run_dir),
        "--eval-root", str(eval_root),
        "--scope", scope,
        "--epochs", str(epoch),
        "--modes", modes,
        "--concat-proj-weights", concat,
        "--exclude-classes", EXCLUDED,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return metrics_path


def concat_stage(incumbent: dict) -> dict:
    run_dir = Path(incumbent["run_dir"])
    epoch = int(incumbent["point"]["epoch"])
    scope = f"dev_concat_epoch{epoch}"
    output_dir = run_dir / "contrastive" / "evaluation" / scope
    metrics_path = output_dir / f"hard42_{scope}_metrics.csv"
    if not metrics_path.exists():
        command = [
            sys.executable,
            "-u",
            str(EVALUATOR),
            "--run-dir", str(run_dir),
            "--eval-root", str(DEV_ROOT),
            "--scope", scope,
            "--epochs", str(epoch),
            "--modes", "backbone,projection,adapter",
            "--concat-proj-weights", "0.2,0.5,1.0,2.0,2.5,3.0,4.0",
            "--exclude-classes", EXCLUDED,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    candidate = best_point(metrics_path)
    if passes_core_gate(candidate, incumbent["point"]):
        updated = deepcopy(incumbent)
        updated["cell"] = f"{incumbent['cell']}+concat"
        updated["point"] = candidate
        return updated
    return incumbent


def seed_validation(root: Path, backbone: str, frozen: dict, incumbent: dict) -> dict:
    fixed_epoch = int(incumbent["point"]["epoch"])
    fixed_mode = str(incumbent["point"]["mode"])
    rows = []
    for seed in (41, 42, 43):
        recipe = deepcopy(incumbent["recipe"])
        recipe["seed"] = seed
        run = execute(root / backbone / "runs", backbone, f"final_s{seed}", recipe)
        run_dir = Path(run["run_dir"])
        metrics_path = evaluate_scope(
            run_dir,
            f"holdout_final_s{seed}_ep{fixed_epoch}",
            HOLDOUT_ROOT,
            fixed_epoch,
            fixed_mode,
        )
        point = best_point(metrics_path, {fixed_mode})
        rows.append({"seed": seed, "run_dir": str(run_dir), "point": point})
    directions = [passes_final_gate(row["point"], frozen["holdout"]) for row in rows]
    means = {
        key: sum(float(row["point"][key]) for row in rows) / len(rows)
        for key in ("p1_min", "noise_max", "p3_min", "p4_min", "ari_min", "ari_mean")
    }
    means_by_method = {
        method: {
            metric: sum(float(row["point"][method][metric]) for row in rows) / len(rows)
            for metric in ("P1_capture", "P2_noise_pct", "P3_completeness", "P4_homogeneity", "ARI")
        }
        for method in ("finch", "louvain")
    }
    mean_pass = True
    for method in ("finch", "louvain"):
        current = means_by_method[method]
        reference = frozen["holdout"][method]
        mean_pass = mean_pass and (
            current["P1_capture"] >= float(reference["P1_capture"])
            and current["P2_noise_pct"] <= float(reference["P2_noise_pct"])
            and current["P3_completeness"] > float(reference["P3_completeness"]) + 1e-4
            and current["P4_homogeneity"] > float(reference["P4_homogeneity"]) + 1e-4
            and current["ARI"] > float(reference["ARI"]) + 1e-4
        )
    result = {
        "fixed_epoch": fixed_epoch,
        "fixed_mode": fixed_mode,
        "frozen_holdout": frozen["holdout"],
        "seeds": rows,
        "directions": directions,
        "means": means,
        "means_by_method": means_by_method,
        "accepted": sum(directions) >= 2 and mean_pass,
    }
    (root / backbone / "seed_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_backbone(root: Path, backbone: str) -> None:
    backbone_root = root / backbone
    runs_root = backbone_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    source_cell = source_winner(backbone)
    frozen = run_frozen(runs_root, backbone)
    incumbent = execute(runs_root, backbone, f"source_{source_cell.lower()}", base_recipe(42, source_cell))
    history = []
    for index, stage in enumerate(
        ("head", "temperature", "queue_size", "ignore_neg_sim", "local_weight", "neco_weight", "lr_head"),
        1,
    ):
        candidates = [incumbent]
        for suffix, recipe in stage_variants(stage, incumbent["recipe"]):
            if recipe == incumbent["recipe"]:
                continue
            candidates.append(execute(runs_root, backbone, f"s{index:02d}_{stage}_{suffix}", recipe))
        passing = [row for row in candidates if row is incumbent or passes_core_gate(row["point"], incumbent["point"])]
        winner = max(passing, key=lambda row: row["point"]["score"])
        history.append({"stage": stage, "winner": winner["cell"], "rows": candidates})
        incumbent = winner
        write_state(root, backbone, frozen, source_cell, history, incumbent)
        print(f"[SELECT] backbone={backbone} stage={stage} winner={incumbent['cell']}", flush=True)
    before_concat = incumbent
    incumbent = concat_stage(incumbent)
    history.append({"stage": "embedding_concat", "winner": incumbent["cell"], "rows": [before_concat, incumbent]})
    write_state(root, backbone, frozen, source_cell, history, incumbent)
    validation = seed_validation(root, backbone, frozen, incumbent)
    print(f"[FINAL] backbone={backbone} accepted={validation['accepted']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backbones", default="cnn_tapt,nocnn")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    if args.plan_only:
        print(json.dumps({backbone: source_winner(backbone) for backbone in backbones}, indent=2))
        return
    for backbone in backbones:
        run_backbone(output_root, backbone)


if __name__ == "__main__":
    main()
