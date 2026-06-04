#!/usr/bin/env python3
"""Run 10 production contrastive/grouping pipeline conditions sequentially.

No CUDA_VISIBLE_DEVICES is set here. Set it outside this script if needed.

Typical:
  python -u scripts/run_pipeline_sweep_10.py \
    --backbone runs/<CNN_RUN>/cnn/best_model.pth \
    --prod-train-dirs data/images/prod_train_A,data/images/prod_train_B \
    --prod-pred-dirs data/images/prod_pred_A,data/images/prod_pred_B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_PROD_TRAIN_DIRS = "data/images/prod_train"
DEFAULT_PROD_PRED_DIRS = "data/images/prod_pred"
DEFAULT_CL_BATCH = 64
DEFAULT_GROUPING_BATCH = 128
DEFAULT_GROUPING_WORKERS = 64
DEFAULT_REPS_PER_CLUSTER = 30


CONDITIONS = [
    {
        "id": "01_current_leaf06",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.2,
        "mcs": 12, "ms": 15, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "02_leaf03_mcs20_ms10",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.2,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.03,
    },
    {
        "id": "03_leaf06_mcs20_ms10",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.2,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "04_leaf10_mcs20_ms10",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.2,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.10,
    },
    {
        "id": "05_temp004_leaf06",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.04, "neco": 0.2,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "06_neg097_leaf06",
        "epochs": 20, "img": 512, "ignore": 0.97, "temp": 0.05, "neco": 0.2,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "07_neco04_leaf06",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.4,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "08_neco01_leaf06",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.1,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "09_epoch30_leaf06",
        "epochs": 30, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.2,
        "mcs": 20, "ms": 10, "method": "leaf", "epsilon": 0.06,
    },
    {
        "id": "10_leaf06_mcs20_ms15",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "neco": 0.2,
        "mcs": 20, "ms": 15, "method": "leaf", "epsilon": 0.06,
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cnn-run-dir", type=str, default=None,
                   help="Existing CNN run dir. If omitted, latest *_cnn_ddp is used.")
    p.add_argument("--backbone", type=str, default=None,
                   help="Existing CNN best_model.pth path. Same as train_pipeline_ddp.py --backbone.")
    p.add_argument("--prod-train-dirs", type=str, default=DEFAULT_PROD_TRAIN_DIRS,
                   help="Comma-separated production train folders.")
    p.add_argument("--prod-pred-dirs", type=str, default=DEFAULT_PROD_PRED_DIRS,
                   help="Comma-separated production prediction folders.")
    p.add_argument("--cl-batch", type=int, default=DEFAULT_CL_BATCH)
    p.add_argument("--grouping-batch", type=int, default=DEFAULT_GROUPING_BATCH)
    p.add_argument("--grouping-workers", type=int, default=DEFAULT_GROUPING_WORKERS)
    p.add_argument("--reps-per-cluster", type=int, default=DEFAULT_REPS_PER_CLUSTER)
    p.add_argument("--only", type=str, default=None,
                   help="Comma list of condition numbers or ids, e.g. 1,3,05_temp004_leaf06.")
    p.add_argument("--start-at", type=int, default=1,
                   help="Start condition number, 1-based.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--allow-train-cnn-each-run", action="store_true",
                   help="If no CNN run is found, allow full CNN training for every condition.")
    return p.parse_args()


def _latest_cnn_run(repo: Path) -> Path | None:
    runs = [
        p for p in (repo / "runs").glob("*_cnn_ddp")
        if (p / "cnn" / "best_model.pth").exists()
    ]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _select_conditions(only: str | None, start_at: int):
    selected = CONDITIONS[max(0, start_at - 1):]
    if not only:
        return selected
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    nums = {str(i) for i in range(1, len(CONDITIONS) + 1) if str(i) in wanted}
    nums |= {f"{int(x):02d}" for x in wanted if x.isdigit()}
    return [
        c for i, c in enumerate(CONDITIONS, start=1)
        if c["id"] in wanted or str(i) in wanted or f"{i:02d}" in nums
    ]


def _run_and_tee(cmd: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def _newest_after(root: Path, pattern: str, started: float) -> str | None:
    hits = [p for p in root.glob(pattern) if p.stat().st_mtime >= started]
    if not hits:
        return None
    return str(sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0])


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    selected = _select_conditions(args.only, args.start_at)
    if not selected:
        raise SystemExit("No conditions selected.")

    if args.backbone and args.cnn_run_dir:
        raise SystemExit("--backbone and --cnn-run-dir cannot be used together.")

    backbone = Path(args.backbone) if args.backbone else None
    if backbone is not None and not backbone.is_absolute():
        backbone = (repo / backbone).resolve()
    if backbone is not None and not backbone.exists():
        raise SystemExit(f"backbone best_model.pth not found: {backbone}")

    cnn_run = None if backbone is not None else (
        Path(args.cnn_run_dir) if args.cnn_run_dir else _latest_cnn_run(repo)
    )
    if cnn_run is not None and not cnn_run.is_absolute():
        cnn_run = (repo / cnn_run).resolve()
    if cnn_run is not None and not (cnn_run / "cnn" / "best_model.pth").exists():
        raise SystemExit(f"cnn best_model.pth not found: {cnn_run / 'cnn' / 'best_model.pth'}")
    if cnn_run is None and not args.allow_train_cnn_each_run:
        raise SystemExit(
            "No CNN run found. Pass --cnn-run-dir runs/<CNN_RUN> "
            "or use --allow-train-cnn-each-run.")

    sweep_dir = repo / "runs" / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_pipeline_sweep_10"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "sweep_summary.json"
    summary: list[dict] = []

    print(f"[sweep] {sweep_dir.resolve()}")
    print(f"[conditions] {len(selected)}")
    print(f"[backbone] {backbone if backbone is not None else '(not set)'}")
    print(f"[cnn_run] {cnn_run if cnn_run is not None else 'FULL CNN EACH CONDITION'}")
    print(f"[prod_train_dirs] {args.prod_train_dirs}")
    print(f"[prod_pred_dirs] {args.prod_pred_dirs}")

    for cond in selected:
        started = time.time()
        log_path = sweep_dir / f"{cond['id']}.log"
        cmd = [
            sys.executable, "-u", str(repo / "scripts" / "train_pipeline_ddp.py"),
            "--prod-train-dirs", args.prod_train_dirs,
            "--prod-pred-dirs", args.prod_pred_dirs,
            "--prod-epochs", str(cond["epochs"]),
            "--cl-img-size", str(cond["img"]),
            "--ignore-neg-sim", str(cond["ignore"]),
            "--nce-temp", str(cond["temp"]),
            "--neco-weight", str(cond["neco"]),
            "--neco-tau", "0.1",
            "--cl-batch", str(args.cl_batch),
            "--grouping-batch", str(args.grouping_batch),
            "--grouping-workers", str(args.grouping_workers),
            "--grouping-reps-per-cluster", str(args.reps_per_cluster),
            "--grouping-min-cluster-size", str(cond["mcs"]),
            "--grouping-min-samples", str(cond["ms"]),
            "--grouping-cluster-selection-method", cond["method"],
            "--grouping-cluster-selection-epsilon", str(cond["epsilon"]),
        ]
        if backbone is not None:
            cmd += ["--backbone", str(backbone)]
        elif cnn_run is not None:
            cmd += ["--cnn-run-dir", str(cnn_run)]

        print("\n" + "=" * 80)
        print(f"[condition] {cond['id']}")
        print(json.dumps(cond, ensure_ascii=False))
        print("$ " + " ".join(cmd))

        if args.dry_run:
            rc = 0
        else:
            rc = _run_and_tee(cmd, repo, log_path)

        elapsed_min = round((time.time() - started) / 60, 2)
        row = {
            "condition": cond,
            "returncode": rc,
            "elapsed_min": elapsed_min,
            "log": str(log_path),
            "pipeline_run": _newest_after(repo / "runs", "*_pipeline_ddp", started),
            "contrastive_run": _newest_after(repo / "runs", "*_contrastive_prod_ddp_pipe", started),
            "grouping_run": _newest_after(repo / "result_grouping", "*_grouping_ddp", started),
            "command": cmd,
        }
        summary.append(row)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[condition done] {cond['id']} rc={rc} elapsed={elapsed_min} min")
        print(f"[summary] {summary_path.resolve()}")

        if rc != 0 and args.stop_on_fail:
            raise SystemExit(rc)

    print(f"\n[OUT] {sweep_dir.resolve()}")
    print(f"[summary] {summary_path.resolve()}")


if __name__ == "__main__":
    main()
