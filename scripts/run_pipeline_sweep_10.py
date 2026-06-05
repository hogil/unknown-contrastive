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
    # 넓게 합쳐지는 현상 확인용 strict sweep.
    # epsilon > 0 은 가까운 cluster 를 다시 붙이는 쪽이라 기본 sweep 에서 제거한다.
    {
        "id": "01_reco_lr5e4_noignore_eom",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 12, "ms": 3, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "02_lr1e3_noignore_eom",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 1e-3, "neco": 0.2,
        "mcs": 12, "ms": 3, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "03_lr3e4_noignore_eom",
        "epochs": 30, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 3e-4, "neco": 0.2,
        "mcs": 12, "ms": 3, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "04_lr5e4_leaf_m8_s2",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 8, "ms": 2, "method": "leaf", "epsilon": 0.0,
    },
    {
        "id": "05_lr5e4_leaf_m6_s2",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 6, "ms": 2, "method": "leaf", "epsilon": 0.0,
    },
    {
        "id": "06_lr5e4_eom_m8_s2",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 8, "ms": 2, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "07_lr5e4_eom_m12_s1",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 12, "ms": 1, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "08_lr5e4_ignore095_eom",
        "epochs": 20, "img": 512, "ignore": 0.95, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 12, "ms": 3, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "09_lr5e4_temp004_eom",
        "epochs": 20, "img": 512, "ignore": 1.01, "temp": 0.04, "lr": 5e-4, "neco": 0.2,
        "mcs": 12, "ms": 3, "method": "eom", "epsilon": 0.0,
    },
    {
        "id": "10_lr5e4_epoch30_eom",
        "epochs": 30, "img": 512, "ignore": 1.01, "temp": 0.05, "lr": 5e-4, "neco": 0.2,
        "mcs": 12, "ms": 3, "method": "eom", "epsilon": 0.0,
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


def _resolve_cnn_best(repo: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (repo / p).resolve()
    candidates = [p]
    if p.is_dir():
        candidates = [p / "cnn" / "best_model.pth", p / "best_model.pth"]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise SystemExit(
        f"CNN best_model.pth not found from: {p}\n"
        f"  accepted forms:\n"
        f"  --backbone runs/<RUN>/cnn/best_model.pth\n"
        f"  --backbone runs/<RUN>\n"
        f"  --cnn-run-dir runs/<RUN>")


def _latest_cnn_best(repo: Path) -> Path | None:
    bests = list((repo / "runs").glob("*/cnn/best_model.pth"))
    if not bests:
        return None
    return sorted(bests, key=lambda p: p.stat().st_mtime, reverse=True)[0]


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


def _load_grouping_stats(grouping_run: str | None) -> dict | None:
    if not grouping_run:
        return None
    p = Path(grouping_run) / "all_summaries.json"
    if not p.exists():
        return None
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not rows:
        return None
    return {
        "targets": len(rows),
        "n_clusters": [r.get("n_clusters") for r in rows],
        "noise_pct": [r.get("noise_pct") for r in rows],
        "largest_group_pct": [r.get("largest_group_pct") for r in rows],
        "top_groups": [r.get("top_groups") for r in rows],
    }


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    selected = _select_conditions(args.only, args.start_at)
    if not selected:
        raise SystemExit("No conditions selected.")

    if args.backbone and args.cnn_run_dir:
        raise SystemExit("--backbone and --cnn-run-dir cannot be used together.")

    backbone = None
    if args.backbone:
        backbone = _resolve_cnn_best(repo, args.backbone)
    elif args.cnn_run_dir:
        backbone = _resolve_cnn_best(repo, args.cnn_run_dir)
    elif not args.allow_train_cnn_each_run:
        backbone = _latest_cnn_best(repo)

    cnn_run = None
    if backbone is not None:
        cnn_run = backbone.parent.parent if backbone.parent.name == "cnn" else backbone.parent

    if backbone is None and not args.allow_train_cnn_each_run:
        raise SystemExit(
            "No CNN best_model.pth found under runs/*/cnn/.\n"
            "Pass --backbone runs/<CNN_RUN>/cnn/best_model.pth "
            "or --cnn-run-dir runs/<CNN_RUN>, "
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
            "--cl-lr-head", str(cond["lr"]),
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

        print("\n" + "=" * 80)
        print(f"[condition] {cond['id']}")
        print(json.dumps(cond, ensure_ascii=False))
        print("$ " + " ".join(cmd))

        if args.dry_run:
            rc = 0
        else:
            rc = _run_and_tee(cmd, repo, log_path)

        elapsed_min = round((time.time() - started) / 60, 2)
        grouping_run = _newest_after(repo / "result_grouping", "*_grouping_ddp", started)
        row = {
            "condition": cond,
            "returncode": rc,
            "elapsed_min": elapsed_min,
            "log": str(log_path),
            "pipeline_run": _newest_after(repo / "runs", "*_pipeline_ddp", started),
            "contrastive_run": _newest_after(repo / "runs", "*_contrastive_prod_ddp_pipe", started),
            "grouping_run": grouping_run,
            "grouping_stats": _load_grouping_stats(grouping_run),
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
