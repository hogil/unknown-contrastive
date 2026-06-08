#!/usr/bin/env python3
"""Run contrastive embedding sweeps and compare kNN/t-SNE metrics.

No CUDA_VISIBLE_DEVICES is set here. Set it outside this script if needed.

Typical:
  python scripts/run_embedding_sweep.py \
    --backbone runs/<CNN_RUN>/cnn/best_model.pth \
    --train-dirs data/images/wm811k_50/train \
    --eval-dir data/images/wm811k_50/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_TRAIN_DIRS = "data/images/wm811k_50/train"
DEFAULT_EVAL_DIR = "data/images/wm811k_50/eval"


CONDITIONS = [
    {
        "id": "01_global_noqueue",
        "desc": "pure SimCLR-style global InfoNCE, encoder fine-tune",
        "queue": False, "queue_size": 0, "ignore": 1.01, "pseudo_neg_remove": False,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "02_moco_queue",
        "desc": "MoCo-style large queue, no false-negative filter",
        "queue": True, "queue_size": 16384, "ignore": 1.01, "pseudo_neg_remove": False,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "03_queue_fn070",
        "desc": "queue + false-negative elimination, strict negative threshold 0.70",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "04_queue_fn085",
        "desc": "queue + false-negative elimination, looser threshold 0.85",
        "queue": True, "queue_size": 16384, "ignore": 0.85, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "05_queue_fn070_temp004",
        "desc": "queue + false-negative elimination + lower temperature",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.04, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "06_queue_fn070_lrbackbone3e5",
        "desc": "queue + false-negative elimination + stronger backbone LR",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 3e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "07_queue_fn070_neco_grid",
        "desc": "queue + false-negative elimination + NeCo grid",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.2, "neco_grid": 6, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "08_queue_fn070_neco_all",
        "desc": "queue + false-negative elimination + NeCo all patches",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.2, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "09_queue_fn070_local_grid",
        "desc": "queue + false-negative elimination + local grid InfoNCE",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 1e-5,
        "neco": 0.0, "neco_grid": 0, "local": 0.2, "local_grid": 6, "local_window": 1,
        "freeze": False,
    },
    {
        "id": "10_headonly_ablation",
        "desc": "head-only ablation: 1024 projection but frozen CNN backbone",
        "queue": True, "queue_size": 16384, "ignore": 0.70, "pseudo_neg_remove": True,
        "pseudo_weight": 0.0, "temp": 0.05, "lr_head": 1e-3, "lr_backbone": 0.0,
        "neco": 0.0, "neco_grid": 0, "local": 0.0, "local_grid": 6, "local_window": 1,
        "freeze": True,
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=None,
                   help="CNN best_model.pth. If omitted, latest runs/*/cnn/best_model.pth is used.")
    p.add_argument("--train-dirs", type=str, default=DEFAULT_TRAIN_DIRS,
                   help="Comma-separated classless contrastive train dirs.")
    p.add_argument("--eval-dir", type=str, default=DEFAULT_EVAL_DIR,
                   help="ImageFolder eval dir. Labels are used only for metric/t-SNE.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--train-sampling-ratio", type=float, default=1.0)
    p.add_argument("--tag-prefix", type=str, default="embedding_sweep")
    p.add_argument("--out-root", type=str, default="result_grouping")
    p.add_argument("--tsne-img-size", type=int, default=512)
    p.add_argument("--tsne-batch", type=int, default=16)
    p.add_argument("--only", type=str, default=None,
                   help="Comma list of condition numbers or ids, e.g. 1,3,07_queue_fn070_neco_grid.")
    p.add_argument("--start-at", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--stop-on-fail", action="store_true")
    return p.parse_args()


def resolve_path(repo: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo / p).resolve()


def latest_cnn_best(repo: Path) -> Path | None:
    bests = list((repo / "runs").glob("*/cnn/best_model.pth"))
    if not bests:
        return None
    return sorted(bests, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def select_conditions(only: str | None, start_at: int):
    selected = CONDITIONS[max(0, start_at - 1):]
    if not only:
        return selected
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    out = []
    for i, cond in enumerate(CONDITIONS, start=1):
        if cond["id"] in wanted or str(i) in wanted or f"{i:02d}" in wanted:
            out.append(cond)
    return out


def run_and_tee(cmd: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def newest_run(repo: Path, tag: str, started: float) -> Path | None:
    hits = [
        p for p in (repo / "runs").glob(f"*_{tag}")
        if (p / "contrastive" / "best_model.pt").exists() and p.stat().st_mtime >= started
    ]
    if not hits:
        return None
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_tsne_metrics(out_dir: Path) -> list[dict]:
    diag = out_dir / "diagnostics.json"
    if not diag.exists():
        return []
    data = json.loads(diag.read_text(encoding="utf-8"))
    rows = []
    for s in data.get("stages", []):
        rates = s.get("knn_same_rate", {})
        rows.append({
            "label": s.get("label"),
            "embedding_dim": s.get("embedding_dim"),
            "top1": rates.get("1"),
            "k3": rates.get("3"),
            "k5": rates.get("5"),
            "k7": rates.get("7"),
            "k9": rates.get("9"),
            "checkpoint": s.get("checkpoint"),
        })
    return rows


def write_metric_csv(rows: list[dict], out: Path) -> None:
    fields = ["label", "embedding_dim", "top1", "k3", "k5", "k7", "k9", "checkpoint"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    selected = select_conditions(args.only, args.start_at)
    if not selected:
        raise SystemExit("No conditions selected.")

    backbone = resolve_path(repo, args.backbone) if args.backbone else latest_cnn_best(repo)
    if backbone is None or not backbone.exists():
        raise SystemExit("CNN best_model.pth not found. Pass --backbone runs/<CNN_RUN>/cnn/best_model.pth")
    eval_dir = resolve_path(repo, args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"eval dir not found: {eval_dir}")

    sweep_dir = repo / "runs" / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{args.tag_prefix}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_dir / "embedding_sweep_summary.json"
    summary: list[dict] = []

    print(f"[sweep] {sweep_dir.resolve()}")
    print(f"[backbone] {backbone.resolve()}")
    print(f"[train_dirs] {args.train_dirs}")
    print(f"[eval_dir] {eval_dir.resolve()}")
    print(f"[conditions] {len(selected)}")

    stage_args = []
    for cond in selected:
        started = time.time()
        tag = f"{args.tag_prefix}_{cond['id']}"
        log_path = sweep_dir / f"{cond['id']}.log"
        cmd = [
            sys.executable, "-u", str(repo / "scripts" / "train_contrastive_ddp.py"),
            "--backbone", str(backbone),
            "--tag", tag,
            "--train-dirs", args.train_dirs,
            "--eval-dirs", str(eval_dir),
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--img-size", str(args.img_size),
            "--proj-dim", str(args.proj_dim),
            "--train-sampling-ratio", str(args.train_sampling_ratio),
            "--ignore-neg-sim", str(cond["ignore"]),
            "--pseudo-pos-weight", str(cond["pseudo_weight"]),
            "--pseudo-pos-min-sim", "0.90",
            "--pseudo-pos-topk", "2",
            "--pseudo-pos-start-epoch", "2",
            "--pseudo-pos-source", "backbone",
            "--infer-embed-mode", "projection",
            "--nce-temp", str(cond["temp"]),
            "--lr-head", str(cond["lr_head"]),
            "--lr-backbone", str(cond["lr_backbone"]),
            "--neco-weight", str(cond["neco"]),
            "--neco-grid", str(cond["neco_grid"]),
            "--local-weight", str(cond["local"]),
            "--local-grid", str(cond["local_grid"]),
            "--local-window", str(cond["local_window"]),
        ]
        if cond["freeze"]:
            cmd.append("--freeze-backbone")
        if not cond["queue"]:
            cmd.append("--no-queue")
        else:
            cmd += ["--queue-size", str(cond["queue_size"])]
        if not cond["pseudo_neg_remove"]:
            cmd.append("--no-pseudo-neg-remove")

        print("\n" + "=" * 88)
        print(f"[condition] {cond['id']} - {cond['desc']}")
        print("$ " + " ".join(cmd))

        rc = 0 if args.dry_run else run_and_tee(cmd, repo, log_path)
        elapsed_min = round((time.time() - started) / 60, 2)
        run_dir = None if args.dry_run else newest_run(repo, tag, started)
        model_path = run_dir / "contrastive" / "best_model.pt" if run_dir else None
        if model_path:
            stage_args.append(f"{cond['id']}={model_path}")
        row = {
            "condition": cond,
            "returncode": rc,
            "elapsed_min": elapsed_min,
            "log": str(log_path.resolve()),
            "run_dir": str(run_dir.resolve()) if run_dir else None,
            "model": str(model_path.resolve()) if model_path else None,
            "command": cmd,
        }
        summary.append(row)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[condition done] {cond['id']} rc={rc} elapsed={elapsed_min} min model={model_path}")
        if rc != 0 and args.stop_on_fail:
            raise SystemExit(rc)

    if not args.dry_run and not args.no_tsne and stage_args:
        tsne_tag = f"{args.tag_prefix}_summary"
        tsne_cmd = [
            sys.executable, "-u", str(repo / "scripts" / "make_tsne_stages.py"),
            "--eval-dir", str(eval_dir),
            "--cnn", str(backbone),
            "--out-root", args.out_root,
            "--tag", tsne_tag,
            "--img-size", str(args.tsne_img_size),
            "--batch", str(args.tsne_batch),
            "--contrastive-embed-mode", "projection",
        ]
        for s in stage_args:
            tsne_cmd += ["--stage", s]
        tsne_log = sweep_dir / "tsne_summary.log"
        started = time.time()
        print("\n" + "=" * 88)
        print("[t-SNE/kNN summary]")
        print("$ " + " ".join(tsne_cmd))
        rc = run_and_tee(tsne_cmd, repo, tsne_log)
        tsne_dir = None
        hits = [p for p in (repo / args.out_root).glob(f"*_{tsne_tag}") if p.stat().st_mtime >= started]
        if hits:
            tsne_dir = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            metrics = load_tsne_metrics(tsne_dir)
            write_metric_csv(metrics, sweep_dir / "embedding_metrics.csv")
            (sweep_dir / "embedding_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[metrics] {(sweep_dir / 'embedding_metrics.csv').resolve()}")
            print(f"[tsne_dir] {tsne_dir.resolve()}")
        summary.append({
            "tsne_returncode": rc,
            "tsne_log": str(tsne_log.resolve()),
            "tsne_dir": str(tsne_dir.resolve()) if tsne_dir else None,
        })
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OUT] {sweep_dir.resolve()}")
    print(f"[summary] {summary_path.resolve()}")


if __name__ == "__main__":
    main()
