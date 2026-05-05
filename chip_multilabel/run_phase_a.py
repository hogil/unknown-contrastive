# -*- coding: utf-8 -*-
"""Phase A — T1 (CE+LS) coordinate-descent hparam sweep.

A1: LS sweep (LR=1e-4, ep=8 fixed)
A2: LR sweep (LS=LS*, ep=8 fixed)
A3: epochs sweep (LS=LS*, LR=LR* fixed)

Sequential — 1 GPU job at a time. nvidia-smi gate before each train.
After each train, run inference (I3, I7, I10) and log macro_f1 / top1_11.

Outputs:
    outputs/phase_a_<TS>/
        sweep_log.csv
        best_config.json
        report.md
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_TRAIN = "chip_multilabel._train_chip_variant"
SCRIPT_INFER = "chip_multilabel.run_stage1"

DEFAULT_LS_GRID = [0.05, 0.10, 0.15, 0.20]
DEFAULT_LR_GRID = [5e-5, 1e-4, 3e-4]
DEFAULT_EP_GRID = [3, 5, 8, 12]

INFER_VARIANTS_DEFAULT = ["I3", "I7", "I10"]


def gpu_gate(mem_pct_threshold: float = 90.0, poll_seconds: int = 30,
             timeout_seconds: int = 300) -> bool:
    """Wait until GPU memory % drops below threshold. Return True if proceeded, False if timeout."""
    waited = 0
    while waited <= timeout_seconds:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            line = (res.stdout or "").strip().splitlines()[0]
            used, total, util = [float(x.strip()) for x in line.split(",")]
            mem_pct = 100.0 * used / max(total, 1.0)
            print(f"[gpu_gate] mem={used:.0f}/{total:.0f} MiB ({mem_pct:.1f} %)  util={util:.0f} %")
            if mem_pct < mem_pct_threshold:
                return True
        except Exception as e:
            print(f"[gpu_gate] nvidia-smi check failed: {e!r} — proceeding")
            return True
        if waited == 0:
            print(f"[gpu_gate] waiting (mem >= {mem_pct_threshold} %), poll every {poll_seconds}s, timeout {timeout_seconds}s")
        time.sleep(poll_seconds)
        waited += poll_seconds
    print(f"[gpu_gate] TIMEOUT after {timeout_seconds}s — proceeding anyway")
    return False


def run_train(variant: str, ls: float, lr: float, epochs: int, batch: int, accum: int,
              tag: str) -> Path:
    cmd = [
        sys.executable, "-X", "utf8", "-m", SCRIPT_TRAIN,
        "--variant", variant,
        "--ls", str(ls),
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--accum", str(accum),
        "--tag", tag,
    ]
    print(f"\n[run_train] {' '.join(cmd[2:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"train failed (rc={proc.returncode})")
    print(proc.stdout[-1500:])
    out_match = [ln for ln in proc.stdout.splitlines() if "out=outputs" in ln]
    if not out_match:
        raise RuntimeError("could not find out= in train stdout")
    out_dir = out_match[0].split("out=")[-1].strip()
    print(f"[run_train] elapsed={elapsed:.1f}s  out={out_dir}")
    return Path(out_dir)


def run_inference(model_ckpt: Path, eval_set: Path, out_root: Path,
                  variants: List[str], batch_size: int = 16) -> Dict[str, Dict[str, float]]:
    """Returns dict {inference_id: {macro_f1, top1_11class}} parsed from stdout."""
    cmd = [
        sys.executable, "-X", "utf8", "-m", SCRIPT_INFER,
        "--model", str(model_ckpt),
        "--eval-set", str(eval_set),
        "--out-root", str(out_root),
        "--variants", ",".join(variants),
        "--batch-size", str(batch_size),
    ]
    print(f"\n[run_inference] {' '.join(cmd[2:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"inference failed (rc={proc.returncode})")
    print(proc.stdout[-1500:])

    pat = re.compile(
        r"T0__([A-Z]\d+)\s+macro_f1=([\d.]+)\s+top1_11=([\d.]+)\s+T=([\d.]+)"
    )
    cells: Dict[str, Dict[str, float]] = {}
    for line in proc.stdout.splitlines():
        m = pat.search(line)
        if m:
            vid, mf1, t11, T = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            cells[vid] = {"macro_f1": mf1, "top1_11": t11, "temperature": T}
    print(f"[run_inference] elapsed={elapsed:.1f}s  cells={cells}")
    return cells


def append_log(rows: List[Dict], log_csv: Path) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "phase", "ls", "lr", "epochs", "train_path",
            "inference_id", "macro_f1", "top1_11", "temperature", "elapsed_train_sec",
        ])
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def best_row(rows: List[Dict], by: str = "macro_f1") -> Dict:
    return max(rows, key=lambda r: r[by])


def write_report(all_rows: List[Dict], best: Dict, out_root: Path,
                 baseline: Dict[str, float]) -> None:
    lines = []
    lines.append("# Phase A - T1 (CE+LS) hparam coordinate-descent sweep\n")
    lines.append(f"baseline (iter 4 winner): T1__I10 macro_f1={baseline.get('macro_f1', 0.8634):.4f} top1_11={baseline.get('top1_11', 0.7006):.4f} (LS=0.10, LR=1e-4, ep=8)\n")
    lines.append("## All sweep rows (sorted macro_f1)\n")
    lines.append("| phase | LS | LR | epochs | inference | macro_f1 | top1_11 | T |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(all_rows, key=lambda x: -x["macro_f1"]):
        lines.append(
            f"| {r['phase']} | {r['ls']:.3f} | {r['lr']:.0e} | {r['epochs']} | {r['inference_id']} | "
            f"{r['macro_f1']:.4f} | {r['top1_11']:.4f} | {r['temperature']:.3f} |"
        )
    lines.append("\n## Best cell\n")
    lines.append(
        f"**LS={best['ls']:.3f}  LR={best['lr']:.0e}  epochs={best['epochs']}  inference={best['inference_id']}**"
        f"  → macro_f1 = {best['macro_f1']:.4f}, top1_11 = {best['top1_11']:.4f}"
    )
    lines.append(f"\nDelta vs baseline: macro_f1 {best['macro_f1'] - baseline.get('macro_f1', 0.8634):+.4f}, "
                 f"top1_11 {best['top1_11'] - baseline.get('top1_11', 0.7006):+.4f}")
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a1", "a2", "a3", "all"], default="all")
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--ls-grid", default=",".join(str(x) for x in DEFAULT_LS_GRID))
    ap.add_argument("--lr-grid", default=",".join(str(x) for x in DEFAULT_LR_GRID))
    ap.add_argument("--ep-grid", default=",".join(str(x) for x in DEFAULT_EP_GRID))
    ap.add_argument("--inference-variants", default=",".join(INFER_VARIANTS_DEFAULT))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--gpu-mem-threshold", type=float, default=90.0)
    args = ap.parse_args()

    ls_grid = [float(x) for x in args.ls_grid.split(",")]
    lr_grid = [float(x) for x in args.lr_grid.split(",")]
    ep_grid = [int(x) for x in args.ep_grid.split(",")]
    inf_variants = args.inference_variants.split(",")

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"phase_a_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_csv = out_root / "sweep_log.csv"

    print(f"[phase_a] out={out_root}")
    print(f"[phase_a] phase={args.phase} ls={ls_grid} lr={lr_grid} ep={ep_grid} inf={inf_variants}")

    all_rows: List[Dict] = []
    cache: Dict[Tuple[float, float, int], Path] = {}

    def step(ls: float, lr: float, epochs: int, phase_label: str) -> Dict:
        gpu_gate(mem_pct_threshold=args.gpu_mem_threshold)
        key = (ls, lr, epochs)
        if key in cache:
            print(f"[phase_a] reusing cached train for ls={ls} lr={lr} ep={epochs}")
            ckpt_dir = cache[key]
            train_elapsed = 0.0
        else:
            tag = f"LS{int(ls*100):02d}_LR{int(round(-1*__import__('math').log10(lr))):02d}_ep{epochs}"
            t0 = time.time()
            ckpt_dir = run_train(
                variant="T1", ls=ls, lr=lr, epochs=epochs,
                batch=args.batch, accum=args.accum, tag=tag,
            )
            train_elapsed = time.time() - t0
            cache[key] = ckpt_dir
        gpu_gate(mem_pct_threshold=args.gpu_mem_threshold)
        ckpt = ckpt_dir / "best_model.pth"
        cells = run_inference(
            model_ckpt=ckpt, eval_set=Path(args.eval_set),
            out_root=Path(args.out_root), variants=inf_variants,
            batch_size=16,
        )
        rows = []
        for vid, m in cells.items():
            row = {
                "phase": phase_label,
                "ls": ls, "lr": lr, "epochs": epochs,
                "train_path": str(ckpt_dir),
                "inference_id": vid,
                "macro_f1": m["macro_f1"],
                "top1_11": m["top1_11"],
                "temperature": m["temperature"],
                "elapsed_train_sec": train_elapsed,
            }
            rows.append(row)
            all_rows.append(row)
        append_log(rows, log_csv)
        b = best_row(rows)
        print(f"[phase_a] {phase_label}  ls={ls} lr={lr} ep={epochs}  best_inf={b['inference_id']}  macro_f1={b['macro_f1']:.4f}  top1_11={b['top1_11']:.4f}")
        return b

    ls_star = 0.10
    lr_star = 1e-4
    ep_star = 8

    if args.phase in ("a1", "all"):
        print("\n=== A1: LS sweep ===")
        a1_rows: List[Dict] = []
        for ls in ls_grid:
            r = step(ls=ls, lr=1e-4, epochs=8, phase_label="A1")
            a1_rows.append(r)
        b = best_row(a1_rows)
        ls_star = b["ls"]
        print(f"\n[A1 done] ls* = {ls_star} (macro_f1={b['macro_f1']:.4f})")

    if args.phase in ("a2", "all"):
        print(f"\n=== A2: LR sweep (ls={ls_star}) ===")
        a2_rows: List[Dict] = []
        for lr in lr_grid:
            r = step(ls=ls_star, lr=lr, epochs=8, phase_label="A2")
            a2_rows.append(r)
        b = best_row(a2_rows)
        lr_star = b["lr"]
        print(f"\n[A2 done] (ls*, lr*) = ({ls_star}, {lr_star}) (macro_f1={b['macro_f1']:.4f})")

    if args.phase in ("a3", "all"):
        print(f"\n=== A3: epochs sweep (ls={ls_star}, lr={lr_star}) ===")
        a3_rows: List[Dict] = []
        for ep in ep_grid:
            r = step(ls=ls_star, lr=lr_star, epochs=ep, phase_label="A3")
            a3_rows.append(r)
        b = best_row(a3_rows)
        ep_star = b["epochs"]
        print(f"\n[A3 done] (ls*, lr*, ep*) = ({ls_star}, {lr_star}, {ep_star}) (macro_f1={b['macro_f1']:.4f})")

    if not all_rows:
        print("[phase_a] no rows produced — exiting")
        return

    best = best_row(all_rows)
    with open(out_root / "best_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "ls": best["ls"], "lr": best["lr"], "epochs": best["epochs"],
            "inference_id": best["inference_id"],
            "macro_f1": best["macro_f1"],
            "top1_11": best["top1_11"],
            "train_path": best["train_path"],
        }, f, indent=2)
    write_report(all_rows, best, out_root,
                 baseline={"macro_f1": 0.8634, "top1_11": 0.7006})
    print(f"\n[phase_a] DONE - best: ls={best['ls']} lr={best['lr']} ep={best['epochs']} {best['inference_id']} macro_f1={best['macro_f1']:.4f}")
    print(f"[phase_a] outputs: {out_root}")


if __name__ == "__main__":
    main()
