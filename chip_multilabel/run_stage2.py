# -*- coding: utf-8 -*-
"""Stage 2 orchestrator: train 4 variants (T1/T4/T5/T6), evaluate each with all
inference variants, aggregate 4 x 9 = 36 cell matrix.

T0 (= existing backbone, iter 2 baseline) is NOT retrained here — re-use iter 2
results parquet for comparison.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .constants import INFERENCE_VARIANTS
from .eval_dataset import (ChipEvalDataset, discover_records,
                           stratified_val_eval_split)
from .inference_variants import (CellResult, _compute_invalid_masks,
                                 evaluate_cell, forward_all_logits)
from .model_io import load_chip_backbone
from .parquet_io import (write_confusion, write_errors, write_eval_summary,
                        write_per_class_metrics, write_preds_parquet,
                        write_results_matrix, write_thresholds_json)


def train_variant(variant: str, epochs: int, batch: int, accum: int) -> Path:
    """Spawn _train_chip_variant.py and return out_dir Path."""
    cmd = [
        sys.executable, "-X", "utf8", "-m", "chip_multilabel._train_chip_variant",
        "--variant", variant,
        "--epochs", str(epochs),
        "--batch", str(batch),
        "--accum", str(accum),
    ]
    print(f"[stage2] TRAIN {variant} - {' '.join(cmd[2:])}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"train {variant} failed (returncode {proc.returncode})")
    print(proc.stdout)
    elapsed = time.time() - t0
    out_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("[train ")]
    out_dir_line = [ln for ln in proc.stdout.splitlines() if "out=outputs" in ln]
    if not out_dir_line:
        raise RuntimeError(f"could not find out= in train output:\n{proc.stdout}")
    out_dir = out_dir_line[0].split("out=")[-1].strip()
    print(f"[stage2] {variant} trained in {elapsed:.1f}s -> {out_dir}")
    return Path(out_dir)


def run_inference_for_model(
    train_id: str,
    ckpt_path: Path,
    eval_set: Path,
    inference_variants: List[str],
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    device: torch.device,
) -> List[CellResult]:
    print(f"[stage2] INFER {train_id} - loading {ckpt_path}")
    model, meta, keep_indices = load_chip_backbone(ckpt_path, device)
    records = discover_records(eval_set)
    val_idx, eval_idx = stratified_val_eval_split(records, val_ratio=val_ratio, seed=seed)
    ds = ChipEvalDataset(records, img_size=meta["img_size"])
    inv_mask, inv_score = _compute_invalid_masks(records)
    t0 = time.time()
    logits_full = forward_all_logits(model, ds, device, batch_size=batch_size,
                                    num_workers=num_workers, tta=False)
    print(f"[stage2]   forward {logits_full.shape}  {time.time() - t0:.1f}s")

    cells: List[CellResult] = []
    for vid in inference_variants:
        cell = evaluate_cell(
            variant_id=vid,
            logits_full=logits_full,
            logits_full_tta=None,
            records=records,
            val_idx=val_idx,
            eval_idx=eval_idx,
            keep_indices=keep_indices,
            invalid_mask=inv_mask,
            invalid_score=inv_score,
            train_id=train_id,
        )
        print(f"[stage2]   {cell.cell_id}  macro_f1={cell.macro_f1:.4f}  top1_11={cell.top1_11class:.4f}  T={cell.temperature:.3f}")
        cells.append(cell)
    return cells


def _write_report(all_cells: List[CellResult], out_root: Path) -> None:
    lines = []
    lines.append("# Stage 2 - chip multi-label training x inference matrix\n")
    lines.append(f"**run dir**: `{out_root}`\n")
    by_train: Dict[str, List[CellResult]] = {}
    for c in all_cells:
        by_train.setdefault(c.train_id, []).append(c)
    lines.append("## Full matrix (sorted macro_f1)\n")
    lines.append("| cell_id | train | inference | macro_f1 | top1_11 | T |")
    lines.append("|---|---|---|---|---|---|")
    for c in sorted(all_cells, key=lambda x: -x.macro_f1)[:30]:
        lines.append(f"| **{c.cell_id}** | {c.train_id} | {c.inference_id} | {c.macro_f1:.4f} | {c.top1_11class:.4f} | {c.temperature:.3f} |")
    lines.append("\n## Per train_id best inference\n")
    lines.append("| train | best inference | macro_f1 | top1_11 |")
    lines.append("|---|---|---|---|")
    for tid, cs in by_train.items():
        b = max(cs, key=lambda x: x.macro_f1)
        lines.append(f"| {tid} | {b.inference_id} | {b.macro_f1:.4f} | {b.top1_11class:.4f} |")
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--variants", default="T1,T4,T5,T6")
    ap.add_argument("--inference-variants", default=",".join(INFERENCE_VARIANTS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--batch-eval", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training; expect --ckpts <T_id>:<path>,...")
    ap.add_argument("--ckpts", default=None,
                    help="comma-separated train_id:ckpt_path for --skip-train")
    args = ap.parse_args()

    train_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    inf_variants = [v.strip() for v in args.inference_variants.split(",") if v.strip()]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"stage2_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[stage2] device={device}  out={out_root}")
    print(f"[stage2] train_variants={train_variants}  inference_variants={inf_variants}")

    train_ckpts: Dict[str, Path] = {}
    if args.skip_train and args.ckpts:
        for pair in args.ckpts.split(","):
            tid, p = pair.split(":", 1)
            train_ckpts[tid.strip()] = Path(p.strip())
    else:
        for tv in train_variants:
            ckpt_dir = train_variant(tv, epochs=args.epochs, batch=args.batch, accum=args.accum)
            train_ckpts[tv] = ckpt_dir / "best_model.pth"

    all_cells: List[CellResult] = []
    for tv in train_variants:
        cells = run_inference_for_model(
            train_id=tv,
            ckpt_path=train_ckpts[tv],
            eval_set=Path(args.eval_set),
            inference_variants=inf_variants,
            batch_size=args.batch_eval,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=device,
        )
        all_cells.extend(cells)

    matrix_rows = []
    pcm_rows = []
    confusion_rows = []
    preds_rows = []
    error_rows = []
    thresholds_payload = {}
    for c in all_cells:
        matrix_rows.append({
            "cell_id": c.cell_id,
            "train_id": c.train_id,
            "inference_id": c.inference_id,
            "n_eval": c.n_eval,
            "macro_f1": c.macro_f1,
            "micro_f1": c.micro_f1,
            "mAP": c.mAP,
            "hamming_loss": c.hamming_loss,
            "subset_accuracy": c.subset_accuracy,
            "top1_11class": c.top1_11class,
            "ece_pre": c.ece_pre,
            "ece_post": c.ece_post,
            "temperature": c.temperature,
            "per_class_f1": json.dumps(c.per_class_f1),
            "threshold_dict": json.dumps(c.threshold_dict),
            "elapsed_sec": c.elapsed_sec,
        })
        pcm_rows.extend(c.per_class_rows)
        confusion_rows.extend(c.confusion_rows)
        preds_rows.extend(c.preds_rows)
        error_rows.extend(c.error_rows)
        thresholds_payload[c.cell_id] = {
            "temperature": c.temperature,
            "thresholds": c.threshold_dict,
        }
    write_results_matrix(matrix_rows, out_root / "results_matrix.parquet")
    write_per_class_metrics(pcm_rows, out_root / "per_class_metrics.parquet")
    write_confusion(confusion_rows, out_root / "confusion_11class.parquet")
    write_preds_parquet(preds_rows, out_root / "preds_chip.parquet")
    write_errors(error_rows, out_root / "errors.parquet")
    write_thresholds_json(thresholds_payload, out_root / "thresholds.json")

    best = max(all_cells, key=lambda c: c.macro_f1)
    write_eval_summary({
        "run_dir": str(out_root),
        "ts": ts,
        "n_classes": 11,
        "train_variants": train_variants,
        "inference_variants": inf_variants,
        "stage2_best": {
            "cell_id": best.cell_id,
            "train_id": best.train_id,
            "inference_id": best.inference_id,
            "macro_f1": best.macro_f1,
            "top1_11class": best.top1_11class,
        },
        "primary_metric": "macro_f1",
        "primary_metric_value": best.macro_f1,
        "train_ckpts": {k: str(v) for k, v in train_ckpts.items()},
    }, out_root / "eval_summary.json")
    _write_report(all_cells, out_root)
    print(f"\n[stage2] DONE - best {best.cell_id} macro_f1={best.macro_f1:.4f}")
    print(f"[stage2] outputs: {out_root}")


if __name__ == "__main__":
    main()
