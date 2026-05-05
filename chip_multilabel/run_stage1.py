# -*- coding: utf-8 -*-
"""Stage 1 orchestrator: existing chip backbone + 6 inference variants -> matrix.

Outputs:
    <out_root>/stage1_<TS>/
        results_matrix.parquet            # 6 rows
        per_class_metrics.parquet
        confusion_11class.parquet
        preds_chip.parquet
        errors.parquet
        errors/<error_type>/<chip>.png    # cap 200 per type per cell (best cell only -> all)
        thresholds.json                   # per-cell T + thresholds
        eval_summary.json
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
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .constants import (DEFAULT_BACKBONE_CKPT, INFERENCE_VARIANTS, TRAIN_CLASSES)
from .eval_dataset import (ChipEvalDataset, discover_records,
                           stratified_val_eval_split)
from .inference_variants import (CellResult, _compute_invalid_masks,
                                 evaluate_cell, forward_all_logits)
from .model_io import load_chip_backbone
from .parquet_io import (write_confusion, write_errors, write_eval_summary,
                        write_per_class_metrics, write_preds_parquet,
                        write_results_matrix, write_thresholds_json)


def _save_errors_for_cell(cell: CellResult, out_root: Path, cap_per_type: int = 200) -> None:
    counts: Dict[str, int] = {}
    for row in cell.error_rows:
        et = row["error_type"]
        n = counts.get(et, 0)
        if n >= cap_per_type:
            continue
        src = Path(row["chip_path"])
        if not src.exists():
            continue
        dst_dir = out_root / "errors" / cell.cell_id / et
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue
        sidecar = {
            "true_class_key": row["true_class_key"],
            "pred_class_key": row["pred_class_key"],
            "true_labels": row["true_labels"],
            "pred_labels": row["pred_labels"],
            "decision_type": row["decision_type"],
            "per_class_probs": row["per_class_probs"],
        }
        with open(dst.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        counts[et] = n + 1


def _write_report(cells: List[CellResult], best_cell: CellResult, out_root: Path) -> None:
    lines: List[str] = []
    lines.append(f"# Stage 1 — chip multi-label inference variant matrix\n")
    lines.append(f"**run dir**: `{out_root}`")
    lines.append(f"**ts**: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("## Results matrix\n")
    lines.append("| cell | macro_f1 | micro_f1 | mAP | hamming | subset_acc | top1_11cls | T | ECE_pre | ECE_post | sec |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda x: -x.macro_f1):
        lines.append(
            f"| **{c.cell_id}** | {c.macro_f1:.4f} | {c.micro_f1:.4f} | {c.mAP:.4f} | "
            f"{c.hamming_loss:.4f} | {c.subset_accuracy:.4f} | {c.top1_11class:.4f} | "
            f"{c.temperature:.3f} | {c.ece_pre:.4f} | {c.ece_post:.4f} | {c.elapsed_sec:.1f} |"
        )
    lines.append("\n## Best cell per-class F1\n")
    lines.append(f"**{best_cell.cell_id}** macro_f1={best_cell.macro_f1:.4f}\n")
    lines.append("| class | F1 | threshold |")
    lines.append("|---|---|---|")
    for c in TRAIN_CLASSES:
        lines.append(f"| {c} | {best_cell.per_class_f1.get(c, 0.0):.4f} | {best_cell.threshold_dict.get(c, 0.0):.4f} |")
    lines.append("\n## Per-cell threshold dict\n")
    for c in cells:
        thr_str = " ".join(f"{k}={v:.3f}" for k, v in c.threshold_dict.items())
        lines.append(f"- **{c.cell_id}** T={c.temperature:.3f}  thresh: {thr_str}")
    lines.append("\n## Files\n")
    lines.append("- `results_matrix.parquet` — 6 row matrix")
    lines.append("- `per_class_metrics.parquet` — per cell × per class")
    lines.append("- `confusion_11class.parquet` — 11x11 confusion per cell")
    lines.append("- `preds_chip.parquet` — per chip × per cell")
    lines.append("- `errors.parquet` — per error row")
    lines.append("- `errors/<cell>/<error_type>/<chip>.png` — error gallery (cap 200/type)")
    lines.append("- `thresholds.json` — per cell T + class thresholds")
    lines.append("- `eval_summary.json` — overall summary")
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--eval-set", required=True, help="path to chip_multilabel_eval_<TS>/ directory")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--variants", default=",".join(INFERENCE_VARIANTS),
                    help="comma-separated subset of I0,I1,I2,I3,I4,I5")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--error-cap", type=int, default=200)
    args = ap.parse_args()

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in requested:
        if v not in INFERENCE_VARIANTS:
            raise SystemExit(f"unknown variant {v} (allowed: {INFERENCE_VARIANTS})")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"stage1_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[stage1] device={device}  out={out_root}")
    print(f"[stage1] loading model from {args.model}")
    model, meta, keep_indices = load_chip_backbone(args.model, device)
    print(f"[stage1] model loaded - backbone={meta['backbone']} img_size={meta['img_size']} "
          f"classes_full={meta['classes_full']} keep_idx={keep_indices}")

    print(f"[stage1] discovering eval set at {args.eval_set}")
    records = discover_records(args.eval_set)
    if not records:
        raise SystemExit(f"no records found at {args.eval_set}")
    print(f"[stage1] discovered {len(records)} chips across {len(set(r.class_key for r in records))} class keys")
    val_idx, eval_idx = stratified_val_eval_split(records, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[stage1] split val={len(val_idx)} eval={len(eval_idx)}")

    ds = ChipEvalDataset(records, img_size=meta["img_size"])

    print(f"[stage1] computing invalid heuristic masks")
    inv_mask, inv_score = _compute_invalid_masks(records)
    print(f"[stage1]   invalid detected pred={int(inv_mask.sum())} (gt={sum(r.is_invalid_gt for r in records)})")

    print(f"[stage1] forward pass")
    t0 = time.time()
    logits_full = forward_all_logits(model, ds, device, batch_size=args.batch_size,
                                    num_workers=args.num_workers, tta=False)
    print(f"[stage1]   {logits_full.shape}  {time.time() - t0:.1f}s")

    cells: List[CellResult] = []
    for vid in requested:
        print(f"[stage1] evaluating cell T0__{vid}")
        cell = evaluate_cell(
            variant_id=vid,
            logits_full=logits_full,
            logits_full_tta=None,  # TTA permanently disabled per Hard Rules
            records=records,
            val_idx=val_idx,
            eval_idx=eval_idx,
            keep_indices=keep_indices,
            invalid_mask=inv_mask,
            invalid_score=inv_score,
            train_id="T0",
        )
        print(f"[stage1]   {cell.cell_id}  macro_f1={cell.macro_f1:.4f}  top1_11={cell.top1_11class:.4f}  T={cell.temperature:.3f}  ({cell.elapsed_sec:.1f}s)")
        cells.append(cell)

    # aggregate writes
    matrix_rows: List[Dict] = []
    pcm_rows: List[Dict] = []
    confusion_rows: List[Dict] = []
    preds_rows: List[Dict] = []
    error_rows: List[Dict] = []
    thresholds_payload: Dict[str, Dict] = {}
    for c in cells:
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
            "invalid_white_ratio": 0.95,
        }

    write_results_matrix(matrix_rows, out_root / "results_matrix.parquet")
    write_per_class_metrics(pcm_rows, out_root / "per_class_metrics.parquet")
    write_confusion(confusion_rows, out_root / "confusion_11class.parquet")
    write_preds_parquet(preds_rows, out_root / "preds_chip.parquet")
    write_errors(error_rows, out_root / "errors.parquet")
    write_thresholds_json(thresholds_payload, out_root / "thresholds.json")

    best = max(cells, key=lambda c: c.macro_f1)
    print(f"\n[stage1] BEST cell: {best.cell_id}  macro_f1={best.macro_f1:.4f}  top1_11={best.top1_11class:.4f}")
    _save_errors_for_cell(best, out_root, cap_per_type=args.error_cap)

    write_eval_summary({
        "run_dir": str(out_root),
        "ts": ts,
        "model_meta": meta,
        "n_eval": len(eval_idx),
        "n_val": len(val_idx),
        "n_classes": 11,
        "stage1": {
            "best_cell_id": best.cell_id,
            "best_macro_f1": best.macro_f1,
            "all_cells": [
                {"cell_id": c.cell_id, "macro_f1": c.macro_f1, "top1_11class": c.top1_11class,
                 "temperature": c.temperature, "ece_pre": c.ece_pre, "ece_post": c.ece_post}
                for c in cells
            ],
        },
        "primary_metric": "macro_f1",
        "primary_metric_value": best.macro_f1,
    }, out_root / "eval_summary.json")

    _write_report(cells, best, out_root)
    print(f"\n[stage1] DONE — outputs at {out_root}")
    print(f"[stage1] report: {out_root / 'report.md'}")


if __name__ == "__main__":
    main()
