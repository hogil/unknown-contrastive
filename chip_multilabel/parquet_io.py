"""Parquet writers + JSON summary writer.

Schemas defined in plan: preds_chip / results_matrix / per_class_metrics /
confusion_11class / errors. JSON: thresholds.json + eval_summary.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_preds_parquet(rows: List[Dict[str, Any]], out: Path) -> None:
    if not rows:
        df = pd.DataFrame(columns=[
            "cell_id", "chip_path", "class_key", "true_labels", "pred_labels",
            "pred_class_key", "prob_bank_boundary", "prob_fork", "prob_scratch",
            "prob_scratch_rot", "max_prob", "invalid_score", "decision_type",
            "correct_multihot", "correct_11class",
        ])
    else:
        df = pd.DataFrame(rows)
    df.to_parquet(_ensure(out), index=False)


def write_results_matrix(rows: List[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(_ensure(out), index=False)


def write_per_class_metrics(rows: List[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(_ensure(out), index=False)


def write_confusion(rows: List[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(_ensure(out), index=False)


def write_errors(rows: List[Dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(_ensure(out), index=False)


def write_thresholds_json(payload: Dict[str, Any], out: Path) -> None:
    with open(_ensure(out), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_eval_summary(payload: Dict[str, Any], out: Path) -> None:
    def _convert(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_convert(x) for x in o]
        return o
    with open(_ensure(out), "w", encoding="utf-8") as f:
        json.dump(_convert(payload), f, indent=2, ensure_ascii=False)


def confusion_to_rows(cell_id: str, cm, labels) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            c = int(cm[i, j])
            if c == 0:
                continue
            rows.append({
                "cell_id": cell_id,
                "true_class_key": t,
                "pred_class_key": p,
                "count": c,
            })
    return rows
