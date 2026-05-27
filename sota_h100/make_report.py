#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate a seed-sweep of single-SOTA runs into RESULTS.md.

Reads each seed's run_stage1 output (results_matrix.parquet) under
  <sweep_root>/seed_<N>/eval/eval_*/results_matrix.parquet
and produces <sweep_root>/RESULTS.md with one consolidated code-block table
(per CLAUDE.md table policy): per-seed bit_F1 / NI-FAR / OOD-FAR / Total-FAR for
the I10 and I13 inference cells, plus mean +- std and the best seed.

Usage:
  python -m sota_h100.make_report --sweep-root outputs/sota_h100_seedsweep_<TS>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CELLS = ["T0__I10", "T0__I13"]


def _load_seed(seed_dir: Path):
    """Return {cell_id: dict(bit_F1, NI, OOD, Total)} for one seed, or None.

    Primary source = bit_far_metrics.json (stable dict keyed by cell_id).
    Fallback = results_matrix.parquet (current schema: eval_bit_F1 / eval_*_FAR).
    """
    js = sorted(glob.glob(str(seed_dir / "eval" / "eval_*" / "bit_far_metrics.json")))
    if not js:
        js = sorted(glob.glob(str(seed_dir / "**" / "bit_far_metrics.json"), recursive=True))
    if js:
        d = json.loads(Path(js[-1]).read_text(encoding="utf-8"))
        out = {}
        for cell, m in d.items():
            out[str(cell)] = {
                "bit_F1": float(m.get("bit_F1", 0.0)),
                "NI": float(m.get("NI_FAR", 0.0)) * 100,
                "OOD": float(m.get("OOD_FAR", 0.0)) * 100,
                "Total": float(m.get("Total_FAR", 0.0)) * 100,
            }
        return out or None

    pq = sorted(glob.glob(str(seed_dir / "eval" / "eval_*" / "results_matrix.parquet")))
    if not pq:
        pq = sorted(glob.glob(str(seed_dir / "**" / "results_matrix.parquet"), recursive=True))
    if not pq:
        return None
    import pandas as pd
    df = pd.read_parquet(pq[-1])
    if "eval_bit_F1" not in df.columns:
        return None
    out = {}
    for _, r in df.iterrows():
        out[str(r["cell_id"])] = {
            "bit_F1": float(r["eval_bit_F1"]),
            "NI": float(r["eval_NI_FAR"]) * 100,
            "OOD": float(r["eval_OOD_FAR"]) * 100,
            "Total": float(r["eval_Total_FAR"]) * 100,
        }
    return out


def _fmt_table(rows, header):
    widths = [len(h) for h in header]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    def line(cols):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)) + " |"
    out = [line(header), "|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--cell", default="T0__I10", choices=CELLS,
                    help="primary cell for mean/std/best (default I10 = SOTA recipe)")
    args = ap.parse_args()
    root = Path(args.sweep_root)
    seed_dirs = sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[1]))

    rows = []
    primary_bitf1, primary_total = [], []
    per_seed = {}
    for sd in seed_dirs:
        seed = sd.name.split("_")[1]
        m = _load_seed(sd)
        if not m:
            rows.append([seed, "-", "-", "-", "-", "-", "-", "-", "-", "no eval"])
            continue
        per_seed[seed] = m
        i10 = m.get("T0__I10", {})
        i13 = m.get("T0__I13", {})
        rows.append([
            seed,
            f"{i10.get('bit_F1', 0):.4f}", f"{i10.get('NI', 0):.2f}",
            f"{i10.get('OOD', 0):.2f}", f"{i10.get('Total', 0):.2f}",
            f"{i13.get('bit_F1', 0):.4f}", f"{i13.get('NI', 0):.2f}",
            f"{i13.get('OOD', 0):.2f}", f"{i13.get('Total', 0):.2f}",
            "ok",
        ])
        pc = m.get(args.cell, {})
        if pc:
            primary_bitf1.append(pc["bit_F1"]); primary_total.append(pc["Total"])

    header = ["seed", "I10 bitF1", "I10 NI", "I10 OOD", "I10 Tot",
              "I13 bitF1", "I13 NI", "I13 OOD", "I13 Tot", "status"]

    lines = []
    lines.append(f"# Single-SOTA seed sweep results\n")
    lines.append(f"sweep root: `{root}`  |  seeds: {len(seed_dirs)}  |  "
                 f"recipe: iter116J (T7 BCE+LS0.30 FCM-PM cmp0.25 g3 masked, val margin)\n")
    lines.append(f"metric: bit_F1 = (single+2combo) macro-F1; FAR split NI / OOD / Total "
                 f"(CLAUDE.md 260512 rule). primary cell = {args.cell}.\n")
    lines.append("```")
    lines.append(_fmt_table(rows, header))
    lines.append("```")

    if primary_bitf1:
        bf = np.array(primary_bitf1); tt = np.array(primary_total)
        best_i = int(np.lexsort((-bf, tt))[0])  # lowest Total FAR, then highest bit_F1
        best_seed = [s for s in per_seed][best_i]
        summ = [
            ["mean", f"{bf.mean():.4f}", f"{tt.mean():.2f}"],
            ["std", f"{bf.std():.4f}", f"{tt.std():.2f}"],
            ["min", f"{bf.min():.4f}", f"{tt.min():.2f}"],
            ["max", f"{bf.max():.4f}", f"{tt.max():.2f}"],
            [f"best (seed {best_seed})", f"{bf[best_i]:.4f}", f"{tt[best_i]:.2f}"],
        ]
        lines.append(f"\n## {args.cell} summary across {len(primary_bitf1)} seeds\n")
        lines.append("```")
        lines.append(_fmt_table(summ, ["stat", "bit_F1", "Total_FAR%"]))
        lines.append("```")

    out_path = root / "RESULTS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OUT] {out_path.resolve()}")


if __name__ == "__main__":
    main()
