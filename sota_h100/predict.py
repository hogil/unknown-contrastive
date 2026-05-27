#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production inference for the chip multi-label SOTA model on REAL chip images.

Pipeline per chip (200x200 PNG):
  1. forward best_model.pth -> 4 sigmoid probs (bank_boundary, fork, scratch, scratch_rot)
  2. Invalid gate: near-white ratio >= --invalid-white -> 'Invalid'
  3. OOD reject (NB): fit Gaussian Naive Bayes on a reference DEFECT prob set
     (--nb-fit parquet, the 10 defect classes' prob vectors); if a chip's max
     joint log-likelihood < --nb-tau -> 'UNKNOWN' (OOD / Normal, no defect).
     This is the method that separated OOD from real weak combos (per-bit
     threshold alone cannot — see docs). Skip with --no-nb.
  4. else per-bit threshold -> multi-label defect set.

Output CSV: chip_path, prob_*, max_prob, nb_loglik, decision, labels

Usage:
  python -m sota_h100.predict --model <best_model.pth> --input <real_chip_dir> \
      --nb-fit <eval preds_chip.parquet> --out preds_real.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from chip_multilabel.model_io import load_chip_backbone
except Exception:  # allow running from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from chip_multilabel.model_io import load_chip_backbone

TRAIN_BITS = ["bank_boundary", "fork", "scratch", "scratch_rot"]
DEFECT_CLASSES = TRAIN_BITS + [
    "bank_boundary+fork", "bank_boundary+scratch", "bank_boundary+scratch_rot",
    "fork+scratch", "fork+scratch_rot", "scratch+scratch_rot",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# per-bit thresholds (override with --thresholds bb,fork,sc,sr)
DEFAULT_THR = {"bank_boundary": 0.46, "fork": 0.26, "scratch": 0.18, "scratch_rot": 0.26}


def _fit_nb(nb_fit_parquet):
    import pandas as pd
    from sklearn.naive_bayes import GaussianNB
    df = pd.read_parquet(nb_fit_parquet)
    if "cell_id" in df.columns:
        cell = "T0__I10" if "T0__I10" in df["cell_id"].unique() else df["cell_id"].unique()[0]
        df = df[df["cell_id"] == cell]
    cols = ["prob_" + b for b in TRAIN_BITS]
    m = df["class_key"].isin(DEFECT_CLASSES)
    X = np.clip(df.loc[m, cols].values, 1e-6, 1 - 1e-6)
    y = df.loc[m, "class_key"].values
    nb = GaussianNB(var_smoothing=1e-6).fit(X, y)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="best_model.pth (default: latest sota_h100 seed-sweep)")
    ap.add_argument("--input", required=True, help="folder of real chip PNGs (recursive)")
    ap.add_argument("--out", default="preds_real.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--thresholds", default=None, help="bb,fork,sc,sr (override per-bit)")
    ap.add_argument("--invalid-white", type=float, default=0.95)
    ap.add_argument("--nb-fit", default=None, help="reference preds_chip.parquet to fit NB OOD-reject")
    ap.add_argument("--nb-tau", type=float, default=-40.0, help="NB log-lik reject threshold (< tau -> UNKNOWN)")
    ap.add_argument("--no-nb", action="store_true", help="disable NB OOD reject")
    args = ap.parse_args()

    # auto-resolve model / nb-fit when not given (option-less run)
    if not args.model:
        cands = sorted(glob.glob("outputs/sota_h100_*/**/best_model.pth", recursive=True),
                       key=lambda p: Path(p).stat().st_mtime)
        if not cands:
            sys.exit("no --model and no outputs/sota_h100_*/**/best_model.pth found")
        args.model = cands[-1]
        print(f"[predict] auto model: {args.model}")
    if args.nb_fit is None and not args.no_nb:
        cands = sorted(glob.glob("outputs/sota_h100_*/**/preds_chip.parquet", recursive=True),
                       key=lambda p: Path(p).stat().st_mtime)
        if cands:
            args.nb_fit = cands[-1]
            print(f"[predict] auto nb-fit: {args.nb_fit}")

    thr = dict(DEFAULT_THR)
    if args.thresholds:
        vals = [float(x) for x in args.thresholds.split(",")]
        thr = dict(zip(TRAIN_BITS, vals))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, keep_idx = load_chip_backbone(args.model, device)
    model.eval()
    img_size = int(meta["img_size"])
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    keep = torch.tensor(keep_idx, device=device)

    nb = None
    if args.nb_fit and not args.no_nb:
        nb = _fit_nb(args.nb_fit)
        print(f"[predict] NB OOD-reject fit on {args.nb_fit} (tau={args.nb_tau})")

    files = sorted(glob.glob(str(Path(args.input) / "**" / "*.png"), recursive=True))
    print(f"[predict] {len(files)} chips at {args.input}  device={device}  img_size={img_size}")
    rows = []
    for i in range(0, len(files), args.batch_size):
        batch = files[i:i + args.batch_size]
        ims, whites = [], []
        for f in batch:
            im = Image.open(f).convert("RGB")
            arr = np.asarray(im)
            whites.append(float((arr.min(axis=2) > 230).mean()))   # near-white ratio
            ims.append(tf(im))
        x = torch.stack(ims).to(device)
        with torch.no_grad():
            logits = model(x)[:, keep]
            probs = torch.sigmoid(logits).cpu().numpy()
        ll = nb._joint_log_likelihood(np.clip(probs, 1e-6, 1 - 1e-6)).max(axis=1) if nb is not None else [None] * len(batch)
        for f, p, w, lg in zip(batch, probs, whites, ll):
            if w >= args.invalid_white:
                dec, labels = "Invalid", ""
            elif nb is not None and lg < args.nb_tau:
                dec, labels = "UNKNOWN", ""
            else:
                bits = [b for b in TRAIN_BITS if p[TRAIN_BITS.index(b)] >= thr[b]]
                dec = "defect" if bits else "UNKNOWN"
                labels = "+".join(bits)
            rows.append({
                "chip_path": f,
                "prob_bank_boundary": round(float(p[0]), 4), "prob_fork": round(float(p[1]), 4),
                "prob_scratch": round(float(p[2]), 4), "prob_scratch_rot": round(float(p[3]), 4),
                "max_prob": round(float(p.max()), 4),
                "nb_loglik": (round(float(lg), 2) if lg is not None else ""),
                "decision": dec, "labels": labels,
            })
        if (i // args.batch_size) % 20 == 0:
            print(f"[predict] {min(i + args.batch_size, len(files))}/{len(files)}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_def = sum(1 for r in rows if r["decision"] == "defect")
    n_unk = sum(1 for r in rows if r["decision"] == "UNKNOWN")
    n_inv = sum(1 for r in rows if r["decision"] == "Invalid")
    print(f"[predict] done: defect={n_def} UNKNOWN={n_unk} Invalid={n_inv}  -> {args.out}")


if __name__ == "__main__":
    main()
