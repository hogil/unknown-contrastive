#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production wafer 33-class predict — walk product/line/date tree, write parquet to mirror result tree.

Input layout:
    <image_root>/<product>/<line>/<date>/*.png            (failbit map wafer images)
    <positions_root>/<product>/<line>/<date>/*.json       (sibling positions JSON, optional metadata)

Output:
    result_wafer/<product>/<line>/<date>/preds.parquet    (1 row / wafer, DB ingestion target)
    logs_predict_wafer/<TS>_<product>_<line>_<date>/      (operational tracking, project-side)
        ├─ _meta.json
        └─ run.log

Walk rule: 3 dirs deep = product → line → date. Subfolders inside the leaf date dir
(classification/, classification_chips/) are NOT processed (training-data containers).

Per-product model resolution: --model-glob default `logs_wafer/{line}/overall/best_model.pth`
where {line} is substituted with each batch's line dir name (e.g. K1AB).
"""
# ===================== CONFIG =====================
DEFAULT_MODEL_GLOB        = "logs_wafer/{line}/overall/best_model.pth"
DEFAULT_RESULT_ROOT       = "result_wafer"
DEFAULT_LOGS_ROOT         = "logs_predict_wafer"
DEFAULT_BATCH_SIZE        = 16
DEFAULT_WORKERS           = 4
DEFAULT_THRESHOLD         = None     # max_prob < threshold -> is_normal=1; None disables
DEFAULT_DEVICE            = None     # None = auto (cuda if available)
KIND_LABEL                = "wafer"
# wafer basename split (synth-format 9-token)
DEFAULT_BASENAME_SCHEMA   = ["prefix","kind","w_idx","date","time","yld","syp","tester","device"]
# JSON top-scalar fields to copy into parquet (sibling positions JSON)
DEFAULT_JSON_FIELDS       = ["partid","pgm","wafer","stime","step","yield","sys","tm","lt","netd","gd"]
# ==================================================

import argparse, glob, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------- common helpers ----------
def resolve_glob_latest(glob_pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(glob_pattern))
    return matches[-1] if matches else None


def print_overall_meta(model_path: str):
    meta_path = Path(model_path).parent / "_overall_meta.json"
    if not meta_path.exists():
        return
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        v = m.get("val_f1")
        v_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        print(f"[*]   sourced from run='{m.get('best_run')}'  val_f1={v_str}  seeded_at={m.get('seeded_at')}",
              file=sys.stderr)
    except Exception as e:
        print(f"[*]   meta read failed: {e}", file=sys.stderr)


def load_wafer_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes: List[str] = ckpt["classes"]
    img_size: int = int(ckpt.get("img_size", 384))
    backbone: str = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))
    state = (ckpt.get("model") if isinstance(ckpt.get("model"), dict)
             else ckpt.get("state_dict") if isinstance(ckpt.get("state_dict"), dict)
             else ckpt)
    model.load_state_dict(state, strict=False)
    if "ema_state" in ckpt:
        ema = ckpt["ema_state"]
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    p.data.copy_(ema[n])
    model.eval().to(device)
    return model, classes, img_size


def find_batches(image_root: Path, target: Optional[str] = None) -> List[tuple]:
    """Walk image_root 3 levels (product/line/date). Returns list of (product, line, date, leaf_path)."""
    if target:
        parts = target.replace("\\", "/").strip("/").split("/")
        if len(parts) != 3:
            raise SystemExit(f"--batch must be <product>/<line>/<date>, got {target!r}")
        leaf = image_root / parts[0] / parts[1] / parts[2]
        if not leaf.is_dir():
            raise SystemExit(f"target batch not found: {leaf}")
        return [(parts[0], parts[1], parts[2], leaf)]
    out = []
    for product_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for line_dir in sorted(p for p in product_dir.iterdir() if p.is_dir()):
            for date_dir in sorted(p for p in line_dir.iterdir() if p.is_dir()):
                out.append((product_dir.name, line_dir.name, date_dir.name, date_dir))
    return out


def split_basename(basename: str, schema: List[str]) -> Dict[str, str]:
    parts = basename.split("_")
    return {name: (parts[i] if i < len(parts) else "") for i, name in enumerate(schema)}


def read_json_meta(json_path: Path, fields: List[str]) -> Dict:
    if not json_path.exists():
        return {f: "" for f in fields}
    try:
        j = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {f: "" for f in fields}
    out = {}
    for f in fields:
        v = j.get(f, "")
        if isinstance(v, (list, dict)):
            v = ""
        out[f] = v
    return out


# --------- Dataset ----------
class WaferList(Dataset):
    def __init__(self, paths: List[Path], img_size: int):
        self.paths = paths
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), norm,
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            x = self.tfm(im.convert("RGB"))
        return x, str(p)


# --------- Per-batch processing ----------
def process_batch(product: str, line: str, date: str, leaf: Path,
                  positions_root: Path, result_root: Path, logs_root: Path,
                  model_glob_template: str, args, device: torch.device) -> Dict:
    """Run wafer predict on one batch; return summary dict for _meta.json."""
    ts = time.strftime("%y%m%d_%H%M%S")
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    log_dir = logs_root / f"{ts}_{product}_{line}_{date}"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / "run.log"

    def log(msg: str):
        line_out = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line_out, flush=True)
        with run_log_path.open("a", encoding="utf-8") as f:
            f.write(line_out + "\n")

    summary: Dict = {
        "kind": KIND_LABEL,
        "product": product, "line": line, "date": date,
        "batch_path": str(leaf),
        "ts": ts,
        "started_at": started_at,
        "status": "pending",
    }

    result_dir = result_root / product / line / date
    result_parquet = result_dir / "preds.parquet"
    summary["result_parquet"] = str(result_parquet)

    if result_parquet.exists() and not args.overwrite:
        log(f"SKIP (preds.parquet exists, no --overwrite): {result_parquet}")
        summary["status"] = "skipped_existing"
        return summary

    # resolve model
    glob_pattern = model_glob_template.format(line=line, product=product)
    log(f"resolve model: {glob_pattern}")
    model_path = resolve_glob_latest(glob_pattern)
    if not model_path:
        log(f"MODEL NOT FOUND (skip batch)")
        summary["status"] = "model_missing"
        summary["model_glob"] = glob_pattern
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    log(f"model: {model_path}")
    print_overall_meta(model_path)
    summary["model_path"] = model_path

    # collect inputs (top-level *.png only, no recursion)
    pngs = sorted(p for p in leaf.glob("*.png") if p.is_file())
    summary["n_input"] = len(pngs)
    if not pngs:
        log("no PNGs in leaf — skip")
        summary["status"] = "empty"
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    log(f"n_input PNG: {len(pngs)}")

    # load model
    model, classes, img_size = load_wafer_model(Path(model_path), device)
    summary["n_classes"] = len(classes)
    summary["img_size"] = img_size

    # inference
    ds = WaferList(pngs, img_size)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type == "cuda"))
    rows = []
    pos_dir = positions_root / product / line / date
    threshold = args.threshold
    n_normal = 0
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for xb, pb in ld:
            xb = xb.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                logits = logits.float()
            else:
                logits = model(xb)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            probs_np = probs.cpu().numpy()
            preds = preds.cpu().numpy(); confs = confs.cpu().numpy()
            for i, p_str in enumerate(pb):
                pi = int(preds[i]); mp = float(confs[i])
                is_normal = int(threshold is not None and mp < threshold)
                if is_normal: n_normal += 1
                wb = Path(p_str).stem
                row = {
                    "path": p_str,
                    "wafer_basename": wb,
                    "batch_product": product,
                    "batch_line": line,
                    "batch_date": date,
                }
                row.update(split_basename(wb, DEFAULT_BASENAME_SCHEMA))
                row.update(read_json_meta(pos_dir / f"{wb}.json", DEFAULT_JSON_FIELDS))
                row["wafer_class"] = classes[pi]
                row["wafer_class_idx"] = pi
                row["max_prob"] = float(mp)
                row["is_normal"] = is_normal
                for k, c in enumerate(classes):
                    row[f"prob_{c}"] = float(probs_np[i, k])
                rows.append(row)

    # write parquet (and optional csv)
    df = pd.DataFrame(rows)
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_parquet, index=False)
    summary["n_processed"] = len(rows)
    summary["n_normal"] = n_normal
    log(f"wrote {result_parquet}  rows={len(rows)}")
    if args.csv:
        csv_path = result_dir / "preds.csv"
        df.to_csv(csv_path, index=False)
        log(f"wrote {csv_path}")
        summary["result_csv"] = str(csv_path)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["status"] = "ok"
    with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--positions-root", required=True)
    ap.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    ap.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    ap.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB,
                    help="default substitutes {line} (and optionally {product}) into the path")
    ap.add_argument("--batch", default=None,
                    help="single-batch override: <product>/<line>/<date>. default = walk all")
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=[None, "cuda", "cpu"])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--csv", action="store_true", help="also write preds.csv (default OFF for prod size)")
    ap.add_argument("--overwrite", action="store_true",
                    help="default: skip batches whose preds.parquet already exists")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    positions_root = Path(args.positions_root)
    result_root = Path(args.result_root)
    logs_root = Path(args.logs_root)

    if not image_root.is_dir():
        raise SystemExit(f"image-root not found: {image_root}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[*] cnn_predict kind={KIND_LABEL}  device={device}", file=sys.stderr)

    batches = find_batches(image_root, args.batch)
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    print(f"[*] batches: {len(batches)}", file=sys.stderr)

    total_summary = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "kind": KIND_LABEL, "n_batches": len(batches),
                     "batches": []}
    for product, line, date, leaf in batches:
        try:
            s = process_batch(product, line, date, leaf,
                              positions_root, result_root, logs_root,
                              args.model_glob, args, device)
        except Exception as e:
            s = {"product": product, "line": line, "date": date,
                 "batch_path": str(leaf), "status": "error", "error": str(e)}
            print(f"[!] batch {product}/{line}/{date} failed: {e}", file=sys.stderr)
        total_summary["batches"].append(s)
        print(f"  [{product}/{line}/{date}] status={s.get('status')} "
              f"n={s.get('n_input', '?')} processed={s.get('n_processed', '?')} "
              f"-> {s.get('result_parquet', '-')}", file=sys.stderr)

    total_summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[*] DONE — {len(batches)} batches processed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
