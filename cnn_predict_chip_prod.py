#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production chip 5-class predict — walk product/line/date tree, crop chips inline from wafer + JSON, parquet.

Input layout:
    <image_root>/<product>/<line>/<date>/*.png            (failbit map wafer images)
    <positions_root>/<product>/<line>/<date>/*.json       (sibling positions JSON, REQUIRED — chip rects)

Output:
    result_chip/<product>/<line>/<date>/preds.parquet     (1 row / chip, defect chips with b ≥ 200)
    logs_predict_chip/<TS>_<product>_<line>_<date>/       (operational tracking)

Per wafer: read JSON → defect chips list → 200x200 crop from wafer PNG → batch through chip CNN → emit chip rows.
Skips wafers whose JSON is missing (logged as n_skipped_no_json).
"""
# ===================== CONFIG =====================
DEFAULT_MODEL_GLOB        = "logs_chip/{line}/overall/best_model.pth"
DEFAULT_RESULT_ROOT       = "result_chip"
DEFAULT_LOGS_ROOT         = "logs_predict_chip"
DEFAULT_BATCH_SIZE        = 128       # chip crops are 200x200 -> 384x384 small, can batch big
DEFAULT_WORKERS           = 0         # workers=0 better since wafer image is reused across many chips
DEFAULT_THRESHOLD         = None
DEFAULT_DEVICE            = None
KIND_LABEL                = "chip"
MIN_DEFECT_BIN            = 200       # match _build_obj_id_maps.MIN_DEFECT_BIN
# wafer basename split
DEFAULT_BASENAME_SCHEMA   = ["prefix","kind","w_idx","date","time","yld","syp","tester","device"]
DEFAULT_JSON_FIELDS       = ["partid","pgm","wafer","stime","step","yield","sys","tm","lt","netd","gd"]
# ==================================================

import argparse, glob, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------- common helpers (mirror cnn_predict_wafer_prod) ----------
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


def load_chip_model(ckpt_path: Path, device: torch.device):
    """Match _build_obj_id_maps.py:load_chip_model — prefers EMA shadow."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes: List[str] = ckpt["classes"]
    img_size: int = int(ckpt.get("img_size", 384))
    backbone: str = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if "ema_state" in ckpt:
        ema = ckpt["ema_state"]
        sd_compat = {k: ema[k] if k in ema else v for k, v in sd.items()}
        model.load_state_dict(sd_compat, strict=False)
    else:
        model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model, classes, img_size


def find_batches(image_root: Path, target: Optional[str] = None) -> List[tuple]:
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


# --------- chip extraction (mirror _build_obj_id_maps collect_wafer_specs but on a single JSON) ----------
def parse_chips_from_json(json_path: Path) -> List[dict]:
    """Returns list of dicts with {gx, gy, b, rect=(x0,y0,x1,y1)} for chips with b >= MIN_DEFECT_BIN."""
    if not json_path.exists():
        return []
    try:
        j = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    chip_entries = []
    for chip in (j.get("chips") or []):
        try:
            b = int(str(chip.get("b", "0")).strip())
        except Exception:
            continue
        if b < MIN_DEFECT_BIN:
            continue
        rect = chip.get("rect") or {}
        try:
            x0 = int(rect["x0"]); y0 = int(rect["y0"])
            x1 = int(rect["x1"]); y1 = int(rect["y1"])
        except Exception:
            continue
        gx = int(chip.get("x_abs", -1))
        gy = int(chip.get("y_abs", -1))
        if gx < 0 or gy < 0:
            continue
        chip_entries.append({"gx": gx, "gy": gy, "b": b, "rect": (x0, y0, x1, y1)})
    return chip_entries


# --------- Per-batch ----------
def process_batch(product: str, line: str, date: str, leaf: Path,
                  positions_root: Path, result_root: Path, logs_root: Path,
                  model_glob_template: str, args, device: torch.device) -> Dict:
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
        "ts": ts, "started_at": started_at,
        "status": "pending",
    }
    result_dir = result_root / product / line / date
    result_parquet = result_dir / "preds.parquet"
    summary["result_parquet"] = str(result_parquet)
    if result_parquet.exists() and not args.overwrite:
        log(f"SKIP (preds.parquet exists): {result_parquet}")
        summary["status"] = "skipped_existing"
        return summary

    glob_pattern = model_glob_template.format(line=line, product=product)
    log(f"resolve model: {glob_pattern}")
    model_path = resolve_glob_latest(glob_pattern)
    if not model_path:
        log(f"MODEL NOT FOUND")
        summary["status"] = "model_missing"; summary["model_glob"] = glob_pattern
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    log(f"model: {model_path}")
    print_overall_meta(model_path)
    summary["model_path"] = model_path

    pngs = sorted(p for p in leaf.glob("*.png") if p.is_file())
    summary["n_input"] = len(pngs)
    if not pngs:
        log("no PNGs"); summary["status"] = "empty"
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    log(f"n_input wafer PNG: {len(pngs)}")

    model, classes, img_size = load_chip_model(Path(model_path), device)
    summary["n_classes"] = len(classes); summary["img_size"] = img_size

    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), norm,
    ])

    pos_dir = positions_root / product / line / date
    use_amp = (device.type == "cuda")
    threshold = args.threshold

    rows = []
    n_skipped_no_json = 0
    n_chip_processed = 0
    n_normal = 0
    t0 = time.time()

    with torch.no_grad():
        for png_path in pngs:
            wb = png_path.stem
            json_path = pos_dir / f"{wb}.json"
            chips = parse_chips_from_json(json_path)
            if not chips:
                n_skipped_no_json += 1
                continue
            json_meta = read_json_meta(json_path, DEFAULT_JSON_FIELDS)
            wb_split = split_basename(wb, DEFAULT_BASENAME_SCHEMA)
            try:
                wafer_img = Image.open(png_path)
            except Exception as e:
                log(f"open failed {png_path}: {e}")
                continue
            for i in range(0, len(chips), args.batch_size):
                batch_chips = chips[i:i+args.batch_size]
                tensors = [tfm(wafer_img.crop(c["rect"]).convert("RGB")) for c in batch_chips]
                tens = torch.stack(tensors, dim=0).to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(tens)
                    logits = logits.float()
                else:
                    logits = model(tens)
                probs = F.softmax(logits, dim=-1)
                confs, preds = probs.max(dim=-1)
                probs_np = probs.cpu().numpy()
                preds = preds.cpu().numpy(); confs = confs.cpu().numpy()
                for k, c in enumerate(batch_chips):
                    pi = int(preds[k]); mp = float(confs[k])
                    is_normal = int(threshold is not None and mp < threshold)
                    if is_normal: n_normal += 1
                    row = {
                        "path": str(png_path),
                        "wafer_basename": wb,
                        "batch_product": product,
                        "batch_line": line,
                        "batch_date": date,
                    }
                    row.update(wb_split)
                    row.update(json_meta)
                    row["chip_x"] = c["gx"]
                    row["chip_y"] = c["gy"]
                    row["chip_b"] = c["b"]
                    row["chip_object_class"] = classes[pi]
                    row["chip_object_class_idx"] = pi
                    row["max_prob"] = mp
                    row["is_normal"] = is_normal
                    for j, cls_name in enumerate(classes):
                        row[f"prob_{cls_name}"] = float(probs_np[k, j])
                    rows.append(row)
                    n_chip_processed += 1

    elapsed = time.time() - t0
    summary["n_skipped_no_json"] = n_skipped_no_json
    summary["n_chip_processed"] = n_chip_processed
    summary["n_normal"] = n_normal
    summary["elapsed_sec"] = round(elapsed, 1)

    if not rows:
        log(f"NO chip rows produced (all wafers skipped)")
        summary["status"] = "empty_after_filter"
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return summary

    df = pd.DataFrame(rows)
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_parquet, index=False)
    log(f"wrote {result_parquet}  rows={len(rows)}  elapsed={elapsed:.1f}s  rate={len(rows)/max(0.1,elapsed):.1f}/s")
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
    ap.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=[None, "cuda", "cpu"])
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
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

    for product, line, date, leaf in batches:
        try:
            s = process_batch(product, line, date, leaf,
                              positions_root, result_root, logs_root,
                              args.model_glob, args, device)
        except Exception as e:
            s = {"product": product, "line": line, "date": date,
                 "status": "error", "error": str(e)}
            print(f"[!] batch {product}/{line}/{date} failed: {e}", file=sys.stderr)
        print(f"  [{product}/{line}/{date}] status={s.get('status')} "
              f"chips={s.get('n_chip_processed', '?')} "
              f"-> {s.get('result_parquet', '-')}", file=sys.stderr)

    print(f"[*] DONE — {len(batches)} batches", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
