#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build per-wafer object_type_id map (uint8 grid) using chip classifier — incremental.

For each wafer in `D:/project/data/wm-811k/unknown/<class>/<basename>.png` + matching
positions JSON `D:/project/data/positions/unknown/<class>/<basename>.json`:

  - Identify all chips with b >= 200 (defect/invalid)
  - Crop 200x200 from PNG using JSON rect
  - Batch predict using chip classifier (5-class)
  - Map predicted class -> obj_id (1..N; 0 = no object / normal chip)
  - Build (grid_h, grid_w) uint8 array — grid auto-derived from JSON `coord.tiles_w_rot/_h_rot`
    (fallback: max(x_abs)+1 / max(y_abs)+1)
  - Save to `D:/project/data/wm-811k/obj_id_maps/<device>_<date>/<basename>.npy`
    (folder name from JSON `device` + basename token 3 = date; falls back to wafer_class
     if either missing)
  - Optionally save a colored PNG visualization next to the .npy
  - Append per-chip prediction to `obj_id_maps/predictions.csv`

Output is INCREMENTAL — each wafer is saved as soon as all its chips are predicted.
Crash mid-run leaves all completed wafers on disk; resume just re-launches the script
and skips wafers whose .npy already exists.

predictions.csv schema (15 cols, header on first creation):
    prefix,kind,w_idx,date,time,yld,syp,tester,device,x_abs,y_abs,b,wafer_class,obj_id,obj_label

Used as G-channel input for cnn_train_compound.py.
"""
from __future__ import annotations
import argparse, colorsys, csv, glob, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


PNG_ROOT  = Path("D:/project/data/wm-811k/unknown")
JSON_ROOT = Path("D:/project/data/positions/unknown")
OUT_ROOT  = Path("D:/project/data/wm-811k/obj_id_maps")

NONE_ID = 0
MIN_DEFECT_BIN = 200

CSV_HEADER = ["prefix","kind","w_idx","date","time","yld","syp","tester","device",
              "x_abs","y_abs","b","wafer_class","obj_id","obj_label"]


# ---------- color palette for PNG visualization (obj_id 0..N) ----------
def make_palette(n_obj: int, sat: float = 0.70, val: float = 0.85) -> np.ndarray:
    """Generate (n_obj+1, 3) uint8 palette. Idx 0 = white (none / normal chip).
    Idx 1..n_obj = evenly spaced HSV hues so the palette extends automatically
    as the chip classifier gains classes.
    """
    palette = np.zeros((n_obj + 1, 3), dtype=np.uint8)
    palette[0] = (255, 255, 255)
    if n_obj <= 0:
        return palette
    for i in range(n_obj):
        h = i / float(n_obj)
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        palette[i + 1] = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
    return palette


def load_chip_model(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, List[str], int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes: List[str] = ckpt["classes"]
    img_size: int = int(ckpt.get("img_size", 384))
    backbone: str = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))
    sd = ckpt["model"]
    if "ema_state" in ckpt:
        ema = ckpt["ema_state"]
        sd_compat = {k: ema[k] if k in ema else v for k, v in sd.items()}
        model.load_state_dict(sd_compat, strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, classes, img_size


def split_basename(basename: str) -> List[str]:
    """Split synth wafer basename. Per _sample_gen:684 the format is:
        {prefix}_{kind}_{w_idx:02d}_{date}_{time}_{yld:.1f}_{syp:.0f}_{tester}_{device}
    9 tokens. If not 9, pad with empty strings to 9.
    """
    parts = basename.split("_")
    if len(parts) < 9:
        parts = parts + [""] * (9 - len(parts))
    elif len(parts) > 9:
        # join overflow into last token to preserve all info
        parts = parts[:8] + ["_".join(parts[8:])]
    return parts


def derive_subfolder(basename_parts: List[str], json_meta: dict, fallback_wafer_class: str) -> str:
    """`<device>_<date>` subfolder. Uses JSON top-level `device` if present, else
    basename token 8. Date = basename token 3. Falls back to wafer_class if either
    field is empty.
    """
    device = (json_meta.get("device") or "").strip() or basename_parts[8].strip()
    date = basename_parts[3].strip()
    if not device or not date:
        return fallback_wafer_class
    return f"{device}_{date}"


def collect_wafer_specs(limit_per_class: Optional[int]) -> Tuple[List[dict], dict]:
    specs: List[dict] = []
    skipped = {"missing_json": 0, "bad_json": 0, "no_defect": 0}
    cls_dirs = sorted(p for p in PNG_ROOT.iterdir() if p.is_dir())
    for cls_dir in cls_dirs:
        wafer_class = cls_dir.name
        pngs = sorted(cls_dir.glob("*.png"))
        if limit_per_class is not None:
            pngs = pngs[:limit_per_class]
        for png_path in pngs:
            json_path = JSON_ROOT / wafer_class / (png_path.stem + ".json")
            if not json_path.exists():
                skipped["missing_json"] += 1
                continue
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                skipped["bad_json"] += 1
                continue
            # auto-derive grid
            coord = j.get("coord") or {}
            grid_w = int(coord.get("tiles_w_rot") or 0)
            grid_h = int(coord.get("tiles_h_rot") or 0)
            if grid_w <= 0 or grid_h <= 0:
                # fallback: max(x_abs)+1, max(y_abs)+1
                xs = [int(c.get("x_abs", 0)) for c in (j.get("chips") or [])]
                ys = [int(c.get("y_abs", 0)) for c in (j.get("chips") or [])]
                if xs and ys:
                    grid_w = max(xs) + 1
                    grid_h = max(ys) + 1
                else:
                    grid_w = grid_h = 32                                                  # safe default
            # parse chip entries
            chip_entries = []
            for chip in (j.get("chips") or []):
                try:
                    b = int(str(chip.get("b", "0")).strip())
                except Exception:
                    continue
                if b < MIN_DEFECT_BIN: continue
                rect = chip.get("rect") or {}
                try:
                    x0 = int(rect["x0"]); y0 = int(rect["y0"])
                    x1 = int(rect["x1"]); y1 = int(rect["y1"])
                except Exception:
                    continue
                gx = int(chip.get("x_abs", -1)); gy = int(chip.get("y_abs", -1))
                if not (0 <= gx < grid_w and 0 <= gy < grid_h): continue
                chip_entries.append({"gx": gx, "gy": gy, "b": b, "rect": (x0, y0, x1, y1)})
            if not chip_entries:
                skipped["no_defect"] += 1
                continue
            basename = png_path.stem
            parts = split_basename(basename)
            subfolder = derive_subfolder(parts, j, fallback_wafer_class=wafer_class)
            specs.append({
                "wafer_class": wafer_class,
                "basename": basename,
                "basename_parts": parts,
                "png_path": str(png_path),
                "subfolder": subfolder,
                "out_npy": str(OUT_ROOT / subfolder / f"{basename}.npy"),
                "out_png": str(OUT_ROOT / subfolder / f"{basename}.png"),
                "grid_w": grid_w,
                "grid_h": grid_h,
                "chips": chip_entries,
            })
    return specs, skipped


def save_obj_map_png(arr: np.ndarray, palette: np.ndarray, out_path: Path, scale: int = 16) -> None:
    """Save a colored PNG visualization (nearest-neighbor upsample for readability).

    palette: (n_obj+1, 3) uint8, indexed by obj_id. arr values clipped to [0, n_obj].
    """
    n_colors = palette.shape[0]
    safe = np.clip(arr, 0, n_colors - 1).astype(np.intp)
    rgb = palette[safe]                                                                  # (h, w, 3)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip-model", required=True, help="chip classifier best_model.pth (glob 가능)")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    ap.add_argument("--limit-per-class", type=int, default=None, help="smoke 모드")
    ap.add_argument("--overwrite", action="store_true", help="기존 .npy 덮어쓰기")
    ap.add_argument("--save-png", action="store_true", default=True,
                    help="(default on) wafer 마다 시각화 PNG 도 저장")
    ap.add_argument("--no-png", dest="save_png", action="store_false",
                    help="PNG 시각화 비활성화 (.npy + CSV 만)")
    ap.add_argument("--csv-path", default=None, help="predictions.csv 경로 (default: <out-root>/predictions.csv)")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.chip_model))
    if not matches:
        raise SystemExit(f"chip-model not found: {args.chip_model}")
    chip_ckpt = Path(matches[-1])
    print(f"[chip-model] {chip_ckpt}", flush=True)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[device] {device}", flush=True)

    model, classes, img_size = load_chip_model(chip_ckpt, device)
    print(f"[model] classes (ImageFolder alphabetical)={classes}  img_size={img_size}", flush=True)
    n_chip_objects = len(classes)

    class_idx_to_obj_id = np.arange(1, n_chip_objects + 1, dtype=np.uint8)
    obj_id_to_label = ["none"] + list(classes)
    print(f"[obj_id mapping] " + ", ".join(f"{i}={lbl}" for i, lbl in enumerate(obj_id_to_label)), flush=True)
    palette = make_palette(n_chip_objects)
    print(f"[palette] HSV-evenly-spaced, n_colors={palette.shape[0]} (idx 0=none white)", flush=True)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv_path) if args.csv_path else (out_root / "predictions.csv")
    csv_new = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_f = csv_path.open("a" if not args.overwrite else "w", encoding="utf-8", newline="")
    csv_w = csv.writer(csv_f)
    if csv_new or args.overwrite:
        csv_w.writerow(CSV_HEADER); csv_f.flush()
    print(f"[csv] {'NEW' if csv_new else 'APPEND'} {csv_path}", flush=True)

    specs, skipped = collect_wafer_specs(args.limit_per_class)
    print(f"[scan] wafers={len(specs)} skipped={skipped}", flush=True)

    if not args.overwrite:
        before = len(specs)
        specs = [s for s in specs if not Path(s["out_npy"]).exists()]
        print(f"[skip-existing] {before - len(specs)} already built; {len(specs)} remaining", flush=True)
    if not specs:
        print("[done] nothing to do", flush=True)
        csv_f.close()
        return 0

    n_total_chips = sum(len(s["chips"]) for s in specs)
    print(f"[plan] wafers={len(specs)}  total_defect_chips={n_total_chips}  batch={args.batch}", flush=True)

    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), norm,
    ])

    use_amp = (device.type == "cuda")
    counts_by_label: Dict[str, int] = {lbl: 0 for lbl in obj_id_to_label}
    t0 = time.time()
    n_done = 0
    last_progress_print = 0

    with torch.no_grad():
        for spec_i, spec in enumerate(specs):
            chips = spec["chips"]
            grid_h = spec["grid_h"]; grid_w = spec["grid_w"]
            obj_map = np.zeros((grid_h, grid_w), dtype=np.uint8)
            wafer_class = spec["wafer_class"]
            basename = spec["basename"]
            parts = spec["basename_parts"]

            try:
                img = Image.open(spec["png_path"])
            except Exception as e:
                print(f"[err] open failed {spec['png_path']}: {e}", flush=True)
                continue

            csv_rows = []
            for i in range(0, len(chips), args.batch):
                batch_chips = chips[i:i+args.batch]
                tensors = [tfm(img.crop(c["rect"]).convert("RGB")) for c in batch_chips]
                tens = torch.stack(tensors, dim=0).to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(tens)
                else:
                    logits = model(tens)
                preds = logits.argmax(dim=-1).cpu().numpy()
                for c, p in zip(batch_chips, preds):
                    oid = int(class_idx_to_obj_id[int(p)])
                    obj_map[c["gy"], c["gx"]] = oid
                    obj_label = obj_id_to_label[oid]
                    counts_by_label[obj_label] += 1
                    csv_rows.append((*parts, c["gx"], c["gy"], c["b"], wafer_class, oid, obj_label))
                n_done += len(batch_chips)
                if n_done - last_progress_print >= max(args.batch * 50, 6400):
                    rate = n_done / max(0.1, time.time() - t0)
                    eta = (n_total_chips - n_done) / max(0.1, rate)
                    print(f"  predicted {n_done}/{n_total_chips}  rate={rate:.1f}/s  eta={eta:.0f}s  "
                          f"(wafer {spec_i+1}/{len(specs)})", flush=True)
                    last_progress_print = n_done

            # save .npy
            out_npy = Path(spec["out_npy"])
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, obj_map)
            # save PNG
            if args.save_png:
                try:
                    save_obj_map_png(obj_map, palette, Path(spec["out_png"]))
                except Exception as e:
                    print(f"[png-err] {spec['out_png']}: {e}", flush=True)
            # flush CSV rows for this wafer
            csv_w.writerows(csv_rows)
            csv_f.flush()

    csv_f.close()

    print(f"[predict] done in {(time.time() - t0):.1f}s", flush=True)
    print(f"[counts_by_label] {counts_by_label}", flush=True)

    meta_path = out_root / "_meta.json"
    meta = {
        "n_chip_objects": n_chip_objects,
        "chip_classes": list(classes),
        "obj_id_to_label": obj_id_to_label,
        "chip_model": str(chip_ckpt).replace("\\", "/"),
        "subfolder_scheme": "<device>_<date>  (fallback: wafer_class)",
        "csv_path": str(csv_path).replace("\\", "/"),
        "csv_header": CSV_HEADER,
        "save_png": bool(args.save_png),
        "min_defect_bin": MIN_DEFECT_BIN,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[meta] {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
