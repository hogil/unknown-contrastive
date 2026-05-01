#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build per-wafer object_type_id map (32x32 uint8) using chip classifier.

For each wafer in `D:/project/data/wm-811k/unknown/<class>/*.png` + matching
positions JSON `D:/project/data/positions/unknown/<class>/<basename>.json`:

  - Identify all chips with b >= 200 (defect/invalid chips)
  - Crop 200x200 from PNG using JSON rect
  - Batch predict using chip classifier (5-class)
  - Map predicted class -> object_type_id (1..5; 0 = no object / normal chip)
  - Build 32x32 uint8 array
  - Save to `D:/project/data/wm-811k/obj_id_maps/<wafer_class>/<basename>.npy`

Used as G channel input for cnn_train_failobj.py 3-channel feature CNN.

OBJECT_TYPE_ID mapping (must match cnn_train_failobj.OBJECT_TYPE_ID):
  none=0, invalid_main=1, scratch=2, scratch_21deg=3, particle_blast=4, bank_boundary=5

Usage:
  # smoke (5 wafer per class)
  python _build_obj_id_maps.py --chip-model log/chip5_n100_w0_*/best_model.pth \
      --limit-per-class 5 --overwrite

  # full
  python _build_obj_id_maps.py --chip-model log/chip5_n100_w0_*/best_model.pth \
      --batch 64 --overwrite

  # CPU fallback (when GPU is busy)
  python _build_obj_id_maps.py --chip-model log/chip5_n100_w0_*/best_model.pth \
      --device cpu --batch 16
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# must match cnn_train_failobj.OBJECT_TYPE_ID
OBJECT_TYPE_ID = {
    "none": 0,
    "invalid_main": 1,
    "scratch": 2,
    "scratch_21deg": 3,
    "particle_blast": 4,
    "bank_boundary": 5,
}
GRID = 32
MIN_DEFECT_BIN = 200


def load_chip_model(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, List[str], int]:
    """Load chip classifier checkpoint. Returns (model, classes, img_size)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes: List[str] = ckpt["classes"]
    img_size: int = int(ckpt.get("img_size", 384))
    backbone: str = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))
    sd = ckpt["model"]
    if "ema_state" in ckpt:
        # prefer EMA shadow weights for inference
        ema = ckpt["ema_state"]
        sd_compat = {k: ema[k] if k in ema else v for k, v in sd.items()}
        model.load_state_dict(sd_compat, strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model, classes, img_size


class WaferChipDataset(Dataset):
    """Iterate ALL defect chips across ALL wafers as a flat sequence.

    Each item = (chip_tensor, wafer_index, gx, gy). For each wafer, the
    chip ordering preserves JSON order.
    """
    def __init__(self, wafer_specs: List[dict], img_size: int):
        self.specs = wafer_specs
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.tfm = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            norm,
        ])
        self.entries: List[Tuple[int, int, int, Tuple[int,int,int,int]]] = []
        for wi, spec in enumerate(wafer_specs):
            for ch in spec["chips"]:
                self.entries.append((wi, ch["gx"], ch["gy"], ch["rect"]))
        self._wafer_imgs: Dict[int, Image.Image] = {}

    def __len__(self): return len(self.entries)

    def _get_img(self, wi: int) -> Image.Image:
        # cache per wafer (sequential access pattern, only need 1 at a time)
        if wi not in self._wafer_imgs:
            if len(self._wafer_imgs) > 1:
                # drop old to free memory
                self._wafer_imgs.clear()
            self._wafer_imgs[wi] = Image.open(self.specs[wi]["png_path"])
        return self._wafer_imgs[wi]

    def __getitem__(self, i):
        wi, gx, gy, rect = self.entries[i]
        img = self._get_img(wi)
        crop = img.crop(rect).convert("RGB")
        tens = self.tfm(crop)
        return tens, wi, gx, gy


def collect_wafer_specs(limit_per_class: Optional[int]) -> List[dict]:
    """Walk PNG_ROOT/<class>/*.png + matching JSON, build per-wafer chip task list."""
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
                if not (0 <= gx < GRID and 0 <= gy < GRID): continue
                chip_entries.append({"gx": gx, "gy": gy, "rect": (x0, y0, x1, y1)})
            if not chip_entries:
                skipped["no_defect"] += 1
                continue
            specs.append({
                "wafer_class": wafer_class,
                "basename": png_path.stem,
                "png_path": str(png_path),
                "out_path": str(OUT_ROOT / wafer_class / f"{png_path.stem}.npy"),
                "chips": chip_entries,
            })
    return specs, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip-model", required=True, help="path to chip classifier best_model.pth (glob 가능)")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader num_workers (0 권장 — 단일 wafer image 캐시 재사용)")
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    ap.add_argument("--limit-per-class", type=int, default=None, help="smoke 모드 — wafer class당 N장만 처리")
    ap.add_argument("--overwrite", action="store_true", help="기존 npy 덮어쓰기")
    args = ap.parse_args()

    # resolve chip model path (glob)
    matches = sorted(glob.glob(args.chip_model))
    if not matches:
        raise SystemExit(f"chip-model not found: {args.chip_model}")
    chip_ckpt = Path(matches[-1])
    print(f"[chip-model] {chip_ckpt}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[device] {device}")

    model, classes, img_size = load_chip_model(chip_ckpt, device)
    print(f"[model] classes={classes} img_size={img_size}")

    # class index -> object_type_id (uint8)
    class_idx_to_obj_id = np.array(
        [OBJECT_TYPE_ID[c] for c in classes],
        dtype=np.uint8,
    )
    print(f"[class_idx_to_obj_id] {dict(zip(classes, class_idx_to_obj_id.tolist()))}")

    # collect wafers
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    specs, skipped = collect_wafer_specs(args.limit_per_class)
    print(f"[scan] wafers={len(specs)} skipped={skipped}")

    # filter: skip already-built (unless --overwrite)
    if not args.overwrite:
        before = len(specs)
        specs = [s for s in specs if not Path(s["out_path"]).exists()]
        print(f"[skip-existing] {before - len(specs)} already built; {len(specs)} remaining")
    if not specs:
        print("[done] nothing to do")
        return 0

    n_total_chips = sum(len(s["chips"]) for s in specs)
    print(f"[plan] wafers={len(specs)}  total_defect_chips={n_total_chips}  batch={args.batch}")

    ds = WaferChipDataset(specs, img_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # accumulate predictions per wafer
    obj_maps = {wi: np.zeros((GRID, GRID), dtype=np.uint8) for wi in range(len(specs))}

    t0 = time.time()
    use_amp = (device.type == "cuda")
    n_done = 0
    with torch.no_grad():
        for tens, wis, gxs, gys in loader:
            tens = tens.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(tens)
            else:
                logits = model(tens)
            preds = logits.argmax(dim=-1).cpu().numpy()
            wis_a = wis.numpy(); gxs_a = gxs.numpy(); gys_a = gys.numpy()
            for k in range(preds.shape[0]):
                obj_maps[int(wis_a[k])][int(gys_a[k]), int(gxs_a[k])] = class_idx_to_obj_id[preds[k]]
            n_done += preds.shape[0]
            if n_done % (args.batch * 50) < args.batch:
                rate = n_done / max(0.1, time.time() - t0)
                eta = (n_total_chips - n_done) / max(0.1, rate)
                print(f"  predicted {n_done}/{n_total_chips}  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    print(f"[predict] done in {(time.time() - t0):.1f}s")

    # save .npy per wafer
    counts_by_label = {n: 0 for n in OBJECT_TYPE_ID}
    inv_map = {v: k for k, v in OBJECT_TYPE_ID.items()}
    for wi, spec in enumerate(specs):
        out_path = Path(spec["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, obj_maps[wi])
        for v, c in zip(*np.unique(obj_maps[wi], return_counts=True)):
            counts_by_label[inv_map[int(v)]] += int(c)
    print(f"[saved] {len(specs)} npy files under {out_root}")
    print(f"[counts_by_label] {counts_by_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
