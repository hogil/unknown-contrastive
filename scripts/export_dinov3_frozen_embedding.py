#!/usr/bin/env python3
"""Export no-train DINOv3 frozen embeddings for an eval image folder."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from _ssl_methods import IMG, TIMM, list_imgs  # noqa: E402
from scripts._common import mask_palette_non_grade_to_white  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--palette-mode", choices=["grade_only", "grade_bg", "raw"], default="grade_only")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--timm", default=TIMM)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["UC_PALETTE_MODE"] = args.palette_mode
    import timm

    dev = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    paths = list_imgs(Path(args.eval_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[frozen] timm={args.timm} imgs={len(paths)} dev={dev} palette={args.palette_mode}", flush=True)

    tf = T.Compose([
        T.Resize((IMG, IMG)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    net = timm.create_model(args.timm, pretrained=True, num_classes=0, global_pool="avg").to(dev).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(paths), args.batch):
            xs = []
            for p in paths[i:i + args.batch]:
                with Image.open(p) as im:
                    img = mask_palette_non_grade_to_white(im).convert("RGB")
                xs.append(tf(img))
            f = net(torch.stack(xs).to(dev))
            if isinstance(f, tuple):
                f = f[0]
            chunks.append(F.normalize(f, dim=1).cpu().numpy())
            if (i // args.batch + 1) % 25 == 0:
                print(f"[frozen] {min(i + args.batch, len(paths))}/{len(paths)}", flush=True)
    np.save(out, np.concatenate(chunks, axis=0).astype(np.float32))
    print(f"[OUT] {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
