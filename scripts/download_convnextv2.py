"""Pre-download convnextv2 backbone weights via timm.

Useful when the target server has no internet access during training — run
this once on a machine with net, copy the resulting pth (or the HF cache
dir) over, and training starts without a network call.

Default model matches what contrastive_com.py / contrastive_rev_com.py use:
``convnextv2_base.fcmae_ft_in22k_in1k_384`` (FCMAE pretrained + IN22k/IN1k
fine-tuned at 384).

Usage::

    python scripts/download_convnextv2.py
    python scripts/download_convnextv2.py --model convnextv2_base.fcmae --out weights/convnextv2_base_fcmae.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import timm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="convnextv2_base.fcmae_ft_in22k_in1k_384",
        help="timm model name (default: convnextv2_base.fcmae_ft_in22k_in1k_384)",
    )
    ap.add_argument(
        "--out",
        default="weights/convnextv2_base_fcmae_ft_in22k_in1k_384.pth",
        help="Output pth path (created if parent missing).",
    )
    ap.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="0 drops the classification head (feature extractor only).",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] timm.create_model({args.model!r}, pretrained=True, num_classes={args.num_classes})")
    model = timm.create_model(args.model, pretrained=True, num_classes=args.num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[download] loaded OK, {n_params:,} params")

    torch.save(model.state_dict(), out)
    mb = out.stat().st_size / (1024 * 1024)
    print(f"[download] saved -> {out}  ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
