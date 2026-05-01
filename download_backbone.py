#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backbone weights를 huggingface에서 1회 다운로드 후 models/ 에 mirror.

사용:
    python download_backbone.py
    # 또는 다른 backbone:
    python download_backbone.py --backbone convnextv2_large.fcmae_ft_in22k_in1k_384

이후 cnn_train.py 는 HF 접근 없이 models/<backbone>.pth 에서 load.
폐쇄망 이전 시 models/ 폴더만 같이 옮기면 끝.
"""
import argparse, sys
from pathlib import Path
import torch
import timm


DEFAULT_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 다운로드")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = args.backbone.replace("/", "__").replace(":", "_")
    wf = out_dir / f"{safe}.pth"

    if wf.exists() and not args.force:
        sz = wf.stat().st_size / 1e6
        print(f"[skip] 이미 존재: {wf} ({sz:.1f} MB). 강제 재다운로드는 --force.")
        return

    print(f"[1/3] timm.create_model('{args.backbone}', pretrained=True) ...", flush=True)
    m = timm.create_model(args.backbone, pretrained=True)
    print(f"[2/3] state_dict() 추출 ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
    torch.save(m.state_dict(), wf)
    sz = wf.stat().st_size / 1e6
    print(f"[3/3] mirror 저장: {wf}  ({sz:.1f} MB)")
    print(f"\n[OK] cnn_train.py는 이제 HF 접근 없이 이 파일에서 load.")
    print(f"     폐쇄망 이전 시 '{out_dir}/' 폴더만 같이 복사.")


if __name__ == "__main__":
    main()
