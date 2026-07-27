#!/usr/bin/env python3
"""E 트랙 1단계 — frozen DINOv3 stage4 feature map 캐싱 (백신-내성: 이미지 단위 저장 = 자연 재개).

train pool: 이미지당 augmentation 8뷰의 stage4 fmap (1024×12×12 fp16) → <out>/<stem>.npz
eval pool:  clean 1뷰 fmap → <out>/<stem>.npz
이미 존재하는 파일은 skip → 죽어도 이어서. 이후 _adaptor_train.py 가 소비.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import _ssl_methods as S  # aug / mask / TIMM / IMG 재사용


def main():
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import timm
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)            # 이미지 풀 dir
    ap.add_argument("--out", required=True)            # 캐시 출력 dir
    ap.add_argument("--views", type=int, default=8)    # aug 뷰 수 (0 = clean 1뷰)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cpu")
    bb = timm.create_model(S.TIMM, pretrained=True, num_classes=0, global_pool="avg").to(dev).eval()
    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    e = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in e)
    if args.limit:
        paths = paths[: args.limit]
    a_tf = S.aug()
    clean_tf = T.Compose([T.Resize((S.IMG, S.IMG)), T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    done = skip = 0
    t0 = time.time()
    with torch.no_grad():
        for p in paths:
            # 파일명 충돌 방지: 상위폴더_파일명
            stem = f"{p.parent.name}__{p.stem}"
            fp = out / f"{stem}.npz"
            if fp.exists():
                skip += 1; continue
            with Image.open(p) as im:
                img = S.mask_palette_non_grade_to_white(im).convert("RGB")
            views = ([a_tf(img) for _ in range(args.views)] if args.views > 0 else [clean_tf(img)])
            x = torch.stack(views).to(dev)
            fm = bb.forward_features(x)                  # [V, C, H, W] (convnext)
            if fm.shape[1] != bb.num_features and fm.shape[-1] == bb.num_features:
                fm = fm.permute(0, 3, 1, 2)
            tmp = out / f"{stem}.tmp.npz"  # 외부 kill 도중 잘린 npz 방지 — 원자적 교체
            np.savez_compressed(tmp, fmap=fm.cpu().numpy().astype(np.float16))
            os.replace(tmp, fp)
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  {done} cached (skip {skip}) — {el/done:.1f}s/img, ETA {(len(paths)-done-skip)*el/done/3600:.1f}h", flush=True)
    print(f"[done] cached {done}, skipped {skip} → {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
