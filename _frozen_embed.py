#!/usr/bin/env python3
"""frozen(무학습) DINOv3 백본 임베딩 추출 — wm_base 등 학습 npy 와 apples-to-apples.
f = backbone avg-pool feature (pre-projection, L500 과 동일), F.normalize 후 저장.
_ssl_methods 의 TIMM/IMG/mask/list_imgs/tf 그대로 재사용. CPU 전용(학습 GPU 무방해).
순서 = S.list_imgs(eval_dir) → _score_umapfree.py --pool 이 같은 순서로 라벨 매칭.
"""
import sys, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ssl_methods as S


def main():
    import torch, timm
    import torchvision.transforms as T
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timm", default=None)
    ap.add_argument("--ckpt", default=None)  # supervised CNN ckpt (TAPT frozen 측정용)
    args = ap.parse_args()
    dev = torch.device("cpu")
    bb = timm.create_model(args.timm or S.TIMM, pretrained=not args.ckpt, num_classes=0, global_pool="avg").to(dev).eval()
    if args.ckpt:  # 옛 트레이너 L255 방식: state_dict/model unwrap + shape-호환 key 만
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        for k in ("state_dict", "model"):
            if isinstance(sd, dict) and k in sd: sd = sd[k]
        m_sd = bb.state_dict()
        compat = {k: v for k, v in sd.items() if k in m_sd and m_sd[k].shape == v.shape}
        bb.load_state_dict(compat, strict=False)
        print(f"[backbone] loaded {len(compat)}/{len(m_sd)} keys from {args.ckpt}", flush=True)
    tf = T.Compose([T.Resize((S.IMG, S.IMG)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    epaths = S.list_imgs(Path(args.eval_dir))
    out = []
    with torch.no_grad():
        for i in range(0, len(epaths), 16):
            xs = [tf(S.mask_palette_non_grade_to_white(Image.open(p)).convert("RGB")) for p in epaths[i:i + 16]]
            f = bb(torch.stack(xs).to(dev))
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    arr = np.concatenate(out, 0).astype(np.float32)
    np.save(args.out, arr)
    print(f"[OUT] {args.out}  shape={arr.shape}  n={len(epaths)}", flush=True)


if __name__ == "__main__":
    main()
