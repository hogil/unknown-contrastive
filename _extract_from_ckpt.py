#!/usr/bin/env python3
"""완주한 ckpt 에서 eval 임베딩만 추출 (가드 kill 로 npy 미저장된 rung 복구용).
트레이너 L496-501 과 동일 recipe: f=backbone avg-pool, L2 norm. CPU 전용."""
import sys, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ssl_methods as S


def main():
    import torch
    import torchvision.transforms as T
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="simclr")
    args = ap.parse_args()
    dev = torch.device("cpu")
    online = S.Net(args.method, K=4096, timm_id=None, head="mlp").to(dev).eval()
    st = torch.load(args.ckpt, map_location=dev, weights_only=False)
    online.load_state_dict(st["model"])
    print(f"[ckpt] gstep={st['gstep']} loaded", flush=True)
    tf = T.Compose([T.Resize((S.IMG, S.IMG)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    epaths = S.list_imgs(Path(args.eval_dir))
    out = []
    with torch.no_grad():
        for i in range(0, len(epaths), 16):
            xs = [tf(S.mask_palette_non_grade_to_white(Image.open(p)).convert("RGB")) for p in epaths[i:i + 16]]
            f, _ = online(torch.stack(xs).to(dev))
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    arr = np.concatenate(out, 0).astype(np.float32)
    np.save(args.out, arr)
    print(f"[OUT] {args.out}  shape={arr.shape}", flush=True)


if __name__ == "__main__":
    main()
