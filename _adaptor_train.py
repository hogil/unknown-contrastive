#!/usr/bin/env python3
"""E 트랙 2단계 — 캐시된 frozen fmap 위 zero-init residual adaptor 학습 (epoch 초 단위, 백신-내성).

구조 (외부답변3 + 사용자 head-warmup + AD small-head 합류 스펙):
  cached fmap [1024,12,12] (+ radial coord 채널)
  → zero-init residual adapter: x + γ·(LN→1×1 C/8→GELU→dw3×3→GELU→1×1 C), γ=0 시작
  → attentive pooling (learnable query 1개) → h' (1024)
  → linear proj → z (256) → InfoNCE + queue
γ=0 이라 시작점 = frozen GAP baseline 근방 — frozen 하한 보장 구조.
평가: mixed29 클린 캐시에 adapter+pooling 적용 → h' npy 저장 (표준 eval 소비).
"""
from __future__ import annotations
import argparse, glob, random, sys, time
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REPO = Path(__file__).resolve().parent


def radial_map(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return (r / r.max()).astype(np.float32)  # [H,W] 0(중심)~1(모서리)


def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", default="cache_fmap/train_v1_aug8")
    ap.add_argument("--eval-cache", default="cache_fmap/mixed29_clean")
    ap.add_argument("--out-dir", default="result_grouping/_field_mixed29/embeddings")
    ap.add_argument("--tag", default="adaptor_v1")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)      # 캐시 위라 큰 batch 공짜
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--queue-size", type=int, default=4096)
    ap.add_argument("--reduction", type=int, default=8)
    ap.add_argument("--pdim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cpu")

    tfiles = sorted(glob.glob(str(REPO / args.train_cache / "*.npz")))
    efiles = sorted(glob.glob(str(REPO / args.eval_cache / "*.npz")))
    assert tfiles and efiles, "캐시 없음 — _cache_features.py 먼저"
    print(f"[adaptor] train {len(tfiles)} files, eval {len(efiles)} files", flush=True)

    # 캐시 전부 RAM 로드 (fp16): train 5149×8×295KB ≈ 12GB → fp16 그대로 들고 다님
    def load(f):
        return np.load(f)["fmap"]  # [V,C,H,W] fp16
    sample = load(tfiles[0])
    V, C, H, W = sample.shape
    rad = torch.from_numpy(radial_map(H, W))[None, None]  # [1,1,H,W]

    class Adapter(nn.Module):
        def __init__(self, c, red, pdim):
            super().__init__()
            hid = c // red
            self.ln = nn.GroupNorm(1, c + 1)  # LN2d 동치 (radial 채널 포함 c+1)
            self.inp = nn.Conv2d(c + 1, c, 1)  # radial 흡수
            self.down = nn.Conv2d(c, hid, 1)
            self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
            self.up = nn.Conv2d(hid, c, 1)
            self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))  # zero-init residual
            self.q = nn.Parameter(torch.randn(1, c) * 0.02)     # attentive pooling query
            self.proj = nn.Linear(c, pdim)
        def fmap_out(self, x):  # x: [B,C,H,W] (radial 미포함)
            xr = torch.cat([x, rad.expand(x.size(0), -1, -1, -1)], 1)
            y = self.inp(self.ln(xr))
            y = self.up(F.gelu(self.dw(F.gelu(self.down(y)))))
            return x + self.gamma * y
        def pool(self, fm):  # attentive pooling → h' [B,C]
            B, C2, H2, W2 = fm.shape
            t = fm.flatten(2).transpose(1, 2)               # [B,N,C]
            att = (t @ self.q.t()).squeeze(-1) / (C2 ** 0.5)  # [B,N]
            w = att.softmax(1).unsqueeze(-1)
            return (t * w).sum(1)                            # [B,C]
        def forward(self, x):
            h = self.pool(self.fmap_out(x))
            return h, F.normalize(self.proj(h), dim=1)

    net = Adapter(C, args.reduction, args.pdim).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-6)
    queue = F.normalize(torch.randn(args.queue_size, args.pdim), dim=1); qptr = 0

    outdir = REPO / args.out_dir; outdir.mkdir(parents=True, exist_ok=True)
    idxs = list(range(len(tfiles)))
    spe = max(1, len(tfiles) // args.batch)
    for ep in range(1, args.epochs + 1):
        random.shuffle(idxs); t0 = time.time(); run = 0.0; n = 0
        for bi in range(spe):
            batch = idxs[bi * args.batch:(bi + 1) * args.batch]
            if len(batch) < 2: continue
            v1l, v2l = [], []
            for i in batch:
                fm = load(tfiles[i])
                a, b = random.sample(range(fm.shape[0]), 2)
                v1l.append(torch.from_numpy(fm[a].astype(np.float32)))
                v2l.append(torch.from_numpy(fm[b].astype(np.float32)))
            x1 = torch.stack(v1l); x2 = torch.stack(v2l)
            _, z1 = net(x1); _, z2 = net(x2)
            zc = torch.cat([z1, z2]); B = z1.size(0)
            sim = zc @ zc.t() / args.temp
            sim.fill_diagonal_(-1e9)
            pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])
            qsnap = queue.detach().clone()
            logits = torch.cat([sim, zc @ qsnap.t() / args.temp], 1)
            loss = F.cross_entropy(logits, pos)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                k = F.normalize(z2, dim=1); end = qptr + k.size(0)
                if end <= queue.size(0): queue[qptr:end] = k
                else:
                    queue[qptr:] = k[:queue.size(0) - qptr]; queue[:end - queue.size(0)] = k[queue.size(0) - qptr:]
                qptr = end % queue.size(0)
            run += float(loss); n += 1
        # 평가 임베딩 (mixed29 클린 캐시 → adapter h')
        net.eval()
        with torch.no_grad():
            outs = []
            for f in efiles:
                fm = torch.from_numpy(load(f)[0].astype(np.float32))[None]
                h, _ = net(fm)
                outs.append(F.normalize(h, dim=1).numpy())
            np.save(outdir / f"{args.tag}_ep{ep}.npy", np.concatenate(outs, 0).astype(np.float32))
        net.train()
        print(f"[{args.tag} ep{ep}] loss={run/max(1,n):.4f} gamma|max|={float(net.gamma.abs().max()):.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[OUT] {outdir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
