#!/usr/bin/env python3
"""다양한 SSL 기법 비교 — DINOv3(convnext_base) all-unfreeze 위에서 method만 교체.

method: moco simclr dcl hardneg alignunif byol simsiam barlow vicreg dino
같은 데이터(cnn_seen_train, novel 제외)·같은 aug·같은 eval(novel k-means k=3). 매 epoch novel 임베딩 저장.
backbone h 로 평가(일관). 목적: 어떤 SSL 기법이 wafer NCD에 best인지 ablation.
"""
from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
from _common import mask_palette_non_grade_to_white  # noqa

TRAIN_DIR = REPO / "data/images/wm811k_novel_disjoint_v1/cnn_seen_train"
EVAL_DIR = REPO / "data/images/wm811k_novel_disjoint_v1/novel_eval"
TIMM = "convnext_base.dinov3_lvd1689m"
IMG = 384
MOMENTUM_METHODS = {"moco", "byol", "dino"}


def list_imgs(d):
    e = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in e)


class GaussNoise:  # 260707: lambda → picklable 클래스 (num_workers>0 Windows spawn 필수). 동작 동일 (σ=0.02 가우시안).
    def __init__(self, sigma=0.02): self.sigma = sigma
    def __call__(self, x): return x + self.sigma * torch.randn_like(x)


def aug(natural=False):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if natural:  # 자연이미지(DTD/Flowers 등) 표준 SimCLR aug — wafer aug 는 positive 가 거의 동일해 무용
        return T.Compose([T.RandomResizedCrop((IMG, IMG), scale=(0.2, 1.0)),
                          T.RandomHorizontalFlip(),
                          T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                          T.RandomGrayscale(p=0.2),
                          T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
                          T.ToTensor(), norm])
    return T.Compose([T.RandomResizedCrop((IMG, IMG), scale=(0.94, 1.0), ratio=(1.0, 1.0)),
                      T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                      T.ToTensor(), GaussNoise(0.02), norm])


class TwoView(torch.utils.data.Dataset):
    def __init__(self, paths, a): self.paths, self.a = paths, a
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = mask_palette_non_grade_to_white(im).convert("RGB")
        return self.a(img), self.a(img)


class OneView(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = mask_palette_non_grade_to_white(im).convert("RGB")
        return self.transform(img)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self, method, pdim=256, K=4096, timm_id=None, head="mlp"):
        super().__init__()
        import timm
        self.method = method
        self.bb = timm.create_model(timm_id or TIMM, pretrained=True, num_classes=0, global_pool="avg")
        fd = self.bb.num_features
        # head 변형 — 완숙 백본 + 소데이터에선 큰 랜덤 head 가 오염원 (AD 문헌 + anomaly-detection repo 실증)
        if head == "linear":   # SimpleNet adaptor 식 최소
            self.proj = nn.Linear(fd, pdim)
        elif head == "ad":     # D:/project/anomaly-detection train.py 의 backbone-winner head
            self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(fd, 512), nn.ReLU(inplace=True), nn.Linear(512, pdim))
        elif head == "adapter":  # ★ zero-init residual adapter (γ=0 → frozen 하한 보장) — h 자체를 다듬어 eval 에 반영
            self.ad = nn.Sequential(nn.Linear(fd, fd // 8), nn.GELU(), nn.Linear(fd // 8, fd))
            self.ad_gamma = nn.Parameter(torch.zeros(1))
            self.proj = mlp(fd, fd, pdim)
        elif head.startswith("adapterN"):  # ★ multi-stage adaptive: N 단 zero-init residual stage 누적 (각 γ=0)
            n_stage = int(head[8:] or "3")
            self.ad_stages = nn.ModuleList(
                nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd, fd // 8), nn.GELU(), nn.Linear(fd // 8, fd))
                for _ in range(n_stage))
            self.ad_gammas = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(n_stage))
            self.adapter_smax = 0.1  # AdaptFormer scale 상한 (s<1 필수)
            self.proj = mlp(fd, fd, pdim)
        else:                  # 기존 SimCLR 표준 (BN 2층)
            self.proj = mlp(fd, fd, pdim)
        self.pred = mlp(pdim, pdim, pdim) if method in ("byol", "simsiam") else None
        self.proto = nn.utils.weight_norm(nn.Linear(pdim, K, bias=False)) if method == "dino" else None
        if self.proto is not None:
            self.proto.weight_g.data.fill_(1); self.proto.weight_g.requires_grad = False

    def forward(self, x):
        f = self.bb(x)
        if getattr(self, "ad_gamma", None) is not None:
            f = f + self.ad_gamma * self.ad(f)   # residual — γ=0 시작이라 초기엔 frozen 그대로
        if getattr(self, "ad_stages", None) is not None:
            for stage, g in zip(self.ad_stages, self.ad_gammas):
                # ★ AdaptFormer: scale s<1 필수 (s>1 불안정) — γ 를 tanh 로 |s|<adapter_smax 로 bound (ep↑ 악화 처방)
                s = torch.tanh(g) * getattr(self, "adapter_smax", 0.1)
                f = f + s * stage(f)             # multi-stage residual — 각 단 bounded scale, γ=0 시작
        z = self.proj(f)
        return f, z

    def forward_patches(self, x):
        fm = self.bb.forward_features(x)
        if fm.ndim == 2:
            fm = fm[:, :, None, None]
        if fm.shape[1] != self.bb.num_features and fm.shape[-1] == self.bb.num_features:
            fm = fm.permute(0, 3, 1, 2)
        B, C, H, W = fm.shape
        p = self.proj(fm.permute(0, 2, 3, 1).reshape(B * H * W, C))
        return p.reshape(B, H * W, -1)


@torch.no_grad()
def ema(o, t, m):
    for po, pt in zip(o.parameters(), t.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)


def off_diag(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def ce_ls(logits, pos_idx, ls, topk=0, ls_uni=0.0):
    """masked-aware CE + label smoothing (옛 검증 L361 방식) — 마스킹(-1e9)된 negative 엔 smoothing 안 줌.
    topk > 0 이면 SoftNCE (arXiv:2212.07158): 균등 분배 대신 top-K hardest negative 에만 ls 분배.
    ls_uni > 0 (topk 와 병용) 이면 hybrid: 균등 ls_uni + top-K ls — positive 는 1-ls-ls_uni."""
    import torch.nn.functional as F
    import torch
    if ls <= 0 and ls_uni <= 0:
        return F.cross_entropy(logits, pos_idx)
    logp = F.log_softmax(logits, dim=1)
    valid = logits > -1e8
    total = ls + (ls_uni if topk > 0 else 0.0)
    if topk > 0:
        neg = logits.masked_fill(~valid, -1e9)
        neg = neg.scatter(1, pos_idx.unsqueeze(1), -1e9)   # positive 제외 negative 만
        idx = neg.topk(topk, dim=1).indices                # top-K hardest
        tgt = torch.zeros_like(logits)
        if ls_uni > 0:  # hybrid: 균등 바닥 + top-K 가산
            n_valid = valid.sum(1, keepdim=True).clamp_min(2).float()
            tgt = valid.float() * (ls_uni / (n_valid - 1))
        tgt.scatter_add_(1, idx, torch.full_like(idx, ls / topk, dtype=logits.dtype))
    else:
        n_valid = valid.sum(1, keepdim=True).clamp_min(2).float()
        tgt = valid.float() * (ls / (n_valid - 1))
        total = ls
    tgt.scatter_(1, pos_idx.unsqueeze(1), 1.0 - total)
    return -(tgt * logp).sum(1).mean()


def koleo(z):
    # DINOv2 KoLeo: 각 점을 최근접 이웃에서 밀어 균일 분산. cdist 대신 dot-product
    # (cdist backward 는 출력 거리행렬을 저장 → inplace 불가, sqrt(0) NaN grad 위험).
    z = F.normalize(z, dim=1)
    sim = z @ z.t()
    sim = sim - torch.eye(z.size(0), device=z.device, dtype=z.dtype) * 2.0  # self 제거(out-of-place)
    nn_sim = sim.max(1).values                       # 최근접 이웃 cos 유사도
    d2 = (2.0 - 2.0 * nn_sim).clamp_min(1e-8)         # NN 까지 squared 거리
    return -0.5 * d2.log().mean()


def local_grid_loss(p1, p2, t):
    # DenseCL식: 각 patch positive = 다른 view에서 feature 최근접 patch
    p1 = F.normalize(p1, dim=2); p2 = F.normalize(p2, dim=2)
    B, N, D = p1.shape
    sim = torch.bmm(p1, p2.transpose(1, 2))
    pos = sim.argmax(2)
    return F.cross_entropy((sim / t).reshape(B * N, N), pos.reshape(-1))


def neco_loss(p1, p2, tau=0.1):
    # ★ 옛 검증 원본 (scripts/train_contrastive.py L379) — patch-neighbor "질서" 일관성:
    # 두 뷰의 patch-patch 유사도 분포(row-softmax, 대각 마스킹)를 대칭 KL 로 일치.
    # (이전 MSE gram 판은 부정확한 이식이었음 — 260612 교체)
    p1 = F.normalize(p1, dim=2); p2 = F.normalize(p2, dim=2)
    n = p1.size(1)
    s1 = torch.bmm(p1, p1.transpose(1, 2)); s2 = torch.bmm(p2, p2.transpose(1, 2))
    eye = torch.eye(n, device=p1.device, dtype=torch.bool).unsqueeze(0)
    s1 = s1.masked_fill(eye, -1e4); s2 = s2.masked_fill(eye, -1e4)
    lp1 = F.log_softmax(s1 / tau, dim=-1); lp2 = F.log_softmax(s2 / tau, dim=-1)
    kl12 = (lp1.exp() * (lp1 - lp2)).sum(-1).mean()
    kl21 = (lp2.exp() * (lp2 - lp1)).sum(-1).mean()
    return 0.5 * (kl12 + kl21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-bb", type=float, default=2e-6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--m", type=float, default=0.99)
    ap.add_argument("--koleo", type=float, default=0.0)
    ap.add_argument("--vicreg-aux", type=float, default=0.0)  # VICReg var+cov 항을 SimCLR 위 AUX 로 (dim-collapse 방지, KoLeo 와 직교/가산). 0=off
    ap.add_argument("--use-queue", action="store_true")
    ap.add_argument("--queue-size", type=int, default=1024)
    ap.add_argument("--ignore", type=float, default=1.01)   # false-neg ignore: cos>thr 인 negative 마스킹 (<1 활성)
    ap.add_argument("--local", type=float, default=0.0)     # DenseCL local-grid loss weight
    ap.add_argument("--neco", type=float, default=0.0)      # NeCo patch-neighbor consistency weight
    ap.add_argument("--cpu", action="store_true")           # 강제 CPU (공유 GPU 회피, 사용자 지시)
    ap.add_argument("--timm", default=None)                  # 백본 init override (백본 2개 비교용)
    ap.add_argument("--dino-tricks", action="store_true")    # DINO 정상화: teacher-proto(EMA)/temp warmup/momentum 스케줄/freeze-proto 1ep
    ap.add_argument("--head", default="mlp")  # mlp|linear|ad|adapter|adapterN<k> (k단 multi-stage adaptive)
    ap.add_argument("--train-dir", default=None)             # 학습 데이터 dir override (cross-dataset 트랙)
    ap.add_argument("--eval-dir", default=None)              # 평가(임베딩 대상) dir override
    ap.add_argument("--out-dir", default=None)               # 임베딩 저장 dir override
    ap.add_argument("--ckpt-every", type=int, default=1000)  # N step 마다 체크포인트 (백신 kill 대비)
    ap.add_argument("--fresh", action="store_true")          # 기존 ckpt 무시하고 처음부터
    ap.add_argument("--nv-filter", type=float, default=0.0)  # NV-Retriever: cos > nv×자기pos_sim 인 neg 마스킹 (분포 자동보정, 예 0.95)
    ap.add_argument("--ls", type=float, default=0.0)         # InfoNCE label smoothing (옛 검증값 0.02)
    ap.add_argument("--ls-topk", type=int, default=0)       # SoftNCE: ls 를 top-K hardest neg 에만 분배 (0=균등)
    ap.add_argument("--ls-uni", type=float, default=0.0)     # hybrid smoothing: 균등 바닥 질량 (topk 와 병용)
    ap.add_argument("--nn-pos", type=float, default=0.0)     # NNCLR: 반대뷰 queue-NN positive loss weight
    ap.add_argument("--barlow-lambda", type=float, default=5e-3)  # Barlow off-diagonal 가중치 (decorrelation 강도)
    ap.add_argument("--dual-temp", type=float, default=0.0)  # dual-temperature τ_β (0=off, 권장 1.0)
    ap.add_argument("--dino-k", type=int, default=4096)      # dino prototype 수 (v2: 128 — 기준점 < 데이터 수 모순 해소)
    ap.add_argument("--dino-kmeans-init", action="store_true")  # v2: 초기 z 공간 k-means 중심으로 proto init (라벨 0)
    ap.add_argument("--dino-head-warmup", type=int, default=0)  # v2: 첫 N ep 동안 backbone lr=0 (head 만 학습)
    ap.add_argument("--sce", type=float, default=0.0)        # SCE λ: target = λ·onehot + (1-λ)·반대뷰 유사도 분포 (queue 필수)
    ap.add_argument("--sce-tau", type=float, default=0.07)   # SCE target sharpening temp
    ap.add_argument("--grad-clip", type=float, default=0.0)  # gradient clipping (옛 검증값 1.0)
    ap.add_argument("--palette-mode", choices=["grade_only", "grade_bg", "raw"], default=None,
                    help="palette PNG masking: grade_only keeps 0-7, grade_bg keeps 0-8 background, raw keeps all")
    ap.add_argument("--natural-aug", action="store_true")   # 자연이미지(DTD/Flowers) 표준 SimCLR aug (crop 0.2-1.0 + colorjitter/grayscale/blur/hflip)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)          # multi-seed robustness (mean±std)
    args = ap.parse_args()
    if args.palette_mode:
        os.environ["UC_PALETTE_MODE"] = args.palette_mode
    torch.manual_seed(args.seed)
    dev = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    meth = args.method
    train_dir = Path(args.train_dir) if args.train_dir else TRAIN_DIR
    eval_dir = Path(args.eval_dir) if args.eval_dir else EVAL_DIR
    paths = list_imgs(train_dir)
    print(f"[ssl] method={meth} {len(paths)} imgs dev={dev} train={train_dir.name} eval={eval_dir.name} palette={os.environ.get('UC_PALETTE_MODE', 'grade_only')}", flush=True)
    dl = torch.utils.data.DataLoader(TwoView(paths, aug(natural=args.natural_aug)), batch_size=args.batch, shuffle=True,
                                     num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True)  # ★ 260707: 0→4 (single-thread 로더가 GPU 0↔100% 진동 병목)
    online = Net(meth, K=args.dino_k, timm_id=args.timm, head=args.head).to(dev).train()
    target = None
    if meth in MOMENTUM_METHODS:
        target = Net(meth, K=args.dino_k, timm_id=args.timm, head=args.head).to(dev).eval()
        target.load_state_dict(online.state_dict())
        for p in target.parameters(): p.requires_grad = False
    params = [p for p in online.parameters() if p.requires_grad]
    bb_ids = {id(p) for n, p in online.bb.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) in bb_ids], "lr": args.lr_bb},
                             {"params": [p for p in params if id(p) not in bb_ids], "lr": args.lr_head}], weight_decay=1e-6)
    pdim = 256
    queue = F.normalize(torch.randn(args.queue_size, pdim, device=dev), dim=1); qptr = 0
    center = torch.zeros(1, args.dino_k, device=dev)  # dino centering (K 파생 — hardcode 금지)
    outdir = Path(args.out_dir) if args.out_dir else REPO / "result_grouping/_dinov3_ncd_autoloop/ssl_embeddings"
    outdir.mkdir(parents=True, exist_ok=True)
    tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    epaths = list_imgs(eval_dir)
    edl = torch.utils.data.DataLoader(
        OneView(epaths, tf), batch_size=16, shuffle=False, num_workers=4,
        persistent_workers=True, pin_memory=True)

    steps_per_ep = max(1, len(dl)); total_steps = args.epochs * steps_per_ep
    gstep = 0  # dino-tricks 스케줄 + 체크포인트 재개용 global step

    # ── 체크포인트 재개 (백신 등 외부 kill 대비) ──
    ckpt_path = outdir / f"{(args.tag or meth)}_ckpt.pt"
    if ckpt_path.exists() and not args.fresh:
        st = torch.load(ckpt_path, map_location=dev, weights_only=False)
        online.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        queue = st["queue"].to(dev); qptr = int(st["qptr"]); gstep = int(st["gstep"])
        center = st["center"].to(dev)
        if target is not None and st.get("target"):
            target.load_state_dict(st["target"])
        print(f"[resume] ckpt 로드 — gstep {gstep}/{total_steps} 부터 재개", flush=True)

    def save_ckpt():
        torch.save({"model": online.state_dict(), "opt": opt.state_dict(),
                    "queue": queue.detach().cpu(), "qptr": qptr, "gstep": gstep,
                    "center": center.detach().cpu(),
                    "target": target.state_dict() if target is not None else None},
                   ckpt_path)

    def loss_fn(f1, z1, f2, z2, tz1=None, tz2=None):
        nonlocal qptr, center
        t = args.temp
        if meth == "simclr":
            zc = F.normalize(torch.cat([z1, z2]), dim=1); B = z1.size(0)
            sim = zc @ zc.t() / t
            sim.fill_diagonal_(-1e9)
            pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(zc.device)
            if args.ignore < 1.0:  # (구) 절대값 false-neg ignore — 분포 의존이라 nv-filter 권장
                fn = (sim * t > args.ignore)
                fn[torch.arange(2 * B, device=zc.device), pos] = False
                sim = sim.masked_fill(fn, -1e9)
            nv_thr = None
            if args.nv_filter > 0:  # ★ NV-Retriever: per-anchor 상대 임계 (자기 pos_sim 의 nv 배)
                with torch.no_grad():
                    nv_thr = args.nv_filter * (zc * zc[pos]).sum(1, keepdim=True)  # [2B,1]
                fn = (sim * t) > nv_thr
                fn[torch.arange(2 * B, device=zc.device), pos] = False
                sim = sim.masked_fill(fn, -1e9)
            if args.use_queue:
                qsnap = queue.detach().clone()  # clone: 이후 inplace 업데이트가 graph 안 건드림
                qneg = zc @ qsnap.t() / t
                if args.ignore < 1.0:
                    qneg = qneg.masked_fill(qneg * t > args.ignore, -1e9)
                if nv_thr is not None:
                    qneg = qneg.masked_fill((qneg * t) > nv_thr, -1e9)
                logits = torch.cat([sim, qneg], 1)
                with torch.no_grad():
                    k = F.normalize(z2, dim=1); end = qptr + k.size(0)
                    if end <= queue.size(0): queue[qptr:end] = k
                    else:
                        queue[qptr:] = k[:queue.size(0) - qptr]; queue[:end - queue.size(0)] = k[queue.size(0) - qptr:]
                    qptr = end % queue.size(0)
                if args.nn_pos > 0:  # ★ NNCLR (arXiv:2104.14548): 반대뷰의 queue 최근접을 positive 로 — 같은 클래스 조각 직접 당김 (W2)
                    with torch.no_grad():
                        z1n = F.normalize(z1, dim=1); z2n = F.normalize(z2, dim=1)
                        nn1 = (z1n @ qsnap.t()).argmax(1)
                        nn2 = (z2n @ qsnap.t()).argmax(1)
                    l_nn = 0.5 * (F.cross_entropy(F.normalize(z2, dim=1) @ qsnap.t() / t, nn1) +
                                  F.cross_entropy(F.normalize(z1, dim=1) @ qsnap.t() / t, nn2))
                    base = ce_ls(logits, pos, args.ls, args.ls_topk, args.ls_uni)
                    # queue 갱신은 아래 공통 블록에서 이미 수행됨
                    return base + args.nn_pos * l_nn
                if args.sce > 0:  # ★ SCE (arXiv:2111.14585): target = λ·onehot + (1-λ)·반대뷰 유사도 분포 (sharpened)
                    with torch.no_grad():
                        s_rel = torch.cat([zc[pos] @ zc.t(), zc[pos] @ qsnap.t()], 1) / args.sce_tau
                        s_rel[:, :2 * B].scatter_(1, torch.stack([pos, torch.arange(2 * B, device=zc.device)], 1), -1e9)
                        rel = F.softmax(s_rel, dim=1)
                        tgt = (1 - args.sce) * rel
                        tgt.scatter_(1, pos.unsqueeze(1), args.sce)
                    return -(tgt * F.log_softmax(logits, dim=1)).sum(1).mean()
                if args.dual_temp > 0:  # ★ dual-temperature (arXiv:2203.17248): few-neg(queue4k) 에서 per-anchor hardness 가중
                    with torch.no_grad():
                        raw = logits * t                                  # /t 복원 → 원 유사도
                        pa = F.softmax(raw / t, 1).gather(1, pos.unsqueeze(1)).squeeze(1)
                        pb = F.softmax(raw / args.dual_temp, 1).gather(1, pos.unsqueeze(1)).squeeze(1)
                        w = ((1 - pb) / (1 - pa).clamp_min(1e-6)).clamp(0.2, 5.0)  # sg(W_β/W_α)
                    return (w * F.cross_entropy(logits, pos, reduction="none")).mean()
                return ce_ls(logits, pos, args.ls, args.ls_topk, args.ls_uni)
            return ce_ls(sim, pos, args.ls, args.ls_topk, args.ls_uni)
        if meth in ("moco", "dcl", "hardneg", "alignunif"):
            q = F.normalize(z1, dim=1); k = F.normalize(tz2 if tz2 is not None else z2, dim=1).detach()
            queue_snapshot = queue.detach().clone()
            if meth == "alignunif":
                align = (q - k).pow(2).sum(1).mean()
                unif = torch.pdist(q, p=2).pow(2).mul(-2).exp().mean().log()
                return align + unif
            pos = (q * k).sum(1, keepdim=True) / t
            neg = q @ queue_snapshot.t() / t
            if meth == "hardneg":
                with torch.no_grad():
                    w = (q @ queue_snapshot.t() / t).softmax(1) * queue_snapshot.size(0)  # importance
                neg = neg + w.log().clamp(-5, 5)
            with torch.no_grad():
                qptr_local = qptr
                end = qptr_local + k.size(0)
                if end <= queue.size(0): queue[qptr_local:end] = k
                else:
                    queue[qptr_local:] = k[:queue.size(0) - qptr_local]; queue[:end - queue.size(0)] = k[queue.size(0) - qptr_local:]
                qptr = end % queue.size(0)
            if meth == "dcl":
                return (-pos.squeeze(1) + torch.logsumexp(neg, 1)).mean()
            logits = torch.cat([pos, neg], 1)
            return F.cross_entropy(logits, torch.zeros(q.size(0), dtype=torch.long, device=q.device))
        if meth == "byol":
            p1 = F.normalize(online.pred(z1), dim=1); p2 = F.normalize(online.pred(z2), dim=1)
            t1 = F.normalize(tz1, dim=1).detach(); t2 = F.normalize(tz2, dim=1).detach()
            return (2 - 2 * (p1 * t2).sum(1).mean()) / 2 + (2 - 2 * (p2 * t1).sum(1).mean()) / 2
        if meth == "simsiam":
            p1 = F.normalize(online.pred(z1), dim=1); p2 = F.normalize(online.pred(z2), dim=1)
            d1 = F.normalize(z1, dim=1).detach(); d2 = F.normalize(z2, dim=1).detach()
            return -((p1 * d2).sum(1).mean() + (p2 * d1).sum(1).mean()) / 2
        if meth == "barlow":
            zn1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5); zn2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
            c = zn1.t() @ zn2 / z1.size(0)
            on = (torch.diagonal(c) - 1).pow(2).sum(); off = off_diag(c).pow(2).sum()
            return on + args.barlow_lambda * off
        if meth == "vicreg":
            inv = F.mse_loss(z1, z2)
            std1 = torch.sqrt(z1.var(0) + 1e-4); std2 = torch.sqrt(z2.var(0) + 1e-4)
            var = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean())
            z1c = z1 - z1.mean(0); z2c = z2 - z2.mean(0)
            cov1 = (z1c.t() @ z1c) / (z1.size(0) - 1); cov2 = (z2c.t() @ z2c) / (z2.size(0) - 1)
            cov = off_diag(cov1).pow(2).sum() / z1.size(1) + off_diag(cov2).pow(2).sum() / z1.size(1)
            return 25 * inv + 25 * var + 1 * cov
        if meth == "dino":
            # tricks: teacher 분포는 target(EMA)의 proto 로 (기존 버그: online.proto 사용),
            #         teacher temp 0.02→0.04 warmup (3 epoch 분량)
            proto_t = target.proto if (args.dino_tricks and target is not None) else online.proto
            tt = (0.02 + 0.02 * min(1.0, gstep / max(1, 3 * steps_per_ep))) if args.dino_tricks else 0.04
            s1 = online.proto(F.normalize(z1, dim=1)); s2 = online.proto(F.normalize(z2, dim=1))
            with torch.no_grad():
                pt1 = proto_t(F.normalize(tz1, dim=1)); pt2 = proto_t(F.normalize(tz2, dim=1))
                t1 = ((pt1 - center) / tt).softmax(1)
                t2 = ((pt2 - center) / tt).softmax(1)
                center = 0.9 * center + 0.1 * torch.cat([pt1, pt2]).mean(0, keepdim=True)
            ls1 = (s1 / 0.1).log_softmax(1); ls2 = (s2 / 0.1).log_softmax(1)
            return (-(t2 * ls1).sum(1).mean() - (t1 * ls2).sum(1).mean()) / 2
        raise ValueError(meth)

    if meth == "dino" and args.dino_kmeans_init and gstep == 0 and online.proto is not None:
        # v2: frozen 초기 z 공간의 k-means 중심으로 prototype 방향 init (DeepCluster/SwAV 계보, 라벨 0)
        import numpy as _np
        rs = _np.random.RandomState(42)
        sample = paths if len(paths) <= 1024 else [paths[i] for i in rs.choice(len(paths), 1024, replace=False)]
        zs = []
        online.eval()
        with torch.no_grad():
            for i in range(0, len(sample), 16):
                xs = torch.stack([tf(mask_palette_non_grade_to_white(Image.open(pp)).convert("RGB")) for pp in sample[i:i + 16]]).to(dev)
                _, zz = online(xs)
                zs.append(F.normalize(zz, dim=1).cpu())
        online.train()
        Z = torch.cat(zs).numpy()
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=args.dino_k, n_init=3, random_state=42).fit(Z)
        cen = torch.tensor(km.cluster_centers_, dtype=online.proto.weight_v.dtype)
        online.proto.weight_v.data.copy_(F.normalize(cen, dim=1))
        if target is not None and target.proto is not None:
            target.proto.weight_v.data.copy_(online.proto.weight_v.data)
        print(f"[dino-v2] kmeans proto init K={args.dino_k} (sample {len(sample)})", flush=True)

    def extract_embeddings(ep):
        online.eval()
        with torch.no_grad():
            out, outz = [], []
            for xs in edl:
                f, zz = online(xs.to(dev, non_blocking=True))
                out.append(F.normalize(f, dim=1).cpu().numpy())
                outz.append(F.normalize(zz, dim=1).cpu().numpy())
        stem = args.tag or meth
        np.save(outdir / f"{stem}_ep{ep}.npy", np.concatenate(out, 0).astype(np.float32))
        np.save(outdir / f"{stem}_ep{ep}_proj.npy", np.concatenate(outz, 0).astype(np.float32))
        online.train()

    start_ep = gstep // steps_per_ep + 1
    previous_ep = start_ep - 1
    previous_out = outdir / f"{(args.tag or meth)}_ep{previous_ep}.npy"
    if previous_ep >= 1 and not previous_out.exists():
        print(f"[resume] extracting missing ep{previous_ep} embeddings", flush=True)
        extract_embeddings(previous_ep)

    for ep in range(start_ep, args.epochs + 1):
        # 재개 시 위치 skip 안 함 — 재셔플로 즉시 계속 (SSL 은 순서 무관, skip 은 PIL 비용만 큼)
        t0 = time.time(); run = 0.0; n = 0
        if args.dino_head_warmup > 0:  # v2: LP-FT — warmup 동안 backbone lr=0
            opt.param_groups[0]["lr"] = 0.0 if ep <= args.dino_head_warmup else args.lr_bb
        ep_target = ep * steps_per_ep
        for x1, x2 in dl:
            if gstep >= ep_target:
                break  # 이번 epoch 분량 완료 (재개 누적 기준)
            x1, x2 = x1.to(dev), x2.to(dev)
            f1, z1 = online(x1); f2, z2 = online(x2)
            tz1 = tz2 = None
            if target is not None:
                with torch.no_grad():
                    _, tz1 = target(x1); _, tz2 = target(x2)
            loss = loss_fn(f1, z1, f2, z2, tz1, tz2)
            if args.koleo > 0:
                loss = loss + args.koleo * koleo(z1)
            if args.vicreg_aux > 0:  # VICReg var(std≥1 hinge)+cov(off-diag decorr) AUX — dim-collapse 방지 (invariance 는 InfoNCE 담당). z1,z2=raw proj
                std1 = torch.sqrt(z1.var(0) + 1e-4); std2 = torch.sqrt(z2.var(0) + 1e-4)
                var = F.relu(1 - std1).mean() + F.relu(1 - std2).mean()
                z1c = z1 - z1.mean(0); z2c = z2 - z2.mean(0)
                cov1 = (z1c.t() @ z1c) / (z1.size(0) - 1); cov2 = (z2c.t() @ z2c) / (z2.size(0) - 1)
                cov = off_diag(cov1).pow(2).sum() / z1.size(1) + off_diag(cov2).pow(2).sum() / z1.size(1)
                loss = loss + args.vicreg_aux * (var + cov)
            if args.local > 0 or args.neco > 0:
                p1 = online.forward_patches(x1); p2 = online.forward_patches(x2)
                if args.local > 0:
                    loss = loss + args.local * local_grid_loss(p1, p2, args.temp)
                if args.neco > 0:
                    loss = loss + args.neco * neco_loss(p1, p2)
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            if args.dino_tricks and ep == 1 and online.proto is not None:
                for p in online.proto.parameters():   # freeze-last-layer: 첫 epoch proto 동결
                    p.grad = None
            opt.step()
            if target is not None:
                m = args.m
                if args.dino_tricks:  # momentum 0.996→1.0 cosine 스케줄
                    m = 1 - (1 - 0.996) * (math.cos(math.pi * gstep / max(1, total_steps)) + 1) / 2
                ema(online, target, m)
            gstep += 1
            if gstep % args.ckpt_every == 0:
                save_ckpt()
            run += float(loss); n += 1
        print(f"[{meth} ep{ep}] loss={run/max(1,n):.4f} ({time.time()-t0:.0f}s)", flush=True)
        save_ckpt()
        online.eval()
        # 무라벨 모니터링 (D-11): alignment(두 뷰 h 거리) + uniformity — epoch 선택 기준 (260611 검증됨)
        with torch.no_grad():
            mon_paths = paths[:: max(1, len(paths)//128)][:128]
            a_tf = aug(natural=args.natural_aug)
            h1l, h2l = [], []
            for p in mon_paths:
                with Image.open(p) as im:
                    img = mask_palette_non_grade_to_white(im).convert("RGB")
                v1 = a_tf(img).unsqueeze(0).to(dev); v2 = a_tf(img).unsqueeze(0).to(dev)
                f1m, _ = online(v1); f2m, _ = online(v2)
                h1l.append(F.normalize(f1m, dim=1)); h2l.append(F.normalize(f2m, dim=1))
            H1 = torch.cat(h1l); H2 = torch.cat(h2l)
            align = float(((H1 - H2) ** 2).sum(1).mean())
            HH = torch.cat([H1, H2])
            sq = torch.cdist(HH, HH).pow(2)
            iu = torch.triu_indices(len(HH), len(HH), offset=1)
            unif = float(torch.log(torch.exp(-2.0 * sq[iu[0], iu[1]]).mean()))
            print(f"[{meth} ep{ep}] align={align:.4f} uniformity={unif:.4f} (무라벨 — 낮은 unif/align 균형이 좋음)", flush=True)
        extract_embeddings(ep)
    print(f"[OUT] {outdir}", flush=True)


if __name__ == "__main__":
    main()
