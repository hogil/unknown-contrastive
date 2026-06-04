#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Initial single-GPU contrastive + HDBSCAN wafer clustering recipe.

This file preserves the earlier "contrastive_init" style:
- train from TRAIN_DIR only, self-supervised
- embed/cluster UNKNOWN_DIR
- global InfoNCE + optional queue + local patch InfoNCE
- save HDBSCAN cluster folders, medoids, summaries

Important:
PyTorch 2.6 changed torch.load default weights_only to True. The original
script failed on checkpoints containing numpy objects. This version explicitly
uses weights_only=False for the trusted local backbone checkpoint.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import shutil
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=FutureWarning)


CFG: dict[str, Any] = {
    "TRAIN_DIR": "/home/sr5/ho.choi/project/unknown-contrastive-main/data/images/P3WN/train",
    "UNKNOWN_DIR": "/home/sr5/ho.choi/project/unknown-contrastive-main/data/images/P3WN/eval",
    "OVERLAY_DIR": "/home/sr5/ho.choi/project/unknown-contrastive-main/data/images/P3WN/train",
    "OUTPUT_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/outputs_hdbscan",
    "IMAGE_SIZE": 384,
    "BACKBONE_NAME": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "LOCAL_BACKBONE_WEIGHTS": "/home/sr5/ho.choi/project/wafer-detection/weights/convnextv2_base.fcmae_ft_in22k_in1k_384_92.6_91.7_250805_1639.pth",
    "PROJ_DIM": 128,
    "FREEZE_BACKBONE": True,
    "EPOCHS": 20,
    "WARMUP_EPOCHS": 2,
    "BATCH": 256,
    "LR_HEAD": 1e-3,
    "WD": 1e-6,
    "TEMP": 0.07,
    "SEED": 42,
    "TRAIN_SAMPLING_RATIO": 0.25,
    "NUM_WORKERS": 32,
    "PIN_MEMORY": True,
    "PREFETCH_FACTOR": 4,
    "PERSISTENT": True,
    "DROP_LAST": False,
    "USE_QUEUE": True,
    "QUEUE_SIZE": 16384,
    "QUEUE_WEIGHT": 1.0,
    "IGNORE_NEG_SIM": 0.72,
    "USE_LOCAL": True,
    "LOCAL_WEIGHT": 0.5,
    "LOCAL_ANCHORS": "grid36_full",
    "LOCAL_SEARCH": "window",
    "LOCAL_WINDOW": 4,
    "LOCAL_POS_TOPK": 12,
    "LOCAL_POS_MIN_SIM": 0.70,
    "LOCAL_EVERY_N": 1,
    "HDBSCAN_METRIC": "euclidean",
    "MIN_CLUSTER_SIZE": 12,
    "MIN_SAMPLES": 4,
    "CLUSTER_SELECTION_METHOD": "leaf",
    "CLUSTER_SELECTION_EPSILON": 0.06,
    "ALLOW_SINGLE_CLUSTER": False,
    "EMBED_LOG_TICKS": 20,
}

RUN_TS = datetime.now().strftime("%y%m%d_%H%M%S")


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("contrastive_init")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _torch_load_trusted(path: str | Path, map_location="cpu"):
    """Load a trusted local checkpoint across PyTorch versions.

    PyTorch 2.6 defaults torch.load(weights_only=True), which can reject older
    checkpoints containing numpy globals. This checkpoint is a local model
    artifact, so use weights_only=False intentionally.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_prefixes(sd: dict[str, Any]) -> dict[str, Any]:
    for pref in ("model.", "backbone.", "module."):
        if any(k.startswith(pref) for k in sd.keys()):
            sd = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}
    return sd


def tfm(train: bool):
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose([
            T.RandomResizedCrop((CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"]),
                                scale=(0.94, 1.0), ratio=(1.0, 1.0)),
            T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Lambda(lambda x: (x + torch.randn_like(x) * 0.02).clamp(0, 1)),
            norm,
        ])
    return T.Compose([
        T.Resize((CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"])),
        T.ToTensor(),
        norm,
    ])


class PairDS(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, _y, cls = self.items[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            return self.transform(im.copy()), self.transform(im.copy()), str(p), cls


class SingleViewDS(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, _y, cls = self.items[idx]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), str(p), cls


def build_loader(ds: Dataset, shuffle: bool, device: torch.device) -> DataLoader:
    kw = {}
    if CFG["NUM_WORKERS"] > 0:
        kw = {
            "num_workers": CFG["NUM_WORKERS"],
            "prefetch_factor": CFG["PREFETCH_FACTOR"],
            "persistent_workers": CFG["PERSISTENT"],
        }
    return DataLoader(
        ds,
        batch_size=CFG["BATCH"],
        shuffle=shuffle,
        drop_last=CFG["DROP_LAST"] if shuffle else False,
        pin_memory=(CFG["PIN_MEMORY"] and device.type == "cuda"),
        **kw,
    )


class Proj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            CFG["BACKBONE_NAME"], pretrained=False, num_classes=0, global_pool=""
        )
        ckpt = _torch_load_trusted(CFG["LOCAL_BACKBONE_WEIGHTS"], map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if not isinstance(sd, dict):
            raise TypeError(f"Unsupported checkpoint format: {type(sd)!r}")
        self.backbone.load_state_dict(_strip_prefixes(sd), strict=False)

        with torch.no_grad():
            dummy = torch.randn(1, 3, CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"])
            fm = self._forward_features(dummy)
            in_dim = fm.shape[1]
        self.proj = Proj(in_dim, CFG["PROJ_DIM"])
        if CFG["FREEZE_BACKBONE"]:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _forward_features(self, x):
        y = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)
        return y if y.ndim == 4 else y[:, :, None, None]

    def forward(self, x, return_local=False):
        x = x.to(memory_format=torch.channels_last)
        fm = self._forward_features(x)
        pooled = fm.mean(dim=(2, 3))
        z = F.normalize(self.proj(pooled), dim=1)
        if return_local:
            return z, F.normalize(fm, dim=1)
        return z


def info_nce_global(z1, z2, temp):
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
    sim = z @ z.T
    n = sim.size(0)
    mask = torch.eye(n, device=sim.device, dtype=torch.bool)
    pos = torch.cat([
        torch.diagonal(sim, offset=z1.size(0)),
        torch.diagonal(sim, offset=-z1.size(0)),
    ], dim=0).unsqueeze(1)
    neg = sim[~mask].view(n, n - 1)
    logits = torch.cat([pos, neg], dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=sim.device))


def info_nce_with_queue(z1, z2, bank, temp, ignore_sim):
    b = z1.size(0)
    pos = (z1 * z2).sum(1, keepdim=True)
    neg_ib = z1 @ z2.T
    neg_ib.fill_diagonal_(-1e9)
    if ignore_sim is not None:
        neg_ib = neg_ib.masked_fill(neg_ib >= ignore_sim, -1e9)
    negs = [neg_ib]
    if bank is not None:
        neg_bank = z1 @ bank.T
        if ignore_sim is not None:
            neg_bank = neg_bank.masked_fill(neg_bank >= ignore_sim, -1e9)
        negs.append(neg_bank)
    logits = torch.cat([pos] + negs, dim=1) / temp
    return F.cross_entropy(logits, torch.zeros(b, dtype=torch.long, device=z1.device))


def _grid36_indices(h: int, w: int, device) -> torch.Tensor:
    ys = [min(h - 1, max(0, int(round(((i + 0.5) * h) / 6)))) for i in range(6)]
    xs = [min(w - 1, max(0, int(round(((j + 0.5) * w) / 6)))) for j in range(6)]
    return torch.tensor([y * w + x for y in ys for x in xs], device=device, dtype=torch.long)


def info_nce_local_multi(f1, f2, temp):
    b, c, h, w = f1.shape
    idx = _grid36_indices(h, w, f1.device)
    losses = []
    k = 2 * CFG["LOCAL_WINDOW"] + 1
    for bi in range(b):
        fm1 = f1[bi].reshape(c, h * w).T.index_select(0, idx)
        patches = F.unfold(f2[bi].unsqueeze(0), kernel_size=(k, k),
                           padding=CFG["LOCAL_WINDOW"], stride=1)[0]
        patches = patches.view(c, k * k, h * w).index_select(2, idx)
        sims = torch.einsum("ac,cna->an", fm1, patches)
        topk = min(CFG["LOCAL_POS_TOPK"], sims.size(1))
        pos_idx = sims.topk(k=topk, dim=1).indices
        pos_mask = torch.zeros_like(sims, dtype=torch.bool)
        pos_mask.scatter_(1, pos_idx, True)
        pos_mask |= sims >= CFG["LOCAL_POS_MIN_SIM"]
        scaled = sims / temp
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        exp_s = scaled.exp()
        losses.append(-torch.log((exp_s * pos_mask).sum(dim=1).clamp_min(1e-12)
                                 / exp_s.sum(dim=1).clamp_min(1e-12)).mean())
    return torch.stack(losses).mean() if losses else f1.new_zeros(())


class QueueBank(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.size = int(size)
        self.register_buffer("queue", F.normalize(torch.randn(self.size, dim), dim=1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, x):
        if x is None or x.numel() == 0:
            return
        x = F.normalize(x.detach(), dim=1)
        k = min(x.size(0), self.size)
        p = int(self.ptr.item())
        end = p + k
        if end <= self.size:
            self.queue[p:end] = x[:k]
        else:
            n1 = self.size - p
            self.queue[p:] = x[:n1]
            self.queue[:end - self.size] = x[n1:k]
        self.ptr[0] = (p + k) % self.size
        self.filled[0] = min(self.size, int(self.filled.item()) + k)

    @torch.no_grad()
    def get(self):
        n = int(self.filled.item())
        return None if n <= 0 else self.queue[:n]


def parse_fields(path_str: str):
    toks = Path(path_str).stem.split("_")
    toks = toks[-5:] if len(toks) >= 5 else toks + [""] * (5 - len(toks))
    return toks


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        try:
            shutil.copy2(src, dst)
        except shutil.SameFileError:
            pass


@torch.no_grad()
def extract(model, items, device, logger):
    ds = SingleViewDS(items, tfm(False))
    ld = build_loader(ds, shuffle=False, device=device)
    model.eval()
    embs, files, classes = [], [], []
    t0 = time.time()
    for i, (xb, fb, cb) in enumerate(ld, 1):
        xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        embs.append(model(xb).float().cpu().numpy())
        files.extend(fb)
        classes.extend(cb)
        if i % max(1, len(ld) // max(1, CFG["EMBED_LOG_TICKS"])) == 0 or i == len(ld):
            logger.info(f"[Embed] {min(i * CFG['BATCH'], len(items))}/{len(items)}")
    emb = np.concatenate(embs, axis=0).astype(np.float32) if embs else np.zeros((0, CFG["PROJ_DIM"]), np.float32)
    logger.info(f"[Embed] done shape={emb.shape} time={time.time() - t0:.1f}s")
    return emb, files, classes


def _pick_export_src(raw_path: str) -> Path:
    raw = Path(raw_path)
    overlay = CFG.get("OVERLAY_DIR")
    if overlay:
        try:
            cand = Path(overlay) / raw.relative_to(CFG["UNKNOWN_DIR"])
            if cand.exists():
                return cand
        except Exception:
            pass
    return raw


def cluster_and_save_hdbscan(emb, files, classes, run_dir: Path, logger):
    params = dict(
        min_cluster_size=CFG["MIN_CLUSTER_SIZE"],
        min_samples=CFG["MIN_SAMPLES"],
        metric=CFG["HDBSCAN_METRIC"],
        cluster_selection_method=CFG["CLUSTER_SELECTION_METHOD"],
        cluster_selection_epsilon=CFG["CLUSTER_SELECTION_EPSILON"],
        allow_single_cluster=CFG["ALLOW_SINGLE_CLUSTER"],
    )
    logger.info(f"[HDBSCAN] {params}")
    clusterer = hdbscan.HDBSCAN(**params)
    t0 = time.time()
    pred = clusterer.fit_predict(emb)
    probs = getattr(clusterer, "probabilities_", np.zeros(len(pred)))
    logger.info(f"[HDBSCAN] clusters={len(set(pred) - {-1})} noise={(pred == -1).sum()} time={time.time() - t0:.1f}s")

    out_root = run_dir / "clusters" / "hdbscan"
    medoid_dir = run_dir / "cluster_summary"
    out_root.mkdir(parents=True, exist_ok=True)
    medoid_dir.mkdir(parents=True, exist_ok=True)
    global_rows = []

    for cid in sorted(c for c in set(pred.tolist()) if c >= 0):
        idxs = np.where(pred == cid)[0]
        tag = f"cluster_{cid:03d}_size_{len(idxs)}"
        cdir = out_root / tag
        cdir.mkdir(parents=True, exist_ok=True)
        hist = Counter(classes[i] for i in idxs)
        center = emb[idxs].mean(axis=0, keepdims=True)
        dist = np.linalg.norm(emb[idxs] - center, axis=1)
        rep = int(idxs[int(np.argmin(dist))])
        rep_src = _pick_export_src(files[rep])
        link_or_copy(rep_src, medoid_dir / f"{tag}__{classes[rep]}__medoid_dist{float(dist.min()):.4f}{rep_src.suffix}")

        with (cdir / f"{tag}.txt").open("w", encoding="utf-8") as f:
            f.write(f"[Cluster {cid:03d}] size={len(idxs)} mean_prob={float(np.mean(probs[idxs])):.3f}\n")
            for cls, cnt in hist.most_common():
                f.write(f"  - {cls}: {cnt}\n")
            f.write("\nclass\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
            for i in idxs:
                src = _pick_export_src(files[i])
                link_or_copy(src, cdir / f"{classes[i]}_{src.name}")
                fields = parse_fields(files[i])
                f.write(f"{classes[i]}\t" + "\t".join(fields) + "\n")
                global_rows.append((f"{cid:03d}", classes[i], *fields))

    with (run_dir / "clusters_global_list.txt").open("w", encoding="utf-8") as f:
        f.write("clusterno\tclass\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
        for row in global_rows:
            f.write("\t".join(map(str, row)) + "\n")
    with (run_dir / "clusters_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Total: {len(files)}\nClusters: {len(set(pred) - {-1})}\nNoise: {int((pred == -1).sum())}\n")


def save_run_info(run_dir: Path, device, class_to_idx):
    info = {
        "timestamp": RUN_TS,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": getattr(timm, "__version__", "unknown"),
        "hdbscan": getattr(hdbscan, "__version__", "unknown"),
        "cfg": CFG,
        "class_to_idx": class_to_idx,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    seed_all(CFG["SEED"])
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(f"{CFG['OUTPUT_DIR']}_{RUN_TS}")
    logger = setup_logger(run_dir)
    logger.info("===== RUN START =====")
    logger.info(json.dumps(CFG, indent=2, ensure_ascii=False))

    train_ds = ImageFolder(CFG["TRAIN_DIR"])
    unknown_ds = ImageFolder(CFG["UNKNOWN_DIR"])
    train_items = [(Path(p), -1, train_ds.classes[int(y)]) for p, y in train_ds.samples]
    eval_items = [(Path(p), -1, unknown_ds.classes[int(y)]) for p, y in unknown_ds.samples]
    logger.info(f"[Train from TRAIN_DIR] full_items={len(train_items)}")
    logger.info(f"[Eval set] unknown_only={len(eval_items)}")

    model = CL().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=CFG["LR_HEAD"], weight_decay=CFG["WD"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    queue = QueueBank(CFG["PROJ_DIM"], CFG["QUEUE_SIZE"]).to(device) if CFG["USE_QUEUE"] else None

    for ep in range(1, CFG["EPOCHS"] + 1):
        lr = CFG["LR_HEAD"] * min(1.0, ep / max(1, CFG["WARMUP_EPOCHS"]))
        for g in opt.param_groups:
            g["lr"] = lr
        rng = random.Random(CFG["SEED"] + ep)
        ratio = float(CFG["TRAIN_SAMPLING_RATIO"])
        items = rng.sample(train_items, max(1, int(len(train_items) * ratio))) if 0 < ratio < 1 else train_items
        ld = build_loader(PairDS(items, tfm(True)), shuffle=True, device=device)
        model.train()
        sums = Counter()
        t0 = time.time()
        for it, (x1, x2, _fb, _cb) in enumerate(ld, 1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=dtype):
                z1, f1 = model(x1, return_local=True)
                z2, f2 = model(x2, return_local=True)
                loss_g = info_nce_global(z1, z2, CFG["TEMP"])
                loss_q = x1.new_zeros(())
                if queue is not None:
                    bank = queue.get()
                    loss_q = 0.5 * (
                        info_nce_with_queue(z1, z2, bank, CFG["TEMP"], CFG["IGNORE_NEG_SIM"])
                        + info_nce_with_queue(z2, z1, bank, CFG["TEMP"], CFG["IGNORE_NEG_SIM"])
                    ) * CFG["QUEUE_WEIGHT"]
                loss_l = x1.new_zeros(())
                if CFG["USE_LOCAL"] and it % CFG["LOCAL_EVERY_N"] == 0:
                    loss_l = info_nce_local_multi(f1, f2, CFG["TEMP"]) * CFG["LOCAL_WEIGHT"]
                loss = loss_g + loss_q + loss_l
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if queue is not None:
                queue.enqueue(z2.detach())
            sums["g"] += float(loss_g)
            sums["q"] += float(loss_q)
            sums["l"] += float(loss_l)
            sums["n"] += 1
        n = max(1, sums["n"])
        logger.info(f"Epoch {ep}/{CFG['EPOCHS']} G={sums['g']/n:.4f} Q={sums['q']/n:.4f} L={sums['l']/n:.4f} time={time.time()-t0:.1f}s")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_dir / "final_infer.pt")
    save_run_info(run_dir, device, {c: i for i, c in enumerate(unknown_ds.classes)})

    emb, files, classes = extract(model, eval_items, device, logger)
    cluster_and_save_hdbscan(emb, files, classes, run_dir, logger)
    logger.info(f"[DONE] {run_dir.resolve()}")


if __name__ == "__main__":
    main()
