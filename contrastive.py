#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wafer Defect Auto-Clustering (Contrastive + HDBSCAN, Auto-K)
- Train: TRAIN_DIR만 사용(자기지도), 클래스별 ratio 샘플링(최소 1장)
- Embed/Cluster: UNKNOWN_DIR 전수
- Aug: small rotate/translate/scale, noise, NO flip
- Loss: Global InfoNCE (+ Queue) + Local InfoNCE
- Embedding L2 normalize
- Saves:
  - clusters/hdbscan/cluster_XXX_size_YYY/... + cluster txt
  - clusters_summary.txt
  - clusters_global_list.txt
  - run.log, run_info.json
  - 대표 이미지: run_dir/cluster_summary/ (medoid)
  - ignored_samples/
- Overlay export: 저장 시 overlay가 있으면 overlay 이미지로 저장(없으면 raw)
"""

import os
import sys
import json
import time
import math
import random
import shutil
import warnings
import platform
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import timm
import hdbscan


# =========================================================
# CONFIG
# =========================================================
CFG: Dict[str, Any] = {
    # -------------------------
    # Paths
    # -------------------------
    "TRAIN_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/data/self",
    "UNKNOWN_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/data/unknown",
    "OVERLAY_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/data/overlay",
    "OUTPUT_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/outputs_hdbscan",
    "IMAGE_SIZE": 384,

    # -------------------------
    # Model / Head
    # -------------------------
    "BACKBONE_NAME": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "LOCAL_BACKBONE_WEIGHTS": "/home/sr5/ho.choi/project/wafer-defect-clustering/weights/convnextv2_base.fcmae_ft_in22k_in1k_384_91.8_91.7_250724_1505.pth",
    "PROJ_DIM": 128,
    "FREEZE_BACKBONE": True,

    # -------------------------
    # Train
    # -------------------------
    "EPOCHS": 20,
    "WARMUP_EPOCHS": 2,
    "BATCH": 256,
    "LR_HEAD": 1e-3,
    "WD": 1e-6,
    "TEMP": 0.07,
    "SEED": 42,
    "TRAIN_SAMPLING_RATIO": 0.25,  # 클래스별 ratio 샘플링

    # -------------------------
    # DataLoader
    # -------------------------
    "NUM_WORKERS": 32,
    "PIN_MEMORY": True,
    "PREFETCH_FACTOR": 4,
    "PERSISTENT": True,
    "DROP_LAST": False,

    # -------------------------
    # Global InfoNCE + Queue
    # -------------------------
    "USE_QUEUE": True,
    "QUEUE_SIZE": 16384,
    "QUEUE_WEIGHT": 1.0,
    "IGNORE_NEG_SIM": 0.72,

    # -------------------------
    # Local InfoNCE
    # -------------------------
    "USE_LOCAL": True,
    "LOCAL_WEIGHT": 0.5,
    "LOCAL_ANCHORS": "grid36_full",
    "LOCAL_SEARCH": "window",  # window or global
    "LOCAL_WINDOW": 4,
    "LOCAL_POS_TOPK": 12,
    "LOCAL_POS_MIN_SIM": 0.70,
    "LOCAL_SUBBATCH": 32,
    "LOCAL_EVERY_N": 1,

    # -------------------------
    # HDBSCAN
    # -------------------------
    "HDBSCAN_METRIC": "euclidean",
    "MIN_CLUSTER_SIZE": 12,
    "MIN_SAMPLES": 4,
    "CLUSTER_SELECTION_METHOD": "leaf",   # leaf / eom
    "CLUSTER_SELECTION_EPSILON": 0.06,
    "ALLOW_SINGLE_CLUSTER": False,

    # -------------------------
    # Keep / Ignore filter
    # -------------------------
    "KEEP_MIN_SIZE": 12,
    "KEEP_MIN_MEDIAN_PROB": 0.55,
    "KEEP_MIN_PERSIST": 0.20,
    "SAVE_IGNORED_SAMPLES": True,
    "IGNORED_TOPN": 100,

    # -------------------------
    # Embed logging
    # -------------------------
    "EMBED_LOG_EVERY_BATCH": True,
    "EMBED_LOG_TICKS": 20,

    # -------------------------
    # Experiments variant markers (experiments/run_experiment.py가 주입)
    # -------------------------
    "__INPUT_MODE__": "palette_rgb",     # "palette_rgb" | "idx_gray_x8" | "idx_1ch"
    "__UNFREEZE_STAGES__": [],           # 예: ["stages.3"]
    "__HARD_NEGATIVES__": False,
    "__STRONG_AUGMENT__": False,
    "__MULTICROP__": False,
    "__DEEPER_HEAD__": False,
}

RUN_TS = datetime.now().strftime("%y%m%d_%H%M%S")


# =========================================================
# Logging
# =========================================================
def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wafer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _fmt_hms(secs: float) -> str:
    secs = int(max(0, secs))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:d}:{m:02d}:{s:02d}"


# =========================================================
# Utils
# =========================================================
def seed_all(x: int = 42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)


def tfm(train: bool = True):
    # __MULTICROP__: TODO — SwAV-style 2 global + 4 local views not implemented.
    # Dataset wiring (PairDS vs a MultiCropDS) is non-trivial; flag raises downstream.
    if CFG.get("__MULTICROP__"):
        raise NotImplementedError("multicrop WIP")

    input_mode = CFG.get("__INPUT_MODE__", "palette_rgb")
    # For idx_1ch we still emit 3-channel float via ToTensor in the dataset pipeline
    # (the dataset produces a PIL RGB image or a synthetic tensor); conv1 surgery
    # (in CL) collapses 3ch->1ch weights where needed.
    if input_mode == "idx_1ch":
        # Single-channel mean/std: use ImageNet luminance-ish average.
        norm = T.Normalize(mean=[0.449], std=[0.226])
    else:
        norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

    strong = bool(CFG.get("__STRONG_AUGMENT__"))

    if train:
        affine_degrees = 15 if strong else 7
        steps = [
            T.RandomResizedCrop(
                (CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"]),
                scale=(0.94, 1.0),
                ratio=(1.0, 1.0),
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.RandomAffine(
                degrees=affine_degrees,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
        ]
        if strong:
            steps.append(T.GaussianBlur(3, sigma=(0.1, 0.8)))
        steps.append(T.ToTensor())
        steps.append(T.Lambda(lambda x: (x + torch.randn_like(x) * 0.02).clamp(0, 1)))
        steps.append(norm)
        if strong:
            steps.append(T.RandomErasing(p=0.3, scale=(0.02, 0.1)))
        return T.Compose(steps)

    return T.Compose([
        T.Resize((CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"]),
                 interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        norm
    ])


# ---------------------------------------------------------------------------
# Palette-index based PIL loading for idx_gray_x8 / idx_1ch variants
# ---------------------------------------------------------------------------
def _load_as_input_mode(p, input_mode: str):
    """Open an image at path `p` and return a PIL Image shaped according
    to the input mode expected by the rest of the pipeline.

    * palette_rgb: standard RGB convert.
    * idx_gray_x8: load palette indices (0..31), clip to 0..31, multiply by 8
      to spread across 0..255, replicate into 3 channels, return as RGB PIL.
    * idx_1ch: load palette indices, clip to 0..31, multiply by 8, return as
      single-channel ('L') PIL. (conv1 surgery is currently NotImplemented,
      see CL.__init__.)
    """
    if input_mode == "palette_rgb":
        with Image.open(p) as im:
            return im.convert("RGB")

    # idx_* modes — use common.palette_io to fetch the raw palette indices.
    try:
        from common.palette_io import load_palette_indices
    except Exception:
        # Fallback to simple P-mode read if common isn't importable.
        with Image.open(p) as im:
            if im.mode == "P":
                arr = np.array(im, dtype=np.int16)
            else:
                arr = np.array(im.convert("L"), dtype=np.int16) // 32
    else:
        arr = load_palette_indices(p)

    arr = np.clip(arr, 0, 31).astype(np.uint8)
    scaled = np.clip(arr.astype(np.int16) * 8, 0, 255).astype(np.uint8)

    if input_mode == "idx_1ch":
        return Image.fromarray(scaled, mode="L")

    # idx_gray_x8: replicate to 3 channels
    rgb = np.stack([scaled, scaled, scaled], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def link_or_copy(src: Path, dst: Path, logger: logging.Logger = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return
    except Exception as e:
        if logger:
            logger.debug(f"Hardlink fail: {e}")

    try:
        shutil.copy2(src, dst)
    except shutil.SameFileError:
        return


def parse_fields_from_filename(path_str: str):
    base = Path(path_str).stem
    toks = base.split('_')
    if len(toks) >= 5:
        root, step, wafer, ymd, hms = toks[-5:]
    else:
        need = 5 - len(toks)
        toks = toks + ([''] * need)
        root, step, wafer, ymd, hms = toks[:5]
    return root, step, wafer, ymd, hms


def strip_prefixes(sd: Dict[str, torch.Tensor],
                   prefixes=("module.", "model.", "backbone.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        out[k] = v
    return out


def sample_per_class(items, ratio: float, seed: int):
    by_class = defaultdict(list)
    for item in items:
        by_class[item[2]].append(item)

    rng = random.Random(seed)
    out = []
    for cls_name, arr in by_class.items():
        k = max(1, int(len(arr) * ratio))
        k = min(k, len(arr))
        out.extend(rng.sample(arr, k))
    rng.shuffle(out)
    return out


# =========================================================
# Data
# =========================================================
class PairDS(Dataset):
    def __init__(self, items, t):
        self.items = items
        self.t = t

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, c = self.items[i]
        mode = CFG.get("__INPUT_MODE__", "palette_rgb")
        im = _load_as_input_mode(p, mode)
        x1 = self.t(im.copy())
        x2 = self.t(im.copy())
        return x1, x2, y, str(p), c


def build_loader(ds, device, shuffle=True):
    kw = {}
    if CFG["NUM_WORKERS"] > 0:
        kw = dict(
            num_workers=CFG["NUM_WORKERS"],
            prefetch_factor=CFG["PREFETCH_FACTOR"],
            persistent_workers=CFG["PERSISTENT"]
        )
    return DataLoader(
        ds,
        batch_size=CFG["BATCH"],
        shuffle=shuffle,
        drop_last=CFG["DROP_LAST"],
        pin_memory=(CFG["PIN_MEMORY"] and device.type == "cuda"),
        **kw
    )


# =========================================================
# Model
# =========================================================
class Proj(nn.Module):
    def __init__(self, d: int, m: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(True),
            nn.Linear(d, m)
        )

    def forward(self, x):
        return self.net(x)


class CL(nn.Module):
    def __init__(self, logger: logging.Logger = None):
        super().__init__()
        self.logger = logger

        self.backbone = timm.create_model(
            CFG["BACKBONE_NAME"],
            pretrained=False,
            num_classes=0,
            global_pool=""
        )

        # load local weights
        wpath = CFG["LOCAL_BACKBONE_WEIGHTS"]
        if wpath and Path(wpath).exists():
            sd = torch.load(wpath, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if not isinstance(sd, dict):
                raise ValueError(f"Unexpected checkpoint format: {type(sd)}")

            sd = strip_prefixes(sd)
            msg = self.backbone.load_state_dict(sd, strict=False)

            if self.logger:
                self.logger.info(f"[Backbone load] weights={wpath}")
                self.logger.info(f"[Backbone load] missing_keys={len(msg.missing_keys)}")
                self.logger.info(f"[Backbone load] unexpected_keys={len(msg.unexpected_keys)}")
                if len(msg.missing_keys) > 0:
                    self.logger.info(f"[Backbone load] missing sample={msg.missing_keys[:20]}")
                if len(msg.unexpected_keys) > 0:
                    self.logger.info(f"[Backbone load] unexpected sample={msg.unexpected_keys[:20]}")
        else:
            if self.logger:
                self.logger.info(f"[Backbone load] local weights not found: {wpath}")

        # idx_1ch: TODO — conv1 surgery (collapse 3ch->1ch) is backbone-dependent
        # and timm has no uniform API for module replacement on ConvNeXtV2.
        # For now this mode is not supported; idx_gray_x8 covers palette-index input
        # without needing any weight surgery.
        if CFG.get("__INPUT_MODE__") == "idx_1ch":
            raise NotImplementedError(
                "idx_1ch input mode requires backbone stem conv1 surgery which is "
                "not implemented yet. Use idx_gray_x8 instead."
            )

        in_ch = 1 if CFG.get("__INPUT_MODE__") == "idx_1ch" else 3
        with torch.no_grad():
            dummy = torch.randn(1, in_ch, CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"])
            fm = self._forward_features(dummy)
            d = fm.shape[1]

        if CFG.get("__DEEPER_HEAD__"):
            self.proj = nn.Sequential(
                nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(True),
                nn.Linear(d, d // 2), nn.BatchNorm1d(d // 2), nn.ReLU(True),
                nn.Linear(d // 2, CFG["PROJ_DIM"]),
            )
        else:
            self.proj = Proj(d, CFG["PROJ_DIM"])

        if CFG["FREEZE_BACKBONE"]:
            for p in self.backbone.parameters():
                p.requires_grad = False
            unfreeze = CFG.get("__UNFREEZE_STAGES__") or []
            if unfreeze:
                unfrozen = 0
                for name, p in self.backbone.named_parameters():
                    for prefix in unfreeze:
                        if name.startswith(prefix):
                            p.requires_grad = True
                            unfrozen += 1
                            break
                if self.logger:
                    self.logger.info(
                        f"[Backbone unfreeze] prefixes={unfreeze} "
                        f"unfrozen_params={unfrozen}"
                    )

    def train(self, mode: bool = True):
        super().train(mode)
        any_trainable = any(p.requires_grad for p in self.backbone.parameters())
        if CFG["FREEZE_BACKBONE"] and not any_trainable:
            self.backbone.eval()
        else:
            self.backbone.train(mode)
        return self

    def _forward_features(self, x):
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        y = self.backbone(x)
        return y if y.ndim == 4 else y[:, :, None, None]

    def forward(self, x, return_local=False):
        x = x.to(memory_format=torch.channels_last)
        fm = self._forward_features(x)           # [B,C,H,W]
        pool = fm.mean(dim=(2, 3))               # GAP
        z = F.normalize(self.proj(pool), dim=1)  # [B,D]

        if return_local:
            fm = F.normalize(fm, dim=1)
            return z, fm
        return z


# =========================================================
# Losses
# =========================================================
def info_nce_global(z1, z2, t=0.07):
    """
    Correct SimCLR-style InfoNCE:
    - self 제외
    - positive 제외 후 negative 구성
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # [2B, D]
    sim = z @ z.T                               # [2B, 2B]

    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    mask = torch.ones((2 * B, 2 * B), dtype=torch.bool, device=z.device)
    mask.fill_diagonal_(False)
    mask[torch.arange(2 * B), pos_idx] = False

    pos = sim[torch.arange(2 * B), pos_idx].unsqueeze(1)   # [2B,1]
    neg = sim[mask].view(2 * B, -1)                        # [2B,2B-2]

    logits = torch.cat([pos, neg], dim=1) / t
    labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


def info_nce_with_queue(z1, z2, bank, t=0.07, ignore_sim=0.8):
    """
    z1 anchor, z2 positive
    bank: normalized negative bank
    """
    B = z1.size(0)
    pos = (z1 * z2).sum(1, keepdim=True)        # [B,1]

    neg_ib = z1 @ z2.T                          # [B,B]
    neg_ib.fill_diagonal_(-1e9)

    if ignore_sim is not None:
        neg_ib = torch.where(
            neg_ib >= ignore_sim,
            torch.full_like(neg_ib, -1e9),
            neg_ib
        )

    if bank is not None:
        neg_bank = z1 @ bank.T                  # [B,N]
        if CFG.get("__HARD_NEGATIVES__"):
            K = min(512, neg_bank.size(1))
            if K > 0:
                top_vals, _ = neg_bank.topk(K, dim=1, largest=True)
                neg_bank = top_vals              # [B,K]
        if ignore_sim is not None:
            neg_bank = torch.where(
                neg_bank >= ignore_sim,
                torch.full_like(neg_bank, -1e9),
                neg_bank
            )
        logits = torch.cat([pos, neg_ib, neg_bank], dim=1) / t
    else:
        logits = torch.cat([pos, neg_ib], dim=1) / t

    labels = torch.zeros(B, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels)


def info_nce_local_multi(
    f1: torch.Tensor,
    f2: torch.Tensor,
    anchors=32,
    search: str = 'global',
    r: int = 1,
    pos_topk: int = -1,
    pos_min_sim: float = 0.0,
    t: float = 0.2,
    subbatch: int = 32
) -> torch.Tensor:
    """
    f1, f2: [B,C,H,W], channel-normalized
    """
    B, C, H, W = f1.shape
    device = f1.device
    search = (search or 'global').lower()

    def _grid_coords(H, W, g=6, shift=(0.0, 0.0)):
        ys = [min(H - 1, max(0, int(round(((i + 0.5 + shift[1]) * H) / g)))) for i in range(g)]
        xs = [min(W - 1, max(0, int(round(((j + 0.5 + shift[0]) * W) / g)))) for j in range(g)]
        return [(y, x) for y in ys for x in xs]

    def _uniq(seq):
        seen, out = set(), []
        for v in seq:
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    def _uniq_pad_random(coords, target, H, W, device=None):
        uniq = _uniq(coords)
        if len(uniq) >= target:
            return uniq[:target]
        existing = set(uniq)
        rest = [(y, x) for y in range(H) for x in range(W) if (y, x) not in existing]
        if rest:
            idx = torch.randperm(len(rest), device=device)[:(target - len(uniq))].tolist()
            uniq += [rest[i] for i in idx]
        return uniq[:target]

    anchor_list_per_b = []

    if isinstance(anchors, str):
        if anchors == "grid36_full":
            coords = _grid_coords(H, W, g=6, shift=(0.0, 0.0))
            j_idx = torch.tensor([y * W + x for (y, x) in coords], device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
        elif anchors == "grid16_shift4":
            shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
            coords = []
            for sh in shifts:
                coords += _grid_coords(H, W, g=4, shift=sh)
            coords_64 = _uniq_pad_random(coords, target=64, H=H, W=W, device=device)
            j_idx = torch.tensor([y * W + x for (y, x) in coords_64], device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
        else:
            j_idx = torch.arange(H * W, device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
    elif anchors is None or (isinstance(anchors, int) and anchors >= H * W):
        j_idx = torch.arange(H * W, device=device, dtype=torch.long)
        anchor_list_per_b = [j_idx] * B
    elif isinstance(anchors, int):
        for _ in range(B):
            idx = torch.randperm(H * W, device=device)[:int(anchors)]
            anchor_list_per_b.append(idx)
    else:
        j_idx = torch.arange(H * W, device=device, dtype=torch.long)
        anchor_list_per_b = [j_idx] * B

    losses = []

    for b in range(B):
        fm1 = f1[b]
        fm2 = f2[b]
        idx = anchor_list_per_b[b]
        A = idx.numel()
        if A == 0:
            continue

        v1_all = fm1.reshape(C, H * W).T.index_select(0, idx)  # [A,C]

        if search == 'global':
            f2_flat = fm2.reshape(C, H * W).T                  # [HW,C]

            for st in range(0, A, subbatch):
                ed = min(A, st + subbatch)
                v1 = v1_all[st:ed]                             # [a,C]
                sims = v1 @ f2_flat.T                          # [a, HW]

                if pos_topk is None or pos_topk < 0:
                    P = sims >= pos_min_sim if pos_min_sim > -1.0 else torch.ones_like(sims, dtype=torch.bool, device=device)
                    best = sims.argmax(dim=1, keepdim=True)
                    P.scatter_(1, best, True)
                else:
                    topk = min(pos_topk, sims.size(1))
                    top_idx = sims.topk(k=topk, dim=1, largest=True).indices
                    P = torch.zeros_like(sims, dtype=torch.bool, device=device)
                    P.scatter_(1, top_idx, True)
                    best = sims.argmax(dim=1, keepdim=True)
                    P.scatter_(1, best, True)

                S = sims / t
                S = S - S.max(dim=1, keepdim=True).values
                expS = torch.exp(S)
                pos_sum = (expS * P).sum(dim=1) + 1e-12
                all_sum = expS.sum(dim=1) + 1e-12
                loss = -torch.log(pos_sum / all_sum).mean()
                losses.append(loss)

        elif search == 'window':
            k = 2 * r + 1
            U = F.unfold(fm2.unsqueeze(0), kernel_size=(k, k), padding=r, stride=1)[0]
            U = U.view(C, k * k, H * W)

            for st in range(0, A, subbatch):
                ed = min(A, st + subbatch)
                idx_sub = idx[st:ed]
                v1 = v1_all[st:ed]                             # [a,C]

                U_sub = U.index_select(2, idx_sub)            # [C,Nc,a]
                sims = torch.einsum('ac,cna->an', v1, U_sub)  # [a,Nc]

                if pos_topk is None or pos_topk < 0:
                    P = sims >= pos_min_sim if pos_min_sim > -1.0 else torch.ones_like(sims, dtype=torch.bool, device=device)
                    best = sims.argmax(dim=1, keepdim=True)
                    P.scatter_(1, best, True)
                else:
                    topk = min(pos_topk, sims.size(1))
                    top_idx = sims.topk(k=topk, dim=1, largest=True).indices
                    P = torch.zeros_like(sims, dtype=torch.bool, device=device)
                    P.scatter_(1, top_idx, True)
                    best = sims.argmax(dim=1, keepdim=True)
                    P.scatter_(1, best, True)

                S = sims / t
                S = S - S.max(dim=1, keepdim=True).values
                expS = torch.exp(S)
                pos_sum = (expS * P).sum(dim=1) + 1e-12
                all_sum = expS.sum(dim=1) + 1e-12
                loss = -torch.log(pos_sum / all_sum).mean()
                losses.append(loss)
        else:
            raise ValueError(f"search must be 'global' or 'window', got: {search}")

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses, dim=0).mean()


# =========================================================
# Queue
# =========================================================
class QueueBank(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.K = int(K)
        self.register_buffer("queue", F.normalize(torch.randn(self.K, dim), dim=1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, x):
        if x is None or x.numel() == 0:
            return
        x = F.normalize(x.detach(), dim=1)
        b = x.size(0)
        k = min(b, self.K)
        p = int(self.ptr.item())
        end = p + k
        if end <= self.K:
            self.queue[p:end] = x[:k]
        else:
            n1 = self.K - p
            self.queue[p:] = x[:n1]
            self.queue[:end - self.K] = x[n1:k]
        self.ptr[0] = (p + k) % self.K
        self.filled[0] = min(self.K, int(self.filled.item()) + k)

    @torch.no_grad()
    def get(self):
        n = int(self.filled.item())
        if n <= 0:
            return None
        return self.queue[:n]


# =========================================================
# Embedding
# =========================================================
@torch.no_grad()
def extract(model, items, device, logger: logging.Logger, log_progress: bool = True, log_name: str = "Embed"):
    class SingleView(Dataset):
        def __init__(self, items, t):
            self.items = items
            self.t = t

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            p, y, c = self.items[i]
            mode = CFG.get("__INPUT_MODE__", "palette_rgb")
            im = _load_as_input_mode(p, mode)
            x = self.t(im)
            return x, y, str(p), c

    ds = SingleView(items, tfm(False))
    kw = {}
    if CFG["NUM_WORKERS"] > 0:
        kw = dict(
            num_workers=CFG["NUM_WORKERS"],
            prefetch_factor=CFG["PREFETCH_FACTOR"],
            persistent_workers=CFG["PERSISTENT"]
        )

    ld = DataLoader(
        ds,
        batch_size=CFG["BATCH"],
        shuffle=False,
        pin_memory=(CFG["PIN_MEMORY"] and device.type == "cuda"),
        **kw
    )

    total_imgs = len(items)
    total_batches = max(1, len(ld))
    tick = max(1, total_batches // max(1, CFG["EMBED_LOG_TICKS"]))

    model.eval()
    embs, labs, files, clss = [], [], [], []
    processed = 0
    start = time.time()

    for it, (xb, yb, fb, cb) in enumerate(ld, 1):
        xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        zb = model(xb)

        embs.append(zb.float().cpu().numpy())
        labs += [int(y) for y in yb]
        files += list(fb)
        clss += list(cb)

        if log_progress:
            processed += xb.size(0)
            elapsed = time.time() - start
            ips = processed / max(1e-6, elapsed)
            remaining = max(0, total_imgs - processed)
            eta = remaining / max(1e-6, ips)
            pct = 100.0 * processed / max(1, total_imgs)
            do_log = CFG["EMBED_LOG_EVERY_BATCH"] or ((it % tick) == 0) or (it == total_batches)

            if do_log:
                if device.type == "cuda":
                    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    logger.info(
                        f"[{log_name}] {processed:7d}/{total_imgs:<7d} imgs ({pct:5.1f}%) | "
                        f"{ips:6.1f} img/s | elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta)} | mem={mem_gb:.1f}GB"
                    )
                else:
                    logger.info(
                        f"[{log_name}] {processed:7d}/{total_imgs:<7d} imgs ({pct:5.1f}%) | "
                        f"{ips:6.1f} img/s | elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta)}"
                    )

    embs = np.concatenate(embs, 0).astype(np.float32, copy=False) if embs else np.zeros((0, CFG["PROJ_DIM"]), dtype=np.float32)
    return embs, labs, files, clss


# =========================================================
# Cluster metrics
# =========================================================
def _cluster_neighbor_cohesion(emb: np.ndarray, idxs: np.ndarray, k: int = 5) -> float:
    if len(idxs) <= 1:
        return 0.0
    E = emb[idxs]
    S = (E @ E.T).astype(np.float32)
    np.fill_diagonal(S, -1e9)
    n = S.shape[0]
    topk = min(k, max(1, n - 1))
    part = np.partition(S, -topk, axis=1)[:, -topk:]
    m = part.mean(axis=1)
    return float(np.median(m))


def _cluster_margin_ratio(emb: np.ndarray, idxs: np.ndarray, centers: np.ndarray, my_idx: int) -> float:
    n = len(idxs)
    if n <= 1:
        return 0.0
    E = emb[idxs]
    ctr = E.mean(0, keepdims=True)
    intra = np.median(np.linalg.norm(E - ctr, axis=1))
    if centers.shape[0] <= 1:
        return 0.0
    dists = np.linalg.norm(ctr - centers, axis=1)
    dists[my_idx] = np.inf
    nearest_other = float(np.min(dists))
    if intra <= 1e-6:
        return float('inf')
    return nearest_other / max(intra, 1e-6)


def _label_purity(idxs: np.ndarray, classes: List[str]) -> float:
    if len(idxs) == 0:
        return 0.0
    hist = {}
    for i in idxs:
        c = classes[i]
        hist[c] = hist.get(c, 0) + 1
    return max(hist.values()) / len(idxs)


def is_kept_cluster(srow: Dict[str, Any]) -> bool:
    return (
        srow["size"] >= CFG["KEEP_MIN_SIZE"] and
        srow["median_prob"] >= CFG["KEEP_MIN_MEDIAN_PROB"] and
        srow["persistence"] >= CFG["KEEP_MIN_PERSIST"]
    )


# =========================================================
# HDBSCAN + Save
# =========================================================
def cluster_and_save_hdbscan(
    emb: np.ndarray,
    files: List[str],
    labels: List[int],
    classes: List[str],
    run_dir: Path,
    logger: logging.Logger
):
    def _pick_export_src(raw_path_str: str) -> Path:
        raw_p = Path(raw_path_str)

        if CFG.get("OVERLAY_DIR"):
            try:
                rel = raw_p.relative_to(CFG["UNKNOWN_DIR"])
                cand = Path(CFG["OVERLAY_DIR"]) / rel
                if cand.exists():
                    return cand
            except Exception:
                pass

        return raw_p

    out_clusters_root = Path(run_dir) / "clusters" / "hdbscan"
    out_clusters_root.mkdir(parents=True, exist_ok=True)

    summary_dir = Path(run_dir) / "cluster_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    ignored_dir = Path(run_dir) / "ignored_samples"
    if CFG["SAVE_IGNORED_SAMPLES"]:
        ignored_dir.mkdir(parents=True, exist_ok=True)

    params = dict(
        min_cluster_size=CFG["MIN_CLUSTER_SIZE"],
        min_samples=CFG["MIN_SAMPLES"],
        metric=CFG["HDBSCAN_METRIC"],
        cluster_selection_method=CFG["CLUSTER_SELECTION_METHOD"],
        cluster_selection_epsilon=CFG["CLUSTER_SELECTION_EPSILON"],
        allow_single_cluster=CFG["ALLOW_SINGLE_CLUSTER"]
    )

    logger.info(f"[HDBSCAN] {params}")
    clusterer = hdbscan.HDBSCAN(**params)

    t0 = time.time()
    cids = clusterer.fit_predict(emb)
    probs = clusterer.probabilities_
    persist = clusterer.cluster_persistence_
    fit_sec = time.time() - t0

    uniq = sorted([int(u) for u in np.unique(cids) if u >= 0])
    noise_idx = np.where(cids == -1)[0].tolist()
    pers_map = {lab: float(persist[i]) for i, lab in enumerate(uniq)}

    stats = []
    for lab in uniq:
        idxs = np.where(cids == lab)[0]
        size = int(len(idxs))
        medp = float(np.median(probs[idxs])) if size > 0 else 0.0
        meanp = float(np.mean(probs[idxs])) if size > 0 else 0.0
        stdp = float(np.std(probs[idxs])) if size > 0 else 0.0
        stats.append({
            "lab": lab,
            "idxs": idxs,
            "size": size,
            "median_prob": medp,
            "mean_prob": meanp,
            "std_prob": stdp,
        })

    centers = []
    for s in stats:
        if s["size"] > 0:
            centers.append(emb[s["idxs"]].mean(0))
        else:
            centers.append(np.zeros((emb.shape[1],), dtype=emb.dtype))
    centers = np.stack(centers, 0) if len(centers) > 0 else np.zeros((0, emb.shape[1]), dtype=emb.dtype)

    for ci, s in enumerate(stats):
        s["persistence"] = pers_map.get(s["lab"], 0.0)
        s["cohesion"] = _cluster_neighbor_cohesion(emb, s["idxs"], k=5)
        s["margin"] = _cluster_margin_ratio(emb, s["idxs"], centers, my_idx=ci)
        s["purity"] = _label_purity(s["idxs"], classes)
        s["status"] = "KEPT" if is_kept_cluster(s) else "IGNORED"

    def _pct_local(a, ps):
        if len(a) == 0:
            return {p: 0.0 for p in ps}
        return {p: float(np.percentile(a, p)) for p in ps}

    sizes = [s["size"] for s in stats]
    medps = [s["median_prob"] for s in stats]
    perss = [s["persistence"] for s in stats]
    p_sizes = _pct_local(sizes, [10, 25, 50, 75, 90]) if sizes else {}
    p_medp = _pct_local(medps, [10, 25, 50, 75, 90]) if medps else {}
    p_pers = _pct_local(perss, [10, 25, 50, 75, 90]) if perss else {}

    logger.info("\n[HDBSCAN Summary]")
    logger.info(f"  samples={len(files)} | fit_time={fit_sec:.1f}s")
    logger.info(f"  clusters_found(pre-filter)={len(uniq)} | noise={len(noise_idx)}")
    if sizes:
        logger.info(f"  size percentiles : P10={p_sizes[10]:.0f} P25={p_sizes[25]:.0f} P50={p_sizes[50]:.0f} P75={p_sizes[75]:.0f} P90={p_sizes[90]:.0f}")
    if medps:
        logger.info(f"  prob percentiles : P10={p_medp[10]:.2f} P25={p_medp[25]:.2f} P50={p_medp[50]:.2f} P75={p_medp[75]:.2f} P90={p_medp[90]:.2f}")
    if perss:
        logger.info(f"  pers percentiles : P10={p_pers[10]:.2f} P25={p_pers[25]:.2f} P50={p_pers[50]:.2f} P75={p_pers[75]:.2f} P90={p_pers[90]:.2f}")

    kept_stats = [s for s in stats if s["status"] == "KEPT"]
    ignored_stats = [s for s in stats if s["status"] == "IGNORED"]

    logger.info(f"  kept_clusters={len(kept_stats)} | ignored_clusters={len(ignored_stats)}")
    logger.info(
        f"  keep thresholds: size>={CFG['KEEP_MIN_SIZE']}, "
        f"median_prob>={CFG['KEEP_MIN_MEDIAN_PROB']:.2f}, "
        f"persistence>={CFG['KEEP_MIN_PERSIST']:.2f}"
    )

    per_cluster_class_counts: Dict[int, Dict[str, int]] = {}
    global_records: List[Tuple[str, str, str, str, str, str, str, str, str]] = []
    ignored_candidates: List[Tuple[float, int, int]] = []  # (prob, idx, lab)

    def write_cluster(srow):
        lab = srow["lab"]
        idxs = srow["idxs"]
        status = srow["status"]
        tag = f"cluster_{lab:03d}_size_{srow['size']}"

        cdir = out_clusters_root / tag
        cdir.mkdir(parents=True, exist_ok=True)

        hist: Dict[str, int] = {}
        recs = []

        for i in idxs:
            cname = classes[i]
            hist[cname] = hist.get(cname, 0) + 1

            src = _pick_export_src(files[i])
            dst = cdir / f"{cname}_{src.name}"
            link_or_copy(src, dst, logger)

            raw_name = Path(files[i]).name
            try:
                rel_path = str(Path(files[i]).relative_to(CFG["UNKNOWN_DIR"]))
            except Exception:
                rel_path = str(Path(files[i]))

            root_f, step_f, wafer_f, ymd, hms = parse_fields_from_filename(files[i])
            recs.append((cname, raw_name, rel_path, root_f, step_f, wafer_f, ymd, hms))
            global_records.append((f"{lab:03d}", status, cname, raw_name, rel_path, root_f, step_f, wafer_f, ymd, hms))

            if status == "IGNORED":
                ignored_candidates.append((float(probs[i]), int(i), int(lab)))

        per_cluster_class_counts[lab] = dict(hist)

        # medoid representative for KEPT only
        if len(idxs) > 0 and status == "KEPT":
            E = emb[idxs]
            ctr = E.mean(0, keepdims=True)
            d = np.linalg.norm(E - ctr, axis=1)
            local = int(np.argmin(d))
            rep_idx = int(idxs[local])
            src = _pick_export_src(files[rep_idx])
            rep_name = f"{tag}__{classes[rep_idx]}__medoid_dist{d[local]:.4f}{src.suffix}"
            link_or_copy(src, summary_dir / rep_name, logger)

        with open(cdir / f"{tag}.txt", "w", encoding="utf-8") as f:
            f.write(
                f"[Cluster {lab:03d}] status={status} size={srow['size']}, "
                f"median_prob={srow['median_prob']:.3f}, mean_prob={srow['mean_prob']:.3f}, std_prob={srow['std_prob']:.3f}, "
                f"persistence={srow['persistence']:.3f}, cohesion={srow['cohesion']:.3f}, margin={srow['margin']:.3f}, purity={srow['purity']:.3f}\n"
            )
            f.write("Class counts:\n")
            for k, v in sorted(hist.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  - {k}: {v}\n")

            f.write("\nclass\tfilename\trelpath\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
            for row in recs:
                f.write("\t".join(map(str, row)) + "\n")

    for s in stats:
        write_cluster(s)

    # Save top ignored samples
    if CFG["SAVE_IGNORED_SAMPLES"] and len(ignored_candidates) > 0:
        ignored_candidates.sort(key=lambda x: -x[0])
        topn = min(CFG["IGNORED_TOPN"], len(ignored_candidates))
        for rank, (p, idx, lab) in enumerate(ignored_candidates[:topn], 1):
            src = _pick_export_src(files[idx])
            cname = classes[idx]
            dst = ignored_dir / f"rank_{rank:03d}__cluster_{lab:03d}__prob_{p:.3f}__{cname}_{src.name}"
            link_or_copy(src, dst, logger)

    with open(Path(run_dir) / "clusters_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total samples: {len(files)}\n")
        f.write(f"HDBSCAN fit_time: {fit_sec:.1f}s\n")
        f.write(f"Clusters found (pre-filter): {len(uniq)}\n")
        f.write(f"Noise: {len(noise_idx)}\n")
        f.write(f"Kept clusters: {len(kept_stats)}\n")
        f.write(f"Ignored clusters: {len(ignored_stats)}\n\n")

        f.write("===== KEEP Thresholds =====\n")
        f.write(f"KEEP_MIN_SIZE = {CFG['KEEP_MIN_SIZE']}\n")
        f.write(f"KEEP_MIN_MEDIAN_PROB = {CFG['KEEP_MIN_MEDIAN_PROB']}\n")
        f.write(f"KEEP_MIN_PERSIST = {CFG['KEEP_MIN_PERSIST']}\n\n")

        f.write("===== Per-cluster summary =====\n")
        for s in stats:
            lab = s["lab"]
            f.write(
                f"[Cluster {lab:03d}] status={s['status']} size={s['size']}, "
                f"median_prob={s['median_prob']:.3f}, mean_prob={s['mean_prob']:.3f}, std_prob={s['std_prob']:.3f}, "
                f"persistence={s['persistence']:.3f}, cohesion={s['cohesion']:.3f}, margin={s['margin']:.3f}, purity={s['purity']:.3f}\n"
            )
            cnts = per_cluster_class_counts.get(lab, {})
            for cname, v in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  - {cname}: {v}\n")
            f.write("\n")

        f.write("===== Global list (tab-separated) =====\n")
        f.write("clusterno\tstatus\tclass\tfilename\trelpath\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
        for row in global_records:
            f.write("\t".join(map(str, row)) + "\n")

    with open(Path(run_dir) / "clusters_global_list.txt", "w", encoding="utf-8") as f:
        f.write("clusterno\tstatus\tclass\tfilename\trelpath\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
        for row in global_records:
            f.write("\t".join(map(str, row)) + "\n")

    return np.asarray(cids), clusterer, [int(s["lab"]) for s in kept_stats]


# =========================================================
# Save run info
# =========================================================
def save_run_info(run_dir: Path, device: torch.device, class_to_idx: Dict[str, int]):
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
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# =========================================================
# Main
# =========================================================
def main():
    seed_all(CFG["SEED"])

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(f"{CFG['OUTPUT_DIR']}_{RUN_TS}")
    logger = setup_logger(run_dir)

    logger.info("===== RUN START =====")
    logger.info(f"Device: {device}")
    logger.info("CFG:")
    logger.info(json.dumps(CFG, indent=2, ensure_ascii=False))

    # -------------------------
    # Data
    # -------------------------
    ds_u = ImageFolder(CFG["UNKNOWN_DIR"])
    ds_t = ImageFolder(CFG["TRAIN_DIR"])

    class_to_idx = {c: i for i, c in enumerate(ds_u.classes)}

    logger.info(f"[TRAIN_DIR classes] {ds_t.classes}")
    logger.info(f"[UNKNOWN_DIR classes] {ds_u.classes}")

    items_train_full = [(Path(p), -1, ds_t.classes[int(y)]) for (p, y) in ds_t.samples]
    logger.info(f"[Train from TRAIN_DIR] full_items={len(items_train_full)}")

    unknown_full = [(Path(p), -1, ds_u.classes[int(y)]) for (p, y) in ds_u.samples]
    items_eval = unknown_full

    if CFG.get("OVERLAY_DIR"):
        miss = 0
        for (p, _, _) in items_eval:
            try:
                rel = Path(p).relative_to(CFG["UNKNOWN_DIR"])
                cand = Path(CFG["OVERLAY_DIR"]) / rel
                if not cand.exists():
                    miss += 1
            except Exception:
                miss += 1
        logger.info(f"[Overlay coverage] missing={miss} / total={len(items_eval)}")

    logger.info(f"[Eval set] unknown_only={len(unknown_full)}")

    # -------------------------
    # Model / Optim
    # -------------------------
    model = CL(logger=logger).to(device).to(memory_format=torch.channels_last)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check FREEZE_BACKBONE / head setup.")

    opt = torch.optim.AdamW(
        trainable_params,
        lr=CFG["LR_HEAD"],
        weight_decay=CFG["WD"]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    neg_bank = QueueBank(CFG["PROJ_DIM"], CFG["QUEUE_SIZE"]).to(device) if CFG["USE_QUEUE"] else None

    # -------------------------
    # Train
    # -------------------------
    logger.info("[학습 시작]")

    for ep in range(1, CFG["EPOCHS"] + 1):
        if CFG["WARMUP_EPOCHS"] > 0 and ep <= CFG["WARMUP_EPOCHS"]:
            warm_lr = CFG["LR_HEAD"] * (ep / max(1, CFG["WARMUP_EPOCHS"]))
            for g in opt.param_groups:
                g["lr"] = warm_lr
            logger.info(f"[LR warmup] epoch {ep} lr={warm_lr:.2e}")

        r = float(CFG.get("TRAIN_SAMPLING_RATIO", 1.0))
        if 0.0 < r < 1.0:
            items_train = sample_per_class(items_train_full, r, CFG["SEED"] + ep)
            logger.info(f"[Train sampling@epoch{ep}] classwise ratio={r:.2f} -> {len(items_train)} items")
        else:
            items_train = items_train_full
            logger.info(f"[Train sampling@epoch{ep}] ratio=1.00 (full) -> {len(items_train)} items")

        train_ds = PairDS(items_train, tfm(True))
        ld = build_loader(train_ds, device, shuffle=True)

        t0 = time.time()
        model.train()

        run_g = 0.0
        run_q = 0.0
        run_l = 0.0
        n = 0

        total = len(ld)
        tick = max(1, total // 20)

        for it, (x1, x2, _, _, _) in enumerate(ld, 1):
            x1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            x2 = x2.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)

            amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                z1, fm1 = model(x1, return_local=True)
                z2, fm2 = model(x2, return_local=True)

                loss_g = info_nce_global(z1, z2, CFG["TEMP"])

                if CFG["USE_QUEUE"] and neg_bank is not None:
                    bank = neg_bank.get()
                    lq1 = info_nce_with_queue(z1, z2, bank, CFG["TEMP"], CFG["IGNORE_NEG_SIM"])
                    lq2 = info_nce_with_queue(z2, z1, bank, CFG["TEMP"], CFG["IGNORE_NEG_SIM"])
                    loss_q = 0.5 * (lq1 + lq2) * CFG["QUEUE_WEIGHT"]
                else:
                    loss_q = torch.tensor(0.0, device=z1.device)

                if CFG["USE_LOCAL"] and ((it % CFG["LOCAL_EVERY_N"]) == 0):
                    loss_l = info_nce_local_multi(
                        fm1, fm2,
                        anchors=CFG["LOCAL_ANCHORS"],
                        search=CFG["LOCAL_SEARCH"],
                        r=CFG["LOCAL_WINDOW"],
                        pos_topk=CFG["LOCAL_POS_TOPK"],
                        pos_min_sim=CFG["LOCAL_POS_MIN_SIM"],
                        t=CFG["TEMP"],
                        subbatch=CFG["LOCAL_SUBBATCH"]
                    ) * CFG["LOCAL_WEIGHT"]
                else:
                    loss_l = torch.tensor(0.0, device=z1.device)

                loss = loss_g + loss_q + loss_l

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                if CFG["USE_QUEUE"] and neg_bank is not None:
                    neg_bank.enqueue(z1)
                    neg_bank.enqueue(z2)

            run_g += float(loss_g.detach())
            run_q += float(loss_q.detach())
            run_l += float(loss_l.detach())
            n += 1

            if (it % tick) == 0 or it == total:
                logger.info(
                    f"  Epoch {ep}/{CFG['EPOCHS']} | {100.0 * it / total:5.1f}% ({it}/{total}) | "
                    f"G={run_g / n:.4f} Q={run_q / n:.4f} L={run_l / n:.4f}"
                )

        logger.info(
            f"Epoch {ep} done | G={run_g / max(1, n):.4f} "
            f"Q={run_q / max(1, n):.4f} L={run_l / max(1, n):.4f} | time={time.time() - t0:.1f}s"
        )

    # -------------------------
    # Save checkpoints
    # -------------------------
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict()}, ckpt_dir / "final_infer.pt")
    torch.save({
        "epoch": CFG["EPOCHS"],
        "model": model.state_dict(),
        "cfg": CFG,
        "class_to_idx": class_to_idx,
    }, ckpt_dir / "last_training.pt")

    logger.info(f"[저장] {(ckpt_dir / 'final_infer.pt').resolve()}")
    save_run_info(run_dir, device, class_to_idx)

    # -------------------------
    # Embedding + Clustering
    # -------------------------
    logger.info("[임베딩 추출]")
    emb, labs, files, clss = extract(model, items_eval, device, logger, log_progress=True, log_name="Embed")

    logger.info("[HDBSCAN + 저장]")
    cluster_labels, clusterer, kept_labels = cluster_and_save_hdbscan(
        emb, files, labs, clss, run_dir, logger
    )

    # Save centroids + clusterer for predict.py
    try:
        from common.centroids import save_cluster_centroids
        centroids_dir = Path(run_dir) / "centroids"
        centroids_dir.mkdir(parents=True, exist_ok=True)
        save_cluster_centroids(
            emb, cluster_labels, centroids_dir,
            clusterer=clusterer, kept_labels=kept_labels,
        )
        logger.info(f"[저장] centroids -> {centroids_dir}")
    except ImportError as e:
        logger.warning(f"centroids 저장 스킵 (common.centroids 없음): {e}")
    except Exception as e:
        logger.warning(f"centroids 저장 실패: {e}")

    # Generate cluster composite maps (KEPT cluster만)
    try:
        from cluster_composite import generate_cluster_composites
        composite_dir = Path(run_dir) / "cluster_summary" / "composite"
        generate_cluster_composites(
            run_dir=Path(run_dir),
            emb=emb,
            files=files,
            cluster_labels=cluster_labels,
            kept_labels=kept_labels,
            out_dir=composite_dir,
            n_per_cluster=10,
            logger=logger,
        )
    except ImportError as e:
        logger.warning(f"cluster_composite 스킵: {e}")
    except Exception as e:
        logger.warning(f"cluster_composite 실패: {e}")

    logger.info(f"[완료] 결과: {run_dir.resolve()}")
    logger.info("===== RUN END =====")


if __name__ == "__main__":
    main()