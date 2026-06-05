#!/usr/bin/env python3
"""Contrastive 단독 학습 — InfoNCE + MoCo Queue + NEG filter + HDBSCAN eval.

사용법:
    python scripts/train_contrastive.py
        → runs/<TS>_contrastive/ 폴더 생성

전제: CNN backbone 학습된 best_model.pth 또는 ImageNet FCMAE backbone.
"""
from __future__ import annotations

# ===================================================================
# === CONFIG (실행 시 이 부분만 수정) ===
# ===================================================================
# ★ 학습/eval 폴더 분리 (사용자 명시 260527 "클래스 보는건 치팅") ★
#   TRAIN_DATA_DIR : flat 또는 ImageFolder — label 무시하고 학습용
#   EVAL_DATA_DIR  : ImageFolder (class subdir) — cluster metric 측정용
TRAIN_DATA_DIR        = "E:/data/images/contrastive_train"  # ★ 절대규: 모든 이미지 E:/data/images/
EVAL_DATA_DIR         = "E:/data/images/contrastive_eval"   # ★ 절대규
ACTIVE_CLASSES_YAML   = None       # eval class subset (선택)
EXCLUDE_CLASSES       = {"classification", "classification_chips"}

# Backbone source — 우선순위:
#   1. CNN_RUN_DIR 지정 시 그 안의 cnn/best_model.pth
#   2. BACKBONE_CKPT 지정 시 그 path
#   3. WEIGHTS_DIR 의 ImageNet FCMAE pretrained
CNN_RUN_DIR           = None       # 예: "runs/260527_120000_cnn"
BACKBONE_CKPT         = None       # 또는 직접 path 지정
WEIGHTS_DIR           = "weights"
BACKBONE              = "convnextv2_base.fcmae_ft_in22k_in1k_384"
FREEZE_BACKBONE       = True       # ★ projection head 만 학습

OUTPUT_ROOT           = "runs"
TAG                   = "contrastive"

IMG_SIZE              = 512
PROJ_DIM              = 128
BATCH                 = 8
NUM_WORKERS           = 4
EPOCHS                = 20
WARMUP_EPOCHS         = 1
TRAIN_SAMPLING_RATIO  = 0.25       # 매 epoch 25% 무작위
LR_HEAD               = 1e-3
LR_MIN                = 1e-6
WEIGHT_DECAY          = 1e-6
NCE_TEMP              = 0.05
GRAD_CLIP             = 1.0
LABEL_SMOOTHING       = 0.02

# 4-tool recipe (Step 2b SOTA)
USE_QUEUE             = True
QUEUE_SIZE            = 4096
IGNORE_NEG_SIM        = 0.95       # NV-Retriever NEG filter
USE_LOCAL             = False      # ★ Local DenseCL OFF (4-tool 최종)
NECO_WEIGHT           = 0.2
NECO_TAU              = 0.1

# Anti-overfit
USE_EMA               = False
USE_AMP               = False
USE_MIXUP             = False

# HDBSCAN eval
MIN_CLUSTER_SIZE      = 12
MIN_SAMPLES           = 15
CLUSTER_SELECTION_METHOD = "leaf"
CLUSTER_SELECTION_EPSILON = 0.06

# Data caps (eval set)
PER_CLASS_CAP         = 500
NORMAL_CAP            = 2000
EVAL_IGNORE_CLASSES   = {"Normal"}   # Normal 은 background/noise pool — metric 계산 제외

SEED                  = 42

# Wrong/outlier 이미지 저장 (eval)
SAVE_WRONG_IMAGES     = True        # ignored class 제외, cluster-mismatch/noise defect 이미지 저장
SAVE_REPRESENTATIVES  = True        # cluster 대표 이미지 몇 장 저장
REPRESENTATIVES_PER_CLUSTER = 5
# ===================================================================

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import shutil
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    AddGaussianNoise,
    ensure_backbone_weights,
    log_stage_metric,
    make_run_dir,
    mask_palette_non_grade_to_white,
    resolve_path,
    snapshot_config,
    system_info,
)


def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def contrastive_lr_for_epoch(ep: int) -> float:
    """Warmup 후 cosine decay. DDP contrastive 와 같은 LR shape."""
    if WARMUP_EPOCHS > 0 and ep <= WARMUP_EPOCHS:
        return LR_HEAD * ep / max(1, WARMUP_EPOCHS)
    decay_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
    t = min(1.0, max(0.0, (ep - WARMUP_EPOCHS) / decay_epochs))
    return LR_MIN + 0.5 * (LR_HEAD - LR_MIN) * (1.0 + math.cos(math.pi * t))


# ---------- Dataset ----------
class SafeImageFolder(ImageFolder):
    """ImageFolder + EXCLUDE + ACTIVE_CLASSES + per-class cap + corrupted skip."""
    def __init__(self, root, transform=None, exclude=None,
                 active_classes=None, per_class_cap=None, normal_cap=None):
        self._exclude = exclude or set()
        self._active = set(active_classes) if active_classes else None
        self._per_cap = per_class_cap
        self._normal_cap = normal_cap
        super().__init__(root, transform=transform)
        # apply cap (sorted file order, deterministic)
        if self._per_cap or self._normal_cap:
            buckets = defaultdict(list)
            for path, lbl in self.samples:
                buckets[lbl].append((path, lbl))
            idx_to_class = {i: c for c, i in self.class_to_idx.items()}
            capped = []
            for lbl, items in sorted(buckets.items()):
                items.sort(key=lambda x: x[0])
                cls = idx_to_class[lbl]
                cap = self._normal_cap if (cls == "Normal" and self._normal_cap) else self._per_cap
                capped.extend(items[:cap] if cap else items)
            self.samples = capped
            self.targets = [t for _, t in capped]
            self.imgs = self.samples
            print(f"[CappedImageFolder] {root}: per_class={self._per_cap} normal={self._normal_cap} → total {len(capped)}")

    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        kept = [c for c in classes if c not in self._exclude]
        if self._active is not None:
            kept = [c for c in kept if c in self._active]
        return kept, {c: i for i, c in enumerate(kept)}

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            from PIL import Image
            with Image.open(path) as im:
                sample = mask_palette_non_grade_to_white(im).convert("RGB")
        except Exception as e:
            from PIL import Image as _PILImage
            print(f"[CORRUPT-SKIP] {path}: {type(e).__name__}", flush=True)
            sample = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class FlatImageDataset(Dataset):
    """Contrastive 학습용 — class label 무시하고 image 만 (recursive glob)."""
    EXTS = (".png", ".jpg", ".jpeg", ".bmp")
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths: list[Path] = []
        for ext in self.EXTS:
            self.paths.extend(self.root.rglob(f"*{ext}"))
        self.paths = sorted(self.paths)
        print(f"[FlatImageDataset] {self.root}: {len(self.paths)} images (class label 숨김)")

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        try:
            from PIL import Image
            with Image.open(p) as im:
                img = mask_palette_non_grade_to_white(im).convert("RGB")
        except Exception:
            from PIL import Image as _PILImage
            img = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, str(p)


class PairDataset(Dataset):
    """Two views of same image — for InfoNCE. Source = flat 또는 ImageFolder."""
    def __init__(self, paths: list, tfm):
        self.paths = list(paths)  # list of (path, label) or just path string
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        item = self.paths[i]
        path = item[0] if isinstance(item, tuple) else item
        try:
            from PIL import Image
            with Image.open(path) as im:
                img = mask_palette_non_grade_to_white(im).convert("RGB")
        except Exception:
            from PIL import Image as _PILImage
            img = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        return self.tfm(img), self.tfm(img), str(path)


def build_aug():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return T.Compose([
        T.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.94, 1.0), ratio=(1.0, 1.0)),
        T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        AddGaussianNoise(0.02),
        norm,
    ])


def build_eval_tf():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        norm,
    ])


# ---------- Model ----------
class ContrastiveModel(nn.Module):
    def __init__(self, backbone_name: str, proj_dim: int, freeze_backbone: bool,
                 backbone_ckpt: Path | None):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        # load weights
        if backbone_ckpt and Path(backbone_ckpt).exists():
            sd = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            m_sd = self.backbone.state_dict()
            compat = {k: v for k, v in sd.items()
                      if k in m_sd and m_sd[k].shape == v.shape}
            self.backbone.load_state_dict(compat, strict=False)
            print(f"[backbone] loaded {len(compat)} keys from {backbone_ckpt}")

        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[backbone] FROZEN — projection head only")

    def _forward_features(self, x):
        if hasattr(self.backbone, "forward_features"):
            fm = self.backbone.forward_features(x)
        else:
            fm = self.backbone(x)
        if fm.ndim == 2:
            fm = fm[:, :, None, None]
        return fm

    def _feature_map_nchw(self, fm):
        if fm.ndim != 4:
            return fm[:, :, None, None]
        feat_dim = self.backbone.num_features
        if fm.shape[1] == feat_dim:
            return fm
        if fm.shape[-1] == feat_dim:
            return fm.permute(0, 3, 1, 2).contiguous()
        return fm

    def _pool_features(self, fm):
        if hasattr(self.backbone, "forward_head"):
            return self.backbone.forward_head(fm, pre_logits=True)
        fm = self._feature_map_nchw(fm)
        return fm.mean(dim=(2, 3))

    def _project_patches(self, fm):
        fm = self._feature_map_nchw(fm)
        b, c, h, w = fm.shape
        patches = fm.permute(0, 2, 3, 1).reshape(b * h * w, c)
        z = F.normalize(self.proj(patches), dim=1)
        return z.view(b, h * w, -1)

    def forward(self, x, return_neco=False):
        fm = self._forward_features(x)
        f = self._pool_features(fm)
        z = F.normalize(self.proj(f), dim=1)
        if return_neco:
            return z, self._project_patches(fm)
        return z


class QueueBank:
    def __init__(self, dim: int, size: int, device):
        self.size = size
        self.buf = F.normalize(torch.randn(size, dim, device=device), dim=1)
        self.ptr = 0
    @torch.no_grad()
    def enqueue(self, z: torch.Tensor):
        b = z.size(0)
        end = self.ptr + b
        if end <= self.size:
            self.buf[self.ptr:end] = z.detach()
        else:
            self.buf[self.ptr:] = z[:self.size - self.ptr].detach()
            self.buf[:end - self.size] = z[self.size - self.ptr:].detach()
        self.ptr = end % self.size


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, queue: QueueBank | None,
                  temp: float, ignore_neg_sim: float):
    """InfoNCE with optional MoCo queue + NV-Retriever NEG filter."""
    b, d = z1.shape
    # similarity between z1 and (z2 anchors + queue)
    logits_pos = (z1 * z2).sum(1, keepdim=True) / temp     # (b, 1)
    neg_bank = z2.detach()
    if queue is not None:
        neg_bank = torch.cat([neg_bank, queue.buf], dim=0)  # (b + Q, d)
    sim = z1 @ neg_bank.T / temp                            # (b, b+Q)
    # NV-Retriever NEG filter: mask out near-positive negatives
    if ignore_neg_sim > 0:
        with torch.no_grad():
            cos_sim = z1 @ neg_bank.T                       # cosine in [-1, 1]
            mask = cos_sim > ignore_neg_sim
            # don't mask the positive (idx == self)
            for i in range(b):
                if i < neg_bank.size(0):
                    mask[i, i] = False
        sim = sim.masked_fill(mask, -1e9)
    diag = torch.arange(b, device=z1.device)
    if b <= neg_bank.size(0):
        sim[diag, diag] = -1e9
    logits = torch.cat([logits_pos, sim], dim=1)            # (b, 1 + b+Q)
    return masked_pos_cross_entropy(logits, LABEL_SMOOTHING)


def masked_pos_cross_entropy(logits: torch.Tensor, label_smoothing: float):
    """Positive is column 0. Label smoothing is assigned only to unmasked negatives."""
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    if label_smoothing <= 0:
        return F.cross_entropy(logits, labels)
    valid = logits > -1e8
    neg_valid = valid.clone()
    neg_valid[:, 0] = False
    neg_count = neg_valid.sum(dim=1)
    eps = torch.where(neg_count > 0,
                      torch.full_like(neg_count, float(label_smoothing), dtype=logits.dtype),
                      torch.zeros_like(neg_count, dtype=logits.dtype))
    logp = F.log_softmax(logits, dim=1)
    pos_loss = -(1.0 - eps) * logp[:, 0]
    neg_logp = logp.masked_fill(~neg_valid, 0.0).sum(dim=1)
    neg_loss = -eps * neg_logp / neg_count.clamp_min(1).to(logits.dtype)
    return (pos_loss + neg_loss).mean()


def neco_loss(p1: torch.Tensor, p2: torch.Tensor, tau: float):
    """B6 NeCo: patch-neighbor order consistency over shared projection patches."""
    if p1.ndim != 3 or p2.ndim != 3:
        raise ValueError("neco_loss expects [B, N, D] patch embeddings")
    n = p1.size(1)
    s1 = torch.bmm(p1, p1.transpose(1, 2))
    s2 = torch.bmm(p2, p2.transpose(1, 2))
    eye = torch.eye(n, device=p1.device, dtype=torch.bool).unsqueeze(0)
    s1 = s1.masked_fill(eye, -1e4)
    s2 = s2.masked_fill(eye, -1e4)
    log_p1 = F.log_softmax(s1 / tau, dim=-1)
    log_p2 = F.log_softmax(s2 / tau, dim=-1)
    p1_soft = log_p1.exp()
    p2_soft = log_p2.exp()
    kl_12 = (p1_soft * (log_p1 - log_p2)).sum(dim=-1).mean()
    kl_21 = (p2_soft * (log_p2 - log_p1)).sum(dim=-1).mean()
    return 0.5 * (kl_12 + kl_21)


# ---------- HDBSCAN eval ----------
def hdbscan_eval(embeddings: np.ndarray, true_labels: list[str], cfg: dict):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg["MIN_CLUSTER_SIZE"],
        min_samples=cfg["MIN_SAMPLES"],
        cluster_selection_method=cfg["CLUSTER_SELECTION_METHOD"],
        cluster_selection_epsilon=cfg["CLUSTER_SELECTION_EPSILON"],
        metric="euclidean",
        allow_single_cluster=False,
    )
    pred = clusterer.fit_predict(embeddings)
    labels_arr = np.array(true_labels)
    measured_mask = ~np.isin(labels_arr, list(EVAL_IGNORE_CLASSES))
    n_total = int(measured_mask.sum())
    n_total_all = len(pred)
    noise_mask = pred == -1
    n_noise = int((noise_mask & measured_mask).sum())
    noise_pct = n_noise / max(1, n_total) * 100

    # tier1: capture / Completeness / Homogeneity / AMI / ARI
    from sklearn.metrics import (adjusted_mutual_info_score,
                                 adjusted_rand_score,
                                 completeness_score,
                                 homogeneity_score)
    # filter noise and ignored classes out for metrics
    keep = (~noise_mask) & measured_mask
    pred_k = pred[keep]
    true_k = labels_arr[keep]
    classes = sorted(set(true_k))
    cl2i = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([cl2i[c] for c in true_k]) if classes else np.array([], dtype=int)

    # capture: per class, max fraction in any cluster (over clustered subset)
    cluster_cls = defaultdict(Counter)
    for p, c in zip(pred_k, true_k):
        cluster_cls[int(p)][c] += 1
    cls_total = Counter(labels_arr[measured_mask])  # over measured classes only
    capture = {}
    for cls, total in cls_total.items():
        max_in = max(
            (cnt for cl, ccnt in cluster_cls.items() for c, cnt in ccnt.items() if c == cls),
            default=0,
        )
        capture[cls] = max_in / total
    can_score = len(true_idx) > 0 and len(set(true_idx.tolist())) > 1 and len(set(pred_k.tolist())) > 1
    return {
        "n_total": int(n_total),
        "n_total_all": int(n_total_all),
        "n_ignored": int((~measured_mask).sum()),
        "ignored_classes": sorted(EVAL_IGNORE_CLASSES),
        "n_clustered": int(keep.sum()),
        "n_clusters": int(len(set(pred_k))),
        "noise_count": n_noise,
        "noise_pct": round(noise_pct, 2),
        "class_capture_rate": round(float(np.mean(list(capture.values()))), 4) if capture else 0.0,
        "completeness": round(float(completeness_score(true_idx, pred_k)), 4) if can_score else 0.0,
        "homogeneity": round(float(homogeneity_score(true_idx, pred_k)), 4) if can_score else 0.0,
        "ami": round(float(adjusted_mutual_info_score(true_idx, pred_k)), 4) if can_score else 0.0,
        "ari": round(float(adjusted_rand_score(true_idx, pred_k)), 4) if can_score else 0.0,
    }, pred


def save_cluster_representatives(cl_dir: Path, embeddings: np.ndarray, pred: np.ndarray,
                                 labels: list[str], paths: list[str], per_cluster: int) -> int:
    rep_dir = cl_dir / "representatives"
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0
    cluster_ids = sorted(int(c) for c in set(pred.tolist()) if int(c) != -1)
    for cl_id in cluster_ids:
        idx = np.where(pred == cl_id)[0]
        if len(idx) == 0:
            continue
        measured_idx = np.array([i for i in idx if labels[i] not in EVAL_IGNORE_CLASSES], dtype=int)
        if len(measured_idx) == 0:
            continue
        cnt = Counter(labels[i] for i in measured_idx)
        top_cls, top_n = cnt.most_common(1)[0]
        purity = top_n / len(measured_idx)
        center = embeddings[measured_idx].mean(axis=0)
        dist = np.linalg.norm(embeddings[measured_idx] - center[None, :], axis=1)
        order = measured_idx[np.argsort(dist)[:per_cluster]]
        sub = rep_dir / (
            f"cluster-{cl_id:03d}_class-{top_cls}_purity-{int(round(purity * 100))}pct_"
            f"n-{len(measured_idx)}_all-{len(idx)}"
        )
        sub.mkdir(parents=True, exist_ok=True)
        for rank_i, i in enumerate(order, 1):
            src = Path(paths[i])
            if not src.exists():
                continue
            dst = sub / f"rep-{rank_i:02d}_true-{labels[i]}_{src.name}"
            try:
                shutil.copy2(src, dst)
                saved += 1
                rows.append({
                    "cluster_id": cl_id,
                    "rank": rank_i,
                    "dominant_class": top_cls,
                    "cluster_purity": f"{purity:.6f}",
                    "cluster_size": len(measured_idx),
                    "cluster_size_all": len(idx),
                    "true_class": labels[i],
                    "distance_to_centroid": f"{float(np.linalg.norm(embeddings[i] - center)):.6f}",
                    "src": str(src),
                    "dst": str(dst),
                })
            except Exception:
                pass

    noise_idx = np.array([i for i in np.where(pred == -1)[0]
                          if labels[i] not in EVAL_IGNORE_CLASSES], dtype=int)[:per_cluster]
    if len(noise_idx):
        sub = rep_dir / "_noise"
        sub.mkdir(parents=True, exist_ok=True)
        for rank_i, i in enumerate(noise_idx, 1):
            src = Path(paths[i])
            if not src.exists():
                continue
            dst = sub / f"noise-{rank_i:02d}_true-{labels[i]}_{src.name}"
            try:
                shutil.copy2(src, dst)
                saved += 1
                rows.append({
                    "cluster_id": -1,
                    "rank": rank_i,
                    "dominant_class": "noise",
                    "cluster_purity": "",
                    "cluster_size": int(sum(labels[i] not in EVAL_IGNORE_CLASSES for i in np.where(pred == -1)[0])),
                    "cluster_size_all": int((pred == -1).sum()),
                    "true_class": labels[i],
                    "distance_to_centroid": "",
                    "src": str(src),
                    "dst": str(dst),
                })
            except Exception:
                pass

    import csv
    with (rep_dir / "representatives.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "rank", "dominant_class",
                                          "cluster_purity", "cluster_size", "cluster_size_all",
                                          "true_class", "distance_to_centroid", "src", "dst"])
        w.writeheader()
        w.writerows(rows)
    return saved


# ---------- Main ----------
def main():
    seed_all(SEED)
    run_dir = make_run_dir(OUTPUT_ROOT, TAG)
    print(f"[run_dir] {run_dir.resolve()}")
    cl_dir = run_dir / "contrastive"; cl_dir.mkdir(exist_ok=True)

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
    cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
    snapshot_config(run_dir, cfg)
    system_info(run_dir)

    # backbone path resolution
    backbone_ckpt = None
    if CNN_RUN_DIR:
        cand = Path(CNN_RUN_DIR) / "cnn" / "best_model.pth"
        if cand.exists(): backbone_ckpt = cand
    if backbone_ckpt is None and BACKBONE_CKPT:
        if Path(BACKBONE_CKPT).exists(): backbone_ckpt = Path(BACKBONE_CKPT)
    if backbone_ckpt is None:
        backbone_ckpt = ensure_backbone_weights(WEIGHTS_DIR, BACKBONE)
    print(f"[backbone source] {backbone_ckpt}")

    active_classes = None
    if ACTIVE_CLASSES_YAML:
        import yaml
        with open(ACTIVE_CLASSES_YAML) as f:
            active_classes = yaml.safe_load(f).get("classes")

    # ========== 학습용 dataset (TRAIN_DATA_DIR — flat, class 무시) ==========
    train_dir = resolve_path(TRAIN_DATA_DIR)
    eval_dir = resolve_path(EVAL_DATA_DIR)
    if not train_dir.exists():
        raise SystemExit(f"TRAIN_DATA_DIR not found: {train_dir}\n"
                         f"  python scripts/generate_data.py && python scripts/_split_data.py")
    if not eval_dir.exists():
        raise SystemExit(f"EVAL_DATA_DIR not found: {eval_dir}")
    eval_tf = build_eval_tf()
    train_aug = build_aug()
    train_ds = FlatImageDataset(str(train_dir), transform=None)
    print(f"[train] {len(train_ds)} images from {train_dir} (class label not used)")

    # ========== eval dataset (EVAL_DATA_DIR — ImageFolder, class 보존) ==========
    eval_base = SafeImageFolder(str(eval_dir), transform=eval_tf,
                                exclude=EXCLUDE_CLASSES, active_classes=active_classes,
                                per_class_cap=PER_CLASS_CAP, normal_cap=NORMAL_CAP)
    classes = eval_base.classes
    print(f"[eval] {len(eval_base)} samples, {len(classes)} classes from {EVAL_DATA_DIR}")
    (run_dir / "classes.json").write_text(
        json.dumps({"classes": classes, "class_to_idx": eval_base.class_to_idx,
                    "train_dir": TRAIN_DATA_DIR, "eval_dir": EVAL_DATA_DIR}, indent=2),
        encoding="utf-8")

    # ---------- Training ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveModel(BACKBONE, PROJ_DIM, FREEZE_BACKBONE, backbone_ckpt).to(device)
    queue = QueueBank(PROJ_DIM, QUEUE_SIZE, device) if USE_QUEUE else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    log_stage_metric(run_dir, "contrastive_setup", {
        "backbone": BACKBONE,
        "backbone_source": str(backbone_ckpt),
        "freeze_backbone": FREEZE_BACKBONE,
        "proj_dim": PROJ_DIM,
        "lr_head": LR_HEAD,
        "lr_min": LR_MIN,
        "lr_schedule": "warmup_cosine_epoch",
        "queue_size": QUEUE_SIZE if USE_QUEUE else 0,
        "ignore_neg_sim": IGNORE_NEG_SIM,
        "use_local": USE_LOCAL,
        "neco_weight": NECO_WEIGHT,
        "neco_tau": NECO_TAU,
        "recipe": "B6: Global InfoNCE + MoCo Queue + NEG + NeCo, no Local",
        "n_train": len(train_ds),
        "n_eval": len(eval_base),
        "n_classes_eval": len(classes),
        "train_dir": TRAIN_DATA_DIR,
        "eval_dir": EVAL_DATA_DIR,
    }, notes="train flat (label hidden) + eval ImageFolder (class for metric) — wafer disjoint")

    print("[train] start")
    history = []
    train_paths_all = [str(p) for p in train_ds.paths]
    for ep in range(1, EPOCHS + 1):
        # epoch 25% sampling
        r = TRAIN_SAMPLING_RATIO
        if 0 < r < 1:
            sub = random.sample(train_paths_all, max(1, int(len(train_paths_all) * r)))
        else:
            sub = train_paths_all
        lr = contrastive_lr_for_epoch(ep)
        for g in opt.param_groups: g["lr"] = lr
        pair_ds = PairDataset(sub, train_aug)
        ld = DataLoader(pair_ds, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

        model.train()
        run_loss = 0.0; run_nce = 0.0; run_neco = 0.0; n = 0
        t0 = time.time()
        for it, batch in enumerate(ld, 1):
            x1, x2, _ = batch
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad()
            bs = x1.size(0)
            if NECO_WEIGHT > 0:
                z_cat, p_cat = model(torch.cat([x1, x2], dim=0), return_neco=True)
                p1, p2 = p_cat[:bs], p_cat[bs:]
                loss_neco = neco_loss(p1, p2, NECO_TAU)
            else:
                z_cat = model(torch.cat([x1, x2], dim=0))
                loss_neco = z_cat.new_zeros(())
            z1, z2 = z_cat[:bs], z_cat[bs:]
            loss_nce = info_nce_loss(z1, z2, queue, NCE_TEMP, IGNORE_NEG_SIM)
            loss = loss_nce + (NECO_WEIGHT * loss_neco)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            if queue is not None:
                queue.enqueue(z2)
            run_loss += loss.item() * x1.size(0)
            run_nce += loss_nce.item() * x1.size(0)
            run_neco += loss_neco.item() * x1.size(0)
            n += x1.size(0)
            if it % max(1, len(ld) // 10) == 0:
                print(f"  Ep {ep}/{EPOCHS} {it/len(ld)*100:>5.1f}% | "
                      f"loss={run_loss/n:.4f} nce={run_nce/n:.4f} "
                      f"neco={run_neco/n:.4f} lr={lr:.2e}", flush=True)
        ep_loss = run_loss / max(1, n)
        ep_nce = run_nce / max(1, n)
        ep_neco = run_neco / max(1, n)
        print(f"  Ep {ep}/{EPOCHS} DONE loss={ep_loss:.4f} "
              f"nce={ep_nce:.4f} neco={ep_neco:.4f} "
              f"time={time.time()-t0:.0f}s", flush=True)
        history.append({"epoch": ep, "loss": ep_loss, "nce": ep_nce,
                        "neco": ep_neco, "lr": lr})

    # save
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": eval_base.class_to_idx,
        "config": cfg,
        "backbone_source": str(backbone_ckpt),
    }, cl_dir / "best_model.pt")
    (cl_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ---------- Eval — embed + HDBSCAN ----------
    print("[eval] embedding...")
    model.eval()
    eval_loader = DataLoader(eval_base, batch_size=BATCH * 4, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    all_z = []; all_label = []; all_path = []
    t0 = time.time()
    with torch.no_grad():
        for imgs, lbls, paths in eval_loader:
            imgs = imgs.to(device, non_blocking=True)
            z = model(imgs).cpu().numpy()
            all_z.append(z)
            all_label.extend([classes[l] for l in lbls.tolist()])
            all_path.extend(paths)
    embeddings = np.concatenate(all_z, axis=0)
    np.save(cl_dir / "embeddings.npy", embeddings)
    (cl_dir / "paths.json").write_text(
        json.dumps({"paths": all_path, "labels": all_label}, indent=2), encoding="utf-8")
    print(f"[eval] embed shape={embeddings.shape}  time={time.time()-t0:.0f}s")

    print("[eval] HDBSCAN clustering...")
    tier1, pred = hdbscan_eval(embeddings, all_label, {
        "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,
        "MIN_SAMPLES": MIN_SAMPLES,
        "CLUSTER_SELECTION_METHOD": CLUSTER_SELECTION_METHOD,
        "CLUSTER_SELECTION_EPSILON": CLUSTER_SELECTION_EPSILON,
    })

    # save cluster assignments
    with open(cl_dir / "clusters_global_list.txt", "w", encoding="utf-8") as f:
        f.write("cluster_id\ttrue_class\tpath\n")
        for p, c, ph in zip(pred, all_label, all_path):
            f.write(f"{int(p)}\t{c}\t{ph}\n")

    (cl_dir / "tier1.json").write_text(json.dumps(tier1, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eval] tier1 = {json.dumps(tier1, indent=2)}")

    if SAVE_REPRESENTATIVES:
        n_rep = save_cluster_representatives(
            cl_dir, embeddings, pred, all_label, all_path, REPRESENTATIVES_PER_CLUSTER)
        tier1["n_representatives_saved"] = n_rep
        (cl_dir / "tier1.json").write_text(json.dumps(tier1, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [representatives] saved {n_rep} to {cl_dir / 'representatives'}")

    # ---------- Cluster outlier 이미지 저장 ----------
    # 각 cluster 의 dominant class 식별 → 그 cluster 안 minor class 이미지 = wrong
    # 또는 noise (cluster_id=-1) 도 wrong 으로 간주
    if SAVE_WRONG_IMAGES:
        wrong_dir = cl_dir / "wrong"
        if wrong_dir.exists():
            import shutil as _sh
            _sh.rmtree(wrong_dir)
        wrong_dir.mkdir(parents=True, exist_ok=True)
        # cluster → dominant measured class + dominant fraction (Normal 등 ignored 제외)
        cluster_dominant: dict[int, tuple[str, float]] = {}
        cluster_cls_cnt = defaultdict(Counter)
        for p, c in zip(pred, all_label):
            if c in EVAL_IGNORE_CLASSES:
                continue
            cluster_cls_cnt[int(p)][c] += 1
        for cl_id, cnt in cluster_cls_cnt.items():
            if cl_id == -1:
                continue
            total = sum(cnt.values())
            top_cls, top_n = cnt.most_common(1)[0]
            cluster_dominant[cl_id] = (top_cls, top_n / total)

        import csv
        import shutil as _sh
        n_wrong = 0
        rows = []
        for p, true_cls, path in zip(pred, all_label, all_path):
            if true_cls in EVAL_IGNORE_CLASSES:
                continue
            cl_id = int(p)
            if cl_id == -1:
                # noise — assign as wrong vs true class
                cluster_pred_cls = "noise"
                purity = None
                purity_tag = "na"
                is_wrong = True
            else:
                if cl_id not in cluster_dominant:
                    continue
                cluster_pred_cls, purity = cluster_dominant[cl_id]
                purity_tag = f"{int(round(purity * 100))}pct"
                is_wrong = (cluster_pred_cls != true_cls)
            if not is_wrong: continue
            sub = wrong_dir / true_cls
            sub.mkdir(exist_ok=True)
            base = Path(path).name
            dst = sub / (
                f"true-{true_cls}___clusterpred-{cluster_pred_cls}_"
                f"cluster-{cl_id}_clusterpurity-{purity_tag}_{base}"
            )
            try:
                _sh.copy2(path, dst)
                n_wrong += 1
                rows.append({
                    "true_class": true_cls,
                    "cluster_pred_class": cluster_pred_cls,
                    "cluster_id": cl_id,
                    "cluster_purity": "" if purity is None else f"{purity:.6f}",
                    "src": str(path),
                    "dst": str(dst),
                })
            except Exception:
                pass
        csv_path = wrong_dir / "wrong_clusters.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["true_class", "cluster_pred_class",
                                              "cluster_id", "cluster_purity",
                                              "src", "dst"])
            w.writeheader()
            w.writerows(rows)
        print(f"  [wrong] saved {n_wrong} cluster-mismatched images to {wrong_dir}")
        tier1["n_wrong_saved"] = n_wrong

    log_stage_metric(run_dir, "contrastive_eval", tier1,
                     notes=f"backbone_freeze={FREEZE_BACKBONE}, queue={USE_QUEUE}, neg_filter={IGNORE_NEG_SIM}, "
                           f"train_dir={TRAIN_DATA_DIR}, eval_dir={EVAL_DATA_DIR}")

    print(f"\n[OUT] {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    main()
