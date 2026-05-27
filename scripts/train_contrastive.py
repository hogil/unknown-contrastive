#!/usr/bin/env python3
"""Contrastive 단독 학습 — InfoNCE + MoCo Queue + NEG filter + HDBSCAN eval.

사용법:
    python scripts/train_contrastive.py
        → runs/<TS>_contrastive/ 폴더 생성

전제: CNN backbone 학습된 best_model.pth 또는 ImageNet FCMAE backbone.
"""
# ===================================================================
# === CONFIG (실행 시 이 부분만 수정) ===
# ===================================================================
# ★ 학습/eval 폴더 분리 (사용자 명시 260527 "클래스 보는건 치팅") ★
#   TRAIN_DATA_DIR : flat 또는 ImageFolder — label 무시하고 학습용
#   EVAL_DATA_DIR  : ImageFolder (class subdir) — cluster metric 측정용
TRAIN_DATA_DIR        = "data/images/contrastive_train"   # 프로젝트 상대, flat
EVAL_DATA_DIR         = "data/images/contrastive_eval"    # 프로젝트 상대, ImageFolder
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

IMG_SIZE              = 384
PROJ_DIM              = 128
BATCH                 = 8
NUM_WORKERS           = 4
EPOCHS                = 5
WARMUP_EPOCHS         = 1
TRAIN_SAMPLING_RATIO  = 0.25       # 매 epoch 25% 무작위
LR_HEAD               = 1e-3
WEIGHT_DECAY          = 1e-6
NCE_TEMP              = 0.07
GRAD_CLIP             = 1.0
LABEL_SMOOTHING       = 0.02

# 4-tool recipe (Step 2b SOTA)
USE_QUEUE             = True
QUEUE_SIZE            = 4096
IGNORE_NEG_SIM        = 0.72       # NV-Retriever NEG filter
USE_LOCAL             = False      # ★ Local DenseCL OFF (4-tool 최종)
NECO_WEIGHT           = 0.2

# Anti-overfit
USE_EMA               = False
USE_AMP               = False
USE_MIXUP             = False

# HDBSCAN eval
MIN_CLUSTER_SIZE      = 12
MIN_SAMPLES           = 3
CLUSTER_SELECTION_METHOD = "eom"
CLUSTER_SELECTION_EPSILON = 0.0

# Data caps (eval set)
PER_CLASS_CAP         = 500
NORMAL_CAP            = 2000

SEED                  = 42

# Wrong/outlier 이미지 저장 (eval)
SAVE_WRONG_IMAGES     = True        # cluster outlier 이미지를 wrong/<true>/<true>_<pred_cluster_class>_<pct>%_<basename>.png
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
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    AddGaussianNoise,
    ensure_backbone_weights,
    log_stage_metric,
    make_run_dir,
    resolve_path,
    snapshot_config,
    system_info,
)


def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


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
            sample = self.loader(path)
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
            img = Image.open(p).convert("RGB")
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
            img = Image.open(path).convert("RGB")
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

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)


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
    logits = torch.cat([logits_pos, sim], dim=1)            # (b, 1 + b+Q)
    labels = torch.zeros(b, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)


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
    n_total = len(pred)
    noise_mask = pred == -1
    n_noise = int(noise_mask.sum())
    noise_pct = n_noise / n_total * 100

    # tier1: capture / Completeness / Homogeneity / AMI / ARI
    from sklearn.metrics import (adjusted_mutual_info_score,
                                 adjusted_rand_score,
                                 completeness_score,
                                 homogeneity_score)
    # filter noise out for metrics
    keep = ~noise_mask
    pred_k = pred[keep]
    true_k = np.array(true_labels)[keep]
    classes = sorted(set(true_k))
    cl2i = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([cl2i[c] for c in true_k])

    # capture: per class, max fraction in any cluster (over clustered subset)
    cluster_cls = defaultdict(Counter)
    for p, c in zip(pred_k, true_k):
        cluster_cls[int(p)][c] += 1
    cls_total = Counter(true_labels)  # over all (incl noise)
    capture = {}
    for cls, total in cls_total.items():
        max_in = max(
            (cnt for cl, ccnt in cluster_cls.items() for c, cnt in ccnt.items() if c == cls),
            default=0,
        )
        capture[cls] = max_in / total
    return {
        "n_total": int(n_total),
        "n_clustered": int(keep.sum()),
        "n_clusters": int(len(set(pred_k))),
        "noise_count": n_noise,
        "noise_pct": round(noise_pct, 2),
        "class_capture_rate": round(float(np.mean(list(capture.values()))), 4),
        "completeness": round(float(completeness_score(true_idx, pred_k)), 4),
        "homogeneity": round(float(homogeneity_score(true_idx, pred_k)), 4),
        "ami": round(float(adjusted_mutual_info_score(true_idx, pred_k)), 4),
        "ari": round(float(adjusted_rand_score(true_idx, pred_k)), 4),
    }, pred


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
        "queue_size": QUEUE_SIZE if USE_QUEUE else 0,
        "ignore_neg_sim": IGNORE_NEG_SIM,
        "neco_weight": NECO_WEIGHT,
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
        # warmup
        lr = LR_HEAD * min(1.0, ep / max(1, WARMUP_EPOCHS))
        for g in opt.param_groups: g["lr"] = lr
        pair_ds = PairDataset(sub, train_aug)
        ld = DataLoader(pair_ds, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

        model.train()
        run_loss = 0.0; n = 0
        t0 = time.time()
        for it, batch in enumerate(ld, 1):
            x1, x2, _ = batch
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad()
            z1 = model(x1); z2 = model(x2)
            loss = info_nce_loss(z1, z2, queue, NCE_TEMP, IGNORE_NEG_SIM)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            if queue is not None:
                queue.enqueue(z2)
            run_loss += loss.item() * x1.size(0); n += x1.size(0)
            if it % max(1, len(ld) // 10) == 0:
                print(f"  Ep {ep}/{EPOCHS} {it/len(ld)*100:>5.1f}% | loss={run_loss/n:.4f} lr={lr:.2e}", flush=True)
        ep_loss = run_loss / max(1, n)
        print(f"  Ep {ep}/{EPOCHS} DONE loss={ep_loss:.4f} time={time.time()-t0:.0f}s", flush=True)
        history.append({"epoch": ep, "loss": ep_loss, "lr": lr})

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

    # ---------- Cluster outlier 이미지 저장 ----------
    # 각 cluster 의 dominant class 식별 → 그 cluster 안 minor class 이미지 = wrong
    # 또는 noise (cluster_id=-1) 도 wrong 으로 간주
    if SAVE_WRONG_IMAGES:
        wrong_dir = cl_dir / "wrong"
        if wrong_dir.exists():
            import shutil as _sh
            _sh.rmtree(wrong_dir)
        wrong_dir.mkdir(parents=True, exist_ok=True)
        # cluster → dominant class + dominant fraction
        cluster_dominant: dict[int, tuple[str, float]] = {}
        cluster_cls_cnt = defaultdict(Counter)
        for p, c in zip(pred, all_label):
            cluster_cls_cnt[int(p)][c] += 1
        for cl_id, cnt in cluster_cls_cnt.items():
            total = sum(cnt.values())
            top_cls, top_n = cnt.most_common(1)[0]
            cluster_dominant[cl_id] = (top_cls, top_n / total)

        import shutil as _sh
        n_wrong = 0
        for p, true_cls, path in zip(pred, all_label, all_path):
            cl_id = int(p)
            if cl_id == -1:
                # noise — assign as wrong vs true class
                pred_cls = "noise"
                pct = 0
                is_wrong = True
            else:
                pred_cls, frac = cluster_dominant[cl_id]
                pct = int(round(frac * 100))
                is_wrong = (pred_cls != true_cls)
            if not is_wrong: continue
            sub = wrong_dir / true_cls
            sub.mkdir(exist_ok=True)
            base = Path(path).name
            dst = sub / f"{true_cls}_{pred_cls}_{pct}%_{base}"
            try:
                _sh.copy2(path, dst)
                n_wrong += 1
            except Exception:
                pass
        print(f"  [wrong] saved {n_wrong} cluster-mismatched images to {wrong_dir}")
        tier1["n_wrong_saved"] = n_wrong

    log_stage_metric(run_dir, "contrastive_eval", tier1,
                     notes=f"backbone_freeze={FREEZE_BACKBONE}, queue={USE_QUEUE}, neg_filter={IGNORE_NEG_SIM}, "
                           f"train_dir={TRAIN_DATA_DIR}, eval_dir={EVAL_DATA_DIR}")

    print(f"\n[OUT] {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    main()
