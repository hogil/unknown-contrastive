#!/usr/bin/env python3
"""Contrastive 학습 DDP — multi-GPU (NCCL).

사용:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_contrastive_ddp.py
"""
from __future__ import annotations

# ===================================================================
# === CONFIG ===
# ===================================================================
TRAIN_DATA_DIR        = "E:/data/images/contrastive_train"  # ★ 절대규. 콤마구분 다중 가능: "a,b,c"
EVAL_DATA_DIR         = "E:/data/images/contrastive_eval"   # ★ 절대규. 콤마구분 다중(class 통합) 가능
ACTIVE_CLASSES_YAML   = None
EXCLUDE_CLASSES       = {"classification", "classification_chips"}

CNN_RUN_DIR           = None
BACKBONE_CKPT         = None
WEIGHTS_DIR           = "weights"
BACKBONE              = "convnextv2_base.fcmae_ft_in22k_in1k_384"
FREEZE_BACKBONE       = True

OUTPUT_ROOT           = "runs"
TAG                   = "contrastive_ddp"

IMG_SIZE              = 384
PROJ_DIM              = 128
BATCH_PER_GPU         = 8
NUM_WORKERS_PER_GPU   = None         # None = auto: os.cpu_count() // world_size (환경 코어 전부 활용)
EPOCHS                = 5
WARMUP_EPOCHS         = 1
TRAIN_SAMPLING_RATIO  = 0.25
LR_HEAD               = 1e-3
WEIGHT_DECAY          = 1e-6
NCE_TEMP              = 0.07
GRAD_CLIP             = 1.0
LABEL_SMOOTHING       = 0.02

USE_QUEUE             = True
QUEUE_SIZE            = 4096
IGNORE_NEG_SIM        = 0.72
USE_LOCAL             = False
NECO_WEIGHT           = 0.2

USE_EMA               = False
USE_AMP               = False

MIN_CLUSTER_SIZE      = 12
MIN_SAMPLES           = 3
CLUSTER_SELECTION_METHOD = "eom"
CLUSTER_SELECTION_EPSILON = 0.0

PER_CLASS_CAP         = 500
NORMAL_CAP            = 2000

SEED                  = 42
SAVE_WRONG_IMAGES     = True
SAVE_REPRESENTATIVES   = True
REPRESENTATIVES_PER_CLUSTER = 5
# ===================================================================

import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
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
from _ddp_utils import (
    all_gather_concat,
    all_reduce_avg,
    cleanup_ddp,
    is_main,
    launch_ddp,
    setup_ddp,
)

# ★ CLI 옵션은 env 로 전달 (mp.spawn 자식은 module 을 새로 import 하므로 module-level 에서 읽어야 상속됨)
BACKBONE_CKPT = os.environ.get("CL_BACKBONE_CKPT") or BACKBONE_CKPT
CNN_RUN_DIR   = os.environ.get("CL_CNN_RUN_DIR") or CNN_RUN_DIR
TAG           = os.environ.get("CL_TAG") or TAG
if os.environ.get("CL_EPOCHS"):
    EPOCHS = int(os.environ["CL_EPOCHS"])
if os.environ.get("CL_BATCH_PER_GPU"):
    BATCH_PER_GPU = int(os.environ["CL_BATCH_PER_GPU"])
TRAIN_DATA_DIR = os.environ.get("CL_TRAIN_DIRS") or TRAIN_DATA_DIR   # 콤마구분 다중 가능
EVAL_DATA_DIR  = os.environ.get("CL_EVAL_DIRS") or EVAL_DATA_DIR


def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------- Dataset ----------
class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None, exclude=None,
                 active_classes=None, per_class_cap=None, normal_cap=None):
        self._exclude = exclude or set()
        self._active = set(active_classes) if active_classes else None
        self._per_cap = per_class_cap
        self._normal_cap = normal_cap
        super().__init__(root, transform=transform)
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
        except Exception:
            from PIL import Image as _PILImage
            sample = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class MultiRootImageFolder(Dataset):
    """여러 eval 폴더의 <class>/<img> 를 ★통합 class_to_idx★ 로 묶음 (단일 root 도 동작).
       SafeImageFolder 와 동일하게 (img, target, path) 반환. class = 모든 root subdir 합집합."""
    def __init__(self, roots, transform=None, exclude=None,
                 active_classes=None, per_class_cap=None, normal_cap=None):
        from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
        if isinstance(roots, (str, Path)):
            roots = [roots]
        roots = [Path(r) for r in roots]
        exclude = exclude or set()
        active = set(active_classes) if active_classes else None
        cls_set = set()
        for r in roots:
            if not r.exists():
                raise SystemExit(f"EVAL dir not found: {r}")
            for d in sorted(r.iterdir()):
                if d.is_dir() and d.name not in exclude and (active is None or d.name in active):
                    cls_set.add(d.name)
        self.classes = sorted(cls_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        exts = set(IMG_EXTENSIONS)
        buckets = defaultdict(list)
        for r in roots:
            for c in self.classes:
                cdir = r / c
                if not cdir.is_dir():
                    continue
                for p in cdir.rglob("*"):
                    if p.suffix.lower() in exts:
                        buckets[self.class_to_idx[c]].append((str(p), self.class_to_idx[c]))
        idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        samples = []
        for lbl in sorted(buckets):
            items = sorted(buckets[lbl], key=lambda x: x[0])
            cls = idx_to_class[lbl]
            cap = normal_cap if (cls == "Normal" and normal_cap) else per_class_cap
            samples.extend(items[:cap] if cap else items)
        self.samples = samples
        self.targets = [t for _, t in samples]
        self.transform = transform
        self.loader = default_loader

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception:
            from PIL import Image as _PILImage
            sample = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class FlatPairDataset(Dataset):
    """flat folder(s) recursive glob → two views (no class label). roots = str 또는 list."""
    EXTS = (".png", ".jpg", ".jpeg", ".bmp")
    def __init__(self, roots, tfm):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        paths = []
        for r in roots:
            for ext in self.EXTS:
                paths.extend(Path(r).rglob(f"*{ext}"))
        self.paths = sorted(set(paths))
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            from PIL import Image
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            from PIL import Image as _PILImage
            img = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        return self.tfm(img), self.tfm(img)


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
        cnt = Counter(labels[i] for i in idx)
        top_cls, top_n = cnt.most_common(1)[0]
        purity = top_n / len(idx)
        center = embeddings[idx].mean(axis=0)
        dist = np.linalg.norm(embeddings[idx] - center[None, :], axis=1)
        order = idx[np.argsort(dist)[:per_cluster]]
        sub = rep_dir / f"cluster-{cl_id:03d}_class-{top_cls}_purity-{int(round(purity * 100))}pct_n-{len(idx)}"
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
                    "cluster_size": len(idx),
                    "true_class": labels[i],
                    "distance_to_centroid": f"{float(np.linalg.norm(embeddings[i] - center)):.6f}",
                    "src": str(src),
                    "dst": str(dst),
                })
            except Exception:
                pass

    noise_idx = np.where(pred == -1)[0][:per_cluster]
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
                    "cluster_size": int((pred == -1).sum()),
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
                                          "cluster_purity", "cluster_size", "true_class",
                                          "distance_to_centroid", "src", "dst"])
        w.writeheader()
        w.writerows(rows)
    return saved


def build_aug():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return T.Compose([
        T.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.94, 1.0), ratio=(1.0, 1.0)),
        T.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ToTensor(),
        AddGaussianNoise(0.02), norm,
    ])


def build_eval_tf():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])


class ContrastiveModel(nn.Module):
    def __init__(self, backbone_name, proj_dim, freeze_backbone, backbone_ckpt):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        if backbone_ckpt and Path(backbone_ckpt).exists():
            sd = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
            m_sd = self.backbone.state_dict()
            compat = {k: v for k, v in sd.items()
                      if k in m_sd and m_sd[k].shape == v.shape}
            self.backbone.load_state_dict(compat, strict=False)
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)


class QueueBank:
    def __init__(self, dim, size, device):
        self.size = size
        self.buf = F.normalize(torch.randn(size, dim, device=device), dim=1)
        self.ptr = 0
    @torch.no_grad()
    def enqueue(self, z):
        b = z.size(0); end = self.ptr + b
        if end <= self.size:
            self.buf[self.ptr:end] = z.detach()
        else:
            self.buf[self.ptr:] = z[:self.size - self.ptr].detach()
            self.buf[:end - self.size] = z[self.size - self.ptr:].detach()
        self.ptr = end % self.size


def info_nce_loss(z1, z2, queue, temp, ignore_neg_sim, label_smoothing):
    b = z1.size(0)
    logits_pos = (z1 * z2).sum(1, keepdim=True) / temp
    neg_bank = z2.detach()
    if queue is not None:
        neg_bank = torch.cat([neg_bank, queue.buf], dim=0)
    sim = z1 @ neg_bank.T / temp
    if ignore_neg_sim > 0:
        with torch.no_grad():
            cos = z1 @ neg_bank.T
            mask = cos > ignore_neg_sim
            for i in range(b):
                if i < neg_bank.size(0): mask[i, i] = False
        sim = sim.masked_fill(mask, -1e9)
    logits = torch.cat([logits_pos, sim], dim=1)
    labels = torch.zeros(b, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def train_worker(rank, world_size):
    setup_ddp(rank, world_size)
    seed_all(SEED + rank)
    import os
    nw = NUM_WORKERS_PER_GPU if NUM_WORKERS_PER_GPU is not None else max(1, (os.cpu_count() or 8) // world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        run_dir = make_run_dir(OUTPUT_ROOT, TAG)
        run_dir_str = str(run_dir)
    else:
        run_dir_str = ""
    obj = [run_dir_str]; dist.broadcast_object_list(obj, src=0)
    run_dir = Path(obj[0])
    cl_dir = run_dir / "contrastive"
    if is_main(rank):
        cl_dir.mkdir(exist_ok=True)
        print(f"[run_dir] {run_dir.resolve()}  (DDP world_size={world_size})")
        cfg = {k: v for k, v in globals().items()
               if k.isupper() and not k.startswith("_")
               and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
        cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
        cfg["WORLD_SIZE"] = world_size
        snapshot_config(run_dir, cfg)
        system_info(run_dir)

    # backbone
    bp = None
    if CNN_RUN_DIR:
        c = Path(CNN_RUN_DIR) / "cnn" / "best_model.pth"
        if c.exists(): bp = c
    if bp is None and BACKBONE_CKPT and Path(BACKBONE_CKPT).exists():
        bp = Path(BACKBONE_CKPT)
    if bp is None:
        bp = ensure_backbone_weights(WEIGHTS_DIR, BACKBONE)
    if is_main(rank): print(f"[backbone] {bp}")
    dist.barrier()

    active_classes = None
    if ACTIVE_CLASSES_YAML:
        import yaml
        with open(ACTIVE_CLASSES_YAML) as f:
            active_classes = yaml.safe_load(f).get("classes")

    # train/eval — 콤마구분 다중 폴더 지원 (학습/eval 따로 여러 개 선택)
    train_roots = [resolve_path(x.strip()) for x in str(TRAIN_DATA_DIR).split(",") if x.strip()]
    eval_roots  = [resolve_path(x.strip()) for x in str(EVAL_DATA_DIR).split(",") if x.strip()]
    for d in train_roots:
        if not d.exists():
            raise SystemExit(f"TRAIN dir not found: {d}\n  python scripts/generate_data.py && python scripts/_split_data.py")
    for d in eval_roots:
        if not d.exists():
            raise SystemExit(f"EVAL dir not found: {d}")
    if is_main(rank):
        print(f"[train dirs] {[str(d) for d in train_roots]}")
        print(f"[eval  dirs] {[str(d) for d in eval_roots]}")
    train_aug = build_aug(); eval_tf = build_eval_tf()
    train_ds = FlatPairDataset(train_roots, train_aug)
    eval_base = MultiRootImageFolder(eval_roots, transform=eval_tf,
                                     exclude=EXCLUDE_CLASSES, active_classes=active_classes,
                                     per_class_cap=PER_CLASS_CAP, normal_cap=NORMAL_CAP)
    classes = eval_base.classes

    if is_main(rank):
        print(f"[train] {len(train_ds)} from {TRAIN_DATA_DIR}")
        print(f"[eval]  {len(eval_base)} from {EVAL_DATA_DIR} ({len(classes)} classes)")
        (run_dir / "classes.json").write_text(
            json.dumps({"classes": classes, "class_to_idx": eval_base.class_to_idx,
                        "train_dir": TRAIN_DATA_DIR, "eval_dir": EVAL_DATA_DIR,
                        "world_size": world_size}, indent=2),
            encoding="utf-8")

    # model + DDP wrap. FREEZE_BACKBONE 면 find_unused_parameters=True
    model = ContrastiveModel(BACKBONE, PROJ_DIM, FREEZE_BACKBONE, bp).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=FREEZE_BACKBONE)
    queue = QueueBank(PROJ_DIM, QUEUE_SIZE, device) if USE_QUEUE else None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    if is_main(rank):
        log_stage_metric(run_dir, "contrastive_ddp_setup", {
            "world_size": world_size,
            "backbone_source": str(bp), "freeze_backbone": FREEZE_BACKBONE,
            "batch_per_gpu": BATCH_PER_GPU, "total_batch": BATCH_PER_GPU * world_size,
            "queue_size": QUEUE_SIZE if USE_QUEUE else 0,
            "ignore_neg_sim": IGNORE_NEG_SIM,
            "n_train": len(train_ds), "n_eval": len(eval_base), "n_classes_eval": len(classes),
        }, notes=f"DDP {world_size} GPUs — train flat (class hidden), eval ImageFolder")

    history = []
    for ep in range(1, EPOCHS + 1):
        # sampling subset (rank 0 결정 후 broadcast indices)
        if is_main(rank):
            r = TRAIN_SAMPLING_RATIO
            n_keep = max(1, int(len(train_ds) * r)) if 0 < r < 1 else len(train_ds)
            idx = sorted(random.sample(range(len(train_ds)), n_keep))
        else:
            idx = []
        obj = [idx]; dist.broadcast_object_list(obj, src=0); idx = obj[0]
        sub = torch.utils.data.Subset(train_ds, idx)
        sampler = DistributedSampler(sub, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED + ep)
        ld = DataLoader(sub, batch_size=BATCH_PER_GPU, sampler=sampler,
                        num_workers=nw, pin_memory=True, drop_last=True)
        # warmup
        lr = LR_HEAD * min(1.0, ep / max(1, WARMUP_EPOCHS))
        for g in opt.param_groups: g["lr"] = lr

        model.train()
        run_loss = 0.0; n = 0
        t0 = time.time()
        for it, (x1, x2) in enumerate(ld, 1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad()
            # ★ DDP: 두 view 를 한 번의 forward 로 (concat). model(x1);model(x2) 처럼
            #   backward 1회 전에 forward 2회 하면 DDP reducer 가 "param ready 한 번만"
            #   기대를 깨서 RuntimeError → process exit 1 (single-GPU 는 정상, multi-GPU 만 crash).
            bs = x1.size(0)
            z_cat = model(torch.cat([x1, x2], dim=0))
            z1, z2 = z_cat[:bs], z_cat[bs:]
            loss = info_nce_loss(z1, z2, queue, NCE_TEMP, IGNORE_NEG_SIM, LABEL_SMOOTHING)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            if queue is not None:
                # 각 rank 의 z2 gather 후 enqueue (queue 가 모든 rank 동일하게)
                z2_all = all_gather_concat(z2.detach(), device)
                queue.enqueue(z2_all)
            run_loss += loss.item() * x1.size(0); n += x1.size(0)
            if is_main(rank) and it % max(1, len(ld) // 10) == 0:
                print(f"  Ep {ep}/{EPOCHS} {it/len(ld)*100:>5.1f}% loss={run_loss/n:.4f} lr={lr:.2e}", flush=True)
        ep_loss = run_loss / max(1, n)
        avg = all_reduce_avg({"loss": ep_loss}, device)
        if is_main(rank):
            print(f"  Ep {ep}/{EPOCHS} DONE loss={avg['loss']:.4f} time={time.time()-t0:.0f}s", flush=True)
            history.append({"epoch": ep, "loss": avg["loss"], "lr": lr})

    # save (rank 0)
    if is_main(rank):
        torch.save({
            "state_dict": model.module.state_dict(),
            "classes": classes,
            "class_to_idx": eval_base.class_to_idx,
            "config": cfg,
            "backbone_source": str(bp),
            "world_size": world_size,
        }, cl_dir / "best_model.pt")
        (cl_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ---------- Eval: embed + HDBSCAN (rank 0 모음, all ranks compute) ----------
    eval_sampler = DistributedSampler(eval_base, num_replicas=world_size, rank=rank, shuffle=False)
    eval_ld = DataLoader(eval_base, batch_size=BATCH_PER_GPU * 4, sampler=eval_sampler,
                         num_workers=nw, pin_memory=True)
    model.eval()
    local_z, local_lbl, local_path = [], [], []
    with torch.no_grad():
        for imgs, lbls, paths in eval_ld:
            imgs = imgs.to(device, non_blocking=True)
            z = model(imgs)
            local_z.append(z); local_lbl.extend(lbls.tolist()); local_path.extend(paths)
    z_local = torch.cat(local_z, dim=0)
    z_all = all_gather_concat(z_local, device)

    # gather labels + paths (object list)
    obj_list = [None] * world_size
    dist.all_gather_object(obj_list, {"lbl": local_lbl, "path": local_path})
    if is_main(rank):
        all_lbl_idx, all_path = [], []
        for o in obj_list:
            all_lbl_idx.extend(o["lbl"]); all_path.extend(o["path"])
        all_label = [classes[i] for i in all_lbl_idx]
        emb = z_all[:len(all_label)].cpu().numpy()
        np.save(cl_dir / "embeddings.npy", emb)
        (cl_dir / "paths.json").write_text(
            json.dumps({"paths": all_path, "labels": all_label}, indent=2), encoding="utf-8")
        print(f"[eval] embed shape={emb.shape}")

        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                                    cluster_selection_method=CLUSTER_SELECTION_METHOD,
                                    cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
                                    metric="euclidean")
        pred = clusterer.fit_predict(emb)

        from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                     completeness_score, homogeneity_score)
        keep = pred != -1
        true_k = np.array(all_label)[keep]; pred_k = pred[keep]
        classes_unique = sorted(set(true_k))
        cl2i = {c: i for i, c in enumerate(classes_unique)}
        true_idx = np.array([cl2i[c] for c in true_k])
        cluster_cls = defaultdict(Counter)
        for p, c in zip(pred_k, true_k):
            cluster_cls[int(p)][c] += 1
        cls_total = Counter(all_label)
        capture = {}
        for cls, total in cls_total.items():
            mx = max((cnt for cl, ccnt in cluster_cls.items() for c, cnt in ccnt.items() if c == cls), default=0)
            capture[cls] = mx / total
        tier1 = {
            "n_total": int(len(pred)), "n_clustered": int(keep.sum()),
            "n_clusters": int(len(set(pred_k))),
            "noise_count": int((~keep).sum()),
            "noise_pct": round(float((~keep).sum() / len(pred) * 100), 2),
            "class_capture_rate": round(float(np.mean(list(capture.values()))), 4),
            "completeness": round(float(completeness_score(true_idx, pred_k)), 4),
            "homogeneity": round(float(homogeneity_score(true_idx, pred_k)), 4),
            "ami": round(float(adjusted_mutual_info_score(true_idx, pred_k)), 4),
            "ari": round(float(adjusted_rand_score(true_idx, pred_k)), 4),
        }
        with open(cl_dir / "clusters_global_list.txt", "w", encoding="utf-8") as f:
            f.write("cluster_id\ttrue_class\tpath\n")
            for p, c, ph in zip(pred, all_label, all_path):
                f.write(f"{int(p)}\t{c}\t{ph}\n")
        (cl_dir / "tier1.json").write_text(json.dumps(tier1, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[eval] tier1 = {json.dumps(tier1, indent=2)}")

        if SAVE_REPRESENTATIVES:
            n_rep = save_cluster_representatives(
                cl_dir, emb, pred, all_label, all_path, REPRESENTATIVES_PER_CLUSTER)
            tier1["n_representatives_saved"] = n_rep
            (cl_dir / "tier1.json").write_text(json.dumps(tier1, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [representatives] saved {n_rep} to {cl_dir / 'representatives'}")

        # wrong outlier save
        if SAVE_WRONG_IMAGES:
            wrong_dir = cl_dir / "wrong"
            if wrong_dir.exists(): shutil.rmtree(wrong_dir)
            wrong_dir.mkdir(parents=True)
            cluster_dom = {cl_id: cnt.most_common(1)[0] for cl_id, cnt in cluster_cls.items()}
            import csv
            n_wrong = 0
            rows = []
            for p, true_cls, path in zip(pred, all_label, all_path):
                cl_id = int(p)
                if cl_id == -1:
                    cluster_pred_cls = "noise"
                    purity = None
                    purity_tag = "na"
                    is_wrong = True
                else:
                    top_cls, top_n = cluster_dom[cl_id]
                    purity = top_n / sum(cluster_cls[cl_id].values())
                    purity_tag = f"{int(round(purity * 100))}pct"
                    cluster_pred_cls = top_cls
                    is_wrong = (cluster_pred_cls != true_cls)
                if not is_wrong: continue
                sub = wrong_dir / true_cls; sub.mkdir(exist_ok=True)
                dst = sub / (
                    f"true-{true_cls}___clusterpred-{cluster_pred_cls}_"
                    f"cluster-{cl_id}_clusterpurity-{purity_tag}_{Path(path).name}"
                )
                try:
                    shutil.copy2(path, dst); n_wrong += 1
                    rows.append({
                        "true_class": true_cls,
                        "cluster_pred_class": cluster_pred_cls,
                        "cluster_id": cl_id,
                        "cluster_purity": "" if purity is None else f"{purity:.6f}",
                        "src": str(path),
                        "dst": str(dst),
                    })
                except Exception: pass
            csv_path = wrong_dir / "wrong_clusters.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["true_class", "cluster_pred_class",
                                                  "cluster_id", "cluster_purity",
                                                  "src", "dst"])
                w.writeheader()
                w.writerows(rows)
            print(f"  [wrong] saved {n_wrong} to {wrong_dir}")
            tier1["n_wrong_saved"] = n_wrong

        log_stage_metric(run_dir, "contrastive_ddp_eval", tier1,
                         notes=f"DDP {world_size} GPUs, freeze={FREEZE_BACKBONE}")
        print(f"\n[OUT] {run_dir.resolve()}")

    if dist.is_initialized():
        dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--backbone", type=str, default=None,
                     help="CNN backbone best_model.pth 경로 (stage1 결과). 생략 시 weights/ ImageNet.")
    _ap.add_argument("--cnn-run-dir", type=str, default=None,
                     help="CNN run 폴더 (안의 cnn/best_model.pth 자동 사용).")
    _ap.add_argument("--tag", type=str, default=None, help="run 폴더 tag override.")
    _ap.add_argument("--epochs", type=int, default=None)
    _ap.add_argument("--train-dirs", type=str, default=None,
                     help="학습 폴더 (콤마구분 다중 가능). 예: a/unknown,a/unknown_archive")
    _ap.add_argument("--eval-dirs", type=str, default=None,
                     help="eval 폴더 (콤마구분 다중 가능, class 통합). 예: a/eval1,a/eval2")
    _ap.add_argument("--batch", type=int, default=None, help="BATCH_PER_GPU override.")
    _a = _ap.parse_args()
    # env 로 세팅 → mp.spawn 자식이 module re-import 시 위 module-level block 에서 읽음
    if _a.backbone:           os.environ["CL_BACKBONE_CKPT"] = _a.backbone
    if _a.cnn_run_dir:        os.environ["CL_CNN_RUN_DIR"] = _a.cnn_run_dir
    if _a.tag:                os.environ["CL_TAG"] = _a.tag
    if _a.epochs is not None: os.environ["CL_EPOCHS"] = str(_a.epochs)
    if _a.train_dirs:         os.environ["CL_TRAIN_DIRS"] = _a.train_dirs
    if _a.eval_dirs:          os.environ["CL_EVAL_DIRS"] = _a.eval_dirs
    if _a.batch is not None:  os.environ["CL_BATCH_PER_GPU"] = str(_a.batch)
    # 부모 프로세스의 현재 module global 도 즉시 반영 (world_size<=1 직접 호출 경로)
    if _a.backbone:           BACKBONE_CKPT = _a.backbone
    if _a.cnn_run_dir:        CNN_RUN_DIR = _a.cnn_run_dir
    if _a.tag:                TAG = _a.tag
    if _a.epochs is not None: EPOCHS = _a.epochs
    if _a.train_dirs:         TRAIN_DATA_DIR = _a.train_dirs
    if _a.eval_dirs:          EVAL_DATA_DIR = _a.eval_dirs
    if _a.batch is not None:  BATCH_PER_GPU = _a.batch
    launch_ddp(train_worker)
