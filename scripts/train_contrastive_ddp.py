#!/usr/bin/env python3
"""Contrastive 학습 DDP — multi-GPU (NCCL).

사용:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_contrastive_ddp.py
"""
# ===================================================================
# === CONFIG ===
# ===================================================================
TRAIN_DATA_DIR        = "data/images/contrastive_train"   # 프로젝트 상대, flat
EVAL_DATA_DIR         = "data/images/contrastive_eval"    # 프로젝트 상대, ImageFolder
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
NUM_WORKERS_PER_GPU   = 4
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
# ===================================================================

from __future__ import annotations

import json
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


class FlatPairDataset(Dataset):
    """flat folder recursive glob → two views (no class label)."""
    EXTS = (".png", ".jpg", ".jpeg", ".bmp")
    def __init__(self, root, tfm):
        self.paths = sorted(p for ext in self.EXTS for p in Path(root).rglob(f"*{ext}"))
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

    # train
    train_dir = resolve_path(TRAIN_DATA_DIR); eval_dir = resolve_path(EVAL_DATA_DIR)
    if not train_dir.exists():
        raise SystemExit(f"TRAIN_DATA_DIR not found: {train_dir}\n  python scripts/generate_data.py && python scripts/_split_data.py")
    if not eval_dir.exists():
        raise SystemExit(f"EVAL_DATA_DIR not found: {eval_dir}")
    train_aug = build_aug(); eval_tf = build_eval_tf()
    train_ds = FlatPairDataset(str(train_dir), train_aug)
    eval_base = SafeImageFolder(str(eval_dir), transform=eval_tf,
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
                        num_workers=NUM_WORKERS_PER_GPU, pin_memory=True, drop_last=True)
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
            z1 = model(x1); z2 = model(x2)
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
                         num_workers=NUM_WORKERS_PER_GPU, pin_memory=True)
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

        # wrong outlier save
        if SAVE_WRONG_IMAGES:
            wrong_dir = cl_dir / "wrong"
            if wrong_dir.exists(): shutil.rmtree(wrong_dir)
            wrong_dir.mkdir(parents=True)
            cluster_dom = {cl_id: cnt.most_common(1)[0] for cl_id, cnt in cluster_cls.items()}
            n_wrong = 0
            for p, true_cls, path in zip(pred, all_label, all_path):
                cl_id = int(p)
                if cl_id == -1:
                    pred_cls = "noise"; pct = 0; is_wrong = True
                else:
                    top_cls, top_n = cluster_dom[cl_id]
                    pct = int(round(top_n / sum(cluster_cls[cl_id].values()) * 100))
                    pred_cls = top_cls
                    is_wrong = (pred_cls != true_cls)
                if not is_wrong: continue
                sub = wrong_dir / true_cls; sub.mkdir(exist_ok=True)
                dst = sub / f"{true_cls}_{pred_cls}_{pct}%_{Path(path).name}"
                try:
                    shutil.copy2(path, dst); n_wrong += 1
                except Exception: pass
            print(f"  [wrong] saved {n_wrong} to {wrong_dir}")
            tier1["n_wrong_saved"] = n_wrong

        log_stage_metric(run_dir, "contrastive_ddp_eval", tier1,
                         notes=f"DDP {world_size} GPUs, freeze={FREEZE_BACKBONE}")
        print(f"\n[OUT] {run_dir.resolve()}")

    cleanup_ddp()


if __name__ == "__main__":
    launch_ddp(train_worker)
