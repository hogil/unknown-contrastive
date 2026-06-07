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
NO_EVAL               = False                               # 현업 class 없음: metric/eval 생략
ACTIVE_CLASSES_YAML   = None
EXCLUDE_CLASSES       = {"classification", "classification_chips"}

CNN_RUN_DIR           = None
BACKBONE_CKPT         = None
WEIGHTS_DIR           = "weights"
BACKBONE              = "convnextv2_base.fcmae_ft_in22k_in1k_384"
FREEZE_BACKBONE       = True

OUTPUT_ROOT           = "runs"
TAG                   = "contrastive_ddp"

IMG_SIZE              = 512
PROJ_DIM              = 128
BATCH_PER_GPU         = 8
NUM_WORKERS_PER_GPU   = None         # None = auto: os.cpu_count() // world_size (환경 코어 전부 활용)
EPOCHS                = 20
WARMUP_EPOCHS         = 1
TRAIN_SAMPLING_RATIO  = 0.25
LR_HEAD               = 1e-3
LR_MIN                = 1e-6
WEIGHT_DECAY          = 1e-6
NCE_TEMP              = 0.05
GRAD_CLIP             = 1.0
LABEL_SMOOTHING       = 0.02

USE_QUEUE             = True
QUEUE_SIZE            = 4096
IGNORE_NEG_SIM        = 0.95
USE_LOCAL             = False
LOCAL_WEIGHT          = 0.0
LOCAL_TAU             = 0.1
LOCAL_GRID            = 6
NECO_WEIGHT           = 0.2
NECO_TAU              = 0.1
NECO_GRID             = 0

USE_EMA               = False
USE_AMP               = False

MIN_CLUSTER_SIZE      = 12
MIN_SAMPLES           = 15
CLUSTER_SELECTION_METHOD = "leaf"
CLUSTER_SELECTION_EPSILON = 0.06

PER_CLASS_CAP         = 500
NORMAL_CAP            = 2000
EVAL_IGNORE_CLASSES   = {"Normal"}   # Normal 은 background/noise pool — metric 계산 제외

SEED                  = 42
SAVE_WRONG_IMAGES     = True
SAVE_REPRESENTATIVES   = True
REPRESENTATIVES_PER_CLUSTER = 5
# ===================================================================

import json
import math
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
    mask_palette_non_grade_to_white,
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
if os.environ.get("CL_IMG_SIZE"):
    IMG_SIZE = int(os.environ["CL_IMG_SIZE"])
if os.environ.get("CL_IGNORE_NEG_SIM"):
    IGNORE_NEG_SIM = float(os.environ["CL_IGNORE_NEG_SIM"])
if os.environ.get("CL_NCE_TEMP"):
    NCE_TEMP = float(os.environ["CL_NCE_TEMP"])
if os.environ.get("CL_LR_HEAD"):
    LR_HEAD = float(os.environ["CL_LR_HEAD"])
if os.environ.get("CL_NECO_WEIGHT"):
    NECO_WEIGHT = float(os.environ["CL_NECO_WEIGHT"])
if os.environ.get("CL_NECO_TAU"):
    NECO_TAU = float(os.environ["CL_NECO_TAU"])
if os.environ.get("CL_NECO_GRID"):
    NECO_GRID = int(os.environ["CL_NECO_GRID"])
if os.environ.get("CL_LOCAL_WEIGHT"):
    LOCAL_WEIGHT = float(os.environ["CL_LOCAL_WEIGHT"])
if os.environ.get("CL_LOCAL_TAU"):
    LOCAL_TAU = float(os.environ["CL_LOCAL_TAU"])
if os.environ.get("CL_LOCAL_GRID"):
    LOCAL_GRID = int(os.environ["CL_LOCAL_GRID"])
if os.environ.get("CL_USE_QUEUE"):
    USE_QUEUE = os.environ["CL_USE_QUEUE"].strip().lower() in {"1", "true", "yes", "y"}
if os.environ.get("CL_QUEUE_SIZE"):
    QUEUE_SIZE = int(os.environ["CL_QUEUE_SIZE"])
TRAIN_DATA_DIR = os.environ.get("CL_TRAIN_DIRS") or TRAIN_DATA_DIR   # 콤마구분 다중 가능
EVAL_DATA_DIR  = os.environ.get("CL_EVAL_DIRS") or EVAL_DATA_DIR
NO_EVAL = NO_EVAL or os.environ.get("CL_NO_EVAL", "").strip().lower() in {"1", "true", "yes", "y"}


def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def contrastive_lr_for_epoch(ep: int) -> float:
    """Warmup 후 cosine decay. CNN/anomaly stage 와 같은 LR shape."""
    if WARMUP_EPOCHS > 0 and ep <= WARMUP_EPOCHS:
        return LR_HEAD * ep / max(1, WARMUP_EPOCHS)
    decay_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
    t = min(1.0, max(0.0, (ep - WARMUP_EPOCHS) / decay_epochs))
    return LR_MIN + 0.5 * (LR_HEAD - LR_MIN) * (1.0 + math.cos(math.pi * t))


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
            from PIL import Image
            with Image.open(path) as im:
                sample = mask_palette_non_grade_to_white(im).convert("RGB")
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
            from PIL import Image
            with Image.open(path) as im:
                sample = mask_palette_non_grade_to_white(im).convert("RGB")
        except Exception:
            from PIL import Image as _PILImage
            sample = _PILImage.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, path


class FlatPairDataset(Dataset):
    """flat folder(s) recursive glob → two views (no class label). roots = str 또는 list."""
    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    def __init__(self, roots, tfm):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        paths = []
        for r in roots:
            paths.extend(p for p in Path(r).rglob("*")
                         if p.is_file() and p.suffix.lower() in self.EXTS)
        self.paths = sorted(set(paths))
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            from PIL import Image
            with Image.open(self.paths[i]) as im:
                img = mask_palette_non_grade_to_white(im).convert("RGB")
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


def write_contrastive_eval_report(cl_dir: Path, tier1: dict, pred: np.ndarray,
                                  labels: list[str], paths: list[str],
                                  ignore_classes: set[str], reps_per_cluster: int) -> dict:
    """사람이 읽는 eval 요약 + class/cluster별 TSV 텍스트 리포트."""
    labels_arr = np.array(labels)
    pred_arr = np.asarray(pred)
    ignore_set = set(ignore_classes)
    measured_mask = ~np.isin(labels_arr, list(ignore_set))

    measured_idx = np.where(measured_mask)[0]
    cluster_cls = defaultdict(Counter)
    cluster_all = Counter(int(p) for p in pred_arr.tolist())
    for i in measured_idx:
        cluster_cls[int(pred_arr[i])][str(labels_arr[i])] += 1

    cluster_dom = {}
    for cl_id, cnt in cluster_cls.items():
        if cl_id == -1:
            continue
        total = sum(cnt.values())
        if total > 0:
            top_cls, top_n = cnt.most_common(1)[0]
            cluster_dom[int(cl_id)] = {
                "dominant_class": top_cls,
                "dominant_n": int(top_n),
                "measured_n": int(total),
                "purity": float(top_n / total),
            }

    class_rows = []
    for cls in sorted(set(labels_arr[measured_mask].tolist())):
        idx = np.where((labels_arr == cls) & measured_mask)[0]
        total = int(len(idx))
        cls_pred = pred_arr[idx]
        noise_n = int((cls_pred == -1).sum())
        clusters = sorted(int(c) for c in set(cls_pred.tolist()) if int(c) != -1)
        captured_n = 0
        other_cluster_n = 0
        best_cluster_id = ""
        best_cluster_n = 0
        for cl_id in clusters:
            n_cls_in_cluster = int((cls_pred == cl_id).sum())
            if n_cls_in_cluster > best_cluster_n:
                best_cluster_n = n_cls_in_cluster
                best_cluster_id = cl_id
            if cluster_dom.get(cl_id, {}).get("dominant_class") == cls:
                captured_n += n_cls_in_cluster
            else:
                other_cluster_n += n_cls_in_cluster
        dominant_clusters = sum(1 for d in cluster_dom.values() if d["dominant_class"] == cls)
        class_rows.append({
            "class": cls,
            "total": total,
            "captured": captured_n,
            "captured_pct": captured_n / max(1, total) * 100,
            "noise": noise_n,
            "noise_pct": noise_n / max(1, total) * 100,
            "other_cluster": other_cluster_n,
            "clusters_with_class": len(clusters),
            "dominant_clusters": dominant_clusters,
            "largest_cluster_id": best_cluster_id,
            "largest_cluster_n": best_cluster_n,
            "largest_cluster_pct": best_cluster_n / max(1, total) * 100,
        })

    cluster_rows = []
    for cl_id in sorted(cluster_dom):
        cnt = cluster_cls[cl_id]
        dom = cluster_dom[cl_id]
        mix = ", ".join(f"{c}:{n}" for c, n in cnt.most_common(8))
        cluster_rows.append({
            "cluster_id": cl_id,
            "dominant_class": dom["dominant_class"],
            "measured_n": dom["measured_n"],
            "all_n": int(cluster_all[cl_id]),
            "purity_pct": dom["purity"] * 100,
            "class_mix_top": mix,
        })

    noise_by_class = Counter(str(labels_arr[i]) for i in measured_idx if int(pred_arr[i]) == -1)
    class_txt = cl_dir / "contrastive_class_report.txt"
    with class_txt.open("w", encoding="utf-8", newline="") as f:
        f.write("# captured = true class 와 같은 dominant_class cluster 에 들어간 수\n")
        f.write("# other_cluster = noise 는 아니지만 다른 class dominant cluster 에 들어간 수\n")
        f.write("# clusters_with_class = 해당 true class 샘플이 흩어진 non-noise cluster 개수\n")
        f.write("# Normal 등 ignored_classes 는 metric/report 계산에서 제외\n")
        f.write("\t".join([
            "class", "total", "captured", "captured_pct", "noise", "noise_pct",
            "other_cluster", "clusters_with_class", "dominant_clusters",
            "largest_cluster_id", "largest_cluster_n", "largest_cluster_pct",
        ]) + "\n")
        for r in class_rows:
            f.write("\t".join([
                str(r["class"]), str(r["total"]), str(r["captured"]),
                f"{r['captured_pct']:.2f}", str(r["noise"]), f"{r['noise_pct']:.2f}",
                str(r["other_cluster"]), str(r["clusters_with_class"]),
                str(r["dominant_clusters"]), str(r["largest_cluster_id"]),
                str(r["largest_cluster_n"]), f"{r['largest_cluster_pct']:.2f}",
            ]) + "\n")

    cluster_txt = cl_dir / "contrastive_cluster_report.txt"
    with cluster_txt.open("w", encoding="utf-8", newline="") as f:
        f.write("# cluster_id=-1(noise)는 class report 의 noise 컬럼과 아래 noise_by_class 참고\n")
        f.write("\t".join([
            "cluster_id", "dominant_class", "measured_n", "all_n",
            "purity_pct", "class_mix_top",
        ]) + "\n")
        for r in cluster_rows:
            f.write("\t".join([
                str(r["cluster_id"]), r["dominant_class"], str(r["measured_n"]),
                str(r["all_n"]), f"{r['purity_pct']:.2f}", r["class_mix_top"],
            ]) + "\n")
        f.write("\n# noise_by_class\n")
        f.write("class\tnoise\n")
        for cls, n in sorted(noise_by_class.items()):
            f.write(f"{cls}\t{n}\n")

    md = cl_dir / "contrastive_eval_report.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Contrastive Evaluation Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- measured total: {tier1.get('n_total', 0)}\n")
        f.write(f"- ignored total: {tier1.get('n_ignored', 0)} ({', '.join(sorted(ignore_set)) or 'none'})\n")
        if tier1.get("ignored_class_counts"):
            ignored_desc = ", ".join(f"{k}={v}" for k, v in tier1["ignored_class_counts"].items())
            f.write(f"- ignored counts: {ignored_desc}\n")
        f.write(f"- clustered: {tier1.get('n_clustered', 0)}\n")
        f.write(f"- noise: {tier1.get('noise_count', 0)} ({tier1.get('noise_pct', 0)}%)\n")
        f.write(f"- clusters: {tier1.get('n_clusters', 0)}\n")
        f.write(f"- class capture rate: {tier1.get('class_capture_rate', 0)}\n")
        f.write(f"- homogeneity/completeness/AMI/ARI: {tier1.get('homogeneity', 0)} / {tier1.get('completeness', 0)} / {tier1.get('ami', 0)} / {tier1.get('ari', 0)}\n")
        f.write(f"- representatives: max {reps_per_cluster} images per cluster\n\n")
        f.write("## Meaning\n\n")
        f.write("- captured: true class 와 같은 dominant_class cluster 에 들어간 샘플 수\n")
        f.write("- noise: HDBSCAN 이 cluster 로 묶지 못하고 -1 로 둔 샘플 수\n")
        f.write("- clusters_with_class: 해당 class 샘플이 흩어진 non-noise cluster 개수\n")
        f.write("- dominant_clusters: 해당 class 가 dominant_class 인 cluster 개수\n\n")
        f.write("## Class Report\n\n")
        f.write("| class | total | captured | captured % | noise | noise % | other cluster | clusters with class | dominant clusters | largest cluster |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in class_rows:
            largest = f"{r['largest_cluster_id']} ({r['largest_cluster_n']}, {r['largest_cluster_pct']:.2f}%)"
            f.write(
                f"| {r['class']} | {r['total']} | {r['captured']} | {r['captured_pct']:.2f} | "
                f"{r['noise']} | {r['noise_pct']:.2f} | {r['other_cluster']} | "
                f"{r['clusters_with_class']} | {r['dominant_clusters']} | {largest} |\n"
            )
        f.write("\n## Files\n\n")
        f.write(f"- class text: `{class_txt.name}`\n")
        f.write(f"- cluster text: `{cluster_txt.name}`\n")
        f.write("- representative images: `representatives/`\n")
        f.write("- wrong images: `wrong/`\n")

    return {
        "class_report": str(class_txt),
        "cluster_report": str(cluster_txt),
        "markdown_report": str(md),
    }


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
    diag = torch.arange(b, device=z1.device)
    if b <= neg_bank.size(0):
        sim[diag, diag] = -1e9
    logits = torch.cat([logits_pos, sim], dim=1)
    return masked_pos_cross_entropy(logits, label_smoothing)


def masked_pos_cross_entropy(logits, label_smoothing):
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


def grid_anchor_indices(n: int, grid_size: int, device):
    if grid_size is None or grid_size <= 0 or grid_size * grid_size >= n:
        return None
    side = int(round(math.sqrt(n)))
    if side * side != n:
        return torch.linspace(0, n - 1, steps=min(n, grid_size * grid_size),
                              device=device).round().long().unique()
    g = min(grid_size, side)
    ys = torch.linspace(0, side - 1, steps=g, device=device).round().long().unique()
    xs = torch.linspace(0, side - 1, steps=g, device=device).round().long().unique()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return (yy.reshape(-1) * side + xx.reshape(-1)).long()


def local_grid_loss(p1, p2, tau, grid_size):
    """Patch-level InfoNCE over matching grid locations only."""
    if p1.ndim != 3 or p2.ndim != 3:
        raise ValueError("local_grid_loss expects [B, N, D] patch embeddings")
    idx = grid_anchor_indices(p1.size(1), grid_size, p1.device)
    if idx is not None:
        p1 = p1.index_select(1, idx)
        p2 = p2.index_select(1, idx)
    a = p1.size(1)
    labels = torch.arange(a, device=p1.device).repeat(p1.size(0))
    logits_12 = torch.bmm(p1, p2.transpose(1, 2)).reshape(-1, a) / tau
    logits_21 = torch.bmm(p2, p1.transpose(1, 2)).reshape(-1, a) / tau
    return 0.5 * (F.cross_entropy(logits_12, labels) +
                  F.cross_entropy(logits_21, labels))


def neco_loss(p1, p2, tau, grid_size=0):
    """B6 NeCo: patch-neighbor order consistency over shared projection patches."""
    if p1.ndim != 3 or p2.ndim != 3:
        raise ValueError("neco_loss expects [B, N, D] patch embeddings")
    idx = grid_anchor_indices(p1.size(1), grid_size, p1.device)
    if idx is not None:
        p1 = p1.index_select(1, idx)
        p2 = p2.index_select(1, idx)
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
    # 현업 데이터는 class label 이 없으므로 --no-eval 에서는 metric/eval 을 완전히 생략.
    train_roots = [resolve_path(x.strip()) for x in str(TRAIN_DATA_DIR).split(",") if x.strip()]
    no_eval = bool(NO_EVAL) or str(EVAL_DATA_DIR).strip().lower() in {"", "none", "null", "skip", "no"}
    eval_roots  = [] if no_eval else [resolve_path(x.strip()) for x in str(EVAL_DATA_DIR).split(",") if x.strip()]
    for d in train_roots:
        if not d.exists():
            raise SystemExit(f"TRAIN dir not found: {d}\n  python scripts/generate_data.py && python scripts/_split_data.py")
    if not train_roots:
        raise SystemExit("--train-dirs is empty")
    if not no_eval:
        for d in eval_roots:
            if not d.exists():
                raise SystemExit(f"EVAL dir not found: {d}")
    if is_main(rank):
        print(f"[train dirs] {[str(d) for d in train_roots]}")
        if no_eval:
            print("[eval  dirs] skipped (--no-eval: 현업 class label 없음)")
        else:
            print(f"[eval  dirs] {[str(d) for d in eval_roots]}")
    train_aug = build_aug(); eval_tf = build_eval_tf()
    train_ds = FlatPairDataset(train_roots, train_aug)
    eval_base = None
    classes = []
    class_to_idx = {}
    if not no_eval:
        eval_base = MultiRootImageFolder(eval_roots, transform=eval_tf,
                                         exclude=EXCLUDE_CLASSES, active_classes=active_classes,
                                         per_class_cap=PER_CLASS_CAP, normal_cap=NORMAL_CAP)
        classes = eval_base.classes
        class_to_idx = eval_base.class_to_idx

    if is_main(rank):
        print(f"[train] {len(train_ds)} from {TRAIN_DATA_DIR}")
        if no_eval:
            print("[eval]  skipped — no metric because production data has no class labels")
        else:
            print(f"[eval]  {len(eval_base)} from {EVAL_DATA_DIR} ({len(classes)} classes)")
        (run_dir / "classes.json").write_text(
            json.dumps({"classes": classes, "class_to_idx": class_to_idx,
                        "train_dir": TRAIN_DATA_DIR, "eval_dir": EVAL_DATA_DIR,
                        "no_eval": no_eval, "world_size": world_size}, indent=2),
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
            "lr_head": LR_HEAD, "lr_min": LR_MIN, "lr_schedule": "warmup_cosine_epoch",
            "queue_size": QUEUE_SIZE if USE_QUEUE else 0,
            "ignore_neg_sim": IGNORE_NEG_SIM,
            "use_local": USE_LOCAL,
            "neco_weight": NECO_WEIGHT,
            "neco_tau": NECO_TAU,
            "recipe": "B6: Global InfoNCE + MoCo Queue + NEG + NeCo, no Local",
            "n_train": len(train_ds), "n_eval": 0 if no_eval else len(eval_base),
            "n_classes_eval": len(classes), "no_eval": no_eval,
        }, notes=f"DDP {world_size} GPUs — train flat (class hidden)"
                 + (", eval skipped" if no_eval else ", eval ImageFolder"))

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
        lr = contrastive_lr_for_epoch(ep)
        for g in opt.param_groups: g["lr"] = lr

        model.train()
        run_loss = 0.0; run_nce = 0.0; run_neco = 0.0; run_local = 0.0; n = 0
        t0 = time.time()
        for it, (x1, x2) in enumerate(ld, 1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad()
            # ★ DDP: 두 view 를 한 번의 forward 로 (concat). model(x1);model(x2) 처럼
            #   backward 1회 전에 forward 2회 하면 DDP reducer 가 "param ready 한 번만"
            #   기대를 깨서 RuntimeError → process exit 1 (single-GPU 는 정상, multi-GPU 만 crash).
            bs = x1.size(0)
            need_patches = NECO_WEIGHT > 0 or LOCAL_WEIGHT > 0
            if need_patches:
                z_cat, p_cat = model(torch.cat([x1, x2], dim=0), return_neco=True)
                p1, p2 = p_cat[:bs], p_cat[bs:]
                loss_neco = neco_loss(p1, p2, NECO_TAU, NECO_GRID) if NECO_WEIGHT > 0 else z_cat.new_zeros(())
                loss_local = local_grid_loss(p1, p2, LOCAL_TAU, LOCAL_GRID) if LOCAL_WEIGHT > 0 else z_cat.new_zeros(())
            else:
                z_cat = model(torch.cat([x1, x2], dim=0))
                loss_neco = z_cat.new_zeros(())
                loss_local = z_cat.new_zeros(())
            z1, z2 = z_cat[:bs], z_cat[bs:]
            loss_nce = info_nce_loss(z1, z2, queue, NCE_TEMP, IGNORE_NEG_SIM, LABEL_SMOOTHING)
            loss = loss_nce + (NECO_WEIGHT * loss_neco) + (LOCAL_WEIGHT * loss_local)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
            if queue is not None:
                # 각 rank 의 z2 gather 후 enqueue (queue 가 모든 rank 동일하게)
                z2_all = all_gather_concat(z2.detach(), device)
                queue.enqueue(z2_all)
            run_loss += loss.item() * x1.size(0)
            run_nce += loss_nce.item() * x1.size(0)
            run_neco += loss_neco.item() * x1.size(0)
            run_local += loss_local.item() * x1.size(0)
            n += x1.size(0)
            if is_main(rank) and it % max(1, len(ld) // 10) == 0:
                print(f"  Ep {ep}/{EPOCHS} {it/len(ld)*100:>5.1f}% "
                      f"loss={run_loss/n:.4f} nce={run_nce/n:.4f} "
                      f"neco={run_neco/n:.4f} local={run_local/n:.4f} lr={lr:.2e}", flush=True)
        ep_loss = run_loss / max(1, n)
        ep_nce = run_nce / max(1, n)
        ep_neco = run_neco / max(1, n)
        ep_local = run_local / max(1, n)
        avg = all_reduce_avg({"loss": ep_loss, "nce": ep_nce, "neco": ep_neco,
                              "local": ep_local}, device)
        if is_main(rank):
            print(f"  Ep {ep}/{EPOCHS} DONE loss={avg['loss']:.4f} "
                  f"nce={avg['nce']:.4f} neco={avg['neco']:.4f} local={avg['local']:.4f} "
                  f"time={time.time()-t0:.0f}s", flush=True)
            history.append({"epoch": ep, "loss": avg["loss"], "nce": avg["nce"],
                            "neco": avg["neco"], "local": avg["local"], "lr": lr})

    # save (rank 0)
    if is_main(rank):
        torch.save({
            "state_dict": model.module.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "config": cfg,
            "backbone_source": str(bp),
            "world_size": world_size,
        }, cl_dir / "best_model.pt")
        (cl_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if no_eval:
        if is_main(rank):
            log_stage_metric(run_dir, "contrastive_ddp_train_done_no_eval", {
                "n_train": len(train_ds),
                "epochs": EPOCHS,
                "final_loss": history[-1]["loss"] if history else None,
                "backbone_source": str(bp),
                "best_model": str(cl_dir / "best_model.pt"),
            }, notes="Production contrastive train only; classless data so metric/eval skipped")
            print("[eval] skipped (--no-eval). Use predict_grouping_prod.py for classless grouping.")
            print(f"\n[OUT] {run_dir.resolve()}")
        if dist.is_initialized():
            dist.barrier()
        cleanup_ddp()
        return

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
        labels_arr = np.array(all_label)
        measured_mask = ~np.isin(labels_arr, list(EVAL_IGNORE_CLASSES))
        ignored_counts = Counter(labels_arr[~measured_mask].tolist())
        keep = (pred != -1) & measured_mask
        true_k = labels_arr[keep]; pred_k = pred[keep]
        classes_unique = sorted(set(true_k))
        cl2i = {c: i for i, c in enumerate(classes_unique)}
        true_idx = np.array([cl2i[c] for c in true_k]) if classes_unique else np.array([], dtype=int)
        cluster_cls = defaultdict(Counter)
        for p, c in zip(pred_k, true_k):
            cluster_cls[int(p)][c] += 1
        cls_total = Counter(labels_arr[measured_mask])
        capture = {}
        for cls, total in cls_total.items():
            mx = max((cnt for cl, ccnt in cluster_cls.items() for c, cnt in ccnt.items() if c == cls), default=0)
            capture[cls] = mx / total
        can_score = len(true_idx) > 0 and len(set(true_idx.tolist())) > 1 and len(set(pred_k.tolist())) > 1
        measured_noise = int(((pred == -1) & measured_mask).sum())
        measured_total = int(measured_mask.sum())
        tier1 = {
            "n_total": measured_total,
            "n_total_all": int(len(pred)),
            "n_ignored": int((~measured_mask).sum()),
            "ignored_classes": sorted(EVAL_IGNORE_CLASSES),
            "ignored_class_counts": dict(sorted(ignored_counts.items())),
            "n_clustered": int(keep.sum()),
            "n_clusters": int(len(set(pred_k))),
            "noise_count": measured_noise,
            "noise_pct": round(float(measured_noise / max(1, measured_total) * 100), 2),
            "class_capture_rate": round(float(np.mean(list(capture.values()))), 4) if capture else 0.0,
            "completeness": round(float(completeness_score(true_idx, pred_k)), 4) if can_score else 0.0,
            "homogeneity": round(float(homogeneity_score(true_idx, pred_k)), 4) if can_score else 0.0,
            "ami": round(float(adjusted_mutual_info_score(true_idx, pred_k)), 4) if can_score else 0.0,
            "ari": round(float(adjusted_rand_score(true_idx, pred_k)), 4) if can_score else 0.0,
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
                if true_cls in EVAL_IGNORE_CLASSES:
                    continue
                cl_id = int(p)
                if cl_id == -1:
                    cluster_pred_cls = "noise"
                    purity = None
                    purity_tag = "na"
                    is_wrong = True
                else:
                    if cl_id not in cluster_dom:
                        continue
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

        report_paths = write_contrastive_eval_report(
            cl_dir, tier1, pred, all_label, all_path,
            EVAL_IGNORE_CLASSES, REPRESENTATIVES_PER_CLUSTER)
        tier1["reports"] = report_paths
        (cl_dir / "tier1.json").write_text(json.dumps(tier1, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [report] {cl_dir / 'contrastive_eval_report.md'}")

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
    _ap.add_argument("--no-eval", action="store_true",
                     help="현업 classless train only. eval/HDBSCAN metric 생략.")
    _ap.add_argument("--batch", type=int, default=None, help="BATCH_PER_GPU override.")
    _ap.add_argument("--img-size", type=int, default=None, help="contrastive input size. 기본 512")
    _ap.add_argument("--ignore-neg-sim", type=float, default=None,
                     help="NEG filter threshold. 예: 0.90 또는 0.95")
    _ap.add_argument("--nce-temp", type=float, default=None,
                     help="InfoNCE temperature. 낮을수록 similarity 차이에 민감. 예: 0.05")
    _ap.add_argument("--lr-head", type=float, default=None,
                     help="projection head learning rate. 예: 5e-4")
    _ap.add_argument("--neco-weight", type=float, default=None,
                     help="B6 NeCo weight. 기본 0.2, 0이면 NeCo OFF.")
    _ap.add_argument("--neco-tau", type=float, default=None,
                     help="NeCo patch-neighbor softmax temperature. 기본 0.1")
    _ap.add_argument("--neco-grid", type=int, default=None,
                     help="NeCo patch grid anchor size. 0=all patches, 6=6x6 grid.")
    _ap.add_argument("--local-weight", type=float, default=None,
                     help="Local grid InfoNCE weight. 0이면 OFF.")
    _ap.add_argument("--local-tau", type=float, default=None,
                     help="Local grid InfoNCE temperature. 기본 0.1")
    _ap.add_argument("--local-grid", type=int, default=None,
                     help="Local grid anchor size. 예: 6이면 6x6 grid.")
    _ap.add_argument("--no-queue", action="store_true",
                     help="MoCo-style queue OFF (ablation).")
    _ap.add_argument("--queue-size", type=int, default=None,
                     help="MoCo-style queue size override.")
    _a = _ap.parse_args()
    # env 로 세팅 → mp.spawn 자식이 module re-import 시 위 module-level block 에서 읽음
    if _a.backbone:           os.environ["CL_BACKBONE_CKPT"] = _a.backbone
    if _a.cnn_run_dir:        os.environ["CL_CNN_RUN_DIR"] = _a.cnn_run_dir
    if _a.tag:                os.environ["CL_TAG"] = _a.tag
    if _a.epochs is not None: os.environ["CL_EPOCHS"] = str(_a.epochs)
    if _a.train_dirs:         os.environ["CL_TRAIN_DIRS"] = _a.train_dirs
    if _a.eval_dirs:          os.environ["CL_EVAL_DIRS"] = _a.eval_dirs
    if _a.no_eval:            os.environ["CL_NO_EVAL"] = "1"
    if _a.batch is not None:  os.environ["CL_BATCH_PER_GPU"] = str(_a.batch)
    if _a.img_size is not None: os.environ["CL_IMG_SIZE"] = str(_a.img_size)
    if _a.ignore_neg_sim is not None: os.environ["CL_IGNORE_NEG_SIM"] = str(_a.ignore_neg_sim)
    if _a.nce_temp is not None:       os.environ["CL_NCE_TEMP"] = str(_a.nce_temp)
    if _a.lr_head is not None:        os.environ["CL_LR_HEAD"] = str(_a.lr_head)
    if _a.neco_weight is not None:    os.environ["CL_NECO_WEIGHT"] = str(_a.neco_weight)
    if _a.neco_tau is not None:       os.environ["CL_NECO_TAU"] = str(_a.neco_tau)
    if _a.neco_grid is not None:      os.environ["CL_NECO_GRID"] = str(_a.neco_grid)
    if _a.local_weight is not None:   os.environ["CL_LOCAL_WEIGHT"] = str(_a.local_weight)
    if _a.local_tau is not None:      os.environ["CL_LOCAL_TAU"] = str(_a.local_tau)
    if _a.local_grid is not None:     os.environ["CL_LOCAL_GRID"] = str(_a.local_grid)
    if _a.no_queue:                   os.environ["CL_USE_QUEUE"] = "0"
    if _a.queue_size is not None:     os.environ["CL_QUEUE_SIZE"] = str(_a.queue_size)
    # 부모 프로세스의 현재 module global 도 즉시 반영 (world_size<=1 직접 호출 경로)
    if _a.backbone:           BACKBONE_CKPT = _a.backbone
    if _a.cnn_run_dir:        CNN_RUN_DIR = _a.cnn_run_dir
    if _a.tag:                TAG = _a.tag
    if _a.epochs is not None: EPOCHS = _a.epochs
    if _a.train_dirs:         TRAIN_DATA_DIR = _a.train_dirs
    if _a.eval_dirs:          EVAL_DATA_DIR = _a.eval_dirs
    if _a.no_eval:            NO_EVAL = True
    if _a.batch is not None:  BATCH_PER_GPU = _a.batch
    if _a.img_size is not None: IMG_SIZE = _a.img_size
    if _a.ignore_neg_sim is not None: IGNORE_NEG_SIM = _a.ignore_neg_sim
    if _a.nce_temp is not None:       NCE_TEMP = _a.nce_temp
    if _a.lr_head is not None:        LR_HEAD = _a.lr_head
    if _a.neco_weight is not None:    NECO_WEIGHT = _a.neco_weight
    if _a.neco_tau is not None:       NECO_TAU = _a.neco_tau
    if _a.neco_grid is not None:      NECO_GRID = _a.neco_grid
    if _a.local_weight is not None:   LOCAL_WEIGHT = _a.local_weight
    if _a.local_tau is not None:      LOCAL_TAU = _a.local_tau
    if _a.local_grid is not None:     LOCAL_GRID = _a.local_grid
    if _a.no_queue:                   USE_QUEUE = False
    if _a.queue_size is not None:     QUEUE_SIZE = _a.queue_size
    launch_ddp(train_worker)
