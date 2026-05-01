#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN classifier training with failbit + object-type feature channels.

Wafer fail-bit defect 분류기. 33 class (31 defect + Starburst + CommaCluster).
Normal class는 학습 제외 — 운영 시 max_prob threshold로 "Normal/unknown" 판정.

입력은 RGB 자연색 이미지가 아니라 3채널 feature tensor:
  R = 기존 palette PNG의 palette-index fail map, BICUBIC resize
  G = position JSON에서 defect chip(b>=200)에 object type id를 칠한 map
  B = all-zero dummy map

object type id:
  0 = registered object 없음
  1 = invalid_main
  2 = scratch
  3 = scratch_21deg
  4 = particle_blast
  5 = bank_boundary

주요 옵션:
- per-class subset (YAML config) — class imbalance 시뮬레이션
- class_weight {none, inverse, effective(default)} 또는 focal loss
- EMA (default ON, decay 0.95)
- val_loss guard + median F1 smoothing → spike best 차단
- 출력 폴더 rename-on-end:
    log/{model_tag}_{YYMMDD_HHMMSS}_{test_f1:.2f}_{val_f1:.2f}/  (3-way split)
    log/{model_tag}_{YYMMDD_HHMMSS}_{val_f1:.2f}/                 (--train-val-only)
  default model_tag = backbone short name (e.g. convnextv2_base)

산출:
    log/<run_dir>/
      hparams.yaml, hparams.txt, best_model.pth,
      best_history.txt          ← 모든 best 갱신 결과 통합 (per-class 포함)
      best_confusion_matrix.png ← test(위)+val(아래) combined
      curves.png, history.json, run.log
      wrong/{val,test}/<true>/<pred>/*.png
      (옵션) predictions/{tp,fp,fn}/<class>/*.png

사용 예:
    # 기본 학습
    python cnn_train.py --epochs 30 --batch 16

    # subset YAML로 class imbalance 시뮬레이션
    python cnn_train.py --epochs 30 --subset-config subset.yaml

    # quick smoke (~5분)
    python cnn_train.py --epochs 2 --subset-config quick.yaml --batch 8
"""
from __future__ import annotations

import os, sys, json, time, copy, random, argparse, logging, platform, shutil

# Windows cp949 console에서 한국어/유니코드 mojibake 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import yaml
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.datasets import ImageFolder
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             classification_report, confusion_matrix)
import timm

from _resource_guard import assess_start, format_assessment, ResourceMonitor

# ===================== CONFIG =====================
DEFAULT_DATA_DIR = "D:/project/data/wm-811k/unknown"
DEFAULT_POSITION_DIR = "D:/project/data/positions/unknown"
EXCLUDE_CLASSES  = {"Normal"}
BACKBONE         = "convnextv2_base.fcmae_ft_in22k_in1k_384"

OBJECT_TYPE_ID = {
    "none": 0,
    "invalid_main": 1,
    "scratch": 2,
    "scratch_21deg": 3,
    "particle_blast": 4,
    "bank_boundary": 5,
}
OBJECT_SUFFIXES = sorted(
    [k for k in OBJECT_TYPE_ID if k != "none"],
    key=len,
    reverse=True,
)
MAX_OBJECT_TYPE_ID = max(OBJECT_TYPE_ID.values())

CFG = {
    "data_dir": DEFAULT_DATA_DIR,
    "position_dir": DEFAULT_POSITION_DIR,
    "log_root": "log",
    "model_tag": "convnextv2_failobj",
    "img_size": 384,
    "batch": 16,
    "epochs": 30,
    "lr_backbone": 1e-5,
    "lr_head": 1e-3,
    "weight_decay": 0.05,
    "warmup_epochs": 2,
    "seed": 42,
    "num_workers": 8,
    "split_ratios": (0.8, 0.1, 0.1),
    "label_smoothing": 0.02,
    "early_stop_patience": 7,
    "amp": True,
    "loss": "ce",
    "focal_gamma": 2.0,
    "class_weight": "effective",                                                       # {none, inverse, effective}
    "effective_beta": 0.999,
    "ema": True,
    "ema_decay": 0.95,
    "ema_warmup": 3,
    "stochastic_depth": 0.05,
    "grad_clip": 0.5,
    "weighted_sampler": False,
    "val_loss_guard": 2.0,
    "val_smooth_window": 3,
    "save_pred_samples": False,
    "mixup": 0.0,
}

RUN_TS = datetime.now().strftime("%y%m%d_%H%M%S")                                       # YYMMDD_HHMMSS (e.g. 260501_104843)
BACKBONE_SHORT = BACKBONE.split(".")[0]                                                  # 폴더명 prefix용 — "convnextv2_base"

# ===================== Logging =====================
def setup_logger(out: Path) -> logging.Logger:
    out.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("cnn_train"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(out / "run.log", encoding="utf-8"); fh.setFormatter(fmt)
    lg.addHandler(sh); lg.addHandler(fh); return lg

# ===================== Subset / Sampling =====================
def load_subset_config(path: Optional[str]) -> Dict[str, int]:
    """YAML format:
        classes:
          Donut_scratch: 30
          Center_bank_boundary: 50
          default: 200    # optional: applies to all unspecified classes
    Returns dict {class_name: n}. 'default' key separate.
    """
    if not path: return {}
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    return d.get("classes", {}) or {}

def apply_subset(samples: List[Tuple[str, int]], idx_to_class: List[str],
                 subset_dict: Dict[str, int], seed: int) -> List[Tuple[str, int]]:
    """Slice samples to per-class caps. subset_dict can have 'default' for unspecified.
    samples: [(path, class_idx), ...] from ImageFolder
    """
    if not subset_dict: return samples
    default_n = subset_dict.get("default", None)
    rng = np.random.default_rng(seed)
    by_cls: Dict[int, List[Tuple[str, int]]] = {}
    for s in samples:
        by_cls.setdefault(s[1], []).append(s)
    out = []
    for ci, items in by_cls.items():
        cname = idx_to_class[ci]
        cap = subset_dict.get(cname, default_n)
        if cap is None or cap >= len(items):
            out.extend(items); continue
        idxs = rng.choice(len(items), size=cap, replace=False)
        out.extend(items[i] for i in sorted(idxs.tolist()))
    return out

def stratified_split(targets: List[int], ratios=(0.8, 0.1, 0.1), seed=42):
    if len(ratios) != 3:
        raise ValueError("ratios must be (train, val, test)")
    tr_ratio, va_ratio, te_ratio = [float(r) for r in ratios]
    if min(tr_ratio, va_ratio, te_ratio) < 0:
        raise ValueError("ratios must be non-negative")
    total_ratio = tr_ratio + va_ratio + te_ratio
    if total_ratio <= 0:
        raise ValueError("at least one split ratio must be positive")
    tr_ratio, va_ratio, te_ratio = (tr_ratio / total_ratio,
                                    va_ratio / total_ratio,
                                    te_ratio / total_ratio)
    rng = np.random.default_rng(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, t in enumerate(targets):
        by_cls.setdefault(t, []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in by_cls.items():
        idxs = list(idxs); rng.shuffle(idxs)
        n = len(idxs)
        if te_ratio == 0:
            n_va = max(1, int(round(n * va_ratio)))
            n_va = min(n_va, n - 1)                                                    # ensure train >= 1
            n_tr = n - n_va
            train_idx.extend(idxs[:n_tr])
            val_idx.extend(idxs[n_tr:])
            continue

        n_tr = max(1, int(n * tr_ratio)); n_va = max(1, int(n * va_ratio))
        if n_tr + n_va >= n:
            n_tr = max(1, n - 2)                                                       # ensure val + test >= 1
            n_va = max(1, n - n_tr - 1)
        train_idx.extend(idxs[:n_tr])
        val_idx.extend(idxs[n_tr:n_tr+n_va])
        test_idx.extend(idxs[n_tr+n_va:])
    return train_idx, val_idx, test_idx

class FilteredImageFolder(ImageFolder):
    """ImageFolder + EXCLUDE_CLASSES 필터."""
    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        kept = [c for c in classes if c not in EXCLUDE_CLASSES]
        new_class_to_idx = {c: i for i, c in enumerate(kept)}
        return kept, new_class_to_idx

def _add_gaussian_noise(t):
    """transforms.Lambda에 박는 module-level callable (Windows DataLoader spawn picklable)."""
    return (t + torch.randn_like(t) * 0.01).clamp(0, 1)


def build_transforms(img_size: int):
    """Wafer fail-bit map augmentation — position + palette + angle safe.
    금지된 augmentation:
      - VFlip / 180° rotation: Edge-Top ↔ Edge-Bottom 클래스 뒤집힘
      - HFlip: scratch_21deg 등 angle 자체가 클래스 정체성 — 21° → -21° 로 의미 변경
      - ColorJitter: palette index의 grade 의미(0=정상, 1-7=결함 강도) 손상
      - MixUp/CutMix/Cutout: palette pixel 평균이 무의미한 grade 생성
    안전한 augmentation:
      - 소각도 rotation ±15°: 검사장비 stage 회전 오차 범위 내 — angle 클래스
        (scratch_21deg) 도 작은 회전 robustness 정도는 OK
      - 작은 translate/scale ±3%: alignment / magnification 검사장비 variability
      - Gaussian noise σ=0.01: sensor pixel-level noise
    """
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        # NO HFlip — scratch_21deg 21° → -21° 같이 angle 클래스 정체성 변경
        transforms.RandomRotation(degrees=15, fill=0),                                 # 소각도만 — stage 회전 오차 모사
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03), fill=0),
        transforms.ToTensor(),
        transforms.Lambda(_add_gaussian_noise),                                        # tiny sensor noise
        norm,
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), norm,
    ])
    return train_tfm, eval_tfm

# ===================== Loss / Class Weight =====================
def compute_class_weights(targets: List[int], num_classes: int, mode: str,
                          beta: float = 0.999) -> Optional[torch.Tensor]:
    if mode == "none":
        return None
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)                                                   # avoid div0
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "effective":
        eff_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"unknown class_weight mode: {mode}")
    w = w / w.sum() * num_classes                                                      # normalize: mean=1
    return torch.tensor(w, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.tensor([]))
        self.label_smoothing = label_smoothing
    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            n = logits.size(-1)
            smooth = self.label_smoothing
            one_hot = F.one_hot(target, n).float()
            target_dist = one_hot * (1 - smooth) + smooth / n
            ce = -(target_dist * log_probs).sum(dim=-1)
        else:
            ce = F.nll_loss(log_probs, target, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.weight is not None and self.weight.numel() > 0:
            w = self.weight.to(logits.device).gather(0, target)
            focal = focal * w
        return focal.mean()

# ===================== EMA =====================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.95, warmup_steps: int = 0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()
    def update(self, model: nn.Module):
        self.steps += 1
        # warmup: decay starts low and ramps up
        d = self.decay if self.steps > self.warmup_steps else min(self.decay, self.steps / max(1, self.warmup_steps))
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)
    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup
    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

# ===================== Model =====================
LOCAL_WEIGHTS_DIR = Path(__file__).parent / "models"


def _backbone_weights_path() -> Path:
    """로컬 mirror 파일 path. backbone 이름에 슬래시/점 포함되니 안전 변환."""
    safe = BACKBONE.replace("/", "__").replace(":", "_")
    return LOCAL_WEIGHTS_DIR / f"{safe}.pth"


def build_model(num_classes: int, drop_path_rate: float = 0.0) -> nn.Module:
    """
    Backbone load 우선순위:
      1. models/<backbone>.pth 있으면 로컬에서 load (HF 접근 X)
      2. 없으면 timm pretrained=True로 다운로드 -> 로컬에 mirror -> 다음부턴 (1)
    폐쇄망 이전 시 models/ 폴더만 같이 복사하면 끝.
    """
    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    wf = _backbone_weights_path()
    if wf.exists():
        m = timm.create_model(BACKBONE, pretrained=False, num_classes=num_classes,
                              drop_path_rate=drop_path_rate)
        full_sd = torch.load(wf, map_location="cpu")
        m_sd = m.state_dict()
        compat = {k: v for k, v in full_sd.items()
                  if k in m_sd and m_sd[k].shape == v.shape}
        skipped = len(full_sd) - len(compat)
        m.load_state_dict(compat, strict=False)
        print(f"[backbone] loaded from local mirror: {wf.name} "
              f"({len(compat)} keys loaded, {skipped} skipped — head re-init for {num_classes} classes)",
              flush=True)
    else:
        print(f"[backbone] local mirror not found -> downloading from HF Hub once...", flush=True)
        m_full = timm.create_model(BACKBONE, pretrained=True)
        torch.save(m_full.state_dict(), wf)
        print(f"[backbone] mirrored to {wf}  (size={wf.stat().st_size/1e6:.1f} MB)", flush=True)
        m = timm.create_model(BACKBONE, pretrained=False, num_classes=num_classes,
                              drop_path_rate=drop_path_rate)
        m_sd = m.state_dict()
        compat = {k: v for k, v in m_full.state_dict().items()
                  if k in m_sd and m_sd[k].shape == v.shape}
        m.load_state_dict(compat, strict=False)
    return m

# ===================== Train / Eval =====================
def train_one_epoch(model, loader, opt, scaler, scheduler, device, lg, ep, total_ep,
                    criterion, ema: Optional[EMA], grad_clip: float):
    model.train()
    losses = []; n_correct = 0; n_total = 0
    t0 = time.time(); tick = max(1, len(loader)//10)
    use_amp = (device.type == "cuda")
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16)
    for it, (imgs, lbls) in enumerate(loader, 1):
        imgs = imgs.to(device, non_blocking=True); lbls = lbls.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(imgs)
            loss = criterion(logits, lbls)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt); scaler.update()
        if scheduler is not None: scheduler.step()
        if ema is not None: ema.update(model)
        losses.append(float(loss))
        with torch.no_grad():
            preds = logits.argmax(1)
        n_correct += int((preds == lbls).sum()); n_total += imgs.size(0)
        if it % tick == 0 or it == len(loader):
            lg.info(f"  Ep {ep}/{total_ep} | {100*it/len(loader):5.1f}% | loss={np.mean(losses):.4f} acc={100*n_correct/n_total:.2f}%")
    return float(np.mean(losses)), n_correct/max(1,n_total)

@torch.no_grad()
def evaluate(model, loader, device, classes, criterion=None, return_paths: bool = False):
    model.eval()
    all_p = []; all_l = []; all_conf = []; all_loss = []; all_paths = []
    for batch in loader:
        if return_paths and len(batch) == 3:
            imgs, lbls, paths = batch
        else:
            imgs, lbls = batch[0], batch[1]; paths = None
        imgs = imgs.to(device, non_blocking=True); lbls_dev = lbls.to(device)
        logits = model(imgs)
        if criterion is not None:
            l = criterion(logits, lbls_dev)
            all_loss.append(float(l) * imgs.size(0))
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_p.append(preds.cpu().numpy()); all_l.append(lbls.numpy()); all_conf.append(confs.cpu().numpy())
        if paths is not None: all_paths.extend(list(paths))
    preds = np.concatenate(all_p); lbls = np.concatenate(all_l); confs = np.concatenate(all_conf)
    acc = accuracy_score(lbls, preds)
    p, r, f1, _ = precision_recall_fscore_support(lbls, preds, average='macro', zero_division=0)
    res = {"acc": float(acc), "macro_p": float(p), "macro_r": float(r), "macro_f1": float(f1),
           "preds": preds.tolist(), "labels": lbls.tolist(), "confs": confs.tolist(),
           "classes": classes}
    if all_loss:
        res["val_loss"] = float(sum(all_loss) / max(1, len(lbls)))
    if return_paths:
        res["paths"] = all_paths
    return res

def save_confusion_matrix(eval_res, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        classes = eval_res["classes"]
        n = len(classes)
        cm = confusion_matrix(eval_res["labels"], eval_res["preds"], labels=list(range(n)))
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        plt.colorbar(im, ax=ax)
        # cell annotations — non-zero만, 진한 셀은 흰 글자
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        font_size = 6 if n <= 36 else 5
        for i in range(n):
            for j in range(n):
                v = int(cm[i, j])
                if v == 0:
                    continue
                ax.text(j, i, str(v),
                        ha="center", va="center",
                        fontsize=font_size,
                        color="white" if v > thresh else "black")
        fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[confusion_matrix save fail] {e}")


def save_confusion_matrix_combined(val_res, test_res, out_path: Path):
    """Test (위) + Val (아래) 2단 combined confusion matrix PNG.

    각 subplot은 셀 숫자 annotation 포함 (진한 셀 흰 글자).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        classes = val_res["classes"]; n = len(classes)
        font_size = 6 if n <= 36 else 5

        cm_test = confusion_matrix(test_res["labels"], test_res["preds"], labels=list(range(n)))
        cm_val  = confusion_matrix(val_res["labels"],  val_res["preds"],  labels=list(range(n)))

        fig, axes = plt.subplots(2, 1, figsize=(14, 24))
        for ax, cm, title in [(axes[0], cm_test, "TEST"),
                              (axes[1], cm_val,  "VAL")]:
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(classes, rotation=90, fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(title, fontsize=14, fontweight="bold")
            plt.colorbar(im, ax=ax)
            thresh = cm.max() / 2.0 if cm.max() > 0 else 0
            for i in range(n):
                for j in range(n):
                    v = int(cm[i, j])
                    if v == 0: continue
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=font_size,
                            color="white" if v > thresh else "black")
        fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    except Exception as e:
        print(f"[confusion_matrix combined save fail] {e}")


def save_curves_png(history: List[dict], out_path: Path):
    """3-axis figure: train_loss / val_loss / val_macro_f1."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps = [r["epoch"] for r in history]
        tr_loss = [r["train_loss"] for r in history]
        va_loss = [r.get("val_loss", float('nan')) for r in history]
        va_f1   = [r["val_macro_f1"] for r in history]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(eps, tr_loss, marker='o', label='train_loss'); axes[0].set_ylabel('train_loss'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(eps, va_loss, marker='o', color='tab:orange', label='val_loss'); axes[1].set_ylabel('val_loss'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(eps, va_f1, marker='o', color='tab:green', label='val_macro_f1'); axes[2].set_ylabel('val_macro_f1'); axes[2].set_xlabel('epoch'); axes[2].grid(True, alpha=0.3)
        for a in axes: a.legend(loc='best')
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    except Exception as e:
        print(f"[curves save fail] {e}")

def format_per_class_report(eval_res: dict) -> str:
    """class별 F1/FP/FN/Support + weighted/macro 요약 문자열 반환."""
    classes = eval_res["classes"]; n = len(classes)
    cm = confusion_matrix(eval_res["labels"], eval_res["preds"], labels=list(range(n)))
    rep = classification_report(eval_res["labels"], eval_res["preds"],
                                target_names=classes, output_dict=True, zero_division=0)
    cls_w = max(8, max((len(c) for c in classes), default=8) + 2)                       # tight to longest name + 2
    col_score = 8                                                                       # F1/P/R: "  0.000"
    col_count = 6                                                                       # FP/FN/Sup
    sep = "  "                                                                          # 2-space gap
    fmt_h = f"{{:<{cls_w}}}{sep}{{:>{col_score}}}{sep}{{:>{col_score}}}{sep}{{:>{col_score}}}{sep}{{:>{col_count}}}{sep}{{:>{col_count}}}{sep}{{:>{col_count}}}"
    fmt_r = f"{{:<{cls_w}}}{sep}{{:>{col_score}.3f}}{sep}{{:>{col_score}.3f}}{sep}{{:>{col_score}.3f}}{sep}{{:>{col_count}d}}{sep}{{:>{col_count}d}}{sep}{{:>{col_count}d}}"
    fmt_avg = f"{{:<{cls_w}}}{sep}{{:>{col_score}.3f}}{sep}{{:>{col_score}.3f}}{sep}{{:>{col_score}.3f}}{sep}{{:>{col_count}}}{sep}{{:>{col_count}}}{sep}{{:>{col_count}d}}"
    header = fmt_h.format("Class", "F1", "P", "R", "FP", "FN", "Sup")
    sepline = "-" * len(header)
    lines = [header, sepline]
    for i, c in enumerate(classes):
        d = rep.get(c, {"precision":0, "recall":0, "f1-score":0, "support":0})
        FP = int(cm[:, i].sum() - cm[i, i])
        FN = int(cm[i, :].sum() - cm[i, i])
        sup = int(d.get("support", 0))
        lines.append(fmt_r.format(c, d['f1-score'], d['precision'], d['recall'], FP, FN, sup))
    lines.append(sepline)
    for k in ("macro avg", "weighted avg"):
        d = rep[k]
        lines.append(fmt_avg.format(k, d['f1-score'], d['precision'], d['recall'], "-", "-", int(d['support'])))
    lines.append(sepline)
    lines.append(f"{'overall acc':<{cls_w}}{sep}{eval_res['acc']:>{col_score}.3f}")
    return "\n".join(lines) + "\n"


def _format_per_class_block(snap: dict) -> str:
    """1개 best의 TEST(위)+VAL(아래) per-class 표만 반환."""
    v = snap["val_res"]; t = snap.get("test_res")
    out = []
    if t is not None:
        out.extend(["[TEST per-class]", format_per_class_report(t).rstrip(), ""])
    out.extend(["[VAL per-class]", format_per_class_report(v).rstrip()])
    return "\n".join(out)


def _format_summary_table(snapshots: List[dict]) -> str:
    """epoch별 total best 요약 표 — VAL/TEST aggregate metrics 한 줄/best."""
    has_test = any(s.get("test_res") is not None for s in snapshots)
    if has_test:
        head = (f"{'ep':>4s}  {'smooth_f1':>9s}  |  "
                f"{'VAL acc':>7s}  {'f1':>6s}  {'p':>6s}  {'r':>6s}  |  "
                f"{'TEST acc':>8s}  {'f1':>6s}  {'p':>6s}  {'r':>6s}")
    else:
        head = (f"{'ep':>4s}  {'smooth_f1':>9s}  |  "
                f"{'VAL acc':>7s}  {'f1':>6s}  {'p':>6s}  {'r':>6s}")
    sep = "-" * len(head)
    lines = [head, sep]
    for s in snapshots:
        v = s["val_res"]; t = s.get("test_res")
        row = (f"{s['epoch']:>4d}  {s['smooth_f1']:>9.4f}  |  "
               f"{v['acc']*100:>7.2f}  {v['macro_f1']*100:>6.2f}  "
               f"{v['macro_p']*100:>6.2f}  {v['macro_r']*100:>6.2f}")
        if t is not None:
            row += (f"  |  {t['acc']*100:>8.2f}  {t['macro_f1']*100:>6.2f}  "
                    f"{t['macro_p']*100:>6.2f}  {t['macro_r']*100:>6.2f}")
        elif has_test:
            row += "  |  " + (" " * 8 + "  " + " " * 6 + "  " + " " * 6 + "  " + " " * 6)
        lines.append(row)
    return "\n".join(lines)


def write_best_history(snapshots: List[dict], out_path: Path):
    """val-best 갱신 시점들을 한 파일에 누적 (유일한 result txt — 개별 val/test
    per-class report 파일과 eval_summary.json 모두 폐지).

    구조:
      0) 맨 윗줄: BEST OVERALL one-liner (TEST + VAL aggregate, FINAL 시점)
      1) FINAL BEST per-class (TEST 위 + VAL 아래)
      2) BEST UPDATES SUMMARY (best 갱신 epoch별 한 줄, VAL/TEST aggregate)
      3) PER-EPOCH PER-CLASS REPORTS (best #1부터 시간순 per-class 상세)

    각 snapshot dict: {epoch, smooth_f1, val_res, test_res(opt), train_loss(opt)}
    """
    if not snapshots:
        return
    total = len(snapshots)
    final = snapshots[-1]
    fv = final["val_res"]; ft = final.get("test_res")
    bar = "=" * 90

    # ===== Section 0: BEST OVERALL 맨윗줄 =====
    sec0 = [bar,
            f"★ BEST OVERALL  |  epoch {final['epoch']}  |  smoothed val F1 = {final['smooth_f1']:.4f}",
            bar]
    if ft is not None:
        sec0.append(f"TEST  acc={ft['acc']*100:6.2f}%   f1={ft['macro_f1']*100:6.2f}%   "
                    f"p={ft['macro_p']*100:6.2f}%   r={ft['macro_r']*100:6.2f}%")
    sec0.append(f"VAL   acc={fv['acc']*100:6.2f}%   f1={fv['macro_f1']*100:6.2f}%   "
                f"p={fv['macro_p']*100:6.2f}%   r={fv['macro_r']*100:6.2f}%")

    # ===== Section 1: FINAL BEST per-class =====
    sec1 = [bar,
            "[1] FINAL BEST per-class",
            bar, "",
            _format_per_class_block(final)]

    # ===== Section 2: epoch별 best update 요약 =====
    sec2 = [bar,
            "[2] BEST UPDATES SUMMARY  (one row per best-val improvement)",
            bar, "",
            _format_summary_table(snapshots)]

    # ===== Section 3: epoch별 per-class 상세 =====
    sec3_blocks = [bar,
                   "[3] PER-EPOCH PER-CLASS REPORTS  (every best-val update, chronological)",
                   bar, ""]
    for i, snap in enumerate(snapshots, 1):
        marker = "  (= FINAL)" if i == total else ""
        head = (f"---- best #{i}{marker}  |  epoch {snap['epoch']}  |  "
                f"smoothed val F1 = {snap['smooth_f1']:.4f}")
        if snap.get("train_loss") is not None:
            head += f"  |  train_loss={snap['train_loss']:.4f}"
        head += " ----"
        sec3_blocks.extend([head, "", _format_per_class_block(snap), ""])

    out_path.write_text(
        "\n".join(sec0) + "\n\n\n" +
        "\n".join(sec1) + "\n\n\n" +
        "\n".join(sec2) + "\n\n\n" +
        "\n".join(sec3_blocks).rstrip() + "\n",
        encoding="utf-8",
    )

def save_pred_samples(eval_res: dict, out_dir: Path, max_per_bucket: int = 20):
    """TP/FP/FN 샘플 이미지를 폴더 구조로 저장."""
    if "paths" not in eval_res:
        return
    classes = eval_res["classes"]
    paths = eval_res["paths"]; labels = eval_res["labels"]; preds = eval_res["preds"]
    counters: Dict[str, int] = {}
    for p, l, pr in zip(paths, labels, preds):
        if l == pr:
            kind = "tp"; cls_dir = classes[l]
        else:
            # FP for predicted class, FN for true class
            for kind, ci in (("fp", pr), ("fn", l)):
                key = f"{kind}/{classes[ci]}"
                if counters.get(key, 0) >= max_per_bucket:
                    continue
                counters[key] = counters.get(key, 0) + 1
                d = out_dir / kind / classes[ci]
                d.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(p, d / f"true-{classes[l]}__pred-{classes[pr]}__{Path(p).name}")
                except Exception:
                    pass
            continue
        key = f"tp/{cls_dir}"
        if counters.get(key, 0) >= max_per_bucket:
            continue
        counters[key] = counters.get(key, 0) + 1
        d = out_dir / "tp" / cls_dir
        d.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, d / Path(p).name)
        except Exception:
            pass

def save_wrong_tree(eval_res: dict, out_dir: Path):
    """
    틀린 예측 이미지를 트리로 저장: out_dir/<true_class>/<pred_class>/<filename>
    best 갱신 시마다 호출, 기존 폴더는 통째로 갈아엎음(stale 방지).
    학습 결과 root는 보존 — 이 함수는 wrong 서브트리만 갱신.
    """
    if "paths" not in eval_res:
        return
    classes = eval_res["classes"]
    paths = eval_res["paths"]; labels = eval_res["labels"]; preds = eval_res["preds"]
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p, l, pr in zip(paths, labels, preds):
        if l == pr:
            continue
        d = out_dir / classes[l] / classes[pr]
        d.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, d / Path(p).name)
            n += 1
        except Exception:
            pass
    return n


def rename_run_dir(out_dir: Path, model_tag: str, ts: str,
                   test_f1: Optional[float], val_f1: float) -> Path:
    """log/{model_tag}_{TS}_{test_f1:.2f}_{val_f1:.2f}/ 로 이름 변경.

    test 없으면 (--train-val-only) 끝 segment 1개 (val_f1만).
    model_tag default = backbone short name (e.g. convnextv2_base).
    """
    if test_f1 is not None:
        new_name = f"{model_tag}_{ts}_{test_f1:.2f}_{val_f1:.2f}"
    else:
        new_name = f"{model_tag}_{ts}_{val_f1:.2f}"
    parent = out_dir.parent
    new_path = parent / new_name
    if new_path.exists():
        new_name += f"_{int(time.time())}"
        new_path = parent / new_name
    try:
        out_dir.rename(new_path)
        return new_path
    except Exception:
        return out_dir

# ===================== Path-aware datasets (for pred sample saving) =====================
class IndexPathSubset(Dataset):
    """Subset wrapper that returns (image, label, path)."""
    def __init__(self, base: ImageFolder, indices: List[int]):
        self.base = base; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        img, lbl = self.base[idx]
        path = self.base.samples[idx][0]
        return img, lbl, path

# ===================== Main =====================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=CFG["data_dir"])
    p.add_argument("--log-root", default=CFG["log_root"])
    p.add_argument("--model-tag", default=CFG["model_tag"])
    p.add_argument("--epochs", type=int, default=CFG["epochs"])
    p.add_argument("--batch", type=int, default=CFG["batch"])
    p.add_argument("--img-size", type=int, default=CFG["img_size"])
    p.add_argument("--workers", type=int, default=CFG["num_workers"])
    p.add_argument("--seed", type=int, default=CFG["seed"])
    p.add_argument("--split-ratios", nargs=3, type=float, default=None,
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="stratified split ratios, e.g. 0.8 0.2 0")
    p.add_argument("--train-val-only", action="store_true",
                   help="use 0.8/0.2/0.0 split and skip test evaluation")
    p.add_argument("--lr-backbone", type=float, default=CFG["lr_backbone"])
    p.add_argument("--lr-head", type=float, default=CFG["lr_head"])
    p.add_argument("--patience", type=int, default=CFG["early_stop_patience"])
    p.add_argument("--subset-config", default=None, help="YAML: classes:{name:N, default:N}")
    p.add_argument("--loss", choices=["ce", "focal"], default=CFG["loss"])
    p.add_argument("--focal-gamma", type=float, default=CFG["focal_gamma"])
    p.add_argument("--class-weight", choices=["none", "inverse", "effective"], default=CFG["class_weight"])
    p.add_argument("--effective-beta", type=float, default=CFG["effective_beta"])
    p.add_argument("--label-smoothing", type=float, default=CFG["label_smoothing"])
    p.add_argument("--ema", action="store_true", default=CFG["ema"])
    p.add_argument("--no-ema", action="store_false", dest="ema")
    p.add_argument("--ema-decay", type=float, default=CFG["ema_decay"])
    p.add_argument("--ema-warmup", type=int, default=CFG["ema_warmup"])
    p.add_argument("--stochastic-depth", type=float, default=CFG["stochastic_depth"])
    p.add_argument("--grad-clip", type=float, default=CFG["grad_clip"])
    p.add_argument("--warmup-epochs", type=int, default=CFG["warmup_epochs"])
    p.add_argument("--weighted-sampler", action="store_true", default=CFG["weighted_sampler"])
    p.add_argument("--val-loss-guard", type=float, default=CFG["val_loss_guard"])
    p.add_argument("--val-smooth-window", type=int, default=CFG["val_smooth_window"])
    p.add_argument("--save-pred-samples", action="store_true", default=CFG["save_pred_samples"])
    p.add_argument("--require-gpu", action="store_true", help="GPU 불가/한계 초과 시 시작 차단 (default: CPU fallback)")
    p.add_argument("--ram-limit", type=float, default=80.0, help="RAM percent limit (시작 차단 + 학습중 abort)")
    p.add_argument("--gpu-mem-limit", type=float, default=90.0, help="GPU mem percent limit (초과 시 CPU fallback)")
    p.add_argument("--monitor-interval", type=float, default=30.0, help="resource monitor 주기 sec")
    args = p.parse_args()
    if args.train_val_only and args.split_ratios is not None:
        p.error("--train-val-only and --split-ratios cannot be used together")

    split_ratios = (0.8, 0.2, 0.0) if args.train_val_only else (
        tuple(args.split_ratios) if args.split_ratios is not None else CFG["split_ratios"])

    # ===== Resource guard (시작 점검) =====
    a = assess_start(ram_limit=args.ram_limit, gpu_mem_limit=args.gpu_mem_limit,
                     require_gpu=args.require_gpu)
    print(format_assessment(a), flush=True)
    if not a["ok"]:
        print("[guard] 시작 차단. RAM/CPU 정리 후 재시도하세요.", file=sys.stderr)
        sys.exit(2)

    # Mutually exclusive: weighted_sampler vs class_weight
    if args.weighted_sampler and args.class_weight != "none":
        print("[!] --weighted-sampler given; forcing --class-weight=none", file=sys.stderr)
        args.class_weight = "none"

    # seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(a["device"])
    out_root = Path(args.log_root); out_root.mkdir(parents=True, exist_ok=True)
    # Initial folder name (will rename at end with test F1/recall)
    init_name = f"{args.model_tag}_{RUN_TS}_running"
    out_dir = out_root / init_name
    lg = setup_logger(out_dir)
    lg.info("===== CNN train start =====")
    lg.info(f"device={device} backbone={BACKBONE} img={args.img_size} batch={args.batch} epochs={args.epochs}")
    lg.info(f"data_dir={args.data_dir}, exclude={EXCLUDE_CLASSES}")
    lg.info(f"loss={args.loss} class_weight={args.class_weight} ema={args.ema} drop_path={args.stochastic_depth}")

    # ===== Data =====
    train_tfm, eval_tfm = build_transforms(args.img_size)
    full_eval = FilteredImageFolder(args.data_dir, transform=eval_tfm)
    full_train = FilteredImageFolder(args.data_dir, transform=train_tfm)
    classes = full_eval.classes; num_classes = len(classes)
    lg.info(f"Classes ({num_classes}): {classes}")
    lg.info(f"Total samples (pre-subset): {len(full_eval)}")

    # subset
    subset_dict = load_subset_config(args.subset_config)
    if subset_dict:
        lg.info(f"Subset config: {subset_dict}")
        full_train.samples = apply_subset(full_train.samples, classes, subset_dict, args.seed)
        full_eval.samples = apply_subset(full_eval.samples, classes, subset_dict, args.seed)
        # ImageFolder targets list 동기화
        full_train.targets = [s[1] for s in full_train.samples]
        full_eval.targets = [s[1] for s in full_eval.samples]
        lg.info(f"Total samples (post-subset): {len(full_eval)}")
        cnt = np.bincount(full_eval.targets, minlength=num_classes)
        for ci, c in enumerate(classes):
            lg.info(f"  - {c}: {int(cnt[ci])}")

    # split
    targets = full_eval.targets
    tr_idx, va_idx, te_idx = stratified_split(targets, split_ratios, args.seed)
    has_test = len(te_idx) > 0
    lg.info(f"Split ratios: train={split_ratios[0]} val={split_ratios[1]} test={split_ratios[2]}")
    lg.info(f"Split sizes: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")
    train_set = Subset(full_train, tr_idx)
    val_set   = Subset(full_eval, va_idx)

    # weighted sampler (optional)
    train_sampler = None; shuffle_train = True
    if args.weighted_sampler:
        tr_targets = [targets[i] for i in tr_idx]
        cnt = np.bincount(tr_targets, minlength=num_classes).astype(np.float64)
        cnt = np.maximum(cnt, 1.0)
        sample_w = np.array([1.0 / cnt[t] for t in tr_targets])
        train_sampler = WeightedRandomSampler(weights=torch.tensor(sample_w, dtype=torch.float64),
                                              num_samples=len(tr_targets), replacement=True)
        shuffle_train = False
        lg.info("WeightedRandomSampler enabled.")

    train_ld = DataLoader(train_set, batch_size=args.batch, shuffle=shuffle_train, sampler=train_sampler,
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)
    val_ld   = DataLoader(val_set,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)

    # test DataLoader with paths (for save_pred_samples)
    test_ld = None
    if has_test:
        test_set_p = IndexPathSubset(full_eval, te_idx)
        test_ld = DataLoader(test_set_p, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True, persistent_workers=args.workers>0)

    # ===== Class weights =====
    cw = compute_class_weights([targets[i] for i in tr_idx], num_classes,
                               args.class_weight, args.effective_beta) if args.class_weight != "none" else None
    if cw is not None:
        lg.info(f"class_weight ({args.class_weight}) min={float(cw.min()):.3f} max={float(cw.max()):.3f}")

    # ===== Loss =====
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=cw, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    criterion = criterion.to(device)

    # ===== Model =====
    model = build_model(num_classes, drop_path_rate=args.stochastic_depth).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # ===== Optim =====
    head_params, backbone_params = [], []
    for n, p_ in model.named_parameters():
        (head_params if n.startswith("head.") else backbone_params).append(p_)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr_backbone},
         {"params": head_params,     "lr": args.lr_head}],
        weight_decay=CFG["weight_decay"])

    steps_per_epoch = max(1, len(train_ld))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    sched_warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_steps))
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[sched_warmup, sched_cos], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    # EMA
    ema = EMA(model, decay=args.ema_decay, warmup_steps=args.ema_warmup * steps_per_epoch) if args.ema else None
    if ema is not None: lg.info(f"EMA enabled (decay={args.ema_decay}, warmup_steps={args.ema_warmup*steps_per_epoch})")

    # ===== Save hparams (snapshot) =====
    hparams_dict = {**CFG, **vars(args), "backbone": BACKBONE, "num_classes": num_classes,
                    "classes": classes, "run_ts": RUN_TS, "device": str(device),
                    "split_ratios_used": split_ratios,
                    "torch_version": str(torch.__version__),
                    "timm_version": str(getattr(timm, "__version__", "?"))}
    def _yaml_safe(v):
        # str subclass(TorchVersion 등)·tuple·기타 객체는 plain str로 직렬화
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if type(v) is str:
            return v
        if isinstance(v, str):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [_yaml_safe(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _yaml_safe(x) for k, x in v.items()}
        return str(v)
    with open(out_dir / "hparams.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: _yaml_safe(v) for k, v in hparams_dict.items()},
                       f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "hparams.txt", "w", encoding="utf-8") as f:
        for k, v in hparams_dict.items():
            f.write(f"{k:<24}: {v}\n")

    # ===== Train loop =====
    history = []; best_score = -1.0; best_ep = 0; no_improve = 0
    best_val_loss = float('inf')
    val_f1_window: List[float] = []
    best_snapshots: List[dict] = []                                                     # 모든 best 갱신 누적 (best_history.txt용)

    # Resource monitor (RAM 한계 초과 시 학습 자동 중단)
    monitor = ResourceMonitor(ram_limit=args.ram_limit,
                              interval_sec=args.monitor_interval, logger=lg)
    monitor.start()
    aborted_reason: Optional[str] = None

    for ep in range(1, args.epochs + 1):
        if monitor.should_abort():
            aborted_reason = monitor.abort_reason
            lg.info(f"  [guard] aborting before epoch {ep}: {aborted_reason}")
            break
        tr_loss, tr_acc = train_one_epoch(model, train_ld, opt, scaler, scheduler, device, lg,
                                          ep, args.epochs, criterion, ema, args.grad_clip)

        # eval (with EMA shadow if enabled)
        if ema is not None:
            backup = ema.apply_to(model)
            va = evaluate(model, val_ld, device, classes, criterion=criterion)
            ema.restore(model, backup)
        else:
            va = evaluate(model, val_ld, device, classes, criterion=criterion)

        rec = {"epoch": ep, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
               "val_loss": va.get("val_loss", float('nan')),
               "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
               "val_macro_p": va["macro_p"], "val_macro_r": va["macro_r"]}
        history.append(rec)
        lg.info(f"[Ep {ep}] tr_loss={tr_loss:.4f} | val_loss={rec['val_loss']:.4f} acc={100*va['acc']:.2f}% f1={100*va['macro_f1']:.2f}%")
        # 매 epoch마다 curves.png + history.json 갱신 — mid-run 종료시에도 결과 보존
        save_curves_png(history, out_dir / "curves.png")
        with open(out_dir / "history.json", "w", encoding="utf-8") as _hf:
            json.dump({"history": history, "best_epoch": best_ep,
                       "best_smoothed_val_f1": best_score}, _hf, indent=2)

        # smoothed val F1 (median of last N)
        val_f1_window.append(va["macro_f1"])
        if len(val_f1_window) > args.val_smooth_window:
            val_f1_window.pop(0)
        smooth_f1 = float(np.median(val_f1_window))

        # val_loss guard: skip best save if val_loss > guard * best_val_loss
        guard_block = (rec["val_loss"] is not None
                       and not np.isnan(rec["val_loss"])
                       and best_val_loss < float('inf')
                       and rec["val_loss"] > args.val_loss_guard * best_val_loss)

        if (smooth_f1 > best_score) and not guard_block:
            best_score = smooth_f1; best_ep = ep; no_improve = 0
            best_val_loss = min(best_val_loss, rec["val_loss"]) if not np.isnan(rec["val_loss"]) else best_val_loss

            # === best 갱신 시점에 val/test path 포함 evaluate ===
            # 6400x6400 palette PNG는 PIL convert 시 122MB/img — 이 시점엔 train/val/test 3개 DataLoader가
            # 이미 prefetch 중이라 추가 worker spawn하면 메모리 폭증. 여기선 num_workers=0 (단일 프로세스).
            best_val_ld = DataLoader(IndexPathSubset(full_eval, va_idx),
                                     batch_size=args.batch, shuffle=False,
                                     num_workers=0, pin_memory=True)
            if ema is not None:
                backup_t = ema.apply_to(model)
                val_res = evaluate(model, best_val_ld, device, classes, return_paths=True)
                test_res = evaluate(model, test_ld, device, classes, return_paths=True) if test_ld is not None else None
                ema.restore(model, backup_t)
            else:
                val_res = evaluate(model, best_val_ld, device, classes, return_paths=True)
                test_res = evaluate(model, test_ld, device, classes, return_paths=True) if test_ld is not None else None
            del best_val_ld
            if test_res is not None:
                lg.info(f"  ✓ best | VAL acc={100*val_res['acc']:.2f}% f1={100*val_res['macro_f1']:.2f}% | TEST acc={100*test_res['acc']:.2f}% f1={100*test_res['macro_f1']:.2f}% r={100*test_res['macro_r']:.2f}%")
            else:
                lg.info(f"  ✓ best | VAL acc={100*val_res['acc']:.2f}% f1={100*val_res['macro_f1']:.2f}% r={100*val_res['macro_r']:.2f}%")

            # save best ckpt
            ckpt = {"epoch": ep, "model": model.state_dict(), "classes": classes,
                    "img_size": args.img_size, "backbone": BACKBONE,
                    "val_macro_f1": float(val_res["macro_f1"]), "val_acc": float(val_res["acc"]),
                    "val_macro_r": float(val_res["macro_r"]),
                    "smoothed_val_f1": float(smooth_f1)}
            if test_res is not None:
                ckpt.update({"test_macro_f1": float(test_res["macro_f1"]),
                             "test_macro_r": float(test_res["macro_r"]),
                             "test_acc": float(test_res["acc"])})
            if ema is not None:
                ckpt["ema_state"] = ema.state_dict()
            torch.save(ckpt, out_dir / "best_model.pth")

            # confusion matrix — combined만 (test 위 / val 아래) 또는 val 단독
            n_wrong_val = save_wrong_tree(val_res, out_dir / "wrong" / "val")
            if test_res is not None:
                save_confusion_matrix_combined(val_res, test_res,
                                               out_dir / "best_confusion_matrix.png")
                n_wrong = save_wrong_tree(test_res, out_dir / "wrong" / "test")
                lg.info(f"  wrong saved: test={n_wrong} val={n_wrong_val} -> wrong/{{test,val}}/<true>/<pred>/")
            else:
                save_confusion_matrix(val_res, out_dir / "best_confusion_matrix.png")
                lg.info(f"  wrong saved: val={n_wrong_val} -> wrong/val/<true>/<pred>/")

            if test_res is not None:
                best_test_res = test_res
            best_val_res = val_res

            # === 역대 best history 누적 — 최신(=최종)이 맨 위 ===
            best_snapshots.append({
                "epoch": ep,
                "smooth_f1": float(smooth_f1),
                "train_loss": float(tr_loss),
                "val_res": val_res,
                "test_res": test_res,
            })
            write_best_history(best_snapshots, out_dir / "best_history.txt")
        else:
            no_improve += 1
            if guard_block:
                lg.info(f"  [guard] val_loss={rec['val_loss']:.4f} > {args.val_loss_guard}×{best_val_loss:.4f} — best NOT updated.")
            if no_improve >= args.patience:
                lg.info(f"  early stop at ep {ep} (no improve for {args.patience}).")
                break

    # stop monitor
    monitor.stop()

    # save history
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_epoch": best_ep, "best_smoothed_val_f1": best_score,
                   "aborted": aborted_reason}, f, indent=2)
    save_curves_png(history, out_dir / "curves.png")

    # final pred samples
    if args.save_pred_samples and 'best_test_res' in dir():
        save_pred_samples(best_test_res, out_dir / "predictions", max_per_bucket=20)
        lg.info(f"  pred samples saved → predictions/")
    elif args.save_pred_samples and 'best_val_res' in dir():
        save_pred_samples(best_val_res, out_dir / "predictions_val", max_per_bucket=20)
        lg.info(f"  pred samples saved → predictions_val/")

    # rename folder log/{model_tag}_{TS}_{test_f1:.2f}_{val_f1:.2f}/ — test 없으면 val_f1만
    final_test_f1 = best_test_res["macro_f1"] if 'best_test_res' in dir() else None
    final_val_f1  = best_val_res["macro_f1"]  if 'best_val_res' in dir() else 0.0
    final_dir = rename_run_dir(out_dir, args.model_tag, RUN_TS, final_test_f1, final_val_f1)
    if aborted_reason:
        # abort 표시 — 결과 폴더는 보존, 이름만 suffix 추가
        aborted_dir = final_dir.with_name(final_dir.name + "_ABORTED")
        try:
            final_dir.rename(aborted_dir)
            final_dir = aborted_dir
        except Exception as e:
            lg.info(f"  [guard] rename to ABORTED failed: {e}")
        lg.info(f"[Aborted] reason: {aborted_reason}")
    lg.info(f"[Metric source] {metric_source}")
    lg.info(f"[Done] outputs: {final_dir.resolve()}")
    lg.info("===== END =====")

if __name__ == "__main__":
    main()
