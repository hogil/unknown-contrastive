# -*- coding: utf-8 -*-
"""Train one Stage 2 chip 4-class CNN variant.

Variants:
- T1: CE + label_smoothing 0.1
- T4: ASL (gamma_pos=1, gamma_neg=4, clip=0.05) on one-hot multi-hot target
- T5: BCE on one-hot multi-hot target
- T6: BCE warmup 5ep -> ASL

Init backbone state_dict from existing chip5_round4_v14 best_model.pth (TAPT).
Drops invalid_main / particle_blast / scratch_21deg from training set.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import DEFAULT_BACKBONE_CKPT, TRAIN_CLASSES, TRAIN_VARIANTS
from .losses import BCEThenASL, build_loss

VARIANT_TO_LOSS = {
    "T1": "ce_ls01",
    "T4": "asl",
    "T5": "bce",
    "T6": "bce_then_asl",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChipFolderDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], img_size: int, train: bool):
        self.samples = samples
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=(0, 0)),
                    transforms.RandomRotation(degrees=(90, 90)),
                    transforms.RandomRotation(degrees=(180, 180)),
                    transforms.RandomRotation(degrees=(270, 270)),
                ]),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


def collect_samples(root: Path) -> List[Tuple[Path, int]]:
    out: List[Tuple[Path, int]] = []
    for ci, cname in enumerate(TRAIN_CLASSES):
        d = root / cname
        if not d.exists():
            raise FileNotFoundError(f"missing class dir: {d}")
        for png in sorted(d.glob("*.png")):
            out.append((png, ci))
    return out


def stratified_split(samples: List[Tuple[Path, int]], val_ratio: float = 0.2, seed: int = 42):
    by = {}
    for p, y in samples:
        by.setdefault(y, []).append(p)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for y, paths in by.items():
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(paths) * val_ratio)))
        for i in idx[:n_val]:
            val.append((paths[i], y))
        for i in idx[n_val:]:
            train.append((paths[i], y))
    return train, val


def build_model(num_classes: int, init_ckpt: Path) -> Tuple[torch.nn.Module, str, int]:
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    backbone = ckpt["backbone"]
    img_size = int(ckpt["img_size"])
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    sd = ckpt["model"]
    msd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    skipped = len(sd) - len(compat)
    model.load_state_dict(compat, strict=False)
    print(f"[init] backbone={backbone} loaded {len(compat)}/{len(sd)} keys (skipped={skipped})")
    return model, backbone, img_size


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    correct = 0
    total = 0
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.size(0))
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_TO_LOSS.keys()))
    ap.add_argument("--ls", type=float, default=None,
                    help="label smoothing (T1 only). default 0.1 if not set.")
    ap.add_argument("--tag", type=str, default="",
                    help="optional tag suffix for out_dir name")
    ap.add_argument("--data-root", default="D:/project/data/wm-811k/classification_chips")
    ap.add_argument("--init-ckpt", default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--out-root", default="outputs/logs_chip_multilabel")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    name = args.variant if not args.tag else f"{args.variant}_{args.tag}"
    out_dir = Path(args.out_root) / f"{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train {args.variant}] device={device}  out={out_dir}")
    samples = collect_samples(Path(args.data_root))
    train_samples, val_samples = stratified_split(samples, val_ratio=0.2, seed=args.seed)
    print(f"[train] data: train={len(train_samples)} val={len(val_samples)} classes={TRAIN_CLASSES}")

    model, backbone, img_size = build_model(num_classes=len(TRAIN_CLASSES), init_ckpt=Path(args.init_ckpt))
    model = model.to(device)

    train_ds = ChipFolderDataset(train_samples, img_size, train=True)
    val_ds = ChipFolderDataset(val_samples, img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)

    loss_name = VARIANT_TO_LOSS[args.variant]
    loss_fn, target_kind = build_loss(loss_name)
    if args.variant == "T1" and args.ls is not None:
        from .losses import CEWithSmoothing
        loss_fn = CEWithSmoothing(smoothing=float(args.ls))
        loss_name = f"ce_ls{args.ls}"
    loss_fn = loss_fn.to(device)
    print(f"[train] loss={loss_name} target_kind={target_kind}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    history = []
    best_val_acc = 0.0
    best_epoch = -1
    t_total = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        if isinstance(loss_fn, BCEThenASL):
            loss_fn.set_epoch(ep - 1)
        running = 0.0
        nb = 0
        optim.zero_grad()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if target_kind == "multi_hot":
                tgt = torch.zeros(y.size(0), len(TRAIN_CLASSES), device=device)
                tgt[torch.arange(y.size(0)), y] = 1.0
                tgt_used = tgt
            else:
                tgt_used = y
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, tgt_used) / args.accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.item()) * args.accum
            nb += 1
            if (step + 1) % args.accum == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                optim.zero_grad()
        sched.step()
        val_acc = evaluate(model, val_loader, device, num_classes=len(TRAIN_CLASSES))
        active = getattr(loss_fn, "last_active", loss_name)
        avg_loss = running / max(nb, 1)
        history.append({"epoch": ep, "train_loss": avg_loss, "val_acc": val_acc,
                        "lr": optim.param_groups[0]["lr"], "loss_active": active})
        print(f"[ep {ep:02d}] loss={avg_loss:.4f} val_acc={val_acc:.4f} loss_active={active}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save({
                "model": model.state_dict(),
                "classes": list(TRAIN_CLASSES),
                "img_size": img_size,
                "backbone": backbone,
                "val_acc": float(val_acc),
                "epoch": ep,
                "variant": args.variant,
                "loss_name": loss_name,
            }, out_dir / "best_model.pth")

    elapsed = time.time() - t_total
    print(f"[train] DONE  best_val_acc={best_val_acc:.4f} @ ep{best_epoch}  elapsed={elapsed:.1f}s")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "variant": args.variant,
            "loss_name": loss_name,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "elapsed_sec": elapsed,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "ts": ts,
            "out_dir": str(out_dir),
        }, f, indent=2)


if __name__ == "__main__":
    main()
