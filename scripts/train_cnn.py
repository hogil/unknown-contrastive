#!/usr/bin/env python3
"""CNN supervised 단독 학습 — ConvNeXtV2 FCMAE + AdamW + warmup + cosine.

사용법:
    python scripts/train_cnn.py
        → runs/<TS>_cnn/ 폴더 생성, best_model.pth + metric 산출

데이터: ImageFolder 형식 (DATA_DIR/<class>/*.png)
"""
from __future__ import annotations

# ===================================================================
# === CONFIG (실행 시 이 부분만 수정) ===
# ===================================================================
DATA_DIR             = "E:/data/images/cnn_train"  # ★ 절대규: 모든 이미지 E:/data/images/
ACTIVE_CLASSES_YAML  = None              # 선택 — class subset YAML path
EXCLUDE_CLASSES      = {"classification", "classification_chips"}

WEIGHTS_DIR          = "weights"         # backbone download path
OUTPUT_ROOT          = "runs"            # 산출 prefix 폴더
TAG                  = "cnn"             # runs/<TS>_<TAG>/

BACKBONE             = "convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG_SIZE             = 384
BATCH                = 16
NUM_WORKERS          = 4                 # Windows safe

EPOCHS               = 30
WARMUP_EPOCHS        = 5
LR_BACKBONE          = 2e-5              # anomaly-detection best: 5e-5 spike 유발
LR_HEAD              = 2e-4              # backbone × 10
WEIGHT_DECAY         = 0.01              # anomaly-detection default
GRAD_CLIP            = 1.0
LABEL_SMOOTHING      = 0.02
EARLY_STOP_PATIENCE  = 7

# anomaly-detection 정책 매치 (사용자 명시)
USE_EMA              = False
USE_AMP              = False
USE_MIXUP            = False
STOCHASTIC_DEPTH     = 0.0

SPLIT_RATIOS         = (0.8, 0.1, 0.1)   # train/val/test
SEED                 = 42
SAVE_WRONG_IMAGES    = True              # FP/FN 저장 → cnn/wrong/<true>/<true>_<pred>_<pct>%_<basename>.png
# ===================================================================

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
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


# ---------- ImageFolder w/ class filter + corrupted-PNG skip ----------
class SafeImageFolder(ImageFolder):
    """ImageFolder + EXCLUDE_CLASSES + ACTIVE_CLASSES_YAML + corrupted PNG skip."""
    def __init__(self, root, transform=None, exclude: set = None,
                 active_classes: list = None, allow_missing: bool = True):
        self._exclude = exclude or set()
        self._active = set(active_classes) if active_classes else None
        self._allow_missing = allow_missing
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        kept = [c for c in classes if c not in self._exclude]
        if self._active is not None:
            missing = self._active - set(classes)
            if missing and not self._allow_missing:
                raise SystemExit(f"missing active classes: {sorted(missing)}")
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


def build_transforms():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomAffine(degrees=15, translate=(0.03,0.03), scale=(0.97,1.03)),
        T.ToTensor(),
        AddGaussianNoise(0.01),
        norm,
    ])
    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


def build_model(num_classes: int, backbone_path: Path) -> nn.Module:
    import timm
    m = timm.create_model(BACKBONE, pretrained=False, num_classes=num_classes,
                          drop_path_rate=STOCHASTIC_DEPTH)
    full_sd = torch.load(backbone_path, map_location="cpu")
    if isinstance(full_sd, dict) and "state_dict" in full_sd:
        full_sd = full_sd["state_dict"]
    m_sd = m.state_dict()
    compat = {k: v for k, v in full_sd.items()
              if k in m_sd and m_sd[k].shape == v.shape}
    skipped = len(full_sd) - len(compat)
    m.load_state_dict(compat, strict=False)
    print(f"[backbone] loaded {len(compat)} keys, {skipped} skipped (head re-init for {num_classes} classes)")
    return m


def split_indices(n: int, ratios=(0.8, 0.1, 0.1), seed=42):
    g = np.random.default_rng(seed)
    idx = np.arange(n); g.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return idx[:n_train].tolist(), idx[n_train:n_train+n_val].tolist(), idx[n_train+n_val:].tolist()


def eval_loop(model, loader, device, criterion, classes=None,
              wrong_save_dir: Path | None = None):
    """eval + (선택) wrong 이미지 저장.

    Wrong file format:
        wrong_save_dir/<true_class>/<true>_<pred>_<pct>%_<basename>.png
    """
    model.eval()
    losses, preds, trues, paths_all, probs_all = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, lbls, paths = batch
            else:
                imgs, lbls = batch; paths = [""] * imgs.size(0)
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, lbls)
            losses.append(loss.item() * imgs.size(0))
            probs = torch.softmax(logits, dim=1)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(lbls.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            paths_all.extend(paths)
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    probs_all = np.concatenate(probs_all, axis=0)
    acc = (preds == trues).mean()
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(trues, preds, average="macro", zero_division=0)
    pr = precision_score(trues, preds, average="macro", zero_division=0)
    rc = recall_score(trues, preds, average="macro", zero_division=0)

    # ----- wrong 이미지 저장 -----
    if wrong_save_dir is not None and classes is not None and SAVE_WRONG_IMAGES:
        wrong_save_dir.mkdir(parents=True, exist_ok=True)
        # clear stale wrong/* from previous epoch (we save best-epoch wrongs)
        for sub in wrong_save_dir.iterdir():
            if sub.is_dir():
                import shutil
                shutil.rmtree(sub)
        import shutil as _sh
        wrong_n = 0
        for i in range(len(preds)):
            if preds[i] == trues[i]: continue
            tcl = classes[trues[i]]; pcl = classes[preds[i]]
            pct = int(round(probs_all[i, preds[i]] * 100))
            sub = wrong_save_dir / tcl; sub.mkdir(exist_ok=True)
            src_path = paths_all[i]
            if not src_path or not Path(src_path).exists():
                continue
            base = Path(src_path).name
            dst = sub / f"{tcl}_{pcl}_{pct}%_{base}"
            try:
                _sh.copy2(src_path, dst)
                wrong_n += 1
            except Exception as e:
                print(f"[wrong-save fail] {src_path}: {e}")
        print(f"  [wrong] saved {wrong_n} mis-classified images to {wrong_save_dir}")

    return {
        "loss": float(sum(losses) / max(1, len(preds))),
        "acc": float(acc),
        "macro_f1": float(f1),
        "macro_p": float(pr),
        "macro_r": float(rc),
        "n_wrong": int((preds != trues).sum()),
    }


def main():
    seed_all(SEED)
    run_dir = make_run_dir(OUTPUT_ROOT, TAG)
    print(f"[run_dir] {run_dir.resolve()}")
    cnn_dir = run_dir / "cnn"; cnn_dir.mkdir(exist_ok=True)

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
    cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
    snapshot_config(run_dir, cfg)
    system_info(run_dir)

    backbone_path = ensure_backbone_weights(WEIGHTS_DIR, BACKBONE)

    # --- active classes from YAML
    active_classes = None
    if ACTIVE_CLASSES_YAML:
        import yaml
        with open(resolve_path(ACTIVE_CLASSES_YAML), "r", encoding="utf-8") as f:
            active_classes = yaml.safe_load(f).get("classes")

    # --- dataset
    data_dir = resolve_path(DATA_DIR)
    if not data_dir.exists():
        raise SystemExit(
            f"DATA_DIR not found: {data_dir}\n"
            f"먼저 데이터 준비:\n"
            f"  python scripts/generate_data.py   # 합성 demo 데이터\n"
            f"  python scripts/_split_data.py     # 분리\n"
        )
    train_tf, eval_tf = build_transforms()
    ds_train_full = SafeImageFolder(str(data_dir), transform=train_tf,
                                    exclude=EXCLUDE_CLASSES,
                                    active_classes=active_classes,
                                    allow_missing=True)
    ds_eval_full = SafeImageFolder(str(data_dir), transform=eval_tf,
                                   exclude=EXCLUDE_CLASSES,
                                   active_classes=active_classes,
                                   allow_missing=True)
    classes = ds_train_full.classes
    n_cls = len(classes)
    print(f"[dataset] {len(ds_train_full)} samples, {n_cls} classes")
    (run_dir / "classes.json").write_text(
        json.dumps({"classes": classes, "class_to_idx": ds_train_full.class_to_idx}, indent=2),
        encoding="utf-8")

    tr_i, va_i, te_i = split_indices(len(ds_train_full), SPLIT_RATIOS, SEED)
    ds_train = Subset(ds_train_full, tr_i)
    ds_val   = Subset(ds_eval_full,  va_i)
    ds_test  = Subset(ds_eval_full,  te_i)
    print(f"[split] train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")

    train_ld = DataLoader(ds_train, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(ds_val, batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    test_ld  = DataLoader(ds_test, batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # --- model + opt + scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_cls, backbone_path).to(device)
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        (head_params if n.startswith("head.") or n.startswith("fc.") else backbone_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params,     "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(train_ld)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps  = EPOCHS * steps_per_epoch
    sched_w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, end_factor=1.0,
                                                total_iters=max(1, warmup_steps))
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps),
                                                         eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[sched_w, sched_c],
                                                  milestones=[warmup_steps])
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    log_stage_metric(run_dir, "cnn_setup", {
        "backbone": BACKBONE, "n_classes": n_cls,
        "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(ds_test),
        "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "label_smoothing": LABEL_SMOOTHING,
        "ema": USE_EMA, "amp": USE_AMP, "mixup": USE_MIXUP,
        "stochastic_depth": STOCHASTIC_DEPTH,
    }, notes="anomaly-detection best matched (사용자 명시)")

    # --- train loop
    best_f1 = -1.0; best_ep = 0; no_improve = 0
    history = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        t0 = time.time()
        for it, (imgs, lbls, _paths) in enumerate(train_ld, 1):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            ep_loss += loss.item() * imgs.size(0)
            ep_correct += (logits.argmax(1) == lbls).sum().item()
            ep_n += imgs.size(0)
            if it % max(1, len(train_ld) // 10) == 0:
                pct = it / len(train_ld) * 100
                print(f"  Ep {ep:>2}/{EPOCHS} | {pct:>5.1f}% | loss={ep_loss/ep_n:.4f} acc={ep_correct/ep_n*100:.2f}%",
                      flush=True)
        tr_metric = {"loss": ep_loss / ep_n, "acc": ep_correct / ep_n}

        val_metric = eval_loop(model, val_ld, device, criterion,
                               classes=classes, wrong_save_dir=None)  # wrong 은 best epoch 에서만 저장
        print(f"  Ep {ep:>2}/{EPOCHS} DONE | train loss={tr_metric['loss']:.4f} acc={tr_metric['acc']*100:.2f}% "
              f"| val loss={val_metric['loss']:.4f} acc={val_metric['acc']*100:.2f}% "
              f"f1={val_metric['macro_f1']*100:.2f}% | time={time.time()-t0:.0f}s",
              flush=True)
        history.append({"epoch": ep, "train": tr_metric, "val": val_metric})

        # save best (val_macro_f1) + best epoch 의 wrong 이미지 갱신
        if val_metric["macro_f1"] > best_f1:
            best_f1 = val_metric["macro_f1"]; best_ep = ep; no_improve = 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": ds_train_full.class_to_idx,
                "epoch": ep,
                "val_metric": val_metric,
                "config": {k: v for k, v in cfg.items()},
            }, cnn_dir / "best_model.pth")
            print(f"  ★ best updated: ep={ep} val_macro_f1={best_f1:.4f}")
            if SAVE_WRONG_IMAGES:
                # re-eval val with wrong save (best epoch 갱신)
                _ = eval_loop(model, val_ld, device, criterion,
                              classes=classes, wrong_save_dir=cnn_dir / "wrong" / "val")
        else:
            no_improve += 1

        (cnn_dir / "history.json").write_text(
            json.dumps({"history": history, "best_epoch": best_ep, "best_val_macro_f1": best_f1},
                       indent=2), encoding="utf-8")

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"[early-stop] no improve {EARLY_STOP_PATIENCE} epochs (best ep {best_ep} f1 {best_f1:.4f})")
            break

    # --- test eval (best model)
    print("[test] loading best ...")
    ck = torch.load(cnn_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    test_metric = eval_loop(model, test_ld, device, criterion,
                            classes=classes,
                            wrong_save_dir=(cnn_dir / "wrong" / "test") if SAVE_WRONG_IMAGES else None)
    print(f"[test] {test_metric}")

    log_stage_metric(run_dir, "cnn_train_done", {
        "best_epoch": best_ep,
        "best_val_macro_f1": best_f1,
        "best_val_acc": history[best_ep - 1]["val"]["acc"] if best_ep > 0 else None,
        "test_macro_f1": test_metric["macro_f1"],
        "test_acc": test_metric["acc"],
        "test_loss": test_metric["loss"],
    }, notes=f"backbone={BACKBONE}, freeze=False (full fine-tune)")

    print(f"\n[OUT] {run_dir.resolve()}")
    return run_dir


if __name__ == "__main__":
    main()
