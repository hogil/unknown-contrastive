#!/usr/bin/env python3
"""CNN supervised 학습 DDP — multi-GPU (NCCL).

사용:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cnn_ddp.py
        → world_size=4 자동 (mp.spawn 4 worker)

    CUDA_VISIBLE_DEVICES=0   python scripts/train_cnn_ddp.py
        → single-GPU fallback

각 worker 가 rank ∈ [0, world_size) 으로 학습. rank=0 만 save/print/metric log.
"""
from __future__ import annotations

# ===================================================================
# === CONFIG ===
# ===================================================================
DATA_DIR             = "E:/data/images/cnn_train"  # ★ 절대규: 모든 이미지 E:/data/images/
ACTIVE_CLASSES_YAML  = None
EXCLUDE_CLASSES      = {"classification", "classification_chips"}

WEIGHTS_DIR          = "weights"
OUTPUT_ROOT          = "runs"
TAG                  = "cnn_ddp"

BACKBONE             = "convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG_SIZE             = 384
BATCH_PER_GPU        = 16            # ★ total batch = BATCH_PER_GPU × world_size
NUM_WORKERS_PER_GPU  = None          # None = auto: os.cpu_count() // world_size (환경 코어 전부 활용)
EPOCHS               = 30
WARMUP_EPOCHS        = 5              # anomaly-detection 조건: warmup 5ep (LinearLR start_factor 0.05)
LR_BACKBONE          = 2e-5           # anomaly-detection 조건: backbone 2e-5
LR_HEAD              = 2e-4           # anomaly-detection 조건: head 2e-4
WEIGHT_DECAY         = 0.01           # anomaly-detection 조건: AdamW wd 0.01
GRAD_CLIP            = 1.0            # anomaly-detection 조건: grad_clip 1.0
LABEL_SMOOTHING      = 0.0            # anomaly-detection 조건: label_smoothing 0.0 (was 0.02)
EARLY_STOP_PATIENCE  = 5             # anomaly-detection 조건: patience 5 (was 7)

USE_EMA              = False         # anomaly-detection 조건: ema_decay 0.0 (OFF — 짧은 학습엔 raw 가 나음)
USE_AMP              = True          # anomaly-detection 조건: use_amp True (autocast fp16 + GradScaler)
USE_MIXUP            = False         # anomaly-detection 조건: use_mixup False
STOCHASTIC_DEPTH     = 0.0           # anomaly-detection 조건: stochastic_depth 0.0
HEAD_DROPOUT         = 0.0           # anomaly-detection 조건: dropout 0.0 (argparse default) — MLP head 의 Dropout

SPLIT_RATIOS         = (0.8, 0.1, 0.1)
SEED                 = 42
SAVE_WRONG_IMAGES    = True
# ===================================================================

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
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
    all_reduce_avg,
    cleanup_ddp,
    is_main,
    launch_ddp,
    setup_ddp,
)

# ★ env override (mp.spawn 자식이 module re-import 시 읽음 → H100 batch 등 주입)
if os.environ.get("CNN_BATCH_PER_GPU"):
    BATCH_PER_GPU = int(os.environ["CNN_BATCH_PER_GPU"])
if os.environ.get("CNN_EPOCHS"):
    EPOCHS = int(os.environ["CNN_EPOCHS"])


def seed_all(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SafeImageFolder(ImageFolder):
    def __init__(self, root, transform=None, exclude=None,
                 active_classes=None, allow_missing=True):
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
        T.ToTensor(), norm,
    ])
    return train_tf, eval_tf


def build_model(num_classes, backbone_path, rank):
    import timm
    m = timm.create_model(BACKBONE, pretrained=False, num_classes=num_classes,
                          drop_path_rate=STOCHASTIC_DEPTH)
    full_sd = torch.load(backbone_path, map_location="cpu")
    if isinstance(full_sd, dict) and "state_dict" in full_sd:
        full_sd = full_sd["state_dict"]
    m_sd = m.state_dict()
    compat = {k: v for k, v in full_sd.items()
              if k in m_sd and m_sd[k].shape == v.shape}
    m.load_state_dict(compat, strict=False)
    # backbone key 가 실제로 로드됐는지 진단 (skipped 가 head 2개보다 많으면 backbone mismatch)
    bb_loaded = sum(1 for k in compat if not (k.startswith("head.") or k.startswith("fc.")))
    if is_main(rank):
        skipped = len(full_sd) - len(compat)
        print(f"[backbone] loaded {len(compat)} keys ({bb_loaded} backbone), "
              f"{skipped} skipped, m_sd total {len(m_sd)} (head re-init for {num_classes})")

    # ★ anomaly-detection 조건: head = MLP (Dropout → Linear(in,512) → ReLU → Linear(512,classes))
    #   단일 Linear probe 대신 2-layer head — anomaly 의 빠른 초기 수렴 recipe.
    if hasattr(m, "head") and hasattr(m.head, "fc") and isinstance(m.head.fc, nn.Linear):
        in_f = m.head.fc.in_features
        m.head.fc = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT), nn.Linear(in_f, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes))
    elif hasattr(m, "head") and isinstance(m.head, nn.Linear):
        in_f = m.head.in_features
        m.head = nn.Sequential(
            nn.Dropout(HEAD_DROPOUT), nn.Linear(in_f, 512),
            nn.ReLU(inplace=True), nn.Linear(512, num_classes))
    if is_main(rank):
        print(f"[head] MLP head (in→512→ReLU→{num_classes}, dropout={HEAD_DROPOUT})")
    return m


def split_indices(n, ratios, seed):
    g = np.random.default_rng(seed)
    idx = np.arange(n); g.shuffle(idx)
    nt = int(n * ratios[0]); nv = int(n * ratios[1])
    return idx[:nt].tolist(), idx[nt:nt+nv].tolist(), idx[nt+nv:].tolist()


def eval_loop(model_ddp, loader, device, criterion, classes=None,
              wrong_save_dir: Path | None = None, rank: int = 0, local_only: bool = False):
    # local_only=True: rank0 단독 호출 시 all_reduce(collective) skip — DDP deadlock 방지
    model_ddp.eval()
    eval_amp = USE_AMP and torch.cuda.is_available()       # anomaly-detection: eval 도 autocast
    losses, preds, trues, paths_all, probs_all = [], [], [], [], []
    with torch.no_grad():
        for imgs, lbls, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=eval_amp, dtype=torch.float16):
                logits = model_ddp(imgs)
                loss = criterion(logits, lbls)
            logits = logits.float()                            # softmax/argmax 는 fp32 로
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

    # 각 rank 의 local metric 평균 (DistributedSampler 가 split)
    metric = {
        "loss": float(sum(losses) / max(1, len(preds))),
        "acc": float(acc),
        "macro_f1": float(f1),
        "macro_p": float(pr),
        "macro_r": float(rc),
    }
    if not local_only:
        metric = all_reduce_avg(metric, device)        # 모든 rank 동시 호출 시만 (collective)
    metric["n_wrong"] = int((preds != trues).sum())   # local only

    if wrong_save_dir is not None and classes is not None and is_main(rank) and SAVE_WRONG_IMAGES:
        wrong_save_dir.mkdir(parents=True, exist_ok=True)
        for sub in wrong_save_dir.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
        wrong_n = 0
        for i in range(len(preds)):
            if preds[i] == trues[i]: continue
            tcl = classes[trues[i]]; pcl = classes[preds[i]]
            pct = int(round(probs_all[i, preds[i]] * 100))
            sub = wrong_save_dir / tcl; sub.mkdir(exist_ok=True)
            src = paths_all[i]
            if not src or not Path(src).exists(): continue
            dst = sub / f"{tcl}_{pcl}_{pct}%_{Path(src).name}"
            try:
                shutil.copy2(src, dst); wrong_n += 1
            except Exception:
                pass
        print(f"  [wrong] saved {wrong_n} images to {wrong_save_dir}", flush=True)

    return metric


def train_worker(rank: int, world_size: int):
    setup_ddp(rank, world_size)
    seed_all(SEED + rank)
    import os
    nw = NUM_WORKERS_PER_GPU if NUM_WORKERS_PER_GPU is not None else max(1, (os.cpu_count() or 8) // world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # run_dir: rank=0 만 만들고 broadcast (path string)
    if is_main(rank):
        run_dir = make_run_dir(OUTPUT_ROOT, TAG)
        run_dir_str = str(run_dir)
    else:
        run_dir_str = ""
    obj_list = [run_dir_str]
    dist.broadcast_object_list(obj_list, src=0)
    run_dir = Path(obj_list[0])
    cnn_dir = run_dir / "cnn"
    if is_main(rank):
        cnn_dir.mkdir(exist_ok=True)
        print(f"[run_dir] {run_dir.resolve()} (DDP world_size={world_size})")
        cfg = {k: v for k, v in globals().items()
               if k.isupper() and not k.startswith("_")
               and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
        cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
        cfg["WORLD_SIZE"] = world_size
        cfg["TOTAL_BATCH"] = BATCH_PER_GPU * world_size
        snapshot_config(run_dir, cfg)
        system_info(run_dir)

    backbone_path = ensure_backbone_weights(WEIGHTS_DIR, BACKBONE)
    dist.barrier()

    active_classes = None
    if ACTIVE_CLASSES_YAML:
        import yaml
        with open(ACTIVE_CLASSES_YAML) as f:
            active_classes = yaml.safe_load(f).get("classes")

    data_dir = resolve_path(DATA_DIR)
    if not data_dir.exists():
        raise SystemExit(f"DATA_DIR not found: {data_dir}\n"
                         f"  python scripts/generate_data.py && python scripts/_split_data.py")
    train_tf, eval_tf = build_transforms()
    ds_train_full = SafeImageFolder(str(data_dir), transform=train_tf,
                                    exclude=EXCLUDE_CLASSES, active_classes=active_classes)
    ds_eval_full = SafeImageFolder(str(data_dir), transform=eval_tf,
                                   exclude=EXCLUDE_CLASSES, active_classes=active_classes)
    classes = ds_train_full.classes
    n_cls = len(classes)
    tr_i, va_i, te_i = split_indices(len(ds_train_full), SPLIT_RATIOS, SEED)
    ds_train = Subset(ds_train_full, tr_i)
    ds_val   = Subset(ds_eval_full,  va_i)
    ds_test  = Subset(ds_eval_full,  te_i)

    if is_main(rank):
        print(f"[dataset] {len(ds_train_full)} total, {n_cls} classes")
        print(f"[split] train={len(ds_train)} val={len(ds_val)} test={len(ds_test)}")
        (run_dir / "classes.json").write_text(
            json.dumps({"classes": classes, "class_to_idx": ds_train_full.class_to_idx,
                        "world_size": world_size, "total_batch": BATCH_PER_GPU * world_size}, indent=2),
            encoding="utf-8")

    # DDP samplers
    tr_sampler = DistributedSampler(ds_train, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    va_sampler = DistributedSampler(ds_val,   num_replicas=world_size, rank=rank, shuffle=False)
    te_sampler = DistributedSampler(ds_test,  num_replicas=world_size, rank=rank, shuffle=False)

    train_ld = DataLoader(ds_train, batch_size=BATCH_PER_GPU, sampler=tr_sampler,
                          num_workers=nw, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(ds_val, batch_size=BATCH_PER_GPU, sampler=va_sampler,
                          num_workers=nw, pin_memory=True)
    test_ld  = DataLoader(ds_test, batch_size=BATCH_PER_GPU, sampler=te_sampler,
                          num_workers=nw, pin_memory=True)

    # model
    model = build_model(n_cls, backbone_path, rank).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False)

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        (head_params if n.startswith("module.head.") or n.startswith("module.fc.") else backbone_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": LR_BACKBONE},
         {"params": head_params,     "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = len(train_ld)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps  = EPOCHS * steps_per_epoch
    sched_w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=max(1, warmup_steps))
    sched_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps),
                                                         eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[sched_w, sched_c],
                                                  milestones=[warmup_steps])
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    amp_on = USE_AMP and torch.cuda.is_available()       # anomaly-detection: AMP fp16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)

    if is_main(rank):
        log_stage_metric(run_dir, "cnn_ddp_setup", {
            "world_size": world_size,
            "batch_per_gpu": BATCH_PER_GPU,
            "total_batch": BATCH_PER_GPU * world_size,
            "backbone": BACKBONE, "n_classes": n_cls,
            "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        }, notes=f"DDP NCCL backend, {world_size} GPUs")

    best_f1 = -1.0; best_ep = 0; no_improve = 0
    history = []
    for ep in range(1, EPOCHS + 1):
        tr_sampler.set_epoch(ep)        # ★ shuffle 매 epoch 재
        model.train()
        ep_loss, ep_n = 0.0, 0
        ep_preds, ep_trues = [], []                    # running train f1 용 누적
        t0 = time.time()
        for it, (imgs, lbls, _) in enumerate(train_ld, 1):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_on, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            if GRAD_CLIP > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += loss.item() * imgs.size(0)
            ep_preds.append(logits.argmax(1).detach().cpu().numpy())
            ep_trues.append(lbls.detach().cpu().numpy())
            ep_n += imgs.size(0)
            if is_main(rank) and it % max(1, len(train_ld) // 10) == 0:
                pct = it / len(train_ld) * 100
                _p = np.concatenate(ep_preds); _t = np.concatenate(ep_trues)
                _f1 = f1_score(_t, _p, average="macro", zero_division=0)
                lr_now = opt.param_groups[1]["lr"]      # head LR (warmup 확인용)
                print(f"  Ep {ep:>2}/{EPOCHS} | {pct:>5.1f}% | loss={ep_loss/ep_n:.4f} "
                      f"f1={_f1*100:.2f}% (train, rank0 local) lr={lr_now:.2e}", flush=True)
        _p = np.concatenate(ep_preds); _t = np.concatenate(ep_trues)
        tr_metric = {"loss": ep_loss / ep_n, "acc": float((_p == _t).mean()),
                     "macro_f1": float(f1_score(_t, _p, average="macro", zero_division=0))}
        tr_metric = all_reduce_avg(tr_metric, device)

        val_metric = eval_loop(model, val_ld, device, criterion,
                               classes=classes, wrong_save_dir=None, rank=rank)
        if is_main(rank):
            print(f"  Ep {ep:>2}/{EPOCHS} DONE | train loss={tr_metric['loss']:.4f} f1={tr_metric['macro_f1']*100:.2f}% "
                  f"| val loss={val_metric['loss']:.4f} f1={val_metric['macro_f1']*100:.2f}% "
                  f"(val acc={val_metric['acc']*100:.2f}%) | time={time.time()-t0:.0f}s",
                  flush=True)
            history.append({"epoch": ep, "train": tr_metric, "val": val_metric})

            if val_metric["macro_f1"] > best_f1:
                best_f1 = val_metric["macro_f1"]; best_ep = ep; no_improve = 0
                # save underlying model (model.module)
                torch.save({
                    "state_dict": model.module.state_dict(),
                    "classes": classes,
                    "class_to_idx": ds_train_full.class_to_idx,
                    "epoch": ep,
                    "val_metric": val_metric,
                    "world_size": world_size,
                }, cnn_dir / "best_model.pth")
                print(f"  ★ best updated: ep={ep} val_macro_f1={best_f1:.4f}")
                # wrong-image 저장은 학습 종료 후 test 구간에서 (전 rank 동일 collective).
                # 루프 중 rank0-only eval_loop 는 collective desync → SIGABRT 유발하므로 제거.
            else:
                no_improve += 1

            (cnn_dir / "history.json").write_text(
                json.dumps({"history": history, "best_epoch": best_ep,
                            "best_val_macro_f1": best_f1, "world_size": world_size},
                           indent=2), encoding="utf-8")

        # broadcast no_improve from rank 0 to all (early stop sync)
        no_improve_t = torch.tensor([no_improve if is_main(rank) else 0],
                                    dtype=torch.long, device=device)
        dist.broadcast(no_improve_t, src=0)
        if no_improve_t.item() >= EARLY_STOP_PATIENCE:
            if is_main(rank):
                print(f"[early-stop] no improve {EARLY_STOP_PATIENCE} epochs")
            break

    # test eval (rank 0 best load, all ranks 평가)
    if is_main(rank):
        print("[test] loading best...")
    dist.barrier()
    map_loc = {"cuda:0": f"cuda:{rank}"} if torch.cuda.is_available() else "cpu"
    ck = torch.load(cnn_dir / "best_model.pth", map_location=map_loc, weights_only=False)
    model.module.load_state_dict(ck["state_dict"])
    # val wrong-image 저장 (전 rank 동일 collective — 학습 끝난 뒤 best model 로 1회)
    if SAVE_WRONG_IMAGES:
        _ = eval_loop(model, val_ld, device, criterion, classes=classes,
                      wrong_save_dir=cnn_dir / "wrong" / "val", rank=rank)
    test_metric = eval_loop(model, test_ld, device, criterion,
                            classes=classes,
                            wrong_save_dir=(cnn_dir / "wrong" / "test") if SAVE_WRONG_IMAGES else None,
                            rank=rank)
    if is_main(rank):
        print(f"[test] {test_metric}")
        log_stage_metric(run_dir, "cnn_ddp_done", {
            "best_epoch": best_ep,
            "best_val_macro_f1": best_f1,
            "test_macro_f1": test_metric["macro_f1"],
            "test_acc": test_metric["acc"],
            "world_size": world_size,
        }, notes=f"DDP {world_size} GPUs, total_batch={BATCH_PER_GPU * world_size}")
        print(f"\n[OUT] {run_dir.resolve()}")

    cleanup_ddp()


if __name__ == "__main__":
    launch_ddp(train_worker)
