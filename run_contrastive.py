#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_contrastive.py — Windows env wrapper for contrastive.py.

contrastive.py 자체 수정 안 함. CFG override + cnn_train backbone state_dict 추출만.

사용:
    python run_contrastive.py                              # 기본 smoke (EPOCHS=2)
    EPOCHS=20 BATCH=32 python run_contrastive.py           # 본 학습
    BACKBONE_CKPT=logs_wafer/<run>/best_model.pth python run_contrastive.py
"""
from __future__ import annotations
import os
import sys
import tempfile
import time
from pathlib import Path
import torch


def install_nv_retriever_perc_pos(alpha: float = 0.95) -> None:
    """NV-Retriever TopK-PercPos α — replace fixed IGNORE_NEG_SIM with per-anchor adaptive threshold.

    Reference: Moreira et al. 2024, "NV-Retriever: Improving Text Embedding Models with Effective
    Hard-Negative Mining" (arXiv:2407.15831), Sec. 4 (TopK-PercPos), Sec. 5 ablation (α=0.95 best).

    Mechanism: thr_i = pos[i] * α (per-anchor). Negatives with sim ≥ thr_i are dropped as suspected
    false-negatives. Replaces uniform `ignore_sim` with per-row mask. Anchors with strong positive
    (clean view) tolerate near-positive negatives; anchors with weak positive tighten threshold
    automatically — directly fits sister-class collapse where some anchors have weak positive
    and need stricter neg filtering.

    Usage: set env NEG_FILTER=perc_pos NEG_FILTER_ALPHA=0.95.
    Effect on Iter 1 weak point (Edge-Bottom_*** family centroid 0.007): per-anchor adaptive
    threshold should keep family-boundary negatives as negatives only when truly close.
    """
    import torch
    import torch.nn.functional as F
    import contrastive  # already imported in main; rebind

    _orig_info_nce = contrastive.info_nce_with_queue

    def info_nce_perc_pos(z1, z2, bank, t=0.2, ignore_sim=None):
        B, D = z1.size()
        pos = (z1 * z2).sum(1, keepdim=True)                 # [B,1]
        thr = pos * alpha                                    # [B,1] — per-anchor threshold
        neg_ib = z1 @ z2.T                                   # [B,B]
        neg_ib.fill_diagonal_(-1e9)
        neg_ib = torch.where(neg_ib >= thr, torch.full_like(neg_ib, -1e9), neg_ib)

        if bank is not None:
            neg_bank = z1 @ bank.T                           # [B,N]
            neg_bank = torch.where(neg_bank >= thr, torch.full_like(neg_bank, -1e9), neg_bank)
            logits = torch.cat([pos, neg_ib, neg_bank], dim=1) / t
        else:
            logits = torch.cat([pos, neg_ib], dim=1) / t

        labels = torch.zeros(B, dtype=torch.long, device=z1.device)
        return F.cross_entropy(logits, labels)

    contrastive.info_nce_with_queue = info_nce_perc_pos
    print(f"[NV-Retriever] TopK-PercPos active: thr_i = pos[i] × {alpha}  (ignore_sim ignored)")


def install_partial_unfreeze(n: int) -> None:
    """Backbone partial unfreeze — last `n` ConvNeXtV2 stages trainable, rest frozen.

    ConvNeXtV2-base has 4 stages (`backbone.stages[0..3]`). With n=2, stages[2] and stages[3]
    become trainable (mid+high-level feature blocks); stages[0..1] + stem stay frozen
    (low-level patterns inherited from supervised TAPT). Motivation: unlock backbone
    capacity without catastrophic forgetting — Iter 31 full unfreeze at LR=5e-5 collapsed
    to noise=62%, suggesting last-layer-only fine-tuning preserves the learned feature
    hierarchy while still allowing task-specific high-level adaptation.

    Implementation: monkey-patch `contrastive.CL.__init__` so that after original init
    completes (which sets `requires_grad=False` for all backbone params when
    FREEZE_BACKBONE=True), we override `requires_grad` per stage. Stem and head are
    untouched — only `stages[i]` parameters are flipped on for i ∈ [n_stages - n, n_stages).

    The optimizer in contrastive.py main() already filters
    `[p for p in model.parameters() if p.requires_grad]`, so flipped stages auto-include.

    Usage: set env BACKBONE_UNFREEZE_LAST_N=2 (default 0 = no-op, full freeze behaviour).
    """
    if n <= 0:
        return
    import contrastive

    _orig_init = contrastive.CL.__init__

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        stages = self.backbone.stages
        n_stages = len(stages)
        n_unfreeze = min(n, n_stages)
        for i, stage in enumerate(stages):
            req = (i >= n_stages - n_unfreeze)
            for p in stage.parameters():
                p.requires_grad = req
        n_train = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.backbone.parameters())
        unfrozen_idx = list(range(n_stages - n_unfreeze, n_stages))
        print(f"[partial-unfreeze] last {n_unfreeze}/{n_stages} stages trainable "
              f"(idx={unfrozen_idx}); backbone trainable params "
              f"{n_train:,} / {n_total:,} ({100.0 * n_train / n_total:.1f}%)")

    contrastive.CL.__init__ = patched_init
    print(f"[patch] install_partial_unfreeze(n={n}) registered")


def install_neco_loss(weight: float, tau: float = 0.1) -> None:
    """NeCo (Pariza et al. 2024, arXiv:2408.11054) — soft patch-neighbor consistency.

    NeCo = "Neighborhood-aware Cluster Order". Original paper uses Sinkhorn-Knopp
    differentiable sorting to enforce that, for each patch, the *rank order* of its
    k-nearest neighbors among other patches is consistent across two augmented views.
    This regularizes dense feature maps to maintain stable spatial relations under
    photometric augmentation, complementary to global InfoNCE which only constrains
    pooled features.

    Implementation here uses the soft-rank approximation noted in the paper's appendix
    (and in concurrent work — DenseCL, VICReg-Local): replace explicit ranking with
    row-softmax over patch-pairwise cosine similarities, yielding distributions
    `P_v[i, :] = softmax(sim(p_i, p_j) / τ)_j` per view v ∈ {1, 2}. Loss is
    symmetric KL between P_1 and P_2 (rows = anchor patches, cols = candidate
    neighbors). This preserves the spirit (relative neighbor ordering should match)
    without the Sinkhorn machinery, keeping the patch into a single forward+matmul pass.

    Why this fits our setup:
    - We already have L2-normed local feature maps `f1, f2 ∈ [B, C, H, W]` from
      `CL.forward(return_local=True)` — no extra forward needed.
    - At 384px input on ConvNeXtV2-base, H=W=12 → N=144 patches per sample, so
      per-sample S ∈ [144, 144] (~80 KB fp16) — cheap.
    - Adds dense regularization on top of Iter 14's frozen backbone (and Iter 32's
      partial unfreeze): exact spatial sister-class boundaries (Edge-Top vs
      Edge-Bottom variants) should benefit from neighbor-order consistency.

    Hook: monkey-patch `contrastive.info_nce_local_multi`. The wrapper calls original
    local loss, then adds `weight * neco_term`. Because contrastive.main() invokes
    info_nce_local_multi only when `CFG["USE_LOCAL"]=True` and `it % LOCAL_EVERY_N == 0`,
    NeCo only fires on those same iterations — same compute pattern as L term.

    Args:
        weight: NeCo loss weight added to local loss term (default 0 = no-op).
        tau: temperature for row-softmax over patch sim. Lower = peakier, more rank-like.
             Paper uses 0.1 with DINOv2 ViT-S; we keep same default.
    """
    if weight <= 0:
        return
    import torch.nn.functional as F
    import contrastive

    _orig_local = contrastive.info_nce_local_multi

    def neco_wrapped(f1: torch.Tensor, f2: torch.Tensor, **kwargs):
        # Original local InfoNCE (multi-positive, anchor-based)
        local_loss = _orig_local(f1, f2, **kwargs)

        # NeCo soft-rank consistency on the same patch maps
        B, C, H, W = f1.shape
        N = H * W
        # CL.forward already L2-normalized along C dim
        p1 = f1.reshape(B, C, N).transpose(1, 2)            # [B, N, C]
        p2 = f2.reshape(B, C, N).transpose(1, 2)
        # Per-sample patch-pairwise similarity
        S1 = torch.bmm(p1, p1.transpose(1, 2))              # [B, N, N]
        S2 = torch.bmm(p2, p2.transpose(1, 2))
        # Drop self-sim from softmax denominator
        eye = torch.eye(N, device=f1.device, dtype=torch.bool).unsqueeze(0)
        S1 = S1.masked_fill(eye, -1e4)
        S2 = S2.masked_fill(eye, -1e4)
        log_P1 = F.log_softmax(S1 / tau, dim=-1)            # [B, N, N]
        log_P2 = F.log_softmax(S2 / tau, dim=-1)
        P1 = log_P1.exp()
        P2 = log_P2.exp()
        # Symmetric KL (KL(P1||P2) + KL(P2||P1)) / 2, mean over patches and batch
        kl_12 = (P1 * (log_P1 - log_P2)).sum(dim=-1).mean()
        kl_21 = (P2 * (log_P2 - log_P1)).sum(dim=-1).mean()
        neco_term = 0.5 * (kl_12 + kl_21)
        return local_loss + weight * neco_term

    contrastive.info_nce_local_multi = neco_wrapped
    print(f"[NeCo] patch-neighbor consistency active: weight={weight} tau={tau}")


def install_gpu_throttle(throttle_ms: float) -> None:
    """Inject sleep after each optimizer.step() to throttle GPU compute utilization.

    Use case: target ~30% GPU util when training shares the GPU with other apps.
    Approach: torch.cuda.synchronize() forces flush, then time.sleep yields.
    """
    if throttle_ms <= 0:
        return
    _orig = torch.optim.Optimizer.step
    sleep_s = throttle_ms / 1000.0

    def _throttled(self, *a, **kw):
        out = _orig(self, *a, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(sleep_s)
        return out

    torch.optim.Optimizer.step = _throttled
    print(f"[throttle] GPU compute throttle: {throttle_ms} ms sleep after each optimizer.step")


def extract_state_dict(ckpt_path: str | Path) -> str:
    """cnn_train save format ({"model": state_dict, "classes": [...], ...}) 에서
    state_dict 만 추출해서 임시 .pth 로 저장. contrastive.py 가 직접 load 가능한 raw state.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"backbone ckpt not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        # try common wrap keys (cnn_train: "model"; some others: "state_dict" / "model_state")
        for k in ("model", "model_state", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                state = sd[k]
                break
        else:
            state = sd  # assume already raw state_dict
    else:
        state = sd
    out = Path(tempfile.gettempdir()) / f"contrastive_init_{ckpt_path.stem}.pth"
    torch.save(state, out)
    print(f"[init] extracted state_dict: {ckpt_path} → {out}  ({len(state)} keys)")
    return str(out)


def main():
    # GPU compute throttle (env GPU_THROTTLE_MS, default 0 = no throttle)
    install_gpu_throttle(float(os.environ.get("GPU_THROTTLE_MS", 0)))

    # Resolve backbone init (TAPT — wafer best_model.pth)
    backbone_ckpt = os.environ.get(
        "BACKBONE_CKPT",
        "D:/project/unknown-contrastive/logs_wafer/overall/best_model.pth",
    )
    backbone_state_path = extract_state_dict(backbone_ckpt)

    # Import contrastive (executes top-level CFG dict)
    sys.path.insert(0, str(Path(__file__).parent))
    import contrastive  # noqa: E402

    # NV-Retriever monkey-patch (env NEG_FILTER=perc_pos to enable, must be after import contrastive)
    if os.environ.get("NEG_FILTER", "fixed") == "perc_pos":
        install_nv_retriever_perc_pos(alpha=float(os.environ.get("NEG_FILTER_ALPHA", 0.95)))

    # Backbone partial unfreeze (env BACKBONE_UNFREEZE_LAST_N, default 0 = no-op).
    # Must run before CL() is instantiated in contrastive.main(). Patches CL.__init__.
    install_partial_unfreeze(int(os.environ.get("BACKBONE_UNFREEZE_LAST_N", 0)))

    # NeCo patch-neighbor consistency (env NECO_WEIGHT, default 0.0 = no-op).
    # Patches contrastive.info_nce_local_multi. Adds soft-rank KL term to local loss.
    install_neco_loss(
        weight=float(os.environ.get("NECO_WEIGHT", 0.0)),
        tau=float(os.environ.get("NECO_TAU", 0.1)),
    )

    # Monkey-patch ImageFolder:
    # 1. classification/classification_chips 빈 폴더 제외 (cnn_train EXCLUDE_CLASSES 동일 정책)
    # 2. PER_CLASS_CAP 으로 각 class 의 sample 수 제한 (env var PER_CLASS_CAP)
    from torchvision.datasets import ImageFolder as _BaseImageFolder
    EXCLUDE = {"classification", "classification_chips"}
    PER_CLASS_CAP = int(os.environ.get("PER_CLASS_CAP", 0)) or None

    class FilteredImageFolder(_BaseImageFolder):
        def find_classes(self, directory):
            classes, _ = super().find_classes(directory)
            classes = [c for c in classes if c not in EXCLUDE]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            return classes, class_to_idx

        def __init__(self, root, **kwargs):
            super().__init__(root, **kwargs)
            if PER_CLASS_CAP:
                from collections import defaultdict
                per_cls = defaultdict(list)
                for path, lbl in self.samples:
                    per_cls[lbl].append((path, lbl))
                capped = []
                for cls_idx, lst in sorted(per_cls.items()):
                    lst.sort(key=lambda x: x[0])  # deterministic
                    capped.extend(lst[:PER_CLASS_CAP])
                self.samples = capped
                self.targets = [lbl for _, lbl in capped]
                self.imgs = self.samples
                print(f"[CappedImageFolder] {root}: capped to {PER_CLASS_CAP} per class "
                      f"→ total {len(capped)} samples")
    contrastive.ImageFolder = FilteredImageFolder
    cap_msg = f"+ cap {PER_CLASS_CAP}/class" if PER_CLASS_CAP else "(no cap)"
    print(f"[patch] ImageFolder filter active: exclude {EXCLUDE}  {cap_msg}")

    # 데이터 디렉토리 — 원본 D:/project/data/wm-811k/unknown 절대 안 건드림.
    # 미리 _make_contrastive_cache.py 로 만든 384×384 RGB cache 사용 (disk 로드 100× 빠름).
    data_dir = os.environ.get("DATA_DIR", "D:/project/data/wm-811k/unknown_cache_384")

    # CFG override for Windows + current synthetic data
    contrastive.CFG.update({
        "TRAIN_DIR":  data_dir,
        "UNKNOWN_DIR": data_dir,
        "OVERLAY_DIR": data_dir,                                # no separate overlay
        "OUTPUT_DIR":  "D:/project/unknown-contrastive/outputs_contrastive",
        "LOCAL_BACKBONE_WEIGHTS": backbone_state_path,
        "IMAGE_SIZE": int(os.environ.get("IMAGE_SIZE", 384)),
        "BATCH":       int(os.environ.get("BATCH",       32)),  # 16GB GPU 안전
        "NUM_WORKERS": int(os.environ.get("NUM_WORKERS",  0)),  # Windows 기본 0
        "EPOCHS":      int(os.environ.get("EPOCHS",       2)),  # smoke default
        "WARMUP_EPOCHS": int(os.environ.get("WARMUP_EPOCHS", 1)),
        "TRAIN_SAMPLING_RATIO": float(os.environ.get("TRAIN_SAMPLING_RATIO", 0.25)),
        "USE_LOCAL":   os.environ.get("USE_LOCAL", "true").lower() == "true",
        "USE_QUEUE":   os.environ.get("USE_QUEUE", "true").lower() == "true",
        "QUEUE_SIZE":  int(os.environ.get("QUEUE_SIZE", contrastive.CFG.get("QUEUE_SIZE", 4096))),
        "LOCAL_WEIGHT": float(os.environ.get("LOCAL_WEIGHT", contrastive.CFG.get("LOCAL_WEIGHT", 0.5))),
        "LOCAL_POS_TOPK": int(os.environ.get("LOCAL_POS_TOPK", contrastive.CFG.get("LOCAL_POS_TOPK", 12))),
        # NOTE: NCE_TEMP not TEMP — Windows TEMP is system temp dir env
        "TEMP":        float(os.environ.get("NCE_TEMP",  contrastive.CFG.get("TEMP", 0.07))),
        "IGNORE_NEG_SIM": float(os.environ.get("IGNORE_NEG_SIM",
                                              contrastive.CFG.get("IGNORE_NEG_SIM", 0.72))),
        "LR_HEAD":     float(os.environ.get("LR_HEAD", contrastive.CFG.get("LR_HEAD", 1e-3))),
        "FREEZE_BACKBONE": os.environ.get("FREEZE_BACKBONE", "true").lower() == "true",
        "SEED":        int(os.environ.get("SEED",   contrastive.CFG.get("SEED", 42))),
        "CLUSTER_SELECTION_METHOD": os.environ.get("CLUSTER_SELECTION_METHOD", "leaf"),
        "CLUSTER_SELECTION_EPSILON": float(os.environ.get("CLUSTER_SELECTION_EPSILON", 0.06)),
        "MIN_CLUSTER_SIZE": int(os.environ.get("MIN_CLUSTER_SIZE", 12)),
        "MIN_SAMPLES": int(os.environ.get("MIN_SAMPLES", 4)),
        "PIN_MEMORY": False,
        "PERSISTENT": False,
    })

    print("[CFG override] paths + Windows defaults applied")
    print(f"  TRAIN_DIR={contrastive.CFG['TRAIN_DIR']}")
    print(f"  UNKNOWN_DIR={contrastive.CFG['UNKNOWN_DIR']}")
    print(f"  OUTPUT_DIR={contrastive.CFG['OUTPUT_DIR']}")
    print(f"  LOCAL_BACKBONE_WEIGHTS={contrastive.CFG['LOCAL_BACKBONE_WEIGHTS']}")
    print(f"  IMAGE_SIZE={contrastive.CFG['IMAGE_SIZE']}, BATCH={contrastive.CFG['BATCH']}, "
          f"NUM_WORKERS={contrastive.CFG['NUM_WORKERS']}, EPOCHS={contrastive.CFG['EPOCHS']}")

    contrastive.main()


if __name__ == "__main__":
    main()
