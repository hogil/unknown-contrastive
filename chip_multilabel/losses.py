"""Loss implementations for Stage 2 training variants.

T1: CE + label_smoothing 0.1 (PyTorch native)
T4: ASL (Asymmetric Loss, Ridnik 2021)
T5: BCE with one-hot multi-hot target (single-positive)
T6: BCE -> ASL switch at warmup_epochs
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """ASL — Ridnik 2021.

    Defaults: gamma_pos=1, gamma_neg=4, clip=0.05.
    Target must be multi-hot (B, C) {0,1}.
    """

    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        target = target.float()
        p_neg = (p - self.clip).clamp(min=self.eps) if self.clip > 0 else p
        log_pos = torch.log(p.clamp(min=self.eps))
        log_neg = torch.log((1 - p_neg).clamp(min=self.eps))
        with torch.no_grad():
            pt0 = (1 - p) ** self.gamma_pos
            p_neg_d = (p - self.clip).clamp(min=0) if self.clip > 0 else p
            pt1 = p_neg_d ** self.gamma_neg
        loss_pos = target * log_pos * pt0
        loss_neg = (1 - target) * log_neg * pt1
        return -(loss_pos + loss_neg).mean()


class BCEMultiHot(nn.Module):
    """BCE on multi-hot target. Single-positive case = one-hot target."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target.float())


class CEWithSmoothing(nn.Module):
    """CE with label smoothing on class-index target."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target.long(), label_smoothing=self.smoothing)


class BCEThenASL(nn.Module):
    """T6: BCE for warmup_epochs, then ASL. Caller must call .set_epoch(ep) each epoch."""

    def __init__(self, warmup_epochs: int = 5, asl_kwargs: dict | None = None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.bce = BCEMultiHot()
        self.asl = AsymmetricLoss(**(asl_kwargs or {}))
        self.epoch = 0
        self.last_active = "bce"

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.epoch < self.warmup_epochs:
            self.last_active = "bce"
            return self.bce(logits, target)
        self.last_active = "asl"
        return self.asl(logits, target)


def build_loss(loss_name: str):
    if loss_name == "ce_ls01":
        return CEWithSmoothing(smoothing=0.1), "class_index"
    if loss_name == "asl":
        return AsymmetricLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05), "multi_hot"
    if loss_name == "bce":
        return BCEMultiHot(), "multi_hot"
    if loss_name == "bce_then_asl":
        return BCEThenASL(warmup_epochs=5), "multi_hot"
    raise ValueError(f"unknown loss: {loss_name}")
