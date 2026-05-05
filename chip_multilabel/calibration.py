"""Temperature scaling — single scalar T learned by minimizing val NLL.

Reference: Guo et al. 2017 "On Calibration of Modern Neural Networks" (arXiv:1706.04599).
For our single-train-multi-predict setup, we apply T as logit/T to the 4-class softmax
fitted against single-label targets on the training distribution val split.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TemperatureScaler:
    """Learn 1 scalar T on (logits, targets) by LBFGS minimizing CE NLL."""

    def __init__(self, init_T: float = 1.0):
        self.T = float(init_T)

    def fit(self, logits: np.ndarray, targets: np.ndarray, max_iter: int = 200) -> "TemperatureScaler":
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2-D, got {logits.shape}")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError(f"targets shape {targets.shape} incompatible with logits {logits.shape}")
        L = torch.from_numpy(np.asarray(logits, dtype=np.float32))
        Y = torch.from_numpy(np.asarray(targets, dtype=np.int64))
        T = torch.nn.Parameter(torch.tensor([float(self.T)]))
        opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            scaled = L / T.clamp(min=1e-3)
            loss = F.cross_entropy(scaled, Y)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(T.detach().clamp(min=1e-3).item())
        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float32) / max(self.T, 1e-3)

    def __repr__(self) -> str:
        return f"TemperatureScaler(T={self.T:.4f})"


class PerClassTemperatureScaler:
    """One scalar T per class fitted on val by minimizing per-class binary CE.

    For each class c, we treat the c-th logit as an independent binary classifier
    (positive=class c, negative=anything else) and fit T_c by binary NLL min on val.
    This lets fork (over-firing) get a different T from scratch (well-calibrated).
    """

    def __init__(self, n_classes: int, init_T: float = 1.0):
        self.n_classes = int(n_classes)
        self.T = np.full(self.n_classes, float(init_T), dtype=np.float64)

    def fit(self, logits: np.ndarray, multihot: np.ndarray, max_iter: int = 200) -> "PerClassTemperatureScaler":
        if logits.shape != multihot.shape:
            raise ValueError(f"shape mismatch: logits={logits.shape} multihot={multihot.shape}")
        if logits.shape[1] != self.n_classes:
            raise ValueError(f"expected {self.n_classes} cols, got {logits.shape[1]}")
        for c in range(self.n_classes):
            L = torch.from_numpy(logits[:, c].astype(np.float32))
            Y = torch.from_numpy(multihot[:, c].astype(np.float32))
            T = torch.nn.Parameter(torch.tensor([1.0]))
            opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

            def closure():
                opt.zero_grad()
                scaled = L / T.clamp(min=1e-3)
                loss = F.binary_cross_entropy_with_logits(scaled, Y)
                loss.backward()
                return loss

            opt.step(closure)
            self.T[c] = float(T.detach().clamp(min=1e-3).item())
        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        if logits.shape[1] != self.n_classes:
            raise ValueError(f"expected {self.n_classes} cols, got {logits.shape[1]}")
        T_safe = np.clip(self.T, 1e-3, None)
        return logits.astype(np.float32) / T_safe[None, :]

    def __repr__(self) -> str:
        return f"PerClassTemperatureScaler(T={[round(float(t),3) for t in self.T]})"
