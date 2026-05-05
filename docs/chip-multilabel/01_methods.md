# 01 — Methods

We describe two orthogonal axes of variation: **inference variants** (I0–I10,
applied to a fixed model checkpoint) and **training variants** (T0–T6,
different ways of fine-tuning the backbone).

The full eval is the cartesian product `T × I`, with each cell evaluated on
the same 2200-chip synthetic 11-class set.

---

## Inference variants

Notation: `s_c = sigmoid(logit_c)` per class. `T` = temperature. Per-class
threshold = `θ_c`. F1-max threshold for class c = `argmax_θ F1(y_c, s_c >= θ)`
on the val split.

### I0 — argmax @ 0.5 (baseline)

Fixed `θ_c = 0.5` for all classes. Pure single-label readout.

```
pred_c = (s_c >= 0.5)
```

### I1 — per-class F1-max thresholds (val-tuned)

Each class gets its own threshold maximising F1 on val.
**Ref**: Lipton et al. 2014, *Optimal Thresholding Classifiers to Maximize F1
Score* (arXiv:1402.1892).

```
θ_c = argmax_θ F1(y_c, s_c >= θ)   # on val split
pred_c = (s_c >= θ_c)
```

### I2 — top-K decision (K=2)

Activate the top-2 classes by sigmoid score regardless of threshold.

```
pred = topK(s, k=2)
```

### I3 — F1-max thresholds + top-K rescue

Combine I1 + I2: union of (`s_c >= θ_c`) and top-K (K=2). Classes whose
F1-max threshold is high but whose score nonetheless dominates the chip are
rescued.

```
pred = (s >= θ_F1max) ∪ topK(s, k=2)
```

### I4 — I3 + temperature scaling

Same as I3 but logits are first divided by a learned temperature `T`
(found by minimising NLL on val). `T < 1` sharpens, `T > 1` smooths.
**Ref**: Guo et al. 2017, *On Calibration of Modern Neural Networks* (arXiv:1706.04599).

```
T  = argmin_T NLL(softmax(logit / T), y)   # on val
s' = sigmoid(logit / T)
pred = (s' >= θ_F1max) ∪ topK(s', k=2)
```

### I5 — I4 + TTA  (PERMANENTLY DISALLOWED)

Test-time rotation 4× averaged with I4 thresholds. **Never re-enabled** — TTA
rotates `scratch` into `scratch_rot` and conflates the two classes
(macro-F1 -0.018 in iter 1).

### I6 — F1-max thresholds + min-threshold floor

I3-style thresholds but each θ_c is clipped from below by 0.30. Stops the
fork threshold from collapsing too low.

```
θ_c = max(0.30, argmax_θ F1)
```

### I7 — F1-max thresholds + per-class step-search (Δ=0.02)

After I3-style F1-max init, perform one local sweep per class on val with
step 0.02 in [0.10, 0.95]. Picks the highest F1 along that grid.
This is the simplest "calibration-free" strict F1 maximisation —
**winner of iter 2** (macro_f1 = 0.8485) and **winner of iter 5** when
combined with T1 LS=0.20.
**Ref**: Lipton et al. 2014.

```
for c in classes:
  θ_c = argmax_{θ ∈ {0.10, 0.12, ..., 0.94}} F1_c(s_c >= θ) on val
pred_c = (s_c >= θ_c)
```

### I8 — F1-max + top-K=1 rescue (single-class fallback)

Like I3 but rescue is restricted to the absolute top-1 class only — preserves
single-defect chips when no class crosses θ_c.

### I9 — F1-max + temperature only (no rescue)

I1 + temperature scaling, no top-K rescue. Tests whether temperature alone
helps without combinatorial decisions.

### I10 — I7 + entropy-based Normal gate

Iter 3 introduces a hard rule: if no `θ_c` is exceeded *and* sigmoid output
entropy is low (i.e. confident-no-defect), label as `Normal`.

```
pred_c = (s_c >= θ_c)                        # I7
H = -Σ s_c log(s_c) - Σ (1-s_c) log(1-s_c)   # binary entropy
if not any(pred_c) and H < H_thresh:
    pred_normal = True
else:
    pred_normal = False
```

`H_thresh` is chosen on val by maximising Normal-class F1 (~0.30 nats in
practice). Pushes Normal recall up without giving up defect precision —
**winner of iter 3** (macro_f1 = 0.8542) and core of iter 4 winner.

---

## Training variants

All variants train on the same 327-image chip dataset (4 defect classes +
`invalid_main`, single-label) with the chip5_round4_v14 backbone topology.
LR schedules and augmentations follow the existing chip CNN trainer
(rotation NEVER applied, same reason as I5).

### T0 — frozen baseline (no retraining)

The reference checkpoint
`chip5_round4_v14_260505_061558_running/best_model.pth`. **Iter 1–3** all run
on top of T0; iter 4 introduces T1/T4/T5/T6.

### T1 — CE + label smoothing 0.10

Cross-entropy with label smoothing `α=0.10`. Default 8 epochs at LR=1e-4.
Reduces over-confidence on the single-label assumption, which was hurting
multi-label sigmoid scores (the model was so confident in 1-of-5 that the
runner-up class was always 1-2 orders of magnitude lower).
**Ref**: Müller, Kornblith, Hinton 2019, *When Does Label Smoothing Help?*
(arXiv:1906.02629).

```
y_smooth = (1 - α) * y_onehot + α / K
loss = -Σ y_smooth_k * log_softmax(logit_k)
```

### T4 — Asymmetric Loss (ASL)

ASL with focusing parameters γ_+=0, γ_-=4, probability-shift m=0.05.
Designed for multi-label imbalance.
**Ref**: Ridnik et al. 2021, *Asymmetric Loss for Multi-Label Classification*
(arXiv:2009.14119).

```
ASL = -[y * (1-p)^γ+ * log p  +  (1-y) * p_m^γ- * log(1 - p_m)]
where p_m = max(0, p - m)
```

### T5 — BCE (per-class binary)

Plain binary cross-entropy treating the trained 5-class problem as 5
independent sigmoids. No smoothing, no asymmetry. Tests whether mismatch
between train (CE softmax) and inference (sigmoid threshold) was the
issue.

```
BCE = -Σ_c [y_c log p_c + (1 - y_c) log (1 - p_c)]
```

### T6 — BCE → ASL (curriculum)

First 4 epochs BCE, then 4 epochs ASL. Idea: BCE establishes basic
multi-label score distribution, ASL then sharpens recall on rare classes.

### T1_LSxx (iter 5)

Sweep of label-smoothing strength on the T1 recipe:
`xx ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}`.
LR=1e-4, epochs=8 fixed. **Iter 5 winner**: `T1_LS20` (α=0.20) +I7.
