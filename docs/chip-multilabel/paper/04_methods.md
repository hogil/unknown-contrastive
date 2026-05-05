# 4. Methods

We describe two orthogonal axes of variation: **inference variants**
(I0–I10), applied to a fixed checkpoint, and **training variants**
(T0–T6), distinct fine-tuning recipes. The full evaluation grid is
the Cartesian product `T × I`. Each cell is named `T<i>__I<j>` (e.g.
`T1_LS20__I7`) and evaluated on the same 2200-chip eval set.

Notation: `z ∈ ℝ^5` is the raw logit vector (5 training classes incl.
`invalid_main`). We project to the 4 defect classes as
`L = z[keep_indices]`, then compute `s = sigmoid(L)` (multi-label) or
`p = softmax(L)` (single-label distribution). Per-class threshold is
`θ_c`; F1-max threshold on val is
`θ_c* = argmax_θ F1(y_c^val, s_c^val ≥ θ)`.

## 4.1 Inference variants

### I0 — argmax with fixed θ=0.5 (baseline)

```
pred_c = (s_c ≥ 0.5)
```

Equivalent to single-label readout. macro-F1 = 0.7302.

### I1 — per-class F1-max threshold (softmax)

`p = softmax(L)`, threshold per class is the val-tuned F1-max point.
**Ref:** Lipton et al. 2014 (arXiv:1402.1892).

### I2 — top-K decision (K=2)

Always activate the top-2 classes by sigmoid score. No thresholding.

### I3 — F1-max threshold + top-K rescue

Union of I1 (sigmoid + F1-max threshold) and I2 (top-K=2). Recovers
chips where one defect is well above its class threshold but the
runner-up has a high score that misses its threshold.

### I4 — I3 + temperature scaling

Logits are first rescaled by a learned `T = argmin_T NLL(softmax(L/T), y)`
on the single-label val subset. Then I3 is applied to the rescaled
sigmoid. **Ref:** Guo et al. 2017 (arXiv:1706.04599).

```
T  = argmin_T NLL(softmax(L_val / T), y_val)
s' = sigmoid(L / T)
pred = (s' ≥ θ_F1max) ∪ topK(s', k=2)
```

### I5 — I4 + 4× rotation TTA — **PERMANENTLY DISALLOWED**

Test-time rotation 4× (identity, hflip, vflip, rot90) averaged. Iter 1
measured -0.018 macro-F1: rotation flips `scratch ↔ scratch_rot`. The
TTA forward path is removed from `forward_all_logits` from iter 2
onward and remains dead in `chip_multilabel/inference_variants.py:62`
for archival reasons.

### I6 — F1-max + min-floor 0.30

I3-style F1-max thresholds clipped from below by 0.30. Rationale: fork's
F1-max threshold collapses to ≈0.12 because non-fork chips still have a
fork sigmoid in the [0.10, 0.30] band, and a low threshold lets that
band cross.

```
θ_c = max(0.30, argmax_θ F1)
```

Empirically a regression (−0.029): fork's *correct* operating point is
the low threshold; the floor throws away ~12% of fork recall.

### I7 — F1-max + per-class step-search (Δ=0.02)

After I3-style F1-max init, perform a fine grid search per class on val
with step Δ=0.02 in [0.10, 0.95]. Selects the F1-maximising step.
**Ref:** Lipton et al. 2014.

```
for c in classes:
    θ_c = argmax_{θ ∈ {0.10, 0.12, ..., 0.94}} F1_c(s_c ≥ θ) on val
pred_c = (s_c ≥ θ_c)
```

We refer to this as "joint coordinate-descent threshold" because the
per-class searches are run independently but evaluated against the
joint multi-hot val labels — see
`chip_multilabel/metrics.py::joint_macro_f1_threshold`.

### I8 — F1-max + top-2 margin gating

I3 + a margin gate: combo is only declared when the second-highest
sigmoid is at least `m=0.6` of the top sigmoid. Suppresses combo
over-firing on chips where one class dominates.

### I9 — F1-max + per-class temperature

Per-class `T_c` fit on val by L-BFGS on per-class binary CE loss. Tests
whether per-class calibration helps where scalar T does not.

### I10 — I7 + entropy-based `Normal` gate

If no `θ_c` is exceeded **and** softmax entropy of the training-class
logits exceeds 0.85·log(C) (i.e. ≥85% of the max entropy for C=4),
declare `Normal`. Else, fall through to I7.

```
pred_c = (s_c ≥ θ_c)                    # I7
H = -Σ p_c log p_c                      # softmax entropy on L
log_C = log(|TRAIN_CLASSES|)            # = log 4
if not any(pred_c) and H ≥ 0.85·log_C:
    pred_normal = True
else:
    pred_normal = False
```

The constant 0.85 is hard-coded
(`chip_multilabel/inference_variants.py:43`,
`I10_ENTROPY_NORMAL_FRAC = 0.85`); we do not sweep it in this paper.

This is the only inference variant that gives `Normal` an explicit
decoder. Without I10, `Normal` chips can only be reached by *all four*
defect sigmoids falling below their thresholds simultaneously, which
the single-label-trained model is not incentivised to produce.

## 4.2 Training variants

All variants train on the same 327-chip / 82-val split (single-label,
5 classes including `invalid_main`) using the chip5_round4_v14 backbone
topology. LR schedule and augmentations follow the existing chip CNN
trainer (rotation NEVER applied). Default 8 epochs at LR=1e-4.

### T0 — frozen baseline (no retraining)

The reference checkpoint. Iters 1–3 all run on T0; iter 4 introduces
T1/T4/T5/T6.

### T1 — CE + label smoothing (α)

```
y_smooth = (1 - α) · y_onehot + α / K
loss = -Σ y_smooth_k · log_softmax(z_k)
```

α=0.10 in iter 4. Iter 5 sweeps α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
0.35}, finding a sharp peak at **α=0.20** (§5.5). **Ref:** Müller,
Kornblith, Hinton (arXiv:1906.02629).

### T4 — Asymmetric Loss (ASL)

```
ASL = -[ y · (1-p)^γ_+ · log p
       + (1-y) · p_m^γ_- · log(1-p_m) ]
where p_m = max(0, p - m)
```

We use the published default `(γ_+=1, γ_-=4, m=0.05)`. **Ref:** Ridnik
et al. 2021 (arXiv:2009.14119).

Note: the codebase actually uses `γ_+=1` (not the published `γ_+=0`)
because ASL with `γ_+=0` reduces to BCE on positives. This is captured
in `chip_multilabel/losses.py::AsymmetricLoss.__init__` defaults. We
flag this as a hyperparameter (Phase B will sweep both γ_+ ∈ {0, 1, 2}
and γ_- ∈ {2, 4, 6}).

### T5 — BCE (per-class binary)

Plain binary cross-entropy on multi-hot targets (single-positive in
practice because train labels are single-label):

```
BCE = -Σ_c [ y_c log p_c + (1 - y_c) log(1 - p_c) ]
```

### T6 — BCE → ASL curriculum

Warmup phase (4 epochs) BCE, then 4 epochs ASL with the T4 defaults.
Idea: BCE first establishes per-class score distribution; ASL then
sharpens the rare-class recall.

### T1_LS<xx> (iter 5)

Sweep of label-smoothing strength on the T1 recipe with `xx ∈ {05, 10,
15, 20, 25, 30, 35}` (i.e. α/100). All other hyperparameters held
(LR=1e-4, epochs=8). **Iter 5 winner:** `T1_LS20` (α=0.20) + I7.

## 4.3 Decision pipeline (per chip)

The full decoder, applied per chip after the inference variant
selection:

1. **Invalid heuristic** (`detect_invalid` on the raw chip image):
   if white-area ratio ≥0.95 and ≥3 of 4 borders contain orange pixels,
   short-circuit to `Invalid` regardless of model output.
2. **(I10 only)** **Entropy gate**: if `H(softmax(L)) ≥ 0.85·log(4)`,
   short-circuit to `Normal`.
3. **Threshold decoding**: `active = { c : s_c ≥ θ_c }`.
4. **Combo collapse**:
   - `|active| = 0` → `Normal`
   - `|active| = 1` → that class
   - `|active| = 2` and the canonical combo key is in `COMBO_KEYS` →
     that combo
   - `|active| = 2` and combo not in COMBO_KEYS (i.e.
     `scratch+scratch_rot` excluded) → fall back to single-class with
     highest probability (`combo_collapsed`)
   - `|active| ≥ 3` → keep top-2 by probability (`truncated_3plus`)

This logic lives in `chip_multilabel/decision_tree.py::decide`.

## 4.4 Metrics

We report macro-F1 (mean F1 over the 4 defect classes), micro-F1, mAP,
hamming loss, subset accuracy, and `top1_11class` (the 11-class
single-label-equivalent accuracy obtained by mapping each chip's
prediction to its class key). Pre/post-temperature ECE is reported
on the single-label val subset for context.

`top1_11class` is the operationally relevant metric in production: each
chip is routed to a single 11-class bin downstream. macro-F1 is the
primary headline because it weighs all 4 defect classes equally
regardless of eval-set class frequencies.
