# 03 — Ablations: what worked / what didn't

All deltas measured on the same 2200-chip 11-class eval set. Pull
quotes use 4-decimal numbers from the canonical sources.

## Inference-side ablations (fixed model = T0)

| change vs baseline               | from cell        | to cell          | Δ macro_f1 | Δ top1_11 | verdict        |
|----------------------------------|------------------|------------------|-----------:|----------:|----------------|
| argmax → F1-max thresholds       | T0__I0 (0.7302)  | T0__I1 (0.8444)  |    +0.1142 |   +0.1852 | huge win       |
| F1-max → top-K=2 alone           | T0__I1 (0.8444)  | T0__I2 (0.7673)  |    -0.0771 |   -0.0585 | regression     |
| F1-max + top-K rescue (I1+I2)    | T0__I1 (0.8444)  | T0__I3 (0.8466)  |    +0.0022 |   -0.0307 | tiny win on F1 |
| I3 + temperature scaling         | T0__I3 (0.8466)  | T0__I4 (0.8466)  |    +0.0000 |   +0.0000 | no-op on F1    |
| I3 + TTA (rotation 4×)           | T0__I3 (0.8466)  | T0__I5 (0.8287)  |    -0.0179 |   -0.0006 | **DISALLOWED** |
| F1-max + min-floor 0.30          | T0__I3 (0.8466)  | T0__I6 (0.8177)  |    -0.0289 |   -0.0136 | regression     |
| F1-max + step-search Δ=0.02      | T0__I3 (0.8466)  | T0__I7 (0.8485)  |    +0.0019 |   +0.0193 | small win      |
| I3 + top-K=1 fallback            | T0__I3 (0.8466)  | T0__I8 (0.8456)  |    -0.0010 |   +0.0000 | flat           |
| I1 + temperature only            | T0__I1 (0.8444)  | T0__I9 (0.7741)  |    -0.0703 |   -0.0983 | regression     |
| I7 + entropy Normal gate (I10)   | T0__I7 (0.8485)  | T0__I10 (0.8542) |    +0.0057 |   +0.0307 | win, durable   |

_Source: outputs/stage1_260505_162842, _165400, _170827 results_matrix.parquet._

### Verdicts

- **F1-max thresholds (I1) is the single biggest single inference change**
  (+0.1142 macro-F1 over argmax). Most of the climb away from baseline is
  this one trick.
- **Step-search (I7)** is a clean micro-improvement on I3 with no extra
  pipeline complexity (Δ=0.02 grid).
- **Entropy Normal gate (I10)** is the only inference idea that *survives*
  retraining — every other variant either ties or trails I7 once the
  model is fine-tuned.
- **TTA (I5) is permanently disallowed** because rotation conflates
  scratch / scratch_rot. Even though it sometimes nudges precision, the
  semantic damage is unacceptable.
- **Temperature scaling (I4, I9)** does not help macro-F1 because the
  threshold sweep already absorbs whatever calibration shift T provides;
  it does help ECE (0.0778 → 0.0129 on I4), so keep it for any
  probability-honest downstream.
- **Min-floor 0.30 (I6)** hurts because the val-tuned fork threshold is
  ~0.12 — clipping it to 0.30 throws away most of fork's recall.

## Training-side ablations (best inference = I10)

| variant | loss            | best cell    | macro_f1 | top1_11 | Δ vs T0__I10 | verdict            |
|---------|-----------------|--------------|---------:|--------:|-------------:|--------------------|
| T0      | none (frozen)   | T0__I10      |   0.8542 |  0.6517 |       (ref)  | baseline           |
| T1      | CE + LS 0.10    | T1__I10      |   0.8634 |  0.7006 |      +0.0092 | win                |
| T4      | ASL             | T4__I10      |   0.7759 |  0.5830 |      -0.0783 | regression         |
| T5      | BCE             | T5__I10      |   0.7589 |  0.5432 |      -0.0953 | regression         |
| T6      | BCE → ASL       | T6__I10      |   0.8193 |  0.6256 |      -0.0349 | regression         |

_Source: outputs/stage2_260505_170121/results_matrix.parquet for T1/T4/T5/T6 ×
I0..I9 grid; outputs/stage1_260505_{173649,173829,173955,174123}/results_matrix.parquet
for the post-hoc I10 inference rows._

### Verdicts

- **T1 (CE + LS 0.10) is the only training intervention that helped**
  on the multi-label benchmark. The single-label CE pretrain provides a
  decent base and label smoothing softens the softmax peak so the
  runner-up class still has a usable score.
- **T4 (ASL), T5 (BCE), T6 (BCE→ASL) all regress** despite being the
  "obvious" multi-label choices. Hypothesis: these losses change the
  distribution of activations enough that the F1-max thresholds tuned
  on val don't transfer cleanly. Specifically T4 and T5 over-suppress
  bank_boundary, dropping its F1 from ~0.96 to ~0.85.
- T6 (BCE→ASL) is the worst hybrid: BCE collapses the softmax structure,
  then ASL doesn't have time to rebuild useful asymmetry in 4 epochs.
- T1 is also the cheapest (~330s on RTX 4090).

## LS sweep (iter 5, T1 only)

| LS    | best inference | macro_f1 | top1_11 | Δ vs LS=0.10 best (T1__I10 = 0.8634) |
|------:|----------------|---------:|--------:|-------------------------------------:|
|  0.05 | I7             |   0.7964 |  0.5591 |                              -0.0670 |
|  0.10 | I3             |   0.8363 |  0.6261 |                              -0.0271 |
|  0.15 | I3             |   0.8961 |  0.7517 |                              +0.0327 |
|  0.20 | **I7**         | **0.9268** | **0.8449** |                          **+0.0634** |
|  0.25 | I3             |   0.8663 |  0.6989 |                              +0.0029 |
|  0.30 | I3             |   0.8185 |  0.6466 |                              -0.0449 |

_Source: outputs/phase_a_260505_175105/sweep_log.csv,
outputs/phase_a_260505_182044/sweep_log.csv._

### Verdicts

- **LS=0.20 is the sweet spot** — too little smoothing leaves the
  single-label collapse intact; too much smoothing erases informative
  margin between classes.
- The curve is **non-monotonic and sharp**: 0.20 → 0.9268 vs 0.15 →
  0.8961 vs 0.25 → 0.8663. ±0.05 around the optimum costs ~0.03 macro-F1.
- The optimum cell is **`T1_LS20 + I7`**, not + I10. The entropy gate
  (I10) helps frozen / mildly-trained models because their Normal logit
  is poor; once LS=0.20 has trained the model into a more
  well-calibrated multi-label state, the explicit Normal gate becomes
  redundant and slightly hurts (0.9268 → 0.8841).

## Things that didn't work — short list

1. **TTA (I5)** — rotation breaks scratch / scratch_rot.
2. **ASL (T4)** — over-suppresses bank_boundary.
3. **BCE (T5)** — drops the softmax shape that was actually useful.
4. **BCE → ASL (T6)** — neither phase converges far enough in 4 epochs.
5. **Min-threshold floor (I6)** — fork needs a low threshold; floor kills it.
6. **Temperature alone (I9)** — without rescue, top-K combo recovery drops.
