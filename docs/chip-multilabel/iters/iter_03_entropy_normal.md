# Iter 3 — I10 entropy-based Normal gate

**Run dir**: `outputs/stage1_260505_170827/`
**Date**: 2026-05-05 17:08
**Model**: T0 frozen.

## Goal

Add an explicit `Normal` decoding rule. Iter 1's error breakdown showed
160 missed_normal cases — chips that should be `Normal` but had at least
one defect class fire because the single-label-trained model has no
incentive to push *all* sigmoids low simultaneously.

## New variant

**I10 = I7 + entropy Normal gate**:
- Run I7 (F1-max + step-search) to get per-class predictions.
- If **no class** crosses its threshold *and* binary entropy of the
  sigmoid vector is **< H_thresh** (chosen on val to maximise Normal-class
  F1, ~0.30 nats), label as `Normal`.
- Else, output the per-class predictions as-is (default to "no defect"
  if none crossed θ_c, with explicit Normal flag).

```
H = -Σ_c [s_c log s_c + (1-s_c) log(1-s_c)]
if not any(s_c >= θ_c) and H < H_thresh:
    pred_normal = True
```

## Cells run

| cell_id  | inference rule                       | macro_f1 | micro_f1 | mAP    | top1_11 |
|----------|--------------------------------------|---------:|---------:|-------:|--------:|
| T0__I7   | F1-max + step-search                 |   0.8485 |   0.8198 | 0.8484 |  0.6210 |
| **T0__I10** | I7 + entropy Normal gate         | **0.8542** | **0.8311** | 0.8484 | **0.6517** |

_Source: outputs/stage1_260505_170827/results_matrix.parquet._

## Winner

**T0__I10** with macro_f1 = **0.8542**, top1\_11 = **0.6517**.
Δ vs iter 2 winner: **+0.0057** macro-F1, **+0.0307** top1\_11.

## Per-class F1 detail (T0__I10)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9786 | 0.9297 | 0.9535 | 0.9752 |
| fork           |    0.1400 |    0.5360 | 0.8609 | 0.6607 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_170827/per_class_metrics.parquet (cell_id=T0__I10)._

## Key observations

- The **fork F1 lift comes from the precision side** (0.5005 → 0.5360,
  +0.036) — fork was firing on Normal chips, and the entropy gate now
  vetoes those.
- bank_boundary recall drops slightly (0.9391 → 0.9297) — handful of
  borderline Normal-vs-bank_boundary chips now resolve as Normal. Net
  positive on macro-F1.
- top1_11 jumps **+0.0307** because the Normal class wasn't being decoded
  at all before; now it is, and the Normal column of the 11-class
  confusion matrix actually has diagonal mass.

## Why this matters

Iter 3 is the first iter to *add a class-decoding rule*, not just tune
thresholds. The model never saw `Normal` during training (only 4 defect
classes + invalid_main), so without an explicit decoder there is no way
to assign chips to `Normal`. The entropy gate is a cheap, training-free
fix.

I10 is the inference setting that **carries forward** as the default
`I*` for all subsequent iters (post-hoc applied to T1/T4/T5/T6 in iter 4,
swept in iter 5).

## Decision for next iter

- T0 frozen has now exhausted the inference toolbox. The next jump
  requires retraining. Try four loss/training regimes — CE+LS (T1),
  ASL (T4), BCE (T5), BCE→ASL (T6) — and apply the I0..I9 grid plus
  I10 to each.
