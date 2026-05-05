# Iter 4 — Stage 2: T1/T4/T5/T6 retrain × inference matrix

**Run dirs**:
- Train: `outputs/logs_chip_multilabel/T1_260505_170126/`,
  `outputs/logs_chip_multilabel/T4_260505_170706/`,
  `outputs/logs_chip_multilabel/T5_260505_171912/`,
  `outputs/logs_chip_multilabel/T6_260505_172459/`
- Stage 2 grid eval: `outputs/stage2_260505_170121/`
- Post-hoc I10 add-on: `outputs/stage1_260505_173649/` (T1+I10),
  `outputs/stage1_260505_173829/` (T4+I10), `outputs/stage1_260505_173955/`
  (T5+I10), `outputs/stage1_260505_174123/` (T6+I10)

**Date**: 2026-05-05 17:01–17:41

## Goal

T0 (frozen) inference is exhausted. Try four loss / training regimes that
multi-label literature suggests, on the *same* chip dataset, and re-apply
the inference variants on each.

## Training variants (recap)

- **T1** — CE + label smoothing α=0.10, 8 epochs, LR=1e-4 — Müller et al. 2019.
- **T4** — Asymmetric Loss (ASL) γ_+=0, γ_-=4, m=0.05 — Ridnik et al. 2021.
- **T5** — pure BCE per-class — multi-label baseline.
- **T6** — BCE→ASL curriculum (4ep BCE then 4ep ASL).

All variants train on the same 327 chip / 82 val split (single-label
ground truth, 5 classes). Compute: 330–720s each on a single GPU.

## Train summary

| variant | loss          | best_val_acc | best_epoch | epochs | elapsed_sec |
|---------|---------------|-------------:|-----------:|-------:|------------:|
| T1      | ce_ls0.1      |       1.0000 |          1 |      8 |       331.8 |
| T4      | asl           |       1.0000 |          2 |      8 |       719.1 |
| T5      | bce           |       1.0000 |          2 |      8 |       340.1 |
| T6      | bce_then_asl  |       1.0000 |          1 |      8 |       339.7 |

_Source: outputs/logs_chip_multilabel/{T1,T4,T5,T6}_*/train_summary.json._

All four hit val-acc 1.0 quickly — the 5-class single-label task is easy.
The discriminator is what happens at the **multi-label benchmark**.

## Stage 2 grid (T × I0..I9, 36 cells)

Top cells per train variant (best inference):

| train | best inference | macro_f1 | top1_11 |
|-------|----------------|---------:|--------:|
| T1    | I1             |   0.8384 |  0.6318 |
| T4    | I3 / I4 / I6 / I9 (tie) | 0.7811 | 0.5881 |
| T5    | T5__I1         |   0.8024 |  0.4591 |
| T6    | T6__I3         |   0.8396 |  0.5108 |

_Source: outputs/stage2_260505_170121/results_matrix.parquet._

Worth noting: **none of T4/T5/T6 beats T0__I10 (0.8542) at this stage**.
T1 alone is competitive (0.8384 vs 0.8542 — slightly behind without I10).

## Post-hoc I10 sweep (one cell per train variant)

| cell     | macro_f1 | micro_f1 | mAP    | top1_11 |
|----------|---------:|---------:|-------:|--------:|
| **T1__I10** | **0.8634** | **0.8518** | 0.8753 | **0.7006** |
| T4__I10  |   0.7759 |   0.7836 | 0.8445 |  0.5830 |
| T5__I10  |   0.7589 |   0.7736 | 0.8270 |  0.5432 |
| T6__I10  |   0.8193 |   0.8291 | 0.8684 |  0.6256 |

_Source: outputs/stage1_260505_{173649,173829,173955,174123}/results_matrix.parquet._

## Winner

**T1__I10** with macro_f1 = **0.8634**, top1\_11 = **0.7006**. First time the
benchmark crosses 0.86 macro-F1.

Δ vs iter 3 (T0__I10): **+0.0092** macro-F1, **+0.0489** top1\_11.

## Per-class F1 detail (T1__I10)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4600 |    1.0000 | 0.7781 | 0.8752 | 0.8969 |
| fork           |    0.2200 |    0.7014 | 0.7891 | 0.7426 | 0.6607 |
| scratch        |    0.6600 |    0.9803 | 0.9354 | 0.9574 | 0.9824 |
| scratch_rot    |    0.5000 |    1.0000 | 0.7833 | 0.8785 | 0.9614 |

_Source: outputs/stage1_260505_173649/per_class_metrics.parquet (cell_id=T0__I10 row, model=T1)._

The headline gain is **fork F1 0.6607 → 0.7426** (+0.082): label
smoothing softens the dominant logit and lifts fork's distinctness.
fork *precision* nearly doubles (0.5360 → 0.7014) at the cost of some
recall (0.8609 → 0.7891) — exactly what we wanted.

## Errors (T1__I10)

| error_type             | count |
|------------------------|------:|
| wrong_combo            |   304 |
| false_positive_fork    |   155 |
| wrong_normal_entropy   |    62 |
| false_positive_scratch |     6 |
| **total**              |   527 |

_Source: outputs/stage1_260505_173649/errors.parquet (cell_id=T0__I10)._

vs iter 1 (T0__I3, 701 total): **-25%** total errors, with fork FPs nearly
halved. `missed_normal` is replaced by `wrong_normal_entropy` — same
underlying Normal/defect ambiguity but now flagged via entropy gate.

## Key observations

- **Only T1 helps.** T4 (ASL), T5 (BCE), T6 (BCE→ASL) all regress vs frozen
  T0__I10. The "obvious" multi-label losses are not the right tool when
  you're starting from a single-label CE pretrain on the same data.
- **CE + label smoothing is the cheap winner**: same training cost as T5,
  same recipe as T0 except for `α=0.1`, **+0.0092 macro-F1**.
- **I10 beats I3/I7 at every train cell**, confirming the entropy gate
  generalises across training regimes.
- The post-hoc workflow (run T0__I0..I9 grid, then add I10 separately)
  was a procedural bug — should have included I10 in the original sweep
  list. Cleaned up by separate `outputs/stage1_260505_17{3649,3829,3955,4123}/`
  runs.

## Decision for next iter

- T1 won; sweep its single hyperparameter (label-smoothing α) and see how
  far this single direction takes us.
- Phase A1: α ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, LR=1e-4, ep=8, eval
  with I3 / I7 / I10 each.
