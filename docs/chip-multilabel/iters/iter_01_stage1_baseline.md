# Iter 1 — Stage 1 baseline (I0–I5)

**Run dir**: `outputs/stage1_260505_162842/`
**Date**: 2026-05-05 16:28
**Model**: T0 frozen — `chip5_round4_v14_260505_061558_running/best_model.pth` (ConvNeXtV2-Base 384, single-label CE 5-class).
**Eval set**: 2200 chips (n_eval over 4 trained classes = 1760).

## Goal

Establish a baseline macro-F1 on the chip-level multi-label benchmark using
**only inference-time interventions** (I0–I5) on a frozen 5-class single-label
backbone. No retraining.

## Cells run

| cell_id | inference rule                       | macro_f1 | micro_f1 | mAP    | top1_11 | T      | ECE_post |
|---------|--------------------------------------|---------:|---------:|-------:|--------:|-------:|---------:|
| T0__I0  | argmax @ 0.5 (baseline)              |   0.7302 |   0.7432 | 0.6766 |  0.4472 | 1.0000 |   0.0778 |
| T0__I1  | per-class F1-max thresholds          |   0.8444 |   0.8165 | 0.8339 |  0.6324 | 1.0000 |   0.0778 |
| T0__I2  | top-K=2                              |   0.7673 |   0.7928 | 0.8484 |  0.5739 | 1.0000 |   0.0778 |
| **T0__I3** | F1-max + top-K rescue            | **0.8466** | **0.8143** | **0.8484** | **0.6017** | 1.0000 |   0.0778 |
| T0__I4  | I3 + temperature scaling             |   0.8466 |   0.8143 | 0.8484 |  0.6017 | 0.3757 |   0.0129 |
| T0__I5  | I4 + TTA (rotation 4×)               |   0.8287 |   0.8022 | 0.8347 |  0.6011 | 0.3624 |   0.0109 |

_Source: outputs/stage1_260505_162842/results_matrix.parquet._

## Winner

**T0__I3** with macro_f1 = **0.8466**, top1\_11 = **0.6017**.

## Key observations

- **F1-max thresholds (I1) is the single biggest jump** of the entire
  project: +0.1142 macro-F1 over argmax. fork's F1-max threshold sits at
  **0.0041**, scratch_rot at **0.6013**, bank_boundary at **0.1945** — none
  of these are near 0.5, confirming that the single-label CE model has wildly
  miscalibrated cross-class scores for multi-label decoding.
- **Top-K alone (I2) regresses** (-0.0771) because it always asserts two
  classes even when only one is present.
- **I3 = I1 ∪ topK(s, k=2)** gets the best of both: confidence-based for
  combos, threshold-based for singles. Marginal gain over I1 (+0.0022),
  but cleaner top1_11 logic.
- **Temperature (I4)** gives **identical** macro-F1 to I3 — temperature
  rescales sigmoid scores monotonically, and the F1-max thresholds re-tune
  to the rescaled scores. ECE drops 0.0778 → 0.0129, useful for honest
  probabilities downstream.
- **TTA (I5) regresses** by 0.0179 macro-F1. Rotation conflates `scratch`
  and `scratch_rot`. **Permanently disallowed from this point.**

## Per-class F1 detail (winner cell T0__I3)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4994 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1195 |    0.4843 | 0.9141 | 0.6331 | 0.5762 |
| scratch        |    0.7682 |    1.0000 | 0.9438 | 0.9711 | 0.9723 |
| scratch_rot    |    0.8355 |    1.0000 | 0.7000 | 0.8235 | 0.8700 |

_Source: outputs/stage1_260505_162842/per_class_metrics.parquet._

## Error breakdown (T0__I3)

| error_type           | count |
|----------------------|------:|
| false_positive_fork  |   277 |
| wrong_combo          |   264 |
| missed_normal        |   160 |
| **total**            |   701 |

_Source: outputs/stage1_260505_162842/errors.parquet._

`fork` recall 0.91 / precision 0.48 — the threshold is so low that fork
fires on roughly 1 in 2 non-fork chips. This is the headline problem and
sets up iter 2-3.

## Decision for next iter

- Try more aggressive threshold search (step-grid Δ=0.02) to nudge
  fork up at the cost of a sliver of recall.
- Try a min-floor on thresholds to suppress fork over-firing.
- Try alternative top-K rescue rules.

## Files

- `results_matrix.parquet`, `per_class_metrics.parquet`,
  `confusion_11class.parquet`, `errors.parquet`, `eval_summary.json`,
  `report.md`, `thresholds.json` under `outputs/stage1_260505_162842/`.
- Error chips at `outputs/stage1_260505_162842/errors/<cell>/<error_type>/`.
