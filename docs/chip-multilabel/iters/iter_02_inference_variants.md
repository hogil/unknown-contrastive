# Iter 2 — Extended inference variants (I6–I9)

**Run dir**: `outputs/stage1_260505_165400/`
**Date**: 2026-05-05 16:54
**Model**: T0 frozen (same as iter 1).

## Goal

Extend the inference grid with four new variants designed to fix the
specific failure modes that iter 1 surfaced (fork over-firing, lost combos).

## New variants (recap)

- **I6** — F1-max thresholds with min-floor 0.30 (suppress fork's runaway low threshold).
- **I7** — F1-max + per-class step-search (Δ=0.02) on val for finer tuning.
- **I8** — F1-max + top-K=1 rescue (single-class fallback) — never reach into combos.
- **I9** — F1-max + temperature only (no top-K rescue).

Plus all I0–I4 cells re-run for sanity.

## Cells run

| cell_id | inference rule              | macro_f1 | micro_f1 | top1_11 | T      | ECE_post |
|---------|-----------------------------|---------:|---------:|--------:|-------:|---------:|
| T0__I0  | argmax @ 0.5                |   0.7302 |   0.7432 |  0.4472 | 1.0000 |   0.0778 |
| T0__I1  | F1-max                      |   0.8444 |   0.8165 |  0.6324 | 1.0000 |   0.0778 |
| T0__I2  | top-K=2                     |   0.7673 |   0.7928 |  0.5739 | 1.0000 |   0.0778 |
| T0__I3  | F1-max + topK rescue        |   0.8466 |   0.8143 |  0.6017 | 1.0000 |   0.0778 |
| T0__I4  | I3 + temperature            |   0.8466 |   0.8143 |  0.6017 | 0.3757 |   0.0129 |
| T0__I6  | F1-max + min-floor 0.30     |   0.8177 |   0.8127 |  0.5881 | 1.0000 |   0.0778 |
| **T0__I7** | F1-max + step-search Δ=0.02 | **0.8485** | **0.8198** | **0.6210** | 1.0000 |   0.0778 |
| T0__I8  | I3 + topK=1 rescue          |   0.8456 |   0.8135 |  0.6017 | 1.0000 |   0.0778 |
| T0__I9  | F1-max + temperature        |   0.7741 |   0.7584 |  0.5341 | 0.7300 |   0.0411 |

_Source: outputs/stage1_260505_165400/results_matrix.parquet._

## Winner

**T0__I7** with macro_f1 = **0.8485**, top1\_11 = **0.6210** — narrowly beats
T0__I3 by **+0.0019** macro-F1 and **+0.0193** top1\_11.

## Per-class F1 detail (winner cell T0__I7)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1400 |    0.5005 | 0.8609 | 0.6330 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_165400/per_class_metrics.parquet (cell_id=T0__I7)._

## Key observations

- **Step-search (I7) wins** — small gain over I3 mostly because fork's
  threshold lifts from 0.1195 → 0.1400 (better precision) and scratch's
  drops from 0.7682 → 0.7400 (better recall).
- **Min-floor (I6) regresses** by 0.0289 macro-F1. fork *needs* a low
  threshold; clipping to 0.30 throws away ~12% of fork recall.
- **Top-K=1 (I8)** ties I3 within rounding. Implies most of I3's top-K
  rescue successes were on truly single-defect chips, not combos.
- **Temperature alone (I9)** loses 0.07 macro-F1 vs I1 — without rescue,
  the rescaled sigmoid drops some defenders below their (rescaled)
  thresholds.

## Errors review

A targeted markdown report `errors_review_T0__I7.md` is checked in alongside
this iter's outputs:
`outputs/stage1_260505_165400/errors_review_T0__I7.md`.

It pastes the top-200 error chips per category alongside per-class
sigmoid scores. Headline takeaway: fork false-positives dominate, and
they are concentrated on chips where a real defect (scratch_rot, bank_boundary)
co-occurs and pulls fork's logit into the borderline band.

## Decision for next iter

- The borderline-fork-on-defect-chip pattern looks like a *normal-side*
  problem in disguise: chips that should fire one defect end up also
  firing fork. Try a Normal gate (entropy-based) and see if it cleans
  up via providing an explicit "no other defect" signal.
