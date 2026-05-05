# 02 — Results

All numbers reported to 4 decimal places. Eval set: 2200 chips, 11-class
multi-label.

## Cross-iter best timeline

| iter | best_cell      | macro_f1   | top1_11    | Δ macro_f1 | Δ top1_11  | source                                                                |
|-----:|----------------|-----------:|-----------:|-----------:|-----------:|-----------------------------------------------------------------------|
|  0\* | T0__I0         |     0.7302 |     0.4472 |          — |          — | outputs/stage1_260505_162842 (argmax baseline)                        |
|    1 | T0__I3         |     0.8466 |     0.6017 |    +0.1164 |    +0.1545 | outputs/stage1_260505_162842                                          |
|    2 | T0__I7         |     0.8485 |     0.6210 |    +0.0019 |    +0.0193 | outputs/stage1_260505_165400                                          |
|    3 | T0__I10        |     0.8542 |     0.6517 |    +0.0057 |    +0.0307 | outputs/stage1_260505_170827                                          |
|    4 | T1__I10        |     0.8634 |     0.7006 |    +0.0092 |    +0.0489 | outputs/stage1_260505_173649                                          |
|    5 | **T1_LS20__I7**| **0.9268** | **0.8449** | **+0.0634**| **+0.1443**| outputs/phase_a_260505_175105                                         |

\*iter 0 = the argmax baseline cell that lives inside iter 1's run.

Cumulative gain vs argmax baseline: **+0.1966 macro-F1**, **+0.3977 top1\_11class**.

_Source: outputs/stage1_260505_162842/results_matrix.parquet,
outputs/stage1_260505_165400/results_matrix.parquet,
outputs/stage1_260505_170827/results_matrix.parquet,
outputs/stage1_260505_173649/results_matrix.parquet,
outputs/phase_a_260505_175105/sweep_log.csv._

## Top-15 all-time cells (by macro_f1)

| rank | iter | cell_id        | macro_f1 | top1_11 | source                                                                  |
|-----:|-----:|----------------|---------:|--------:|-------------------------------------------------------------------------|
|    1 |    5 | T1_LS20__I7    |   0.9268 |  0.8449 | outputs/phase_a_260505_175105/sweep_log.csv                             |
|    2 |    5 | T1_LS20__I3    |   0.9239 |  0.8324 | outputs/phase_a_260505_175105/sweep_log.csv                             |
|    3 |    5 | T1_LS15__I3    |   0.8961 |  0.7517 | outputs/phase_a_260505_175105/sweep_log.csv                             |
|    4 |    5 | T1_LS15__I7    |   0.8959 |  0.7517 | outputs/phase_a_260505_175105/sweep_log.csv                             |
|    5 |    5 | T1_LS15__I10   |   0.8900 |  0.7449 | outputs/phase_a_260505_175105/sweep_log.csv                             |
|    6 |    5 | T1_LS20__I10   |   0.8841 |  0.8136 | outputs/phase_a_260505_175105/sweep_log.csv                             |
|    7 |    5 | T1_LS25__I3    |   0.8663 |  0.6989 | outputs/phase_a_260505_182044/sweep_log.csv                             |
|    8 |    5 | T1_LS25__I7    |   0.8647 |  0.6903 | outputs/phase_a_260505_182044/sweep_log.csv                             |
|    9 |    4 | T1__I10        |   0.8634 |  0.7006 | outputs/stage1_260505_173649/results_matrix.parquet                     |
|   10 |    3 | T0__I10        |   0.8542 |  0.6517 | outputs/stage1_260505_170827/results_matrix.parquet                     |
|   11 |    2 | T0__I7         |   0.8485 |  0.6210 | outputs/stage1_260505_165400/results_matrix.parquet                     |
|   12 |    1 | T0__I4         |   0.8466 |  0.6017 | outputs/stage1_260505_162842/results_matrix.parquet                     |
|   13 |    1 | T0__I3         |   0.8466 |  0.6017 | outputs/stage1_260505_162842/results_matrix.parquet                     |
|   14 |    2 | T0__I8         |   0.8456 |  0.6017 | outputs/stage1_260505_165400/results_matrix.parquet                     |
|   15 |    1 | T0__I1         |   0.8444 |  0.6324 | outputs/stage1_260505_162842/results_matrix.parquet                     |

_Source: docs/chip-multilabel/tables/all_runs_macro_f1.csv (all rows incl. ranks 16+)._

## Per-iter winner — per-class F1 detail

### iter 1 — T0__I3 (frozen, F1-max + top-K rescue)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4994 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1195 |    0.4843 | 0.9141 | 0.6331 | 0.5762 |
| scratch        |    0.7682 |    1.0000 | 0.9438 | 0.9711 | 0.9723 |
| scratch_rot    |    0.8355 |    1.0000 | 0.7000 | 0.8235 | 0.8700 |

_Source: outputs/stage1_260505_162842/per_class_metrics.parquet (cell_id=T0__I3)._

### iter 2 — T0__I7 (frozen, F1-max + step-search Δ=0.02)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1400 |    0.5005 | 0.8609 | 0.6330 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_165400/per_class_metrics.parquet (cell_id=T0__I7)._

### iter 3 — T0__I10 (frozen, I7 + entropy Normal gate)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9786 | 0.9297 | 0.9535 | 0.9752 |
| fork           |    0.1400 |    0.5360 | 0.8609 | 0.6607 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_170827/per_class_metrics.parquet (cell_id=T0__I10)._

The Normal-gate gain comes mostly from `fork` precision (0.5005 → 0.5360, recall held).

### iter 4 — T1__I10 (CE+LS=0.10 retrain, I7+entropy)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4600 |    1.0000 | 0.7781 | 0.8752 | 0.8969 |
| fork           |    0.2200 |    0.7014 | 0.7891 | 0.7426 | 0.6607 |
| scratch        |    0.6600 |    0.9803 | 0.9354 | 0.9574 | 0.9824 |
| scratch_rot    |    0.5000 |    1.0000 | 0.7833 | 0.8785 | 0.9614 |

_Source: outputs/stage1_260505_173649/per_class_metrics.parquet (cell_id=T0__I10 row, but the model was T1)._

The big jump is **fork F1 0.6607 → 0.7426** (+0.082): label smoothing
flattens the runner-up logit so multi-label thresholding actually has a
distinguishable score for fork-in-combo chips.

### iter 5 — T1_LS20__I7 (CE+LS=0.20 retrain, I7) — overall best

`per_class_metrics.parquet` is not stored for sweep cells; per-class
breakdown is the next thing to capture in iter 6 if needed.
Aggregate: macro_f1 = 0.9268, top1\_11 = 0.8449.

_Source: outputs/phase_a_260505_175105/sweep_log.csv (LS=0.20, inference_id=I7)._

## Table dump

`tables/all_runs_macro_f1.csv` contains every iter-1-through-5 cell (69 rows)
with columns: `iter, cell_id, train_id, inference_id, macro_f1, micro_f1,
mAP, top1_11class, temperature, ece_post, source`.
