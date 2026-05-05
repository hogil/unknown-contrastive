# Chip Multi-Label — Experiment Log

Documentation for the chip-level multi-label defect-classification iterations
on the synthetic 11-class evaluation set. Source of truth for paper-grade
numbers, ablations, and decisions. All runs live under
`D:/project/unknown-contrastive/outputs/`.

## Abstract

We turn an existing single-label chip CNN (5-class CE, ConvNeXtV2-Base 384,
trained on 4 defect classes + `invalid_main`) into a multi-label predictor over
an 11-class chip benchmark (4 single-defect, 5 defect combos, Normal, Invalid,
2200 chips). Starting from `argmax >= 0.5` (macro-F1 = 0.7302), inference-only
tricks — F1-max thresholds, sweep-based decision rules, and an entropy-based
Normal gate — push the unchanged backbone to **0.8542** (I10 stack). Stage 2
retraining (CE+LS / ASL / BCE / BCE→ASL) gives a further **+0.0092** at
T1__I10 = **0.8634**. Phase A1 then sweeps label-smoothing and lands at
**LS=0.20, I7 → macro-F1 = 0.9268, top1\_11 = 0.8449**: a **+0.1966 macro-F1**
gain over the argmax baseline using a single-label-trained backbone, no TTA,
and ~6 minutes of fine-tuning.

## Table of contents

- [00_problem_setup.md](00_problem_setup.md) — task, 11-class set, data synthesis, sister-repo refs
- [01_methods.md](01_methods.md) — I0-I10 inference variants, T0-T6 training variants (with formulas + paper refs)
- [02_results.md](02_results.md) — cross-iter timeline, top-15 all-time cells, per-iter winners
- [03_ablations.md](03_ablations.md) — what worked / what didn't, with deltas
- [04_error_analysis.md](04_error_analysis.md) — fork over-firing, scratch_rot diffuse prior, missed_normal
- [iters/iter_01_stage1_baseline.md](iters/iter_01_stage1_baseline.md) — Stage 1 baseline (I0-I5)
- [iters/iter_02_inference_variants.md](iters/iter_02_inference_variants.md) — Extended inference variants (I6-I9)
- [iters/iter_03_entropy_normal.md](iters/iter_03_entropy_normal.md) — I10 entropy-based Normal gate
- [iters/iter_04_stage2_matrix.md](iters/iter_04_stage2_matrix.md) — Stage 2: T1/T4/T5/T6 retrain × inference matrix
- [iters/iter_05_phase_a1_ls_sweep.md](iters/iter_05_phase_a1_ls_sweep.md) — Phase A1 label-smoothing sweep
- [tables/all_runs_macro_f1.csv](tables/all_runs_macro_f1.csv) — flat dump of every cell

## Cross-iter best timeline

| iter | best_cell      | macro_f1   | top1_11    | Δ macro_f1 vs prev | source                                                                         |
|-----:|----------------|-----------:|-----------:|-------------------:|--------------------------------------------------------------------------------|
|    1 | T0__I3         |     0.8466 |     0.6017 |          baseline  | outputs/stage1_260505_162842                                                   |
|    2 | T0__I7         |     0.8485 |     0.6210 |            +0.0019 | outputs/stage1_260505_165400                                                   |
|    3 | T0__I10        |     0.8542 |     0.6517 |            +0.0057 | outputs/stage1_260505_170827                                                   |
|    4 | T1__I10        |     0.8634 |     0.7006 |            +0.0092 | outputs/stage2_260505_170121 + outputs/stage1_260505_173649                    |
|    5 | **T1_LS20__I7**| **0.9268** | **0.8449** |        **+0.0634** | outputs/phase_a_260505_175105                                                  |

Argmax baseline (I0): macro_f1 = 0.7302, top1\_11 = 0.4472. Iter-5 best is **+0.1966 macro-F1 / +0.3977 top1\_11** over baseline.

## Hard rules (apply across all iters)

- **TTA permanently disallowed.** Rotation augmentation flips `scratch ↔ scratch_rot` semantics. Single ablation showed −0.018 macro-F1; never re-enabled.
- **GPU = 1 job at a time** (user directive). Sweeps are strictly sequential.
- **`scratch + scratch_rot` combo excluded** from the 11-class set (same defect family, ill-defined ground truth).

## Citations (key papers)

- Müller, Kornblith, Hinton (2019). *When Does Label Smoothing Help?* arXiv:1906.02629 — basis for **T1** (CE + LS) and the LS sweep in iter 5.
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks.* arXiv:1706.04599 — temperature scaling, **I4**.
- Lipton, Elkan, Naryanaswamy (2014). *Optimal Thresholding Classifiers to Maximize F1 Score.* arXiv:1402.1892 — per-class F1-max thresholds, **I3 / I7**.
- Ridnik et al. (2021). *Asymmetric Loss for Multi-Label Classification.* arXiv:2009.14119 — **T4 (ASL)**, **T6 (BCE → ASL)**.

## Sister repo

- `D:/project/known-cnn` — supervised open-set CNN, data synthesis (`dist_apply/`), TAPT backbone (`outputs/logs_<kind>/overall/best_model.pth`).
