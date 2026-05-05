# 00 — Problem setup

## Task statement

A wafer chip image (200×200 grayscale) may contain **zero, one, or more
defects** simultaneously. We have a chip-level CNN that was trained
**single-label, 5-way cross-entropy** (4 defect classes + `invalid_main`),
and we wish to repurpose it as a **multi-label** predictor over the
real evaluation distribution: 4 single defects, 5 plausible defect
**combinations**, plus `Normal` and `Invalid` — eleven classes total.

Single-label train, multi-label predict.

## Class set (11)

| Group              | Class                       | Notes                                  | n eval |
|--------------------|-----------------------------|----------------------------------------|-------:|
| Single defect      | `bank_boundary`             | Trained class                          |    240 |
|                    | `fork`                      | Trained class                          |    240 |
|                    | `scratch`                   | Trained class                          |    160 |
|                    | `scratch_rot`               | Trained class (rotated scratch)        |    160 |
| Combo (2 defects)  | `bank_boundary+fork`        | Combo                                  |    160 |
|                    | `bank_boundary+scratch`     | Combo                                  |    160 |
|                    | `bank_boundary+scratch_rot` | Combo                                  |    160 |
|                    | `fork+scratch`              | Combo                                  |    160 |
|                    | `fork+scratch_rot`          | Combo                                  |    160 |
| Other              | `Normal`                    | No defect                              |    160 |
|                    | `Invalid`                   | Invalid wafer (predicted via 5th head) |     40 |
| **Total**          |                             |                                        | **2200** |

`scratch + scratch_rot` is **excluded** (same defect family — labels are
ill-defined).

_Counts derived from `outputs/stage1_260505_162842/preds_chip.parquet`
n_eval=1760 over the four trained classes plus combos; Invalid handled
separately._

## Data synthesis pipeline (sister repo)

1. **Source distributions**: WM-811K wafer maps grouped by class.
2. **Sister repo `D:/project/known-cnn` `dist_apply/`** stamps fail-bits on a
   blank wafer using the per-class chip distributions, producing 200×200 chip
   images. For combo classes, two distributions are stamped on the same chip.
3. **Output tree**: `D:/project/data/wm-811k/unknown/<class>/*.png`.
4. The chip CNN backbone (`chip5_round4_v14_…`) is a TAPT variant: ImageNet
   FCMAE→supervised on the same synthetic chips.

## Backbone under test (T0)

- ckpt: `D:/project/known-cnn/outputs/logs_chip/chip5_round4_v14_260505_061558_running/best_model.pth`
- backbone: `convnextv2_base.fcmae_ft_in22k_in1k_384`
- img_size: **384** (chips are 200×200 → upsampled)
- training: **single-label CE**, 5 classes (`bank_boundary`, `fork`,
  `invalid_main`, `scratch`, `scratch_rot`)
- val_macro_f1 on 5-class single-label val set: **1.000** at epoch 1
- training compute: trivial (327 train / 82 val chips)

This is a strong but narrow model: it can recognise each defect on its own
but has never been asked to assert two defects in one chip.

## Evaluation harness

- File: `chip_multilabel_eval.py` (skill: `chip-multilabel-pipeline`)
- Inputs: `(model checkpoint)` × `(inference variant)`
- Output: `outputs/stage{1,2}_<TS>/`
  - `results_matrix.parquet` — one row per (train_id, inference_id) cell
  - `per_class_metrics.parquet`, `confusion_11class.parquet`,
    `errors.parquet`, `errors/<cell>/<error_type>/*.png`
  - `eval_summary.json`, `report.md`, `thresholds.json`

## Evaluation metrics (per cell)

- **`macro_f1`** — mean of per-class F1 over the 4 defect classes (combos
  decompose into multi-hot ground truth). Primary headline.
- **`micro_f1`**, **`mAP`**, **`hamming_loss`**, **`subset_accuracy`** — standard multi-label aggregates.
- **`top1_11class`** — argmax-decoded 11-class accuracy (combo recovery).
- **`ece_pre` / `ece_post`** — calibration before/after temperature.

Unless stated otherwise, “best cell” = highest `macro_f1`.

## Hard rules (constant across iters)

- **TTA permanently disallowed.** Rotation augmentation collapses
  `scratch` and `scratch_rot` (one ablation: −0.018 macro-F1). Never
  re-enabled even where it might help, because the rotation invariance
  destroys a class boundary.
- **GPU = 1 job at a time.** Sweeps run strictly sequentially.
- **`scratch + scratch_rot` combo excluded** from the 11-class set.

## Sister repos

- `D:/project/known-cnn` — supervised CNN (TAPT backbone supplier), data synthesis.
- `D:/project/mapviewer` — composite-map visualization (read-only).
