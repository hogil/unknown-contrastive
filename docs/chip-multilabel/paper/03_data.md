# 3. Data

## 3.1 Chip image format

A *chip* is a 200×200 image cropped from a wafer fail-bit map. Pixels
are encoded by a fixed palette where grade 0 = white (no failure),
grade 1 = grey (mild failure), grades 2 and 3 = saturated colours
(severe). In the current iteration the source distribution is
heavily concentrated at grades 0 and 1; iterations introducing grade
2/3 elevation are queued for follow-up (§9).

## 3.2 Data synthesis pipeline

All training and evaluation data is *synthesised* in the sister repo
`D:/project/known-cnn`, file `_sample_gen.py`. The pipeline uses
WM-811K wafer maps grouped by class as the source distribution and
then composes 200×200 chips using one of three rules:

1. **Single-defect chips** (TRAIN_CLASSES): for each of
   `bank_boundary`, `fork`, `scratch`, `scratch_rot`, sample a
   class-conditional chip-coordinate distribution and stamp the
   corresponding fail-bit pattern onto a blank chip.
2. **Combo chips** (5 entries): `min`-blend two single-defect chips
   from distinct TRAIN_CLASSES (excluding `scratch+scratch_rot`,
   which is ill-defined as the rotation makes the two defects
   pixel-overlapping on the rotated stamp).
3. **`Normal` chips** (`_make_normal_chip`, sister-repo
   `_sample_gen.py:151`): sample a `BASELINE` background — low-grade
   speckle that mimics no-defect wafer noise.
4. **`Invalid` chips** (sister-repo `_sample_gen.py:915`): stamp the
   QC orange border (RGB ≈ (240, 160, 0)) on a near-white chip; the
   inference-side detector for these is a deterministic colour
   heuristic (`chip_multilabel/decision_tree.py:36`, `detect_invalid`),
   not a learned head.

## 3.3 Train / val / eval splits

| split | n chips | source                                                        |
|-------|--------:|---------------------------------------------------------------|
| train |     327 | sister repo `classification_chips/`, single-label, 5 classes  |
| val   |      82 | same source, 4:1 split with train                             |
| eval  |    2200 | `D:/project/data/wm-811k/chip_multilabel_eval_full/`          |

The **training data is single-label**: each training chip has exactly
one of the 4 defect classes (or `invalid_main`) as its ground truth,
and there are no `Normal` chips in train. The val set is also
single-label and is used purely for threshold tuning and temperature
scaling (no model selection beyond loss curves).

The **eval set is multi-label** by construction. It contains 11
logical classes:

| Group            | Class                       | n eval |
|------------------|-----------------------------|-------:|
| Single defect    | `bank_boundary`             |    240 |
|                  | `fork`                      |    240 |
|                  | `scratch`                   |    160 |
|                  | `scratch_rot`               |    160 |
| Combo (2 defect) | `bank_boundary+fork`        |    160 |
|                  | `bank_boundary+scratch`     |    160 |
|                  | `bank_boundary+scratch_rot` |    160 |
|                  | `fork+scratch`              |    160 |
|                  | `fork+scratch_rot`          |    160 |
| Other            | `Normal`                    |    160 |
|                  | `Invalid`                   |     40 |
| **Total**        |                             | **2200** |

Combo classes are encoded as multi-hot labels (e.g. `bank_boundary+fork`
sets both `bank_boundary` and `fork` bits to 1). `Normal` has all
defect bits 0. `Invalid` is a special class whose ground truth is
established by the QC border heuristic (chip excluded from the
defect-class bitmap entirely).

The combo `scratch + scratch_rot` is **excluded from the eval set** —
the same rotation invariance that makes `scratch` and `scratch_rot`
distinguishable (when present alone) makes the combo ill-defined: a
rotated scratch stamped on a non-rotated scratch overlaps pixel-wise.

## 3.4 Sanity checks (sister repo)

The synthesis pipeline runs three sanity checks before publishing
chips:

- **Per-class fail-bit density.** Each chip's grade-1+ pixel ratio
  must lie in the per-class histogram window measured from real
  WM-811K samples.
- **Combo orthogonality.** For a 2-defect combo, each contributing
  defect's pixel set must overlap by at most 30% with the other —
  we are simulating co-occurrence, not duplication.
- **Border purity.** `Invalid` chips must satisfy the
  `detect_invalid` heuristic (white-area ratio ≥0.95 + ≥3 of 4
  borders containing orange pixels within tolerance).

## 3.5 Backbone (T0)

The reference checkpoint under test, henceforth **T0**, is

```
D:/project/known-cnn/outputs/logs_chip/chip5_round4_v14_260505_061558_running/best_model.pth
```

a `convnextv2_base.fcmae_ft_in22k_in1k_384` initialised from
ImageNet FCMAE pretrain → ImageNet supervised → TAPT (task-aligned
pretraining) on the same synthetic chip distribution → final
single-label CE on 5 classes. Val 5-class accuracy is 1.0000 at
epoch 1; we view the multi-label benchmark as the *only*
discriminative test of the model and treat val accuracy as a
hyperparameter-selection signal rather than a quality signal
(§6 documents that single-label val accuracy is a poor predictor
of multi-label macro-F1: T1_LS25 hits val 1.0 but only 0.8663
multi-label, while T1_LS20 hits val 0.9756 and 0.9268 multi-label).

**Why TAPT instead of pure ImageNet?** The chip distribution is far
from natural images; pretrain on the same synthetic distribution
gives the backbone several percent of multi-label headroom on the
eval set. We retain TAPT throughout this paper and treat it as part
of T0. Re-pretraining experiments are deferred.

## 3.6 Limitations of the synthesis pipeline

The synthesis pipeline has two known limitations that bound the
upper macro-F1 we can achieve:

1. **Combo difficulty.** `min`-blend produces combo chips whose
   per-class fail-bit pattern is *weaker* than the source single
   chips (because `min` zeroes overlapping cells). Phase B+ work
   plans a `--source-strength-pct` filter to use only top-strength
   source chips when blending, which we hypothesise will lift the
   combo-class macro-F1 by up to 0.03.
2. **Grade variation.** Source chips are concentrated at grades 0–1.
   Generating chips with elevated grade-2/3 pixel populations
   (`--grade-mode {default, elevated_2, elevated_3}`) is queued; we
   expect this to test scratch vs scratch_rot under saturated
   colour conditions, where the two are visually most distinct.

These two are deferred until Phase A (this paper) is closed.
