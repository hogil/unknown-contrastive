# Iter 5 — Phase A1: T1 (CE+LS) label-smoothing sweep

**Run dirs**:
- Initial sweep (LS 0.05/0.10/0.15/0.20): `outputs/phase_a_260505_175105/`
- Extension (LS 0.25/0.30): `outputs/phase_a_260505_182044/`
- Per-LS train dirs:
  `outputs/logs_chip_multilabel/T1_LS{05,10,15,20,25,30}_LR04_ep8_<TS>/`

**Date**: 2026-05-05 17:51 – 18:30

## Goal

Iter 4 surfaced T1 (CE + LS=0.10) as the only training intervention that
helped. Sweep the label-smoothing strength α to see whether 0.10 was the
sweet spot or whether stronger smoothing helps further.

## Sweep design (Phase A1)

- **α (LS)** ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
- **LR** = 1e-4 (held)
- **Epochs** = 8 (held)
- **Inference** = I3 / I7 / I10 evaluated for each train run (3 cells per α).
- 6 trains × 3 inferences = **18 cells**.
- Strictly sequential (GPU = 1 job at a time).

## Sweep log (sorted by macro_f1)

| LS    | inference | macro_f1 | top1_11 | T      | train_path                                                          |
|------:|-----------|---------:|--------:|-------:|---------------------------------------------------------------------|
| 0.20  | **I7**    | **0.9268** | **0.8449** | 1.0000 | outputs/logs_chip_multilabel/T1_LS20_LR04_ep8_260505_181242 |
| 0.20  | I3        |   0.9239 |  0.8324 | 1.0000 | outputs/logs_chip_multilabel/T1_LS20_LR04_ep8_260505_181242 |
| 0.15  | I3        |   0.8961 |  0.7517 | 1.0000 | outputs/logs_chip_multilabel/T1_LS15_LR04_ep8_260505_180531 |
| 0.15  | I7        |   0.8959 |  0.7517 | 1.0000 | outputs/logs_chip_multilabel/T1_LS15_LR04_ep8_260505_180531 |
| 0.15  | I10       |   0.8900 |  0.7449 | 1.0000 | outputs/logs_chip_multilabel/T1_LS15_LR04_ep8_260505_180531 |
| 0.20  | I10       |   0.8841 |  0.8136 | 1.0000 | outputs/logs_chip_multilabel/T1_LS20_LR04_ep8_260505_181242 |
| 0.25  | I3        |   0.8663 |  0.6989 | 1.0000 | outputs/logs_chip_multilabel/T1_LS25_LR04_ep8_260505_182048 |
| 0.25  | I7        |   0.8647 |  0.6903 | 1.0000 | outputs/logs_chip_multilabel/T1_LS25_LR04_ep8_260505_182048 |
| 0.25  | I10       |   0.8398 |  0.6824 | 1.0000 | outputs/logs_chip_multilabel/T1_LS25_LR04_ep8_260505_182048 |
| 0.10  | I3        |   0.8363 |  0.6261 | 1.0000 | outputs/logs_chip_multilabel/T1_LS10_LR04_ep8_260505_175823 |
| 0.10  | I10       |   0.8317 |  0.5631 | 1.0000 | outputs/logs_chip_multilabel/T1_LS10_LR04_ep8_260505_175823 |
| 0.10  | I7        |   0.8220 |  0.4767 | 1.0000 | outputs/logs_chip_multilabel/T1_LS10_LR04_ep8_260505_175823 |
| 0.30  | I3        |   0.8185 |  0.6466 | 1.0000 | outputs/logs_chip_multilabel/T1_LS30_LR04_ep8_260505_182757 |
| 0.30  | I7        |   0.8048 |  0.6210 | 1.0000 | outputs/logs_chip_multilabel/T1_LS30_LR04_ep8_260505_182757 |
| 0.05  | I7        |   0.7964 |  0.5591 | 1.0000 | outputs/logs_chip_multilabel/T1_LS05_LR04_ep8_260505_175109 |
| 0.05  | I10       |   0.7941 |  0.5574 | 1.0000 | outputs/logs_chip_multilabel/T1_LS05_LR04_ep8_260505_175109 |
| 0.05  | I3        |   0.7899 |  0.5534 | 1.0000 | outputs/logs_chip_multilabel/T1_LS05_LR04_ep8_260505_175109 |
| 0.30  | I10       |   0.7680 |  0.6051 | 1.0000 | outputs/logs_chip_multilabel/T1_LS30_LR04_ep8_260505_182757 |

_Source: outputs/phase_a_260505_175105/sweep_log.csv (rows 1–12),
outputs/phase_a_260505_182044/sweep_log.csv (rows 13–18)._

## Winner

**T1_LS20__I7** — α=0.20, LR=1e-4, epochs=8, inference=I7.
- macro_f1 = **0.9268**
- top1_11  = **0.8449**

```json
{"ls": 0.2, "lr": 0.0001, "epochs": 8, "inference_id": "I7",
 "macro_f1": 0.9268, "top1_11": 0.8449,
 "train_path": "outputs/logs_chip_multilabel/T1_LS20_LR04_ep8_260505_181242"}
```

_Source: outputs/phase_a_260505_175105/best_config.json._

Δ vs iter 4 winner (T1__I10, 0.8634 / 0.7006):
**+0.0634 macro-F1**, **+0.1443 top1\_11**.

Δ vs argmax baseline (T0__I0, 0.7302 / 0.4472):
**+0.1966 macro-F1**, **+0.3977 top1\_11**.

## LS curve

```
LS    | best_macro_f1 (across I3/I7/I10)
0.05  | 0.7964
0.10  | 0.8363
0.15  | 0.8961
0.20  | 0.9268   ← peak
0.25  | 0.8663
0.30  | 0.8185
```

Sharply non-monotonic peak at α=0.20. ±0.05 around the peak loses
~0.03–0.06 macro-F1.

## Inference variant per LS — surprising flip

For **frozen** and **mildly-trained** models, **I10 ≥ I7** consistently
(iter 3, iter 4). For **strongly-trained** models (LS=0.20), **I7 > I10**:

| LS    | I3       | I7       | I10      | best   |
|------:|---------:|---------:|---------:|--------|
|  0.05 |   0.7899 |   0.7964 |   0.7941 | I7     |
|  0.10 |   0.8363 |   0.8220 |   0.8317 | I3     |
|  0.15 |   0.8961 |   0.8959 |   0.8900 | I3     |
|  0.20 |   0.9239 | **0.9268** |   0.8841 | I7     |
|  0.25 |   0.8663 |   0.8647 |   0.8398 | I3     |
|  0.30 |   0.8185 |   0.8048 |   0.7680 | I3     |

The entropy Normal gate (I10) **starts to hurt** once the model has been
trained well. Once LS=0.20 has spread the softmax mass, the chip's
binary entropy is *naturally* in the "uncertain" band even for chips
with a real defect, so the entropy gate over-fires Normal. I7 (no
explicit Normal decoding) wins.

## Train summaries

| variant       | best_val_acc | best_epoch | epochs | elapsed_sec |
|---------------|-------------:|-----------:|-------:|------------:|
| T1_LS05       |       0.9756 |          1 |      8 |       342.7 |
| T1_LS10       |       1.0000 |          1 |      8 |       340.1 |
| T1_LS15       |       0.9756 |          6 |      8 |       341.7 |
| T1_LS20       |       0.9756 |          1 |      8 |       340.3 |
| T1_LS25       |       1.0000 |          1 |      8 |       340.6 |
| T1_LS30       |       1.0000 |          3 |      8 |       335.9 |

_Source: outputs/logs_chip_multilabel/T1_LS*_LR04_ep8_*/train_summary.json._

Note: best_val_acc is the **single-label** 5-class val accuracy and
**does not predict** multi-label macro-F1. T1_LS25 hits val 1.0 but
multi-label 0.8663; T1_LS20 only hits val 0.9756 but multi-label 0.9268.
Single-label val is a poor selector for the multi-label task.

## Decision for next iter

- Phase A2: hold LS=0.20, sweep LR ∈ {5e-5, 1e-4, 2e-4} at ep=8.
- Phase A3: hold (LS, LR) winner, sweep epochs ∈ {6, 8, 10, 12}.
- Phase B onwards: re-visit T4 (ASL), T5 (BCE), T6 (BCE→ASL) hyperparameters
  to see if any of those losses can match or beat T1_LS20 with proper
  tuning.

## Files

- `outputs/phase_a_260505_175105/` — initial sweep + report.md + best_config.json
- `outputs/phase_a_260505_182044/` — extension (LS=0.25/0.30); sweep_log.csv only
- Per-train: `outputs/logs_chip_multilabel/T1_LS{05,10,15,20,25,30}_LR04_ep8_<TS>/`
