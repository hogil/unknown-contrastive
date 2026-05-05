# Abstract

We adapt a strong single-label chip CNN (ConvNeXtV2-Base 384, trained
single-label CE on 4 defect classes plus an `invalid_main` head) into a
multi-label predictor over an 11-class chip benchmark (4 single defects,
5 plausible defect combinations, `Normal`, `Invalid`; 2200 chips). On a
pure argmax baseline the model achieves macro-F1 = **0.7302**;
inference-time interventions alone — per-class F1-max thresholds, joint
coordinate-descent thresholds, and an entropy-based `Normal` gate — lift
the unchanged backbone to **0.8542**. A single training intervention
(CE with label smoothing α=0.10) further reaches **0.8634**. A targeted
sweep of label-smoothing strength surfaces a sharp peak at **α=0.20**,
yielding macro-F1 = **0.9268** and top-1 11-class accuracy =
**0.8449** — a **+0.1966** absolute macro-F1 over the argmax baseline,
with no TTA and ≈6 min of fine-tuning. We attribute the gain to a
re-shaped softmax that exposes runner-up logits useful for sigmoid
threshold decoding, and document negative results (ASL, BCE, BCE→ASL,
TTA, min-floor thresholds) as ablations of equal value.
