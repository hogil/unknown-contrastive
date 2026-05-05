# Abstract

> 250 단어 목표. 매 milestone 시 paper-recorder agent 가 갱신.

## v0.1 (2026-05-05)

We present a self-supervised contrastive learning pipeline for **unknown wafer defect
discovery via density-based clustering**. The system trains a 128-dim projection head
on top of a frozen ConvNeXtV2-base FCMAE backbone using **InfoNCE with momentum queue**,
adapted from a sister repo's supervised classifier (TAPT — task-adaptive pre-training).
After training, embeddings are clustered with HDBSCAN, producing operational
defect groups for production review.

The dataset is synthesized from WM-811K wafer-map distributions, augmented with
chip-level defect patterns drawn from five object categories (`bank_boundary`, `fork`,
`scratch`, `scratch_rot`, `invalid_main`) plus 9 wafer-canvas patterns and a Normal
class. We report on a baseline run with **8,357 wafers across 39 ground-truth classes**
(38 defect + Normal).

Evaluation uses **Tier 1 official metrics only** (rejecting custom or classifier-style
scores): **Completeness** (Rosenberg-Hirschberg 2007), **AMI** (Vinh 2010),
**HDBSCAN noise rate**, and a class-fragmentation summary capturing (A) every defect
class is captured by ≥1 cluster, (B) per-class non-noise coverage, and (C) the
fraction of classes concentrated in a single cluster.

On the baseline run, the system achieves **Completeness 0.9466, AMI 0.9288**, with
only **0.71% of defect samples falling into HDBSCAN noise** and **all 38 defect
classes captured by at least one cluster**. **34 of 38 classes (89.5%) collapse to a
single cluster**; the four split classes (`Full_*` × 3 + `Thick-Edge_fork`) are shown
to reflect a true bimodal sub-style in the synthesis itself rather than encoder
weakness, validated by HDBSCAN parameter sweep, intra/inter cluster distance ratio
(2-9×), and Gaussian-mixture BIC analysis.

We outline planned improvements: **production-realistic class-size sampling**,
**Wang-Isola alignment+uniformity training-time monitoring**, and **hard negative
mining** (Robinson et al. 2021) on top of InfoNCE.

> 갱신 history: `ITERATIONS.md`. 거부한 옵션 (Multi-crop, SupCon 주력): `docs/contrastive-eval/DECISIONS.md`.
