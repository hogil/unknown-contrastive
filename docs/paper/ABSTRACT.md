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

## v0.2 (2026-05-09)

We present a self-supervised contrastive learning pipeline for **unknown wafer defect
discovery via density-based clustering**, integrating a 128-dim projection head over
a frozen ConvNeXtV2-base FCMAE backbone (TAPT from a sister supervised classifier),
**InfoNCE** with global + local-grid + queue contrast, and **NeCo patch-neighbor
consistency** (arXiv:2408.11054) as a fifth lever discovered through systematic ablation.

The dataset is synthesized from WM-811K wafer-map distributions and **5 chip-internal
object categories** plus **9 wafer-canvas patterns**, organized as a 43-class anchor
(42 defect classes + Normal_bank_boundary, 2,146 wafer total).

Evaluation strictly follows **Tier 1+2 official metrics only**: Completeness (Rosenberg-Hirschberg
2007), AMI (Vinh 2010), ARI, Silhouette, HDBSCAN noise rate (defect-only scope), and
class-fragmentation summary. Custom or classifier-style scores are rejected by policy.

On the new anchor SOTA (iter 37, 2026-05-09), we achieve **Completeness 0.991, AMI 0.960,
ARI 0.870, defect-noise 0.61%, all 43 classes captured at 100%**. The breakthrough is
attributable to **5 atomic levers**: (1) LOCAL_WEIGHT 0.5→1.0 (-50% noise), (2) LR_HEAD
1e-3→5e-4 (+12pp Comp), (3) IGNORE_NEG_SIM 0.72→0.65 (sister-class separation), (4)
TEMP 0.07→0.05 (+0.3pp AMI), and (5) **NECO_WEIGHT 0→0.2 (-70% noise, 2.01%→0.61%)**.
The fifth lever (NeCo) is base-cfg-dependent: it requires the P2-King base (LR=1e-3,
NEG=0.72, TEMP=0.07) and shows negative interaction on Quality King base.

We rule out partial backbone unfreeze as a viable axis (permanent reject, iter 36/42).
Future code-level changes target **Zone-Aware NeCo** (iter 43, in progress), which
restricts patch-neighbor consistency to vertical zones to better separate Edge-Top vs
Edge-Bottom sub-styles.

> 갱신 history: `ITERATIONS.md`. 거부한 옵션 (backbone unfreeze, Multi-crop, SupCon, HDBSCAN forcing
> w/ capture cost): `docs/contrastive-eval/DECISIONS.md`.

## v0.3 (2026-05-10) — IEEE Transactions on Semiconductor Manufacturing (submit ready)

We present a self-supervised contrastive learning + density-based clustering pipeline
for **open-set wafer defect grouping** that is hardened over **58 atomic-change
ablation iterations** against four primary objectives: P1 class capture, P2 defect
noise rate, P3 Completeness, P4 Homogeneity / AMI. The system uses a frozen
ConvNeXtV2-base FCMAE encoder with task-adaptive pre-training (TAPT from a sister
supervised classifier), a 128-dimension projection head trained with InfoNCE plus
DenseCL-style local grid contrast plus NeCo patch-neighbor consistency, and HDBSCAN
clustering with `eom` selection and `min_samples=3`.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect classes plus
Normal, 2,146 wafers, with 5 chip-internal object types and 9 wafer-canvas patterns),
the iter-37 configuration achieves **single-seed ARI 0.870, 3-seed mean ARI 0.866 +-
0.014, Completeness 0.991, AMI 0.960, defect-noise 0.61%, and 43/43 class capture**.

We make five contributions. **(N1)** A 43-class open-set wafer benchmark with
documented sub-style splits (Full_* bimodality, sister-class rotation pairs)
validated by GMM-BIC and intra/inter ratio evidence. **(N2)** Multi-seed honesty:
the single-seed ARI 0.870 is a high-tail draw, and the 3-seed mean 0.866 +- 0.014 is
the headline. We further document that two completely independent ablation axes
(Zone-Aware NeCo z=4 and LOCAL_POS_TOPK=16) reproduce the **same +0.010 lucky
variance pattern** (seed=42 ARI 0.880, seed=1 ARI 0.852), giving strong empirical
support for the multi-seed protocol. **(N3)** Five atomic encoder levers
(LOCAL_WEIGHT, LR_HEAD, IGNORE_NEG_SIM, NCE_TEMP, NECO_WEIGHT) plus three
HDBSCAN-side axes plus 14 dead axes — a full ablation map for practitioners.
**(N4)** A domain-specific reinterpretation of NeCo's mechanism: NeCo at weight 0.2
acts as **Normal-defect boundary repulsion** rather than uniform compactness
(per-class centroid shift +0.05 to +0.07 cosine units away from Normal, while
inter-defect centroid distances stay within +- 0.01). **(N5)** A
**comprehensive saturation point lock-in**: 6 hyperparameter axes (LW / LR / NEG /
TEMP / TOPK / QUEUE) plus two Spatial-NeCo variants (Hierarchical 1,2,4 pools and
Zone-Aware z=3) all sweep to within multi-seed std of iter 37, establishing iter 37
as a multi-axis sweet-spot saturation point. Further encoder-side gains require a
paradigm shift, which we sketch as a Cluster-Aware Synthesis Loop in future work.

> 갱신 history: `ITERATIONS.md` (append-only). Decision history: `docs/contrastive-eval/DECISIONS.md`.

## v0.4 (2026-05-11) — Real Baseline Component Isolation 추가 (★ N6 NEW)

We present a self-supervised contrastive learning + density-based clustering pipeline
for **open-set wafer defect grouping**, hardened over **65 atomic-change ablation
iterations** plus a **six-step Real Baseline component isolation** (B0 to B5) against
four primary objectives: P1 class capture, P2 defect noise rate, P3 Completeness, P4
Homogeneity / AMI. The system uses a frozen ConvNeXtV2-base FCMAE encoder with
task-adaptive pre-training (TAPT) from a sister supervised classifier, a 128-dimension
projection head trained with InfoNCE plus DenseCL-style local grid contrast plus NeCo
patch-neighbor consistency, and HDBSCAN clustering with `eom` selection and
`min_samples=3`.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect + Normal,
2,146 wafers, 5 chip-internal object types + 9 wafer-canvas patterns), the iter-37
configuration achieves **single-seed ARI 0.870, 3-seed mean ARI 0.866 +/- 0.014,
Completeness 0.991, AMI 0.960, defect-noise 0.61%, and 43/43 class capture**.

We make **six contributions**. **(N1)** A 43-class open-set wafer benchmark with
documented sub-style splits validated by GMM-BIC and intra/inter ratio evidence.
**(N2)** Multi-seed honesty: the single-seed ARI 0.870 is a high-tail draw, and the
3-seed mean 0.866 +/- 0.014 is the headline; **a Real Baseline B5 reproduce of iter-37
(same seed=42, same cfg) lands at ARI 0.856, exactly one std away** — strong evidence
that same-seed run-to-run variance is non-negligible. Two independent ablation axes
(Zone-Aware NeCo z=4 and LOCAL_POS_TOPK=16) further reproduce the same +0.010 lucky
variance pattern. **(N3)** Five atomic encoder levers (LOCAL_WEIGHT, LR_HEAD,
IGNORE_NEG_SIM, NCE_TEMP, NECO_WEIGHT) plus three HDBSCAN-side axes plus 14 dead axes
— a full ablation map for practitioners. HDBSCAN-side tuning alone (`leaf` to `eom` +
`ms=4` to `ms=3`) reduces defect noise from 12.6% to 0.61% at fixed encoder (-91%).
**(N4)** A domain-specific reinterpretation of NeCo's mechanism: NeCo at weight 0.2
acts as **Normal-defect boundary repulsion** (per-class centroid shifts +0.05 to +0.07
cosine away from Normal). **(N5)** A **comprehensive saturation point**: 6
hyperparameter axes + 2 Spatial-NeCo variants all sweep to within multi-seed std of
iter 37, establishing iter 37 as a multi-axis sweet-spot. **(N6 NEW) Component
Interaction Matters**: A six-step Real Baseline ablation (B0 Global-only to B5 iter-37
cfg) reveals that **LW lever's isolated effect is negative** (B1 to B2: ARI -0.028)
and its real contribution **only materializes through Queue interaction** (B2 to B3:
ARI +0.023). NeCo's isolated effect (B4 to B5) is **-0.004 ARI**, within run-to-run
variance; B4 (no NeCo) is in fact the best single-step Real Baseline configuration.
This sharpens the contribution map: the dominant work is done by the **TAPT backbone**
(B0 ARI 0.823 = 94.6% of iter-37 ARI), and the contrastive head + HDBSCAN tuning
together is the 5%-of-ARI polish on top. The Real Baseline B0 path makes the
per-component contribution explicit and exposes a Component Interaction lesson that
isolated lever-ablation tables miss.

The Real Baseline B0 to B5 matrix is published as the per-component evidence backing
all five lever claims (RESULTS table 13). Further encoder-side gains require a
paradigm shift, which we sketch as a **Cluster-Aware Synthesis Loop** in future work.

> 갱신 history: `ITERATIONS.md` (append-only, iter 0 through iter 65). Decision history:
> `docs/contrastive-eval/DECISIONS.md`. Component isolation: `RESULTS.md` table 13,
> `ABLATION_PLAN.md`, `DISCUSSION.md` §7.9.
