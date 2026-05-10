# Introduction

> Target venue: IEEE Transactions on Semiconductor Manufacturing.

## 1.1 Motivation

Wafer defect classification is a yield-critical step in modern semiconductor
manufacturing. Each fab generates millions of wafer maps per month, and the
ability to triage failure patterns automatically — rather than via human
classification engineers — directly affects two operational KPIs: (i) yield
recovery time (mean time between a process excursion and its root-cause
identification) and (ii) inspection throughput (wafers reviewed per
engineer-hour). State-of-the-art supervised CNN classifiers, including the
ConvNeXtV2-base baseline used as a backbone in this study (sister repo
`known-cnn`, validation F1 0.9946 on 33-class wafer maps), have reached the
known-class regime where additional gains are saturated.

The remaining bottleneck is the **unknown-defect regime**. Production wafers
periodically present failure modes that are not in the supervised label
inventory: novel rotation sub-styles of a known scratch, sub-style splits
inside an existing class (e.g., `Full_*` bimodality), or genuinely new
process-tool signatures. A supervised classifier responds to these with
high-confidence misclassification or low-confidence rejection, neither of
which gives the engineer a corrective handle. The operational requirement is
**defect grouping that survives the closed label set**: every wafer that
deviates from Normal must end up in some interpretable cluster, and no class
of true defects must collapse into a Normal-flagged or noise bucket.

This work targets that requirement with a **self-supervised contrastive
encoder + density-based clustering** pipeline that has been hardened over 58
ablation iterations against four primary objectives (P1-P4):

- **P1 — class capture**: every defect class produces at least one HDBSCAN
  cluster (no defect class swallowed by Normal or HDBSCAN noise).
- **P2 — defect noise rate**: HDBSCAN noise fraction restricted to the
  defect subpopulation must be small (sub-1% target).
- **P3 — Completeness** (Rosenberg-Hirschberg 2007): single ground-truth
  defect class is concentrated in a single cluster.
- **P4 — Homogeneity / AMI**: each cluster contains a single defect class.

P1 is hard-locked: any iteration that reduces capture below 1.000 is
rejected regardless of P2-P4 gains (iter 36, iter 41 both rejected on this
basis).

## 1.2 Limitations of supervised CNN baselines

The sister-repo supervised classifier reaches val_f1 0.9946 on the 33-class
wafer map task. Three structural limits motivate the contrastive replacement:

1. **Closed label set**: the classifier outputs probabilities over a fixed
   33-vocabulary. New defect modes either get absorbed into the nearest
   visual neighbor (silent error) or pushed below a softmax threshold
   (low-confidence reject without grouping).
2. **Sub-style invisibility**: classes such as `Full_bank_boundary`,
   `Full_scratch_rot`, `Full_fork`, and `Thick-Edge_fork` carry a true
   bimodal sub-style inside the synthesis itself (intra/inter cluster
   distance ratio 2-9x, GMM BIC bimodal — see RESULTS table 4). A
   classifier collapses these into a single softmax row by construction.
3. **Rotation as identity**: the `scratch` versus `scratch_rot` pair
   differs only by a 21-degree rotation. A classifier with rotation
   augmentation enabled would erase this distinction. This is the same
   reason rotation-invariant contrastive methods (e.g., DECOR, AAAI 2026,
   arXiv:2510.03328) cannot be directly adopted in our domain.

## 1.3 Why contrastive + HDBSCAN

Self-supervised contrastive learning produces an embedding whose geometry
is dictated by visual similarity rather than by an a-priori label
vocabulary. HDBSCAN then derives an open-cardinality partition over that
embedding, so new defect modes appear as new clusters rather than as
classifier confusions. Three design choices make this combination practical
for the wafer-defect setting:

- **InfoNCE with momentum queue + dense local grid contrast (DenseCL-style)
  + patch-neighbor consistency (NeCo)**. Three contrastive components feed
  the same projection head, with weights tuned by atomic ablation
  (LOCAL_WEIGHT, NECO_WEIGHT — see METHOD section 3).
- **Frozen TAPT backbone**. The ConvNeXtV2-base FCMAE encoder is
  task-adaptively pre-trained as a 33-class supervised classifier on the
  same wafer dataset (sister repo) and then frozen during contrastive
  head training. Joint fine-tuning of the last backbone stage was tested
  twice (iter 36, iter 42) and rejected: with only 2,146 anchor wafers,
  partial unfreeze induces supervised-collapse-style overfit.
- **HDBSCAN with `eom` selection method and `min_samples=3`**. The
  HDBSCAN configuration is treated as a separate axis from encoder
  ablation. Within the same iter-37 embedding, switching from `leaf` to
  `eom` (with `mcs=12, ms=4`) reduces defect-noise from 12.6% to 4.28%
  (-66%); further switching `ms=4` to `ms=3` reduces it to 2.79% (-50%
  more). Encoder-side improvements (5 levers) and HDBSCAN-side
  improvements (3 axes) are reported separately to keep ablation
  attribution clean.

## 1.4 Contributions

This paper contributes the following five items:

**(C1) A 43-class open-set wafer benchmark synthesized from WM-811K**.
The benchmark uses 8 base WM-811K distribution classes as priors, applies
five chip-internal object types (`bank_boundary`, `fork`, `scratch`,
`scratch_rot`, `invalid_main`), and adds 9 wafer-canvas patterns
(BrokenRing, CenterDonut, CrescentArc, CrossScratch, DiagonalSmear,
ParallelScratches, RingDots, Row, Starburst). The result is a 42 defect +
1 Normal anchor of 2,146 wafers (`avg30_new_260508_123037`). Synthesis
sub-style splits (Full_* bimodality, sister-class rotation pairs) are
documented as deliberate design rather than encoder weakness, with
GMM-BIC and intra/inter ratio evidence.

**(C2) Five atomic contrastive levers and nine dead axes — a
practitioner's ablation map**. Through 49 iterations on a single fixed
data anchor, we identify five levers that each produce a Tier 1 metric
movement larger than the iter-to-iter standard deviation:
LOCAL_WEIGHT (0.5 to 1.0, -50% noise), LR_HEAD (1e-3 to 5e-4 in old
anchor), IGNORE_NEG_SIM (0.72 to 0.65 conditional on base cfg),
NCE_TEMP (0.07 to 0.05 in old anchor / 0.07 in new anchor), and
NECO_WEIGHT (0 to 0.2, -70% noise). We further enumerate nine axes that
were swept and rejected — partial backbone unfreeze (permanent reject),
EPOCHS, WARMUP, LOCAL_POS_TOPK, QUEUE_SIZE, BATCH, NV-Retriever
PercPos alpha, HDBSCAN forcing at capture cost, and Quality-King +
NeCo cross-cfg combination. RESULTS table 10 lists each.

**(C3) NeCo's mechanism reinterpreted as Normal-defect boundary
repulsion**. The original NeCo paper (Pariza et al. 2024,
arXiv:2408.11054) frames patch-neighbor consistency as a generic
compactness loss for spatial representations. On wafer data we find a
sharper mechanism via cluster-analyzer post-hoc inspection: NeCo at
weight 0.2 displaces 54 wafers out of a Normal supercluster and pushes
each defect-class centroid 0.05 to 0.07 cosine units further from the
Normal centroid, while leaving inter-defect-class distances roughly
unchanged. This is a directional signal (Normal versus defect),
not a uniform compactness boost. We make this distinction explicit in
DISCUSSION because it changes how a practitioner should expect NeCo
to behave on a domain whose Normal class is well-defined and dense.

**(C4) State-of-the-art Tier 1 numbers under a strict P1-P4 policy**.
On the new-anchor track (iter 37, NECO_WEIGHT=0.2, P2-King base
configuration), the system achieves Completeness 0.991, AMI 0.960,
ARI 0.870, defect-noise 0.61%, and 43/43 class capture. Multi-seed
verification (3 seeds, iter 44-46) gives ARI 0.866 plus or minus
0.014 — a single-seed reading of 0.880 was lucky and is not the
mean. We report the multi-seed result as the headline and discuss
the single-seed hazard in DISCUSSION. A second axis
(LOCAL_POS_TOPK=16, iter 54/55) reproduces the same +0.010 lucky
variance pattern (seed=42 ARI 0.880, seed=1 ARI 0.852, mean 0.866 —
identical to Zone-Aware NeCo z=4 to the third decimal place),
strengthening the multi-seed protocol claim.

**(C5) A comprehensive multi-axis saturation point for the iter-37
configuration**. Iterations 50-58 sweep six hyperparameter axes
(LOCAL_WEIGHT 1.2, LR_HEAD 7e-4, IGNORE_NEG_SIM 0.65, NCE_TEMP 0.06,
LOCAL_POS_TOPK 16, QUEUE_SIZE 8192) plus two Spatial-NeCo variants
(Hierarchical NeCo at "1,2,4" pools, Zone-Aware NeCo at z=3) on top
of iter 37. Every atomic step lands inside the iter-37 multi-seed
standard deviation (0.014 ARI). The iter-37 configuration is therefore
a robust sweet spot that holds within plus or minus 20% on every
hyperparameter axis. We publish this saturation map as the
practitioner's "do not bother sweeping these" guide and as the
empirical justification for the Cluster-Aware Synthesis Loop being
the natural next paradigm rather than another encoder-side sweep.

## 1.5 Paper outline

Section 2 (RELATED_WORK) compares to recent self-supervised
contrastive, dense / patch-consistency, and wafer-defect-clustering
literature, with explicit differentiation against rotation-invariant
methods. Section 3 (METHOD) describes data synthesis, the contrastive
loss composition, NeCo integration, frozen-backbone TAPT, and HDBSCAN
configuration. Section 4 (DATASET) lists the 43-class composition.
Section 5 (EXPERIMENTS) lists hyperparameter rows for all reported
iterations. Section 6 (RESULTS) gives Tier 1+2 metric tables. Section 7
(DISCUSSION) covers multi-seed importance, NeCo mechanism, the
zone-aware NeCo negative result, HDBSCAN configuration sensitivity,
and a Cluster-Aware Synthesis Loop proposal. Section 8 (CONCLUSION)
summarizes contributions and future work.

> Iteration history is in `ITERATIONS.md` (append-only). Decision history
> for every accepted and rejected option is in `docs/contrastive-eval/DECISIONS.md`.
