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

This paper contributes the following seven items:

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

**(C6) Component Interaction matters — a Real Baseline isolation
map (NEW, 2026-05-11)**. A six-step Real Baseline ablation (B0 to B5)
isolates each contrastive component from a minimal Global-InfoNCE-only
baseline. LW lever's isolated atomic step (B1 LW=0.5 to B2 LW=1.0)
is **negative** (ARI -0.028, noise +2.27pp); its real contribution
**only materializes through Queue interaction** (B2 to B3: ARI +0.023,
noise -4.89pp). NeCo's isolated effect (B4 to B5) is **-0.004 ARI**,
within same-seed run-to-run variance; B4 (no NeCo) is in fact the
best single-step Real Baseline configuration. The B0 Global-only
baseline already reaches ARI 0.8231, capture 1.000, indicating that
the TAPT ConvNeXtV2 backbone alone delivers 94.6% of the iter-37 ARI.
This Component Interaction lesson sharpens the contribution map and
exposes a hazard of lever-isolated atomic ablation on top of a
non-minimal baseline (which the original Iter A0 baseline was). The
full B0-to-B5 matrix with Tier 1+2 metrics is published as RESULTS
table 13.

**(C7) Component Dependency Hierarchy and single-cfg recommendation
(NEW, 2026-05-12; ★ N1 v7 FINAL revised 2026-05-12 — supersedes v6 dual-cfg)**.

★ **v7 FINAL CORRECTION (iter 84, 2026-05-12)**: The v6 sub-claim "B5 (Local + NeCo
combined) is the absolute SOTA at known-K Agglomerative Ward K=42 with ARI 0.9358"
is retracted on multi-seed reproducibility grounds. B5 seed=1 (same cfg, same
protocol) gave Agglo K=42 ARI **0.8482** (Δ −0.0876 from seed=42 0.9358). B5
2-seed avg ARI = **0.8920 ± 0.062**, **below** NEW 3-seed avg **0.9014 ± 0.022**
(Δ −0.0094, std 2.8× higher). Across all three benchmarked clustering methods,
NEW (NeCo only, no Local) > B5 (Local + NeCo) on multi-seed average:
HDBSCAN +0.0245, Agglo Ward K=42 +0.0094, KMeans K=42 +0.0138. **The v6 dual-cfg
recipe (B5 for known-K, NEW for unknown-K) collapses to a single-cfg
recommendation: NEW for both clustering frontier targets**. Local DenseCL is
**operationally optional, not required for SOTA**. Per-class purity flips at
single-seed Agglo K=42 (RESULTS §16) are preserved as observations but DO NOT
propagate to multi-seed averages — the v6 "complementary inductive bias" framing
applies only to single-seed observations. The reproducibility evidence (B5
seed=42 → seed=1 Δ ARI −0.088) is the **largest cross-seed flip in this paper's
84-iteration cycle** and the strongest paper-grade evidence for N2 (multi-seed
methodology obligation). RESULTS §17 publishes the full B5 seed=1 measurement
and multi-seed avg comparison.

The v6 content below (substitutability-on-aggregate, complementary-on-per-class)
remains a valid characterization of the **single-seed** behavior, useful for
mechanism interpretation, but is **not the recipe for SOTA**.

An eleven-iteration
four-component lattice exploration (iter 67-77, Local x Queue x NEG x NeCo)
plus a per-class K=42 Agglomerative Ward purity breakdown maps each
contrastive component's dependency on every other component, and produces a
**dual-cfg recipe** rather than a single SOTA. Five key findings:
(i) **Aggregate HDBSCAN equality of NeCo (Pariza et al. 2024) and DenseCL
Local InfoNCE (Wang et al. 2021)** — iter 69 (Global + NeCo only) gives ARI
0.8514 = iter B1 (Global + Local LW=0.5 only) ARI 0.8514 to four decimal
places, with identical defect-noise (3.93%) and identical cluster count (37).
The v5 framing "they substitute each other on partitioning" applies to
HDBSCAN aggregate ARI / noise / n_cl scope.
(ii) **★ N1 v6 NEW — Local DenseCL and NeCo are COMPLEMENTARY at per-class
scope, not substitutable**. Under Agglomerative Ward K=42 (defect-only,
oracle K=K_gt=42, RESULTS §16), per-GT-class dominant cluster purity reveals
class-by-class winner flips that aggregate ARI hides. Local DenseCL strength
is sub-pattern variant integration: `Edge-Ring_fork` (n=31) B5 100% vs NEW
64.5% (Δ −35.5pp), `Center_scratch` (n=40) 95% vs 75% (−20pp), `Donut_fork`
(n=37) 100% vs 81.1% (−18.9pp), `Edge-Top_scratch` (n=19) 100% vs 84.2%
(−15.8pp). NeCo strength is uniform-pattern consolidation: `CenterCircle`
(n=42) NEW 100% vs B5 54.8% (+45.2pp), `Edge-Top_fork` (n=20) 100% vs 90%
(+10pp). Net average per-class purity (single-seed=42): B5 = 97.0%, NEW = 96.2%,
Δ −0.83pp. The single-seed=42 max ARI is **B5 (Local + NeCo combined) Agglomerative
Ward K=42 = 0.9358**, above iter 70 NEW 0.9200 (Δ +0.0158) — ★ **v7 retracted as
multi-seed SOTA**: seed=1 reproduce = 0.8482 (Δ −0.0876), B5 2-seed avg 0.8920 ±
0.062 BELOW NEW 3-seed avg 0.9014 ± 0.022 (std 2.8× higher). The v5 "substitutable"
framing is refined in v6 to **single-seed complementary** (per-class purity flips
preserved as single-seed observation only); the v7 multi-seed correction collapses
to **NEW alone covers both frontiers** — Local DenseCL is **operationally optional,
not required for SOTA** (see §1.4 ★ v7 FINAL header above).
(iii) Replacing Local with NeCo (no Local + NeCo + Queue + NEG = iter 70
NEW cfg) gives seed=42 ARI 0.8797 versus B4 (with Local, no NeCo) 0.8605,
a +0.019 single-seed lift on HDBSCAN; **3-seed mean ARI 0.859 +/- 0.018 over
B5 0.856 +/- 0.012 is marginal +0.003, within multi-seed std** on HDBSCAN.
Under apples-to-apples HDBSCAN protocol (eom mcs=12 ms=3, defect-only),
Silhouette is **equivalent within ±0.013 (B5 0.7988 vs NEW 0.7860)**. The
previously-claimed "Silhouette +30% robust across seeds" was an HDBSCAN
protocol-mismatch artefact (B5 measured under leaf+ms=4, NEW under eom+ms=3)
and is **retracted** in this version. The genuine NEW-vs-B5 full-set
differentiator is **Normal/defect boundary stability** — under full-set
clustering (with Normal class), NEW gives Completeness 0.917 versus B5 0.851
and full-set ARI 0.83 versus B5 0.69, via Normal-cluster consolidation (859 of
1000 Normals merge into a single dense cluster, Normal noise 77.7% → 14.1%).
The defect-cluster geometry actually **widens** under NeCo (intra_p95 +26%),
confirming that NeCo's mechanism is **two-pronged** — Normal/defect boundary
stability (full-set, N1 v5) plus uniform-pattern consolidation (per-class
Agglo K=42, N1 v6) — not generic defect-cluster compactness.
(iii) **NEG filter requires Queue** — iter 74 (NeCo + NEG, no Queue)
reproduces iter 69 (NeCo only) to four decimals (ARI 0.8514, noise 3.93%,
Sil 0.7071, n_cluster 37 all identical), meaning NEG's false-negative
protection has zero effect when the large negative pool (MoCo Queue 4096)
is absent.
(iv) **Cross-component TEMP interaction sharpens C6 / N6**: TEMP 0.05
helps the Local-based cfg (+0.014 ARI in B5 path to iter 37) but **hurts
the NEW cfg** (-0.024 ARI in iter 73), demonstrating that best
hyperparameter values depend on component context. The resulting
Component Dependency Hierarchy is: **Required** (Global + {Local or NeCo}
patch-neighbor), **Significant** (MoCo Queue), **Conditional** (NEG filter,
requires Queue), **Substitutable** (Local DenseCL ↔ NeCo on ARI/noise;
apples-to-apples Silhouette comparison pending). The NEW four-component
cfg is one component fewer than the original five-component iter-37 cfg,
with **equivalent ARI and Silhouette under apples-to-apples protocol**;
the practitioner choice between NEW and B5 is operational (Normal/defect
boundary stability vs lower defect-noise floor), not strict superiority.
The full lattice matrix, multi-seed comparison, NeCo weight sweep, TEMP
interaction, retraction notes, and Normal-cluster consolidation evidence
are published as RESULTS table 14 (§14c, §14h, §14i, §14k).

**(C8) HDBSCAN Protocol Mismatch Methodology (NEW, 2026-05-12)**. An
initially-claimed "Silhouette +30% robust across three seeds" headline
in v0.5 (B5 vs NEW) was traced to a cross-protocol comparison — B5 was
measured under leaf+ms=4, NEW under eom+ms=3. Apples-to-apples
re-measurement (eom + mcs=12 + ms=3, defect-only) gives Sil 0.7988 vs
0.7860, a regression of −0.013 within seed variance. We preserve the
retracted claim as documented evidence (RESULTS §14k) and list it as
N8: any contrastive-clustering paper claiming Silhouette / ARI / noise
differences across cfg families must explicitly fix every HDBSCAN axis
(`selection_method`, `mcs`, `ms`, `epsilon`) and the metric scope
(full-set vs defect-only). Multi-seed robustness within a fixed
protocol does **not** detect cross-protocol artefacts.

**(C9) Clustering Algorithm Dependency — dual-frontier framework
(NEW, 2026-05-12)**. A five-method clustering benchmark (HDBSCAN,
DP-GMM, KMeans K=42, Agglomerative Ward K=42, Spectral K=42) on the
same three contrastive embeddings (B4, B5, iter 70 NEW) shows that
the ARI claim depends on the clustering algorithm by **+0.04 to +0.10
magnitude** at fixed embedding. For example, on B5 the ARI moves from
0.8564 (HDBSCAN) to 0.9358 (Agglomerative Ward K=42), a +0.079 lift
that comes purely from the clustering algorithm. Density-based methods
(HDBSCAN, DP-GMM) rank iter 70 NEW above B5, reflecting NEW's noise/
outlier-handling advantage. Linkage- and centroid-based methods with
oracle K (Agglomerative, KMeans K=42) rank B5 above iter 70 NEW,
reflecting B5's tighter defect-cluster geometry. Spectral K=42 is
unstable across cfg (ARI 0.23 to 0.79, with graph-not-fully-connected
warnings). The practitioner consequence is a **dual-frontier
framework**: (i) the **unknown-K real-world frontier** uses HDBSCAN +
iter 70 NEW (ARI 0.880 single-seed, 0.859 ± 0.018 3-seed, with
Normal/defect boundary stability via Normal-cluster consolidation per
paper N1 v5); (ii) the **known-K oracle benchmark frontier** uses
Agglomerative Ward + B5 / iter 37 (ARI 0.9358 single-seed, 0.9014 ±
0.022 NEW 3-seed). We publish §15 of RESULTS as the five-method ×
three-cfg ARI/NMI matrix and §15e as the practitioner choice tree.
This finding clarifies that headline ARI numbers across the
contrastive-clustering literature are not comparable unless both
encoder cfg and clustering algorithm are matched — a methodology
disclosure obligation that this paper makes explicit.

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
