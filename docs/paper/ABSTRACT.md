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

## v0.5 (2026-05-12) — ★ DEPRECATED 2026-05-12 (Sil +30% protocol mismatch)

> **Deprecated**: v0.5 의 "Silhouette +30% robust geometry lift" headline 은
> HDBSCAN protocol mismatch artefact 였음. apples-to-apples 재측정 결과 NEW vs B5 Sil
> 차이는 −0.013 (equivalent within seed variance). 본 v0.5 는 historical reference 로
> 보존. 현행 paper headline 은 v0.6 사용. 정정 detail: RESULTS §14c / §14h / §14i.

We present a self-supervised contrastive learning + density-based clustering pipeline
for **open-set wafer defect grouping**, hardened over **77 atomic-change ablation
iterations** plus a **four-component lattice exploration** (Local x Queue x NEG x NeCo)
against four primary objectives: P1 class capture, P2 defect noise rate, P3 Completeness,
P4 Homogeneity / AMI. The system uses a frozen ConvNeXtV2-base FCMAE encoder with
task-adaptive pre-training (TAPT) from a sister supervised classifier, a 128-dimension
projection head trained with InfoNCE plus NeCo patch-neighbor consistency, MoCo queue,
and NV-Retriever-style negative filtering. HDBSCAN with `eom` selection and
`min_samples=3` clusters the resulting embedding.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect + Normal,
2,146 wafers, 5 chip-internal object types + 9 wafer-canvas patterns), the
**iter-70 NEW SOTA cfg (Global + NeCo + Queue + NEG, no Local DenseCL)** achieves
**3-seed mean ARI 0.859 +/- 0.018, Silhouette 0.7941 +/- 0.017 (+30% over the
Local-based iter-37 B5 cfg, robust across seeds), Completeness 0.9825, AMI 0.9503,
defect-noise 1.48%, 43/43 class capture**. The headline novelty is **Silhouette +30%
robust geometry lift** at marginal ARI equivalence — a Pareto frontier between
partitioning quality and cluster compactness.

We make **seven contributions**. **(N1)** A 43-class open-set wafer benchmark with
documented sub-style splits validated by GMM-BIC and intra/inter ratio evidence.
**(N2)** Multi-seed honesty across two cfg families: iter-37 B5 (Local-based) at ARI
0.866 +/- 0.014, NEW (NeCo-replaces-Local) at ARI 0.859 +/- 0.018; same-seed run-to-run
variance is non-negligible. **(N3)** Five atomic encoder levers plus three HDBSCAN-side
axes plus 14 dead axes — a full ablation map. HDBSCAN-side tuning alone reduces defect
noise from 12.6% to 0.61% at fixed encoder (-91%). **(N4)** A domain-specific
reinterpretation of NeCo's mechanism: Normal-defect boundary repulsion (per-class
centroid shifts +0.05 to +0.07 cosine away from Normal). **(N5)** A **comprehensive
saturation point** for iter 37 — 6 hyperparameter axes + 2 Spatial-NeCo variants all
sweep within multi-seed std. **(N6) Component Interaction Matters**: a six-step Real
Baseline ablation (B0 to B5) reveals that LW lever isolated effect is **negative**
(B1 to B2: ARI -0.028) and only materializes through Queue interaction (B2 to B3: ARI
+0.023). NeCo's isolated effect within Local-based cfg (B4 to B5) is **-0.004 ARI**,
within run-to-run variance. **(N7 NEW) Component Dependency Hierarchy**: a
four-component lattice (Local x Queue x NEG x NeCo, iter 67-77) maps each component's
dependency: (i) **NeCo functionally equivalent to DenseCL Local InfoNCE** (iter 69 ARI
0.8514 = B1 ARI 0.8514 to four decimal places, identical noise/n_cluster), with NeCo
strictly superior on Silhouette (+0.193 in isolation); (ii) **Local + NeCo simultaneously
= redundant + slightly harmful** (B5 < B4, replace Local with NeCo for NEW SOTA);
(iii) **NEG filter requires Queue** — iter 74 (NeCo + NEG, no Queue) = iter 69 (NeCo
only) **exact identical** to four decimals, meaning NEG effect = 0 when Queue absent;
(iv) **TEMP x component interaction**: TEMP 0.05 helps Local-based cfg (+0.014 ARI in
B5) but **hurts NEW cfg** (-0.024 ARI in iter 73), confirming N6 that best hparam
depends on component context; (v) **NeCo weight inverse-U** with peak ARI at 0.2 but
**monotonic Silhouette ascent**, exposing a geometry-vs-partitioning Pareto frontier.

The dependency hierarchy that emerges is: Required (Global + patch-neighbor either-or),
Significant (Queue, +0.029 lift with NeCo), Conditional (NEG, requires Queue), and
Deprecated (Local DenseCL, NeCo strictly equivalent or better). The four-component
NEW cfg is **more parsimonious (4 vs 5 components) and Silhouette-stronger** than the
original iter-37 five-component cfg, at equivalent partitioning quality. Future work
remains the Cluster-Aware Synthesis Loop (F2).

> 갱신 history: `ITERATIONS.md` (append-only, iter 0 through iter 77). Decision history:
> `docs/contrastive-eval/DECISIONS.md`. NEW cfg + N7 evidence: `RESULTS.md` table 14,
> `DISCUSSION.md` §7.10, `CONCLUSION.md` §8.6.

## v0.6 (2026-05-12) — Sil +30% retract + paper N1 v5 + N8 NEW (★ CURRENT)

We present a self-supervised contrastive learning + density-based clustering pipeline for
**open-set wafer defect grouping**, hardened over **77 atomic-change ablation iterations**
plus a **four-component lattice exploration** (Local x Queue x NEG x NeCo) and a
**six-step Real Baseline component isolation** (B0 to B5) against four primary objectives
(P1 class capture, P2 defect noise, P3 Completeness, P4 Homogeneity / AMI). The system uses
a frozen ConvNeXtV2-base FCMAE encoder with task-adaptive pre-training (TAPT), a
128-dimension projection head trained with InfoNCE plus NeCo patch-neighbor consistency,
MoCo queue, and NV-Retriever-style negative filtering. HDBSCAN with `eom` selection and
`min_samples=3` clusters the resulting embedding.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect + Normal, 2,146
wafers), the iter-37 B5 configuration (Local + Queue + NEG + NeCo) achieves **3-seed mean
ARI 0.866 +/- 0.014, defect-only Silhouette 0.7988, defect-noise 0.61%, 43/43 class
capture**, and the iter-70 NEW configuration (NeCo + Queue + NEG, no Local) achieves
**3-seed mean ARI 0.859 +/- 0.018, Silhouette 0.7860, defect-noise 1.48%**.

**On apples-to-apples comparison (same HDBSCAN protocol `eom mcs=12 ms=3` defect-only),
NEW vs B5 gives ARI +0.023 (single-seed=42) / +0.003 (3-seed avg, marginal within multi-seed
std); Silhouette equivalent (±0.013, slight regression); noise +0.52pp slight regression.**
A previously-claimed "Silhouette +30%" / "+0.184 Sil" headline in v0.5 was an artefact of
HDBSCAN protocol mismatch (B5/B4 measured under leaf+ms=4, NEW under eom+ms=3) and is
retracted in this version.

We make **eight contributions**. **(N1 v5 final)** NeCo's primary mechanism on this domain
is **Normal/defect boundary stability**, not defect-cluster geometry: cluster-analyzer
post-hoc inspection shows Normal-cluster consolidation (Normal noise 77.7% → 14.1%, 859 of
1000 Normals merged into a single dense cluster, full-set Completeness 0.917 vs B5 0.851,
full-set ARI 0.83 vs B5 0.69) while defect-only metrics are functionally equivalent (ARI
±0.003 multi-seed, Sil ±0.013, defect-cluster intra_p95 actually +26% widening). **(N2)**
Multi-seed honesty across two cfg families; single-seed +0.010 lucky variance reproduces on
three independent axes. **(N3)** Five atomic encoder levers + three HDBSCAN-side axes + 14
dead axes; HDBSCAN-side tuning alone reduces defect noise 91% at fixed encoder. **(N4)**
Domain-specific NeCo mechanism — Normal-defect boundary repulsion. **(N5)** Comprehensive
saturation point at iter 37 across six hparam axes and two Spatial-NeCo variants.
**(N6)** Component Interaction Matters — Real Baseline ablation shows LW lever isolated
effect is negative and only materializes through Queue interaction. **(N7)** Component
Dependency Hierarchy — Local DenseCL and NeCo are substitutes on partitioning (ARI/noise/n_cl
4-decimal identical at iter 69 vs B1), NEG filter requires Queue (iter 74 vs iter 69
4-decimal identical), TEMP sign flips across cfg families. **(N8 NEW) HDBSCAN Protocol
Mismatch Methodology**: comparing clustering metrics across different HDBSCAN configurations
or metric scopes (full-set vs defect-only) produces spurious headline differences. The
retracted v0.5 "+30% Sil" claim is itself the documented evidence; future contrastive-clustering
papers must explicitly fix `selection_method`, `mcs`, `ms`, `epsilon`, and the metric scope
before reporting cross-cfg deltas.

The honest practitioner-facing summary: **NEW (4-component) and B5 (5-component, iter 37)
are essentially equivalent on defect-only clustering**, with NEW offering Normal/defect
boundary stability (paper N1 v5) and B5 offering a slightly lower defect-noise floor (0.61%
vs 1.48%). The choice is operational, not strict superiority.

> 갱신 history: `ITERATIONS.md` (append-only). Decision history: `docs/contrastive-eval/DECISIONS.md`.
> Retracted-claims index: `RESULTS.md` §14k.

## v0.7 (2026-05-12) — ★ Dual-frontier + N9 NEW (clustering algorithm dependency) — superseded by v0.8

We present a self-supervised contrastive learning + clustering pipeline for **open-set wafer
defect grouping**, hardened over **77 atomic-change ablation iterations**, a **four-component
lattice exploration** (Local × Queue × NEG × NeCo), a **six-step Real Baseline component
isolation** (B0 to B5), and a **five-method clustering benchmark** (HDBSCAN, DP-GMM,
KMeans K=42, Agglomerative Ward K=42, Spectral K=42) against four primary objectives (P1
class capture, P2 defect noise, P3 Completeness, P4 Homogeneity / AMI). The encoder is a
frozen ConvNeXtV2-base FCMAE backbone with task-adaptive pre-training (TAPT), with a
128-dimension projection head trained on InfoNCE plus NeCo patch-neighbor consistency, MoCo
queue, and NV-Retriever-style negative filtering.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect + Normal, 2,146
wafers), the work reports **dual frontiers**, not a single SOTA:

- **Unknown-K frontier (real-world deployment, HDBSCAN)**: iter 70 NEW cfg (NeCo + Queue +
  NEG, no Local) gives **single-seed ARI 0.8797 / 3-seed 0.859 ± 0.018, Normal-cluster
  consolidation (Normal noise 77.7% → 14.1%), full-set ARI 0.83 vs B5 0.69**.
- **Known-K frontier (oracle benchmark, Agglomerative Ward K=42)**: B5 (iter 37 cfg, Local
  + Queue + NEG + NeCo) gives **single-seed ARI 0.9358 / NEW 3-seed 0.9014 ± 0.022,
  NMI 0.9704** — linkage-based clustering recovers fine defect sub-structure when K is known.

The same embedding produces ARI differences of **+0.04 to +0.10 depending on the clustering
algorithm** (HDBSCAN vs Agglomerative on B5: 0.8564 vs 0.9358, +0.079). This is a methodology
finding (paper N9): any ARI claim on contrastive-cluster pipelines must specify the
clustering algorithm and the K-discovery regime.

We make **nine contributions**. **(N1 v5 final)** NeCo's wafer-domain mechanism is
Normal/defect boundary stability (Normal-cluster consolidation, full-set Completeness
0.917 vs B5 0.851), not defect-cluster compactness — defect-cluster intra_p95 actually
widens +26% under NeCo. **(N2)** Multi-seed honesty across two cfg families and three
clustering axes; single-seed +0.010 lucky variance reproduces on HDBSCAN, Agglomerative,
and KMeans seed=42 vs seed=1 drops. **(N3)** Five atomic encoder levers + three HDBSCAN
axes + 14 dead encoder axes; HDBSCAN-side tuning reduces defect noise 91% at fixed encoder.
**(N4)** NeCo mechanism reinterpreted as Normal-defect boundary repulsion. **(N5)**
Comprehensive saturation point at iter 37 across six hparam axes and two Spatial-NeCo
variants. **(N6)** Component Interaction Matters — Real Baseline B0 to B5 reveals LW
isolated effect is negative; the lift only materializes via Queue interaction.
**(N7)** Component Dependency Hierarchy — Local DenseCL ↔ NeCo substitutable on partitioning;
NEG requires Queue. **(N8)** HDBSCAN Protocol Mismatch Methodology — cross-cfg metric
comparisons require explicit unification of `selection_method`, `mcs`, `ms`, `epsilon`,
and metric scope. **(N9 NEW) Clustering Algorithm Dependency** — five-method benchmark
shows ARI claims depend on clustering algorithm by +0.04 to +0.10 magnitude; the unknown-K
SOTA (HDBSCAN) and the known-K SOTA (Agglomerative) live on different cfg points (NEW vs
B5), and Spectral K=42 is unstable (ARI 0.23 to 0.79 across cfg). The honest practitioner
recipe is a two-frontier decision tree: HDBSCAN + NEW for real-world unknown-K deployment
with Normal-dominant streams, Agglomerative Ward + B5 for oracle-K lab benchmarks.

> 갱신 history: `ITERATIONS.md` (append-only). Decision history: `docs/contrastive-eval/DECISIONS.md`.
> Five-method benchmark: `RESULTS.md` §15. Retracted-claims index: `RESULTS.md` §14k.

## v0.8 (2026-05-12) — ★ N1 v6 complementary + dual-cfg dual-frontier — ★ superseded by v0.9 (B5 seed=1 retraction)

We present a self-supervised contrastive learning + clustering pipeline for **open-set
wafer defect grouping**, hardened over **77 atomic-change ablation iterations**, a
**four-component lattice exploration** (Local × Queue × NEG × NeCo), a **six-step Real
Baseline component isolation** (B0 to B5), a **five-method clustering benchmark** on
defect-only embeddings, and a **per-class K=42 Agglomerative Ward purity breakdown**.
The encoder is a frozen ConvNeXtV2-base FCMAE backbone with task-adaptive pre-training
(TAPT), with a 128-dimension projection head trained on InfoNCE plus optionally Local
DenseCL grid contrast plus NeCo patch-neighbor consistency, MoCo queue, and
NV-Retriever-style negative filtering.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect + Normal, 2,146
wafers), the work reports a **dual-cfg dual-frontier framework**, not a single SOTA:

- **Frontier 1 — Unknown-K real-world deployment (HDBSCAN)**: iter 70 NEW cfg
  (4-component, NeCo + Queue + NEG, **no Local**) gives **single-seed ARI 0.8797 /
  3-seed 0.859 ± 0.018**, Normal-cluster consolidation (Normal noise 77.7% → 14.1%,
  full-set ARI 0.83 vs B5 0.69), and uniform-pattern consolidation (CenterCircle 100%
  purity vs B5 54.8%, Edge-Top_fork 100% vs 90%).
- **Frontier 2 — Known-K oracle benchmark (Agglomerative Ward)**: **B5 cfg
  (5-component, Local + NeCo + Queue + NEG)** is the **absolute SOTA** with
  single-seed ARI **0.9358** / NMI **0.9704**, strictly above NEW 0.9200
  (Δ +0.0158), because Local DenseCL and NeCo encode **complementary per-class
  inductive biases**: Local integrates fork/scratch sub-pattern variants
  (`Edge-Ring_fork` 100% B5 vs 64.5% NEW, `Center_scratch` 95% vs 75%, `Donut_fork`
  100% vs 81.1%, `Edge-Top_scratch` 100% vs 84.2%) while NeCo consolidates
  uniform-pattern classes.

The previously-claimed "Local DenseCL functionally equivalent to NeCo —
substitutable on partitioning" (paper N1 v5) is **refined in v6 to complementary,
not substitutable**: aggregate HDBSCAN ARI equality (iter 69 vs B1 4-decimal
identity) is necessary-but-not-sufficient; per-class K=42 Agglomerative Ward purity
reveals class-by-class winner flips on both sides with B5 marginal aggregate win
(97.0% vs 96.2%, Δ −0.83pp). Local DenseCL is therefore **not deprecated** — it
carries sub-pattern variant integration that NeCo cannot recover alone, and the
**absolute SOTA configuration retains both mechanisms** under known-K linkage
clustering.

We make **nine contributions**. **(N1 v6 FINAL)** Local DenseCL and NeCo carry
**complementary per-class inductive biases** (RESULTS §16, NEW): per-class purity
flips reveal class-by-class winners on both sides; aggregate ARI hides the
complementarity. Combining both (B5) reaches absolute SOTA ARI 0.9358 single-seed
on Agglomerative Ward K=42. NeCo's full-set differentiator (Normal/defect boundary
stability, N1 v5) is preserved. **(N2)** Multi-seed honesty across two cfg families
and three clustering axes. **(N3)** Five atomic encoder levers + three HDBSCAN axes
+ 14 dead encoder axes; HDBSCAN-side tuning reduces defect noise 91% at fixed
encoder. **(N4)** NeCo's two-pronged mechanism — Normal-defect boundary repulsion
(full-set, v5) + uniform-pattern consolidation (per-class, v6). **(N5)**
Comprehensive saturation point at iter 37 across six hparam axes and two
Spatial-NeCo variants. **(N6)** Component Interaction Matters — Real Baseline B0
to B5 reveals LW isolated effect is negative. **(N7)** Component Dependency
Hierarchy — Local DenseCL and NeCo are **complementary per-class** (refined v6),
NEG requires Queue. **(N8)** HDBSCAN Protocol Mismatch Methodology — cross-cfg
metric comparisons require explicit unification of `selection_method`, `mcs`,
`ms`, `epsilon`, and metric scope. **(N9) Clustering Algorithm Dependency** —
five-method benchmark shows ARI claims depend on clustering algorithm by +0.04 to
+0.10 magnitude. The honest practitioner recipe is a **dual-cfg dual-frontier
decision tree**: HDBSCAN + NEW for real-world unknown-K with Normal-dominant
streams; Agglomerative Ward K=42 + B5 for oracle-K lab benchmarks where the
complementary per-class purity is recoverable.

> 갱신 history: `ITERATIONS.md` (append-only). Decision history: `docs/contrastive-eval/DECISIONS.md`.
> Five-method benchmark: `RESULTS.md` §15. Per-class K=42 purity breakdown: `RESULTS.md` §16.
> Retracted-claims index: `RESULTS.md` §14k (v5 substitutability framing now refined in v6).

## v0.9 (2026-05-12) — ★ N1 v7 FINAL — B5 seed=1 retraction + NEW unified multi-seed SOTA — ★ CURRENT

We present a self-supervised contrastive learning + clustering pipeline for **open-set
wafer defect grouping**, hardened over **84 atomic-change ablation iterations**, a
**four-component lattice exploration** (Local × Queue × NEG × NeCo), a **six-step
Real Baseline component isolation** (B0 to B5), a **five-method clustering benchmark**
on defect-only embeddings, a **per-class K=42 Agglomerative Ward purity breakdown**,
and a **B5 reproducibility test (seed=1) that retracts the v6 single-cfg absolute SOTA
claim**. The encoder is a frozen ConvNeXtV2-base FCMAE backbone with task-adaptive
pre-training (TAPT), with a 128-dimension projection head trained on InfoNCE plus
optionally Local DenseCL grid contrast plus NeCo patch-neighbor consistency, MoCo
queue, and NV-Retriever-style negative filtering.

On a 43-class new-anchor benchmark synthesized from WM-811K (42 defect + Normal,
2,146 wafers), the work reports a **single recommended encoder cfg (iter 70 NEW —
Global + NeCo + Queue + NEG, no Local) with two clustering frontier targets**:

- **Frontier 1 — Unknown-K real-world deployment (HDBSCAN eom mcs=12 ms=3)**:
  3-seed mean ARI **0.859 ± 0.018**, defect-noise 1.48% mean, 43/43 class capture.
  Normal-cluster consolidation (Normal noise 77.7% → 14.1%, paper N1 v5) and full-set
  ARI 0.83 vs B5 0.69 retained.
- **Frontier 2 — Known-K oracle benchmark (Agglomerative Ward K=42, defect-only)**:
  NEW 3-seed mean ARI **0.9014 ± 0.022**. B5 (Local + Queue + NEG + NeCo, the v6
  "absolute SOTA" cfg) 2-seed avg = **0.8920 ± 0.062**, **below NEW** by Δ −0.0094 and
  with 2.8× higher std.

**The v6 claim that B5 (5-component, Local + NeCo) is the absolute SOTA at single-seed
Agglo Ward K=42 ARI 0.9358 is retracted in v7**. The seed=42 0.9358 reading was a
cherry-picked outlier: seed=1 reproduction (iter 84, same cfg, same protocol) gave
Agglo K=42 ARI **0.8482** — a Δ −0.0876 single-seed drop, vs NEW's Δ −0.0346 drop on
the same method. Across all three clustering methods (HDBSCAN, Agglo Ward K=42,
KMeans K=42), **NEW > B5 on multi-seed average ARI** by +0.009 to +0.025 with
1.7–2.8× lower std. The "dual-cfg dual-frontier" recipe (v0.8) collapses to a
**single-cfg recommendation (NEW) with two clustering frontier targets**.

We make **nine contributions**. **(N1 v7 FINAL)** NEW cfg (NeCo + Queue + NEG, no
Local) is the unified multi-seed SOTA on both HDBSCAN unknown-K and Agglomerative
Ward K=42 oracle-K frontiers; the v6 "complementary inductive bias" per-class
breakdown (RESULTS §16) is preserved as a single-seed observation but does not
propagate to multi-seed averages — Local DenseCL is **not required** for SOTA.
**(N2)** Multi-seed honesty — strongest evidence to date: B5 seed=42 vs seed=1
Agglomerative Ward K=42 ARI drop = **−0.088**, larger than any encoder lever in this
paper. Single-seed ablation across cfg families produces false winner claims.
**(N3)** Five atomic encoder levers + three HDBSCAN axes + 14 dead encoder axes.
**(N4)** NeCo's two-pronged mechanism — Normal-defect boundary repulsion (full-set,
v5) + uniform-pattern consolidation (per-class single-seed Agglo K=42, v6) — but
**both behaviors are operational, not strict SOTA prerequisites** (v7). **(N5)**
Comprehensive saturation point at iter 37 across six hparam axes and two Spatial-NeCo
variants. **(N6)** Component Interaction Matters — Real Baseline B0 to B5 reveals
LW lever isolated effect is negative; Queue interaction realizes the lift.
**(N7)** Component Dependency Hierarchy — NEG requires Queue; Local DenseCL and NeCo
are substitutable on HDBSCAN aggregate (v5) and complementary on per-class single-seed
Agglo K=42 (v6), but multi-seed avg gives **NEW (NeCo only) ≥ B5 (Local + NeCo)**
on all three clustering methods (v7). **(N8)** HDBSCAN Protocol Mismatch Methodology
— cross-cfg metric comparisons require explicit unification of `selection_method`,
`mcs`, `ms`, `epsilon`, and metric scope. **(N9) Clustering Algorithm Dependency** —
five-method benchmark shows ARI claims depend on clustering algorithm by +0.04 to
+0.10 magnitude. The honest practitioner recipe is a **single-cfg (NEW) two-frontier
decision tree**: HDBSCAN + NEW for real-world unknown-K with Normal-dominant streams;
Agglomerative Ward K=42 + NEW for oracle-K lab benchmarks. The v0.8 "B5 for known-K"
sub-recommendation is retracted.

**Production inference is 14 ms / wafer on a single NVIDIA RTX 4060 Ti (BATCH = 1,
ConvNeXtV2-base 87.7 M params frozen), with HDBSCAN `approximate_predict` adding
~10 ms — total ≈ 24 ms / wafer (≈ 40 wafers/sec, real-time deployable). NEW
training is 23.7 min for 5 epochs on n=2,146, ~30% faster than the B5 recipe
(28–49 min) due to Local DenseCL removal.**

> 갱신 history: `ITERATIONS.md` (append-only, iter 84 entry). Decision history:
> `docs/contrastive-eval/DECISIONS.md`. Five-method benchmark: `RESULTS.md` §15.
> Per-class K=42 purity breakdown: `RESULTS.md` §16. **B5 reproducibility +
> multi-seed avg comparison: `RESULTS.md` §17 (NEW)**. Computational performance +
> dataset statistics: `RESULTS.md` §18 (NEW). Retracted-claims index:
> `RESULTS.md` §14k + §17c (v6 absolute SOTA retraction).
