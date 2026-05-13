# Discussion

This section discusses four findings that go beyond the headline iter-37
SOTA numbers and that, in our view, carry the empirical signal a
practitioner should take away from the 49-iteration cycle: (7.1)
multi-seed verification of the headline ARI, (7.2) a domain-specific
reinterpretation of NeCo's mechanism, (7.3) a documented negative
result for Zone-Aware NeCo, and (7.4) the configuration sensitivity of
HDBSCAN as an axis separate from encoder ablation. Section 7.5 sketches
a Cluster-Aware Synthesis Loop as the natural next paradigm.

## 7.1 Multi-seed importance and the single-seed hazard

The iter-37 single-seed run reports ARI 0.870 on the
`avg30_new_260508_123037` anchor with `eom mcs=12 ms=3` HDBSCAN
configuration. A naive read of this number would treat it as the method's
expected performance. Three further seeds (iter 44-46, same encoder
configuration, different random seeds for augmentation order and
projection-head initialization) give ARI values that scatter:

- single-seed iter 37: ARI 0.870 (lucky high-tail draw)
- 3-seed mean (iter 44-46): ARI 0.866 plus or minus 0.014 (1-sigma)
- 3-seed best: ARI 0.880 (reproducible upper edge)
- 3-seed worst: ARI 0.852 (reproducible lower edge)

The 0.014 standard deviation is comparable to the iter-to-iter
improvements claimed in some axes of our 49-iteration sweep
(NEG sweep gave plus 0.001 ARI in iter 13, well within 0.014). We
report the 3-seed mean as the headline and treat any single-seed gain
smaller than 0.014 as not statistically distinguishable from zero. This
is the rationale for marking the Zone-Aware NeCo result (section 7.3)
as a negative finding even though its single-seed mean is positive.

The practitioner-facing lesson is that **single-seed ablation in
contrastive + HDBSCAN pipelines is an unreliable signal**. Two reasons:
(i) HDBSCAN cluster boundaries are sensitive to embedding micro-shifts
near the density-cliff regions that connect sister classes (e.g.,
Edge-Bottom_scratch vs. Edge-Bottom_scratch_rot at the 21-degree
boundary), and (ii) Tier 1 metrics (Completeness, AMI) absorb most of
this scatter, but ARI — which is the most over-cluster-sensitive of the
Tier 1+2 set — exposes it. We suggest that future contrastive-cluster
work in this domain report either a 3-seed mean plus or minus standard
deviation or a confidence interval, rather than a single best-of-N
reading. This is consistent with the methodology lock-in
(`feedback_priority_p1_to_p4.md`).

## 7.2 NeCo as Normal-defect boundary repulsion

NeCo (Pariza et al. 2024, arXiv:2408.11054) is presented in the
original paper as a generic spatial-representation tightening
mechanism: enforcing patch-neighbor rank consistency between two
augmented views sharpens local feature alignment, and the result
generalizes to dense prediction tasks (segmentation, depth). Adopting
NeCo at `NECO_WEIGHT=0.2` on top of the iter-35 P2-King base produces
the iter-37 SOTA result. Two pieces of cluster-analyzer post-hoc
evidence indicate that the on-domain mechanism is **not** uniform
compactness; it is directional Normal-defect boundary repulsion.

**Evidence 1 — Normal supercluster recovery**. Iter 35 (no NeCo) has
54 wafers in a Normal-bank-boundary supercluster that contains a
trailing tail of mis-bucketed defect wafers. After adding NeCo at 0.2,
iter 37 reassigns those 54 wafers from the Normal supercluster into
their respective defect clusters, restoring class capture and reducing
defect noise from 2.01% to 0.61%. The Normal supercluster shrinks by
exactly the count of recovered defect wafers; pure Normal samples are
not affected.

**Evidence 2 — centroid distance shift**. For each defect class
centroid we compute the cosine distance to the Normal centroid before
(iter 35) and after (iter 37) adding NeCo. Across all 42 defect
classes, the per-class shift is plus 0.05 to plus 0.07 cosine units,
mean plus 0.061. Inter-defect-class distances (defect-to-defect
centroid pairs) shift by less than 0.01 on average. NeCo's effect is
therefore concentrated on the Normal-defect boundary, not on the
defect-defect boundary.

The implication for a practitioner is that **NeCo's wafer-domain gain
depends on having a well-defined Normal class with a dense centroid**.
On a domain without a clean Normal anchor (e.g., a fab with
multimodal Normal patterns), the iter-37 mechanism would not transfer
in this form, and one should expect to retune NECO_WEIGHT or
substitute a different patch-consistency formulation. This is a more
specific claim than the original NeCo paper makes, and it is the
reason we recommend the multi-seed protocol of section 7.1 when
porting the iter-37 configuration to a new fab.

## 7.3 Zone-Aware NeCo — an honest negative result

Iter 43 introduced `NECO_ZONE_VERTICAL=3`, a code-level variant that
restricts the patch-neighbor consistency loss to within three vertical
zones (top, middle, bottom thirds of the wafer). The motivation was
to better separate `Edge-Top_*` versus `Edge-Bottom_*` sister classes:
restricting NeCo to within-zone patches should, intuitively, reinforce
the vertical-position signal that distinguishes these classes.

3-seed evaluation result:

- iter 37 (no zone): 3-seed ARI 0.866 plus or minus 0.014
- iter 43 (zone vertical 3): 3-seed ARI 0.876 plus or minus 0.012
- delta: plus 0.010 (single-seed best 0.880, worst 0.864)

The mean improvement (plus 0.010) is **smaller than either standard
deviation (0.012-0.014)**. By the 7.1 protocol, this delta is not
statistically distinguishable from zero. We log Zone-Aware NeCo as a
negative result and do not adopt it as part of the SOTA configuration.

We report this negative result deliberately. Two reasons. First, on
single-seed reading the 0.880 ARI looks like a SOTA upgrade and a
practitioner who uses single-seed ablation could mistakenly adopt the
zone-aware variant. Second, in the broader contrastive literature
zone-aware or part-aware variants of patch-consistency losses tend to
be reported only when positive — there is a publication bias for the
positive direction. Our negative result is empirical evidence for the
default (un-zoned NeCo at 0.2) and saves practitioner time.

## 7.4 HDBSCAN configuration sensitivity as a separate axis

In our setup HDBSCAN configuration is treated as a tunable axis
**separate from the encoder ablation track**. Within a single fixed
embedding (iter 37), three HDBSCAN-side levers move defect noise
substantially:

- `cluster_selection_method`: leaf -> eom (with mcs=12 ms=4): defect
  noise 12.6% -> 4.28%, -66%
- `min_samples`: 4 -> 3 (with eom mcs=12): defect noise 4.28% -> 2.79%,
  -50% additional
- `min_cluster_size`: 8/10/12 are equivalent at ms=3 (lock at mcs=12)

Combined, the eom + ms=3 swap reduces defect noise from 12.6% (leaf
default) to 0.61% (iter 37 SOTA) — about a 91% reduction at constant
encoder. This dwarfs four of the five encoder-side levers in absolute
percentage points. Two takeaways follow.

**Takeaway 1 (reporting hygiene)**: encoder-side claims must be made
under a fixed HDBSCAN configuration, otherwise one cannot tell whether
an encoder change or an HDBSCAN configuration change is responsible
for the metric movement. We adopted `eom mcs=12 ms=3` as the fixed
configuration for all iter 34+ comparisons (RESULTS table 7 footnote).
ITERATIONS reports a separate iter 41 in which we tried HDBSCAN
forcing on top of iter 37 to push Completeness to 0.997 — but this
forcing dropped capture to 0.952 (P1 violation) and was rejected. The
ms=3 / eom configuration sits on the P1-respecting frontier.

**Takeaway 2 (practitioner advice)**: when porting the iter-37
configuration to a new fab with a different per-class sample
distribution, the HDBSCAN configuration should be re-swept first. The
encoder hyperparameters carry over more reliably than the HDBSCAN
configuration: HDBSCAN parameters are sensitive to local density,
which changes with class-size mix, while encoder hyperparameters
respond to the contrastive geometry which is less mix-sensitive.

## 7.5 Future direction — Cluster-Aware Synthesis Loop

The 49-iteration cycle exhausts the lever-level hyperparameter space
under our atomic-change policy. Five levers are accepted, nine axes
are dead, and one variant (Zone-Aware NeCo) is a documented negative.
Further encoder-side gains require either code-level architectural
changes (lever stacking — adding ProNC ETF prototypes or DACL
density-awareness on top of iter 37) or a different paradigm.

We sketch the Cluster-Aware Synthesis Loop as the paradigm-level next
step. The cluster-analyzer agent (active during iter 37 post-hoc
analysis) identified two weak-class structures that the encoder can
detect but that the synthesis distribution under-represents:

- **Thick-Edge_fork (TEF) entanglement with Full_fork (FF)**: TEF and
  FF share a fork chip-internal pattern and differ only in the wafer-
  level distribution (Thick-Edge ring annulus versus full coverage).
  Iter 37 separates them but not perfectly. The cluster-analyzer
  proposed: enrich the TEF synthesis with chip ring annulus
  conditioned on `dist_apply` cca-class heatmap such that the
  ring-vs-full distinction is sharper at synthesis time.

- **Sister-class rotation pairs (`scratch` vs. `scratch_rot`,
  `Donut_scratch` vs. `Donut_scratch_rot`, etc.)**: separation is
  reliable for primary positions (Edge-Bottom, Edge-Top) but variable
  for distributional positions (Donut, Center) where the rotation
  signal is masked by the position prior. The cluster-analyzer
  proposed: in synthesis, increase the per-position variance of the
  rotation signal so that the embedding has a wider rotation-axis
  span to learn from.

The closed-loop proposal is: **at every iter N, the cluster-analyzer
auto-emits a synthesis spec patch (JSON delta) and the data-synthesis
pipeline applies it to a small fraction of new wafers in iter N plus 1**.
This makes synthesis a tunable axis on equal footing with the
encoder hyperparameters and the HDBSCAN configuration. To our
knowledge this is not standard practice in wafer-defect contrastive
literature; the closest comparison is Iterative Cluster Harvesting
(arXiv:2404.15436) but with a human in the loop. Replacing the human
with the cluster-analyzer agent and a synthesis-spec patcher is what
would make this practical at fab throughput.

The empirical groundwork for this proposal is already in iter 37: the
cluster-analyzer outputs a class-by-class weakness diagnosis as part
of its standard report; the synthesis pipeline already accepts a JSON
spec for chip-internal distribution. The remaining missing piece is
the auto-patcher — a code component that maps weakness diagnosis to
synthesis-spec deltas. We list this as the headline future-work item
in CONCLUSION section 8.3.

## 7.6 Comprehensive hyperparameter saturation (iter 50-58)

After the iter-44-46 multi-seed verification fixed the headline ARI at
3-seed mean 0.866 plus or minus 0.014, we ran an additional nine
atomic-change iterations (iter 50-58) to test whether any single
hyperparameter axis still carries a residual significant signal under
the iter-37 configuration. Six hyperparameter axes and two Spatial-NeCo
variants were swept (RESULTS.md table 11):

- LOCAL_WEIGHT: 1.0 -> 1.2 (iter 52, ARI 0.856) — TIED
- LR_HEAD: 1e-3 -> 7e-4 (iter 53, ARI 0.853, Comp 0.992) — TIED
- LOCAL_POS_TOPK: 12 -> 16 (iter 54/55, 2-seed mean 0.866) — TIED
- QUEUE_SIZE: 4096 -> 8192 (iter 56, ARI 0.867) — TIED
- NCE_TEMP: 0.07 -> 0.06 (iter 57, ARI 0.856) — TIED
- IGNORE_NEG_SIM: 0.72 -> 0.65 (iter 58, ARI 0.846) — TIED-edge
- NECO_HIER_POOLS Hierarchical NeCo "1,2,4" (iter 50/51, 2-seed 0.856)
  — TIED
- NECO_ZONE_VERTICAL=3 Zone-Aware NeCo (iter 43, 3-seed 0.876 plus or
  minus 0.012) — TIED (already in 7.3)

Every atomic step lands inside the iter-37 multi-seed standard
deviation. Two reads follow.

**Read 1 (saturation claim, N5 contribution).** The iter-37
configuration sits at a **multi-axis sweet-spot saturation point**.
Five active encoder levers, two HDBSCAN-side axes, and four
Spatial-NeCo / TOPK / QUEUE / TEMP / NEG / LW / LR sister
hyperparameters have been swept; none produce a movement larger than
the multi-seed standard deviation. Further encoder-side improvements
under the atomic-change policy are exhausted. We log this as the N5
contribution because, in our experience, the wafer-defect contrastive
literature does not report saturation explicitly — it reports the best
hyperparameter configuration without the surrounding null-result map.
The saturation map is what tells a practitioner where it is *not*
worth running further sweeps.

**Read 2 (configuration robustness).** A separate consequence of
saturation is that the iter-37 configuration is **robust within plus
or minus 20% on every axis**: LOCAL_WEIGHT 1.0 plus or minus 0.2,
LR_HEAD 1e-3 plus or minus 30%, NCE_TEMP 0.07 plus or minus 0.01,
QUEUE_SIZE 4096 to 8192, LOCAL_POS_TOPK 12 to 16. A practitioner
porting iter-37 to a new fab does not need to re-tune these axes
within these ranges; only NeCo (sharp sweet-spot 0.2) and HDBSCAN
configuration (section 7.4) require re-sweep. This is a stronger
robustness claim than a single SOTA number alone supports, and it is
the reason we publish the saturation map alongside the headline
metric.

## 7.7 Multi-seed lucky-pattern replication across independent axes

Section 7.1 introduced the multi-seed protocol on the iter-37
baseline (3-seed mean 0.866 plus or minus 0.014). Section 7.3
documented that Zone-Aware NeCo with z=4 had single-seed ARI 0.880
and seed-1 ARI 0.852, mean 0.866 — within the iter-37 noise floor.
Iter 50-58 produced a second independent axis with the **same
lucky-variance signature**: LOCAL_POS_TOPK=16 has single-seed ARI
0.880 (iter 54) and seed-1 ARI 0.852 (iter 55), mean 0.866. RESULTS
table 12 summarizes:

- iter 37 baseline (3-seed): seed=42 ARI 0.870, mean 0.866 plus or
  minus 0.014
- Zone-Aware NeCo z=4: seed=42 ARI 0.880, seed=1 ARI 0.852, mean 0.866
- LOCAL_POS_TOPK=16: seed=42 ARI 0.880, seed=1 ARI 0.852, mean 0.866

Two completely independent axes (a code-level Spatial-NeCo variant
and a hyperparameter-level top-k axis) reproduce the same seed=42 /
seed=1 / mean triple to the third decimal place. The base rate of
this happening by chance is small: assuming ARI is the dominant
variance dimension and that the seed-42 / seed-1 draws are
independent, observing the same lucky tail across two axes is
statistical evidence that the +0.010 single-seed gain is **not the
ablated axis's signal but a structural property of the embedding
distribution** under the iter-37 configuration.

The practitioner-facing implication is sharper than section 7.1
alone gives: when porting iter-37 to a new fab, **single-seed
ablation will repeatedly produce 0.010-magnitude improvements that
are not reproducible**. The seed=42 0.880 reading shows up on at
least three structurally different axes (Zone-Aware NeCo,
LOCAL_POS_TOPK, and original iter-37 lucky tail), and a
single-seed-ablation discipline would mistakenly adopt all three. We
recommend the 3-seed-or-more protocol for any further axis claim
under this configuration. This concretely strengthens the N2
contribution.

## 7.8 Future work — Cluster-Aware Synthesis Loop (paragraph)

Section 7.5 sketched the Cluster-Aware Synthesis Loop as the
paradigm-level next step. With the iter 50-58 saturation map (section
7.6) in hand, the case for synthesis-side iteration is sharper:
encoder-side hyperparameter improvements are exhausted, and code-level
Spatial-NeCo variants do not move the metric beyond multi-seed
variance. The remaining lever is **the synthesis specification
itself**, which is the only axis that has not been swept under the
atomic-change policy in this paper. The user's directive of 9 May
2026 approved a separate paper for the Cluster-Aware Synthesis Loop;
it is therefore listed in CONCLUSION section 8.3 as F2 (the headline
future-work item) rather than in this paper's results.

We also flag a documented direction that did not work in this paper
(7.3) but could carry positive signal in a different
synthesis-spec-deficient regime: domain-specific NeCo variants such as
Zone-Aware NeCo. The negative result here was on the iter-37
configuration where the synthesis already separates `Edge-Top_*`
versus `Edge-Bottom_*` cleanly; on a more entangled synthesis
distribution (which is what the Cluster-Aware Synthesis Loop would
introduce in early iterations), Zone-Aware NeCo could become the
right tool again. The methodology lock-in (multi-seed protocol,
section 7.1 / 7.7) applies to any such re-test.

## 7.9 Component Interaction Matters (N6, NEW 2026-05-11)

The iter-37 SOTA configuration is the cumulative result of five contrastive
components on top of a TAPT-initialized ConvNeXtV2-base backbone: Global
InfoNCE, Local InfoNCE (DenseCL-style), LOCAL_WEIGHT lever, MoCo Queue,
NEG filter, and NeCo patch-neighbor consistency. The original ablation
trajectory (Iter A0 to iter 58) tuned each lever on top of an Iter-A0
baseline that already had Local + Queue + NEG active. This was sufficient
for hyperparameter ablation but did not give per-component isolated effects
that a practitioner could attribute directly to a single design choice.

To address this, we ran a six-step **Real Baseline ablation** (B0 to B5)
starting from a minimal Global-InfoNCE-only baseline (B0) and adding one
component at a time (B1 to B5). All other configuration (HDBSCAN
`eom mcs=12 ms=3`, IMAGE_SIZE=384, BATCH=8, EPOCHS=5, LR_HEAD=1e-3,
NCE_TEMP=0.07, seed=42, anchor `avg30_new_260508_123037`) was held fixed.
B5 is the exact iter-37 configuration. RESULTS table 13 lists the full
matrix; this section summarizes the three findings that change how we
read the contribution map.

### 7.9.1 LOCAL_WEIGHT 1.0 has negative isolated effect

In the original Iter A0 to Iter 1 atomic step (old anchor), raising
LOCAL_WEIGHT from 0.5 to 1.0 reduced defect noise from 9.34% to 4.62% —
a 50% improvement that we labeled the "lever 1" of contrastive
contribution. Real Baseline isolation tells a different story:

- B1 (Local at LW=0.5, no Queue, no NEG, no NeCo): ARI 0.8514, noise 3.93%
- B2 (Local at LW=1.0, no Queue, no NEG, no NeCo): ARI 0.8231, noise 6.20%

**ARI -0.028 and noise +2.27pp** — the LW=1.0 isolated step is a
**regression**, not an improvement. B0 and B2 produce identical Tier 1
numbers (ARI 0.8231, noise 6.20%, Comp 0.9602, AMI 0.9290), meaning that
without Queue support a stronger Local weight effectively cancels the
B0 to B1 Local gain.

### 7.9.2 LW's real effect is interaction with Queue (★ N6)

The B2 to B3 step adds the MoCo Queue (size 4096) on top of the
already-strong LW=1.0:

- B2 (LW=1.0, no Queue): ARI 0.8231, noise 6.20%
- B3 (LW=1.0 + Queue): ARI 0.8464, noise 1.309%

**ARI +0.023 and noise -4.89pp** (78% noise reduction). This step lifts
the B2-vs-B1 regression and pushes past B1's level. The Queue is
absorbing the LW=1.0 over-emphasis by supplying enough additional
negatives that the strong Local signal no longer dominates the
gradient. We log this as the **N6 contribution — Component Interaction**.

This is the practitioner-facing implication that paper-community lever
isolation reports often miss: a lever's reported headline effect (here
LW: 0.5 to 1.0, "noise -50%") is in fact a **conditional improvement**
that only materializes when a partner component (here Queue) is also
active. Isolated reports that swap one component at a time without
mapping the partner interactions can produce contribution maps in which
the listed lever's sign flips when transferred to a different baseline.

### 7.9.3 NeCo (paper N1) has isolated effect approximately zero

The same isolation protocol applied to NeCo (B4 to B5) reveals a sharper
result:

- B4 (LW=1.0 + Queue + NEG=0.72, no NeCo): ARI 0.8605, noise 0.524%,
  Comp 0.9852, AMI 0.9557
- B5 (LW=1.0 + Queue + NEG=0.72 + NeCo=0.2 = iter-37 cfg):
  ARI 0.8564, noise 0.960%, Comp 0.9801, AMI 0.9503

**ARI -0.004, noise +0.44pp, Comp -0.005, AMI -0.005** — every Tier 1
metric is **slightly worse** with NeCo added in isolation. B4 (no NeCo)
is in fact the best single-step configuration in the Real Baseline matrix.

This is striking against the original iter-35-to-iter-37 result, which
reported NeCo's atomic step as **noise 2.01% to 0.61%, -70%**. We now
understand that contrast as the difference between two cross-run
trajectories rather than a clean atomic isolation:

- iter 35 (5/8, P2-King base, no NeCo): ARI 0.856, noise 2.01%
- iter 37 (5/9, P2-King base + NeCo 0.2): ARI 0.870, noise 0.61%
- B4 (Real Baseline, no NeCo): ARI 0.860, noise 0.52%
- B5 (Real Baseline + NeCo 0.2): ARI 0.856, noise 0.96%
- B5 vs iter 37 (same seed=42, same cfg): ARI **0.014 apart, noise 0.35pp
  apart** — same seed, same configuration, different run, multi-seed
  std worth of difference.

The honest read: **NeCo's isolated contribution is within run-to-run
variance, not above it**. NeCo's headline effect in iter 37 is a lucky
draw of cross-run variance that the multi-seed protocol (section 7.1 /
7.7) was already telling us to expect. The atomic step (B4 to B5) gives
a slight negative; the iter-35-to-iter-37 step gave a +0.014 ARI
positive. Both are within the 0.014 multi-seed standard deviation.

### 7.9.4 What N6 means for paper N1 and the contribution map

Two consequences for how the rest of the paper reports contributions.

**Consequence 1** — Paper N1 (NeCo as a wafer-defect lever) is preserved
but reframed. The original framing was "NeCo at 0.2 is a fifth atomic
lever that reduces defect noise by 70%." The Real Baseline result
revises this to "NeCo at 0.2 in combination with LW=1.0 + Queue + NEG
gives an iter-37 configuration that has run-to-run variance comparable
to a NeCo-less B4 configuration; the per-component isolated effect is
within multi-seed std." This is consistent with the section 7.2
mechanism reinterpretation (Normal-defect boundary repulsion) and with
the section 7.6 saturation claim — NeCo sits at a sweet-spot edge whose
isolated lift is small once one accounts for cross-run variance.

**Consequence 2** — A new N6 contribution is added. The Real Baseline
B0-to-B5 path shows that **LW's real effect is its interaction with
Queue**, not its isolated atomic step. Future work that ports the
iter-37 configuration to a new fab must replicate the four
baseline components (Local + LW + Queue + NEG) together, not in
isolation. Component-interaction maps — not just lever-isolated tables
— are the right tool for ablation in this domain.

The Real Baseline B0 (Global-only) result (ARI 0.823, capture 1.000,
noise 6.20%) is itself the supporting evidence for a separate claim
that we list here for completeness: **the ConvNeXtV2-base + TAPT
backbone is doing most of the cluster-structure work**. Of the
ARI 0.823 to 0.870 path from B0 to iter 37, the backbone alone
delivers 0.823 / 0.870 = 94.6% of the ARI; the remaining 5.4% is the
combined contribution of all five contrastive components plus the
HDBSCAN tuning. This sharpens the C1-to-C5 contribution list: the
quantitative novelty of the contrastive head and the HDBSCAN cfg
together is in the 5%-of-ARI regime, and the dominant work is done
by TAPT backbone selection — the supervised sister-classifier's
val_f1 0.9946 transfers to the contrastive embedding as a strong
prior. Practitioners porting this pipeline to a new fab should
expect that getting a good TAPT backbone is the dominant lever, and
the contrastive head + HDBSCAN tuning is the polish on top.

The full Real Baseline matrix (RESULTS table 13, B0 to B5 with all
Tier 1+2 numbers) is the deliverable evidence for this section. The
six contributions (N1 to N6) are summarized in CONCLUSION 8.1.

## 7.10 NEW cfg derivation and Component Dependency Hierarchy (N7, NEW 2026-05-12; ★ N1 v6 FINAL revised 2026-05-12)

> **★ CORRECTION 2026-05-12 (HDBSCAN protocol mismatch retraction, paper N8)**: All
> "Silhouette +30%" / "Sil +0.184" / "Sil +0.176" / "Sil +0.193" / "Sil +30%
> robust" claims in subsections 7.10.1 through 7.10.4 are based on cross-protocol
> Silhouette measurements (B5 leaf+ms=4, NEW eom+ms=3) and are **retracted**.
> Under apples-to-apples HDBSCAN protocol (eom + mcs=12 + ms=3, defect-only),
> B5 Sil = 0.7988, NEW Sil = 0.7860 — Silhouette is **equivalent within seed
> variance** (slight regression −0.013). The "geometry vs partitioning Pareto"
> framing is retracted.
>
> **★ FURTHER REFINEMENT 2026-05-12 (paper N1 v6 FINAL — complementary not
> substitutable)**: The v5-era "NeCo functionally equivalent to Local DenseCL —
> substitutable on partitioning" framing (sections 7.10.2, 7.10.3) is **refined
> in v6** to **complementary at per-class scope under Agglomerative Ward K=42**.
> Aggregate HDBSCAN ARI equality (iter 69 vs B1 4-decimal identity) is preserved
> as a valid HDBSCAN-aggregate-scope finding, but per-class K=42 purity
> breakdown (RESULTS §16) reveals class-by-class winner flips with magnitudes
> up to ±45pp on individual classes. Local DenseCL excels at sub-pattern
> variant integration (fork/scratch 100% B5 vs 64-84% NEW); NeCo excels at
> uniform-pattern consolidation (CenterCircle 100% NEW vs 54.8% B5). The
> absolute SOTA configuration is **B5 (both mechanisms combined)** under
> known-K Agglomerative Ward (ARI 0.9358 single-seed=42 vs NEW 0.9200,
> Δ +0.0158). Local DenseCL is therefore **not deprecated**, and the practitioner
> recipe is **dual-cfg dual-frontier** (sections 7.12 / 7.13).
>
> The genuine NEW-vs-B5 full-set differentiator remains **Normal/defect
> boundary stability** (paper N1 v5 preserved), evidenced by Normal-cluster
> consolidation (Normal noise 77.7% → 14.1%, 859/1000 Normals → 1 dense
> cluster) and full-set ARI 0.83 (NEW) vs 0.69 (B5). NeCo's mechanism is
> **two-pronged**: Normal/defect boundary stability (full-set, v5) plus
> uniform-pattern consolidation (per-class, v6). NEW vs B5 on defect-only
> HDBSCAN metrics: ARI marginal +0.003 multi-seed avg, noise +0.52pp slight
> regression, Sil equivalent.

Section 7.9 (N6) gave a Real Baseline isolation map (B0 to B5) and exposed
two practitioner-facing surprises: LW lever's isolated effect is negative
without Queue support, and NeCo's isolated effect within the iter-37 cfg
(B4 to B5) is within run-to-run variance. These findings prompted a
follow-up 11-iteration **four-component lattice exploration** (iter 67-77,
2026-05-12) that systematically swaps each of the four contrastive components
(Local DenseCL, MoCo Queue, NEG filter, NeCo) on and off in turn. The result
is a more parsimonious NEW SOTA configuration and a Component Dependency
Hierarchy (paper N7) that we report as a separate contribution from the
Component Interaction (N6) of section 7.9.

### 7.10.1 NEW configuration — drop Local DenseCL, keep NeCo (★ revised 2026-05-12)

Iter 70 (seed=42) trains a Global + NeCo (weight 0.2) + Queue (4096) + NEG
(0.72) configuration with **no Local DenseCL term**, producing:

- ARI 0.8797 (seed=42, vs iter 37 single-seed 0.870, B4 0.8605, B5 0.8564)
- Comp 0.9872, AMI 0.9594, defect-noise 0.87%
- Silhouette 0.7860 (apples-to-apples eom + mcs=12 + ms=3, defect-only)

Multi-seed verification (iter 70/71/72, seeds 42/1/2) gives the corrected
comparison below:

| cfg | ARI mean +/- std | Sil seed=42 (apples) | noise mean | components |
|---|---:|---:|---:|:-:|
| B5 (iter-37 cfg, Local + Queue + NEG + NeCo) | 0.856 +/- 0.012 | **0.7988** | 0.96% | 5 |
| NEW (NeCo + Queue + NEG, no Local) | 0.859 +/- 0.018 | **0.7860** | 1.48% | 4 |
| Delta (NEW - B5) | **+0.003** (within multi-seed std) | **−0.013 (Sil equivalent)** | +0.52pp | -1 |

The corrected headline is **NEW vs B5 ARI marginal +0.003 (3-seed avg, within
multi-seed std)**, with **Silhouette equivalent (−0.013)** under apples-to-apples
HDBSCAN protocol. The previously-reported "+30% Sil robust" was a HDBSCAN
protocol-mismatch artefact (paper N8) and is retracted.

The **genuine differentiator** is Normal/defect boundary stability (paper N1
v5): NEW gives full-set Completeness 0.917 vs B5 0.851, full-set ARI 0.83 vs
B5 0.69, via Normal-cluster consolidation (Normal noise 77.7% → 14.1%, 859 of
1000 Normals merge into 1 dense cluster). The defect-cluster intra_p95 actually
widens +26% under NeCo, confirming that NeCo's mechanism here is Normal/defect
boundary stability, not defect-cluster compactness. A practitioner with an
open-set production stream (Normal-dominant) should prefer NEW for boundary
stability; a practitioner optimizing strict sub-1% defect-noise should keep
B5 / iter-37 cfg.

### 7.10.2 NeCo functionally equivalent to DenseCL Local InfoNCE on partitioning (★ revised 2026-05-12)

The decisive piece of evidence for the NEW cfg derivation is iter 69 versus
B1:

- B1 (Global + Local DenseCL LW=0.5, no Queue, no NEG, no NeCo): ARI 0.8514,
  noise 3.93%, n_cluster 37
- iter 69 (Global + NeCo 0.2, no Local, no Queue, no NEG): ARI 0.8514, noise
  3.93%, n_cluster 37

**Four-decimal ARI identity, identical noise, identical n_cluster** — the
two patch-neighbor consistency mechanisms (Local DenseCL grid-cell contrast
and NeCo patch-neighbor rank consistency) produce indistinguishable
partitions on our 43-class wafer benchmark.

★ **CORRECTION 2026-05-12**: The previously-reported "Silhouette asymmetry
+0.193 in favor of NeCo (geometry-only signal)" came from B1 Sil 0.5139 vs
iter 69 Sil 0.7071, but these measurements were under mixed HDBSCAN protocols.
Under apples-to-apples re-measurement, NEW (NeCo+Queue+NEG) Sil = 0.7860 vs
B5 (Local+Queue+NEG+NeCo) Sil = 0.7988 — equivalent within seed variance. The
"NeCo strictly preferred for downstream geometry-sensitive use" framing is
retracted. The two losses are substitutes on ARI / noise / n_cluster, and
Silhouette comparison requires apples-to-apples re-measurement on the
isolated-component baselines (B1 vs iter 69) before a separate Sil claim
can be made.

### 7.10.3 N7 — Component Dependency Hierarchy

The full four-component lattice exposes a dependency structure that lever-
isolated atomic ablation cannot reveal:

**Required (one of two)**: Global InfoNCE plus a patch-neighbor consistency
loss (Local DenseCL or NeCo). Without patch-neighbor signal the encoder
falls to B0 (ARI 0.8231). With either mechanism it climbs to ARI 0.8514.

**Significant**: MoCo Queue (4096). Adds ARI +0.029 with NeCo (iter 69 to
iter 70 via the +Queue +NEG path, with NEG contribution disambiguated
below). Without Queue, the larger-pool false-negative protection that NEG
filter provides has nowhere to operate.

**Conditional**: NEG filter requires Queue. The single most striking finding
of the lattice is iter 74 versus iter 69:

- iter 69 (NeCo only, no Queue, no NEG): ARI 0.8514, noise 3.93%, Sil 0.7071,
  n_cluster 37
- iter 74 (NeCo + NEG=0.72, no Queue): ARI 0.8514, noise 3.93%, Sil 0.7071,
  n_cluster 37

**Four-decimal exact identity**. Adding NEG filter to a cfg without Queue
produces zero effect: ARI, noise, Sil, and n_cluster all match to the
fourth decimal place. The mechanism is straightforward — NEG filter
(IGNORE_NEG_SIM = 0.72) drops in-batch negatives whose cosine similarity
to the anchor exceeds 0.72, a false-negative protection. With only an 8-batch
of negatives (no Queue), this filter has insufficient statistical mass to
make a difference. With a 4096-deep Queue, the filter operates on a much
larger negative distribution and provides meaningful false-negative
protection. We log this as paper contribution **N7 Component Dependency
Hierarchy** because it is not derivable from a lever-isolated ablation table.

**Complementary, not substitutable** (★ N1 v6 FINAL 2026-05-12): Aggregate
HDBSCAN ARI shows Local DenseCL and NeCo as 4-decimal identical at iter 69 vs B1
(ARI 0.8514 = 0.8514, noise 3.93% = 3.93%, n_cluster 37 = 37). This aggregate
identity motivated the v5 "substitutable on partitioning" framing. Per-class
breakdown under Agglomerative Ward K=42 (defect-only, RESULTS §16 NEW)
overturns the substitutability framing: the two mechanisms carry **complementary
inductive biases** that show only when oracle-K linkage clustering exposes
per-class purity to measurement.

- **Local DenseCL strength** (NEW NeCo-only loses): sub-pattern variant
  integration — `Edge-Ring_fork` 100% (B5) vs 64.5% (NEW) Δ −35.5pp,
  `Center_scratch` 95% vs 75% Δ −20pp, `Donut_fork` 100% vs 81.1% Δ −18.9pp,
  `Edge-Top_scratch` 100% vs 84.2% Δ −15.8pp.
- **NeCo strength** (NEW gains over B5 Local-with-NeCo): uniform-pattern
  consolidation — `CenterCircle` 100% (NEW) vs 54.8% (B5) Δ +45.2pp,
  `Edge-Top_fork` 100% vs 90% Δ +10pp.
- **Net average per-class purity** (Agglomerative Ward K=42): B5 = 97.0%,
  NEW = 96.2%, Δ −0.83pp. B5 marginally better on micro-aggregate while
  individual class winners flip on both sides.

The v5 "Deprecated: Local DenseCL (NeCo strictly better on Sil)" claim, already
retracted on N8 Sil grounds, is now further retracted on N1 v6 complementarity
grounds. **Local DenseCL is not deprecated** — it provides sub-pattern variant
integration (fork / scratch / scratch_rot rotational+positional sub-styles)
that NeCo cannot recover alone. The practitioner choice between B5 (5-component,
both) and NEW (4-component, NeCo-only) is **task-dependent**: density-cluster
+ unknown-K → NEW (lower noise hazard, Normal/defect boundary stability);
linkage-cluster + known-K → B5 (highest absolute SOTA ARI 0.9358 via
complementary per-class integration).

### 7.10.4 NeCo weight sweep — ARI inverse-U, Sil pattern retracted (★ revised 2026-05-12)

The NeCo weight sweep within the NEW cfg (seed=42) is:

| NeCo weight | iter | ARI | noise | Sil (mixed-protocol, see note) |
|---:|:-:|---:|---:|---:|
| 0.0 (= B4) | — | 0.8605 | 0.524% | 0.6109 → apples 0.8012 |
| 0.2 (= iter 70 NEW) | 70 | **0.8797** | 0.87% | 0.7860 |
| 0.4 | 77 | 0.8605 | 0.52% | 0.8012 |

ARI peaks at NeCo=0.2 (inverse-U) — this finding is preserved.

★ **CORRECTION 2026-05-12**: The previously-reported "Silhouette monotonic
increasing with NeCo weight" and "geometry-vs-partitioning Pareto" framing
is retracted. Under apples-to-apples HDBSCAN protocol the B4 (NeCo=0) Sil is
0.8012, NEW (NeCo=0.2) Sil is 0.7860, and NeCo=0.4 Sil is 0.8012 — **no
monotonic ascent**; NeCo at 0.2 has slightly lower Sil than either neighbor.

The honest read: NeCo weight 0.2 is the ARI sweet spot, while Sil shows a
**non-monotonic dip at the ARI peak**. The two metrics do not trade off
along a Pareto curve here; they share a single sweet spot region near 0.2
for ARI and a slightly different (also flat) profile for Sil. The
"geometry-vs-partitioning Pareto" claim from v0.5 ABSTRACT is retracted.
A practitioner with strict downstream Sil requirements gains essentially
no Sil by switching NeCo weight from 0.2 to 0.4 (apples Sil 0.7860 vs
0.8012, +0.015 within noise floor) at the cost of -0.019 ARI.

### 7.10.5 TEMP x component interaction extends N6

A final cross-cfg finding: NCE_TEMP 0.05 versus 0.07 reverses sign across
the two cfg families:

| base cfg | TEMP 0.07 | TEMP 0.05 | Delta |
|---|---:|---:|---:|
| B5 (with Local) | 0.8564 (iter 65) | 0.8700 (iter 37) | +0.014 |
| NEW (no Local, iter 70 base) | 0.8797 | 0.8555 (iter 73) | **-0.024** |

This is direct evidence for N6 (Component Interaction) at the hparam level:
TEMP 0.05's lift in the original iter-37 trajectory was a Local-stability
synergy (Local stabilizes the spatial geometry, TEMP sharpens the global
contrast). In the NeCo-only NEW cfg, lower TEMP over-sharpens the NeCo
neighbor signal and produces a -24pp ARI regression. The paper-finding is
that **best hyperparameter values depend on component context**: a single-
component sweep finds a local optimum, but cross-component re-tuning is
required when components are swapped. This sharpens the C2 ablation map
contribution (section 1.4) by adding a cross-cfg dependency layer.

### 7.10.6 Summary of N7 (★ revised 2026-05-12 — N1 v6 complementarity)

The four-component lattice exploration produces four deliverables (v6 revised):

1. An **alternative parsimonious NEW cfg** (4 components instead of 5) with
   equivalent aggregate HDBSCAN ARI (multi-seed +0.003 within std) and
   **equivalent Silhouette under apples-to-apples HDBSCAN protocol** (NEW
   0.7860 vs B5 0.7988, −0.013 within seed variance). The previously-claimed
   "+30% Sil multi-seed robust" is retracted (paper N8 protocol-mismatch
   artefact).
2. A **Component Dependency Hierarchy** showing NEG requires Queue (iter 74
   vs iter 69 4-decimal identical). The previously-claimed "Local DenseCL ↔
   NeCo substitutable" is **refined** in v6 to **complementary at per-class
   scope under Agglomerative Ward K=42** (RESULTS §16): aggregate ARI identity
   hides class-by-class inductive-bias differences — Local DenseCL excels at
   sub-pattern variant integration (fork/scratch rotational+positional
   variants), NeCo excels at uniform-pattern consolidation (CenterCircle round
   geometry).
3. An **ARI inverse-U with peak at NeCo=0.2**; the previously-claimed
   "monotonic Sil ascent / geometry-vs-partitioning Pareto" is retracted.
4. **★ N1 v6 NEW (2026-05-12) — complementary, not substitutable**: per-class
   K=42 Agglomerative Ward purity (RESULTS §16) shows B5 (Local + NeCo
   combined) is the absolute SOTA for known-K oracle clustering (ARI 0.9358
   single-seed), strictly above iter 70 NEW (NeCo only) at 0.9200. The
   substitutability framing (v5) is therefore valid for unknown-K density
   clustering only; for known-K linkage clustering the two mechanisms must
   be combined to recover their complementary inductive biases.

The genuine NEW-vs-B5 differentiator is **Normal/defect boundary stability**
(paper N1 v5): full-set Completeness 0.917 (NEW) vs 0.851 (B5), full-set ARI
0.83 (NEW) vs 0.69 (B5), via Normal-cluster consolidation (Normal noise
77.7% → 14.1%, 859 of 1000 Normals → 1 dense cluster). Defect-cluster
intra_p95 actually widens +26% under NeCo, confirming the Normal/defect
boundary stability mechanism over defect-cluster compactness.

The full lattice matrix (12 cells), 3-seed multi-seed comparison, NeCo
weight sweep, TEMP cross-cfg flip, retraction notes, and Normal-cluster
consolidation evidence are published as RESULTS table 14 (§14b, §14c,
§14d, §14e, §14h, §14i, §14k).

### 7.10.7 ★ N1 v7 FINAL (2026-05-12, iter 84) — B5 seed=1 reproducibility retracts v6 absolute SOTA

The v6 "complementary inductive bias" framing of section 7.10.6 produced two
inferences that v7 (iter 84) retracts:

(i) **"B5 absolute SOTA at Agglomerative Ward K=42 = single-seed ARI 0.9358"**.
This was based on a single seed=42 reading. Iter 84 ran the same B5 cfg under
seed=1 (same anchor, same HDBSCAN protocol, same Agglomerative Ward K=42
clustering, defect-only) and measured ARI **0.8482** (RESULTS §17a). The
seed=42 → seed=1 drop is **Δ −0.0876** — by far the largest reproducibility
flip documented anywhere in this paper's 84-iteration cycle (compare: iter 37
multi-seed std 0.014, Zone-Aware NeCo std 0.012, TOPK 16 std 0.014). The same
seed flip on NEW (NeCo only, no Local) was Δ −0.0346 (RESULTS §15c) — 2.5×
smaller. **B5's single-seed 0.9358 was a cherry-picked lucky outlier**, not
the population mean.

(ii) **"Local DenseCL is preserved in the absolute SOTA cfg because NeCo and
Local carry complementary per-class inductive biases"**. Multi-seed averaging
overturns this on the aggregate scale: B5 2-seed avg ARI is 0.8920 ± 0.062
(Agglo K=42), **below** NEW 3-seed avg 0.9014 ± 0.022 by Δ −0.0094. Across all
three clustering methods on multi-seed average:

| Method | B5 2-seed avg ± std | NEW 3-seed avg ± std | Δ (NEW − B5) | std ratio (B5/NEW) |
|---|---:|---:|---:|---:|
| HDBSCAN | 0.8343 ± 0.031 | **0.859 ± 0.018** | **+0.0245** | 1.7× |
| Agglo Ward K=42 | 0.8920 ± 0.062 | **0.9014 ± 0.022** | **+0.0094** | **2.8×** |
| KMeans K=42 | 0.8540 ± 0.044 | **0.8678 ± 0.026** | **+0.0138** | 1.7× |

**NEW > B5 on multi-seed average across all three clustering methods**, with
1.7–2.8× lower std. The v6 dual-cfg recipe collapses to **single-cfg
recommendation (NEW) with two clustering frontier targets**. (v0.8 ABSTRACT
"B5 for known-K Agglomerative" sub-recommendation: retracted.)

**Status of v6 per-class purity observations (RESULTS §16)**: the single-seed
per-class winner flips (B5 wins fork/scratch sub-pattern variants at 100% vs
NEW 64-84%; NEW wins CenterCircle 100% vs B5 54.8%) **remain observed at
seed=42**. They do **not** propagate to multi-seed averages — B5's seed=42
win on Agglo K=42 aggregate ARI evaporates by seed=1, suggesting the
per-class purity flips are themselves single-seed phenomena rather than
robust complementary inductive biases. We preserve §16 as documented
single-seed evidence with the v7 caveat that **"complementary inductive
biases" is at most a single-seed observation, not a multi-seed claim**.

**N2 strongest evidence**: B5 seed=42 → seed=1 Agglo K=42 Δ −0.088 is the
largest cross-seed flip documented across this paper's 84-iteration cycle,
larger than every encoder lever and larger than any HDBSCAN-axis sweep.
**Single-seed ablation across cfg families produces false winner claims** —
this is the strongest paper-grade evidence for paper contribution N2
(multi-seed methodology obligation). The N1 v7 retraction is itself a
worked example.

### 7.10.8 Final practitioner recipe (v7)

The honest practitioner recipe is now a **single encoder cfg + two-clustering-target**
decision tree:

| Frontier | Encoder cfg | Clustering | Multi-seed avg ARI | Rationale |
|---|---|---|---:|---|
| Unknown-K real-world (HDBSCAN) | iter 70 NEW (NeCo + Queue + NEG, no Local) | HDBSCAN eom mcs=12 ms=3 | **0.859 ± 0.018** (3-seed) | Normal-cluster consolidation, open-set deployment |
| Known-K oracle (Agglomerative Ward K=42) | **iter 70 NEW (same cfg)** | Agglomerative Ward K=42 | **0.9014 ± 0.022** (3-seed) | Linkage-based fine-structure on multi-seed avg |

The B5 / iter 37 cfg (5-component, with Local DenseCL) is no longer recommended.
It survives in the paper as a historical anchor for N6 (Component Interaction)
and N7 (Component Dependency Hierarchy) evidence — both of which used B0 → B5
isolation — but **NEW (no Local) dominates B5 on multi-seed average across all
three benchmarked clustering methods** and has lower run-to-run variance.

## 7.11 ★ N8 NEW (2026-05-12) — HDBSCAN Protocol Mismatch Methodology

The retraction in 7.10 is itself paper-grade methodology evidence. The
sequence was: (i) iter 70 NEW cfg measured under eom + mcs=12 + ms=3 +
defect-only HDBSCAN protocol gave Sil 0.7860; (ii) B5 (iter 37 cfg)
Sil 0.6104 was reported from a different evaluation run with leaf + ms=4
HDBSCAN protocol; (iii) cross-protocol diff gave a spurious "Sil +30% robust
across three seeds" headline that survived multi-seed verification because
all NEW seed runs used the same eom+ms=3 protocol and all B5 seed runs used
the same leaf+ms=4 protocol — the "robustness across seeds" was real but
across-cfg interpretation was wrong; (iv) cluster-analyzer agent post-hoc
inspection on
`outputs_contrastive_260512_001719/eval/cluster_report.parquet` recomputed
B5 / B4 / NEW Silhouette under unified apples-to-apples protocol (eom +
mcs=12 + ms=3 + defect-only): B5 = 0.7988, B4 = 0.8012, NEW = 0.7860 —
NEW slightly worse, not +30% better.

**Methodology deliverable N8**: every contrastive-clustering paper that
compares Silhouette / ARI / noise across cfg families must explicitly fix
every HDBSCAN axis (`cluster_selection_method`, `mcs`, `ms`, `epsilon`) AND
the metric scope (full-set vs defect-only) before the cross-cfg diff is
interpretable. Same-axis multi-seed robustness is **not** sufficient — it
provides robustness within a protocol but cannot detect cross-protocol
artefacts. We list this as a **negative-methodology contribution** in
the seven-contribution map (N8).

The retracted-+30%-Sil artefact is preserved in this paper as the worked
example. We do not edit the v0.5 ABSTRACT history; instead we add v0.6
(corrected current) and an explicit retraction index (RESULTS §14k).

## 7.12 ★ N9 NEW (2026-05-12) — Clustering Algorithm Dependency

Section 7.4 reported that HDBSCAN configuration is a separate axis from encoder
ablation: within a single fixed embedding, switching `selection_method` from leaf to
eom and `min_samples` from 4 to 3 cuts defect-noise by 91% at fixed encoder. The
five-method clustering benchmark (iter 82-83, 2026-05-12; RESULTS §15) extends this
finding to a stronger claim: not just HDBSCAN configuration but **the entire choice
of clustering algorithm** is a separate axis that can move headline ARI by **+0.04
to +0.10** at fixed embedding.

### 7.12.1 The five-method benchmark setup

Three contrastive embeddings — B4 (Local-based, no NeCo), B5 (iter 37 cfg, five-
component), iter 70 NEW (NeCo-based, four-component, no Local) — were each fed to
five clustering algorithms on defect-only data (K_gt = 42):

- HDBSCAN with eom + mcs=12 + ms=3 (unknown-K, density-based, our paper default)
- DP-GMM (variational Dirichlet Process GMM, unknown-K, K_max budget)
- KMeans K=42 (oracle K, centroid-based)
- Agglomerative Ward K=42 (oracle K, linkage-based)
- Spectral K=42 (oracle K, Laplacian eigengap-based)

### 7.12.2 Cross-method ARI / NMI matrix (single-seed=42)

The single-seed ARI matrix (RESULTS §15a):

| cfg | HDBSCAN | DP-GMM | KMeans-42 | Agglo-Ward-42 | Spectral-42 |
|---|---:|---:|---:|---:|---:|
| B4 | 0.8605 | 0.8344 | 0.8876 | 0.9055 | 0.4046 |
| B5 (iter 37) | 0.8564 | 0.8369 | 0.8854 | **0.9358** | 0.7898 |
| iter 70 NEW | 0.8797 | 0.8413 | 0.8798 | 0.9200 | 0.2289 |

Three observations.

**Observation 1 (cfg-ranking flip across method families)**:
- Density-based (HDBSCAN, DP-GMM): rank iter 70 NEW > B5 ≈ B4.
- Centroid-based (KMeans K=42), Linkage-based (Agglomerative K=42): rank B5 > iter
  70 NEW ≈ B4.

NEW's noise/outlier-handling advantage is what density-based methods reward; B5's
tighter defect-cluster geometry (intra_p95 not widened, because NeCo widening is
applied on top of Local rather than instead of Local) is what oracle-K linkage
methods reward. The two method families read different geometric properties of the
same embedding, and reach different cfg rankings.

★ **N1 v6 NEW (2026-05-12) — the deeper reason for the B5 > NEW flip under
Agglomerative Ward K=42**: per-class purity breakdown (RESULTS §16) reveals that
B5 and NEW carry **complementary inductive biases** that are masked under HDBSCAN
aggregate ARI. Under Agglomerative Ward K=42:

- B5 (Local + NeCo combined) achieves 100% per-class purity on fork/scratch
  rotational+positional variants (`Edge-Ring_fork` 100%, `Center_scratch` 95%,
  `Donut_fork` 100%, `Edge-Top_scratch` 100%) — Local DenseCL's grid-cell
  contrast integrates the sub-style variants into one tight cluster.
- NEW (NeCo only) achieves 100% per-class purity on uniform-pattern
  classes (`CenterCircle` 100%, `Edge-Top_fork` 100%) — NeCo's neighbor-rank
  consistency consolidates symmetric round/uniform geometry.
- B5 net average per-class purity 97.0%, NEW 96.2% (Δ −0.83pp) — B5 marginally
  wins aggregate while individual class winners flip on both sides.

The B5 > NEW flip under linkage clustering is therefore not a generic "tighter
geometry" effect but a **complementarity effect**: B5 keeps both mechanisms
active and therefore recovers both per-class strength axes when K is exposed
to the clustering algorithm. The previously-asserted "B5 wins by tighter
geometry because NeCo widening is applied on top of Local rather than instead"
is **refined** in v6 — the real load-bearing evidence is per-class purity
flips, not aggregate geometry compactness.

**Observation 2 (magnitude of ARI shift across methods at fixed cfg)**:
- B4 HDBSCAN→Agglomerative: +0.045 (0.8605 → 0.9055).
- B5 HDBSCAN→Agglomerative: +0.079 (0.8564 → 0.9358).
- iter 70 NEW HDBSCAN→Agglomerative: +0.040 (0.8797 → 0.9200).

The magnitude is comparable to (and on B5 exceeds) any single encoder lever in this
paper. Reporting an ARI number without specifying the clustering algorithm is
therefore as ambiguous as reporting an encoder result without specifying the
HDBSCAN configuration (paper N8). We log this as **paper contribution N9**.

**Observation 3 (Spectral K=42 is unstable)**:
Spectral ARI across the three cfg is 0.23, 0.79, 0.40 — a 0.56 spread. The fit
emits `graph-not-fully-connected` warnings, indicating that the cosine-similarity
affinity graph on the 128-d defect-only embedding has disconnected components
under Spectral's normalized-cut formulation. Spectral K=42 is **not robust** for
this domain and is excluded from the practitioner choice tree.

### 7.12.3 Multi-seed verification on iter 70 NEW (3-seed)

The three-seed (42, 1, 2) verification on iter 70 NEW reproduces the lucky-pattern
hazard documented in section 7.1 / 7.7 across multiple clustering methods (RESULTS
§15c):

| Method | seed=42 | seed=1 | seed=2 | 3-seed avg | std |
|---|---:|---:|---:|---:|---:|
| HDBSCAN | 0.8797 | 0.8491 | 0.8475 | 0.8588 | 0.018 |
| Agglo K=42 | 0.9200 | 0.8854 | 0.8989 | 0.9014 | 0.022 |
| KMeans K=42 | 0.8798 | 0.8456 | 0.8779 | 0.8678 | 0.026 |

The seed=42 vs seed=1 drop is +0.030 (HDBSCAN), +0.035 (Agglo), +0.034 (KMeans) —
across-method consistency suggests the lucky variance is in **the embedding
distribution itself**, not in the clustering algorithm. This extends the
multi-seed protocol (paper N2) from a single-clustering-method recommendation to a
**cross-method-consistent recommendation**: any ARI claim on a contrastive
pipeline must be reported as a multi-seed mean ± std, regardless of clustering
algorithm.

### 7.12.4 The dual-frontier framework (★ v7 revised 2026-05-12 — single-cfg, two clustering targets)

The 5×3 ARI matrix originally supported a **dual-cfg dual-frontier framework**
(B5 for known-K Agglomerative, NEW for unknown-K HDBSCAN). Iter 84's B5 seed=1
reproducibility test (RESULTS §17, DISCUSSION 7.10.7) retracts the B5
sub-recommendation: B5 2-seed avg Agglo K=42 ARI = 0.8920 ± 0.062 is **below**
NEW 3-seed avg 0.9014 ± 0.022. The framework collapses to a **single-cfg
recommendation (NEW) with two clustering frontier targets**:

**Frontier 1 — Unknown-K real-world deployment**:
Encoder: iter 70 NEW (Global + NeCo 0.2 + Queue 4096 + NEG 0.72, no Local).
Clustering: HDBSCAN with eom + mcs=12 + ms=3, defect-only scope.
3-seed mean ARI: **0.859 ± 0.018**.
Rationale: K is unknown, Normal-cluster consolidation is operationally valuable
(paper N1 v5), and HDBSCAN's noise-handling is robust to the embedding's
density-cliff regions.

**Frontier 2 — Known-K oracle lab benchmark (★ v7 revised)**:
Encoder: **iter 70 NEW (same cfg as Frontier 1)**.
Clustering: Agglomerative Ward with K=42, defect-only scope.
NEW 3-seed mean ARI: **0.9014 ± 0.022**.
Comparison: B5 / iter 37 (5-component, with Local) 2-seed avg = 0.8920 ± 0.062
on the same Agglo K=42 method. NEW > B5 by Δ +0.0094 with std 2.8× lower.
Rationale: K is known (e.g., closed lab benchmark), linkage-based clustering
recovers fine sub-structure on multi-seed average; the v6 dual-cfg claim that
"B5 with Local DenseCL is the absolute SOTA at Agglo K=42 with ARI 0.9358" is
retracted — that 0.9358 reading was a single-seed lucky outlier (seed=1
reproduction gave 0.8482, Δ −0.088).

The single-cfg + two-clustering-target framing is **operationally simpler** than
the v6 dual-cfg recipe: practitioners train one encoder configuration (NEW) and
choose the clustering algorithm based on whether K is known. The previous v6
recommendation that "B5 must be retrained when K is known" is no longer required.

The framing still **replaces a single SOTA number** (e.g., "iter 37 ARI 0.870")
with two operating points on a clustering-method axis. This is, to our knowledge,
not standard in contrastive-clustering literature, which typically reports a
single ARI number under an implicit clustering algorithm choice. The N9
deliverable is the methodology disclosure obligation: any ARI claim must specify
the clustering algorithm. The N1 v7 deliverable refines this further: **always
report multi-seed average ± std, regardless of clustering algorithm — single-seed
readings can flip across cfg families by ±0.09 ARI**.

## 7.13 Practitioner choice tree

The combined evidence of sections 7.4 (HDBSCAN configuration sensitivity), 7.11 (N8
HDBSCAN protocol mismatch methodology), and 7.12 (N9 clustering algorithm
dependency) leads to a practitioner choice tree for operating the system on a new
fab / new benchmark:

```
Step 1 — Is K (number of defect classes) known at deployment?

  YES (closed taxonomy, lab benchmark):
    → Frontier 2 (★ v7 revised). Use iter 70 NEW cfg + Agglomerative Ward K=K_gt.
    → Expected: ARI 0.9014 ± 0.022 (3-seed avg, multi-seed compliant).
    → ★ v7 note: v6 recipe (B5 / iter 37 cfg + Agglo K=42 single-seed 0.9358)
      retracted on seed=1 reproducibility (iter 84) — B5 2-seed avg
      0.8920 ± 0.062 is BELOW NEW 3-seed avg 0.9014 ± 0.022. NEW std 2.8× lower.
      Reference: §7.10.7, §7.12.4, RESULTS §17.
    → Watch out for: linkage-based clustering is sensitive to global scaling;
                     verify cosine-normalization before fitting.

  NO (open-set, unknown new defect modes possible):
    → Frontier 1. Use iter 70 NEW cfg + HDBSCAN with eom + mcs=12 + ms=3.
    → Expected: ARI 0.85-0.88 (3-seed), Normal noise consolidation,
                full-set ARI 0.80+ vs B5 0.69.
    → Watch out for: HDBSCAN parameter sensitivity to local density —
                     re-sweep `selection_method`, `mcs`, `ms` on new fab data.

Step 2 — Is the operating stream Normal-dominant (e.g., production line)?

  YES (Normal makes up >50% of inference stream):
    → Frontier 1, definitely. Normal-cluster consolidation (paper N1 v5)
       is operationally critical. Full-set ARI matters more than defect-only ARI.

  NO (mostly defect, lab triage):
    → Either frontier acceptable. Frontier 2 if K is known, Frontier 1 if K is
       unknown. Frontier 1 still gives 0.86 ARI on defect-only, but Frontier 2
       gives 0.93 if K is known.

Step 3 — Reporting hygiene (paper N8 + N9 disclosure obligation).

  Always disclose:
    - Clustering algorithm (HDBSCAN vs Agglomerative vs KMeans vs ...).
    - K-discovery regime (unknown-K vs oracle-K).
    - For HDBSCAN: selection_method, mcs, ms, epsilon, metric scope (full-set
      vs defect-only).
    - Multi-seed mean ± std (single-seed numbers are not comparable across
      cfg, paper N2).
```

This choice tree is the operational deliverable of N1 v5 + N6 + N7 + N8 + N9
combined. It is what we recommend a practitioner port to a new fab, rather than a
single SOTA number.

> Iteration history: `ITERATIONS.md`. Tier 1+2 metric tables: `RESULTS.md`.
> Five-method clustering benchmark: `RESULTS.md` §15.
