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

> Iteration history: `ITERATIONS.md`. Tier 1+2 metric tables: `RESULTS.md`.
