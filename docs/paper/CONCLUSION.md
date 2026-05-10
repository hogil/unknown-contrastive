# Conclusion

## 8.1 Summary of contributions

We presented a self-supervised contrastive learning + density-based
clustering pipeline for open-set wafer defect grouping, hardened over
49 atomic-change ablation iterations against four primary objectives
(P1 class capture, P2 defect-noise rate, P3 Completeness, P4
Homogeneity / AMI). On the 43-class new-anchor benchmark
(`avg30_new_260508_123037`, 2,146 wafers — 42 defect classes plus
Normal), the iter-37 configuration achieves:

- **Completeness 0.991** (P3, Rosenberg-Hirschberg 2007)
- **AMI 0.960** (Vinh 2010, chance-corrected)
- **ARI 0.870** single-seed; **3-seed mean 0.866 plus or minus 0.014**
- **defect-noise 0.61%** (P2, HDBSCAN noise restricted to defect
  subpopulation)
- **class capture 1.000** (P1, every defect class produces at least one
  cluster)

The system uses a frozen ConvNeXtV2-base FCMAE backbone with task-adaptive
pre-training from the sister-repo supervised classifier, a 128-dimension
projection head trained on InfoNCE plus DenseCL-style local grid contrast
plus NeCo patch-neighbor consistency, and HDBSCAN clustering with `eom`
selection method and `min_samples=3`. Training ran for 5 epochs at
batch 8, image size 384, on a single GPU.

Four contributions correspond to the four content sections:

**(C1)** A 43-class open-set wafer benchmark synthesized from WM-811K
distribution priors plus five chip-internal object types plus nine
wafer-canvas patterns, with documented sub-style splits. (Section 4 / 
DATASET.md)

**(C2)** Five atomic levers — LOCAL_WEIGHT, LR_HEAD, IGNORE_NEG_SIM,
NCE_TEMP, NECO_WEIGHT — and nine dead axes including permanent
rejection of partial backbone unfreeze. The lever-and-dead-axis map
serves as a practitioner guide. (Section 6 / RESULTS table 8 and
table 10)

**(C3)** A domain-specific mechanism reinterpretation: NeCo at weight
0.2 acts as Normal-defect boundary repulsion (per-class centroid shifts
plus 0.05 to plus 0.07 cosine units away from Normal, while inter-defect
centroid distances stay within plus or minus 0.01), recovering 54
wafers from a Normal supercluster. This is a sharper claim than the
original NeCo paper's generic compactness framing.
(Section 7.2 / DISCUSSION.md)

**(C4)** Multi-seed honesty: the single-seed iter-37 ARI 0.870 is a
high-tail draw; the 3-seed mean is 0.866 plus or minus 0.014.
Zone-Aware NeCo (iter 43) gives a single-seed best of 0.880 but a
3-seed mean of 0.876 plus or minus 0.012, which is not statistically
distinguishable from iter 37 — reported as an honest negative.
This claim is strengthened in iter 54/55 where a second independent
axis (LOCAL_POS_TOPK=16) reproduces the same +0.010 seed=42 lucky
tail (single-seed 0.880, seed-1 0.852, 2-seed mean 0.866 — identical
to Zone-Aware NeCo z=4 to the third decimal place). (Section 7.1 /
7.3 / 7.7 / DISCUSSION.md)

**(C5)** A **comprehensive multi-axis saturation point**: 6
hyperparameter axes (LOCAL_WEIGHT, LR_HEAD, IGNORE_NEG_SIM, NCE_TEMP,
LOCAL_POS_TOPK, QUEUE_SIZE) plus two Spatial-NeCo variants
(Hierarchical NeCo at "1,2,4" pools, Zone-Aware NeCo at z=3) all
sweep to within the iter-37 multi-seed standard deviation
(0.014 ARI). The iter-37 configuration is therefore not just a SOTA
single point but a **multi-axis sweet-spot** that is robust within
plus or minus 20% on every hyperparameter axis. We publish the
saturation map (RESULTS table 11, ITERATIONS iter 50-58 entries) as
the practitioner's "do not bother sweeping these" guide and as the
empirical justification for the Cluster-Aware Synthesis Loop being
the natural next paradigm rather than another encoder-side sweep.
(Section 7.6 / DISCUSSION.md)

## 8.2 The five-lever ablation map (one-line summary)

| Lever | Step | Tier-1 effect (atomic) | Iter |
|---|---|---|---|
| 1 LOCAL_WEIGHT | 0.5 -> 1.0 | defect noise 9.34% -> 4.62% (-50%) | Iter 1 (old) |
| 2 LR_HEAD | 1e-3 -> 5e-4 | Completeness 0.83 -> 0.948 (plus 12pp) | Iter 11 (old) |
| 3 IGNORE_NEG_SIM | 0.72 -> 0.65 | sister-class separation, conditional on base cfg | Iter 13 (old) |
| 4 NCE_TEMP | 0.07 -> 0.05 (old) / hold 0.07 (new) | AMI / ARI / Comp / Hom plus 0.3pp | Iter 14 (old) |
| 5 NECO_WEIGHT | 0 -> 0.2 | defect noise 2.01% -> 0.61% (-70%) | iter 37 (new) |

Plus three HDBSCAN-side axes (separate track):

| HDBSCAN axis | Step | Effect at fixed encoder |
|---|---|---|
| `cluster_selection_method` | leaf -> eom | defect noise -66% |
| `min_samples` | 4 -> 3 | defect noise -50% additional |
| `min_cluster_size` | 8/10/12 | equivalent at ms=3, lock at 12 |

Combined HDBSCAN configuration alone (leaf-default to eom-mcs12-ms3) at
fixed encoder reduces defect noise from approximately 12.6% to 0.61% —
a 91% reduction. The HDBSCAN axis dwarfs four of the five encoder
levers in absolute percentage points. Practitioner consequence: report
encoder-side claims under a fixed HDBSCAN configuration, otherwise the
attribution is ambiguous.

## 8.3 Future work

Three directions, ordered by paradigm distance from iter 37:

**(F1) Lever stacking — code-level extensions**.
Two existing public methods plug into the iter-37 architecture without
disturbing the levers and could compound the iter-37 result:
**ProNC** (arXiv:2505.24254) ETF prototype geometry, and **DACL**
(arXiv:2412.19871) density-aware re-weighting. **Mean Teacher / EMA
teacher** (arXiv:2411.18533) is a third candidate. Each could be
introduced as a new atomic-step lever (e.g., `PRONC_WEIGHT`,
`DACL_BETA`, `EMA_TEACHER_TAU`) under the same single-change-per-iter
policy.

**(F2) Cluster-Aware Synthesis Loop**.
Current pipeline: synthesize wafers (fixed spec) -> train contrastive
encoder -> HDBSCAN cluster -> evaluate. The cluster-analyzer agent
already emits a per-class weakness diagnosis at iter end. We propose
an auto-patcher that translates the weakness diagnosis into a
synthesis-spec delta (JSON patch) and applies it to a small fraction
of synthesis at the next iter. Concrete first targets identified by
iter-37 cluster-analyzer: (i) Thick-Edge_fork chip-internal ring
annulus enrichment to disambiguate from Full_fork; (ii) sister-class
rotation pair (`scratch` vs. `scratch_rot`) per-position rotation-axis
variance enrichment so that distributional positions (Donut, Center)
get the same separation as primary positions (Edge-Bottom, Edge-Top).
This closes the synthesis-encoder-cluster loop without a human in the
loop; to our knowledge this paradigm has not been reported in
wafer-defect contrastive literature. The closest analog (Iterative
Cluster Harvesting, arXiv:2404.15436) keeps a human label step in the
loop.

**(F3) Production deployment**.
The iter-37 configuration is amenable to daily inference via the
existing `predict_contrastive_daily.py` entry. Two production-side
tasks are flagged: (i) per-day HDBSCAN
re-clustering versus `approximate_predict` against a frozen iter-37
clustering — empirical comparison required, recall the configuration
sensitivity discussed in section 7.4; (ii) drift detection via a
running Wang-Isola alignment + uniformity monitor (METHOD section 8) —
out-of-distribution wafer batches should show alignment metric
degradation before they show cluster-membership change.

## 8.4 Open problem — head-only contrastive sister-class entanglement

One residual issue is exposed by the iter-50-58 saturation map. The
iter-37 configuration achieves 3-seed ARI 0.866 plus or minus 0.014,
not 0.95+, even though Completeness (0.991) and AMI (0.960) are near
ceiling. The gap is concentrated in **sister-class entanglement** —
specifically the rotation pairs (`scratch` vs. `scratch_rot`,
`Donut_scratch` vs. `Donut_scratch_rot`) at distributional positions
(Donut, Center) where the rotation signal is masked by the position
prior. Six hyperparameter axes plus two Spatial-NeCo variants do not
move ARI past 0.880, and no single seed reproducibly clears 0.880.
This suggests the residual gap is not in the hyperparameter space but
in **what a frozen-backbone head-only contrastive head can geometrically
represent** under the current 128-d projection: a rotation-axis
distinction that is a 21-degree rotation only carries weak signal
through a frozen ConvNeXt backbone whose features were learned for
rotation-augmented supervised classification (sister repo, val_f1
0.9946 with rotation augmentation).

We log this as an open problem because the obvious fix — partial
backbone unfreeze — was tested twice (iter 36, iter 42) and rejected
on different LR_SCALE values, and TTA / rotation augmentation are
forbidden by domain policy (rotation is identity-bearing in our
classes). The Cluster-Aware Synthesis Loop is one path: increase
per-position rotation-axis variance at synthesis time so the frozen
backbone has more rotation signal to align. Lever stacking (ProNC
ETF prototypes, DACL density-aware re-weight) is another: change the
geometry of the projection-head output so rotation-distinct sister
classes occupy distinct ETF slots even with weak backbone signal.
Both are listed in 8.3 as F1 / F2 future work.

## 8.5 Closing remark

The 58-iteration cycle exhausts hyperparameter-level encoder ablation
under our atomic-change policy. Three empirical lessons survive
across the cycle and we expect them to transfer across fabs.

First, **HDBSCAN configuration matters as much as encoder
hyperparameters** (section 7.4): in our setup the HDBSCAN axis carries
about 91% of the defect-noise improvement at fixed encoder. Reporting
encoder ablation under a fixed HDBSCAN configuration is necessary for
clean attribution.

Second, **single-seed reading is unreliable in this domain** (section
7.1 / 7.7): the iter-37 ARI 0.870 is a 3-seed best-tail draw; the
mean is 0.866 plus or minus 0.014. The same +0.010 lucky variance
reproduces on Zone-Aware NeCo z=4 and on LOCAL_POS_TOPK=16 — two
completely independent axes with identical seed=42 / seed=1 / mean
triples to the third decimal place. We treat single-seed deltas
smaller than 0.014 ARI as not distinguishable from zero, and we
recommend the 3-seed-or-more protocol for any further claim.

Third, **iter-37 is a multi-axis saturation point, not a single
SOTA hyperparameter combination** (section 7.6): six hyperparameter
axes and two Spatial-NeCo variants all sweep to within multi-seed
standard deviation, establishing iter-37 as a robust sweet spot
(plus or minus 20% on every axis). Further encoder-side improvements
require a paradigm shift, not another sweep.

The five-lever map plus the dead-axis map (now extended to 14 axes
with iter 50-58) plus the multi-seed discipline plus the NeCo
mechanism reinterpretation plus the saturation lock-in are the
five contributions (N1-N5) of this work. The **Cluster-Aware
Synthesis Loop** (8.3 F2) is the paradigm-level next step and the
subject of a follow-up paper approved by the project user on 9 May
2026. The negative result for Zone-Aware NeCo (section 7.3) is
preserved as a conditional positive — it could carry signal in a
synthesis-spec-deficient regime that the Cluster-Aware Synthesis
Loop would itself create in early iterations, with the multi-seed
protocol applied at every re-test.

> Iteration history: `ITERATIONS.md` (append-only, 58 iterations through
> 2026-05-10). Tier 1+2 metrics: `RESULTS.md`. Decision history:
> `docs/contrastive-eval/DECISIONS.md`.
