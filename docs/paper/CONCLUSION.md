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

**(C6 NEW, 2026-05-11) Component Interaction Matters — a Real Baseline
isolation map**. A six-step Real Baseline ablation (B0 Global-only to
B5 iter-37 cfg) added on top of the 58-iteration original sweep
reveals that **LW lever's isolated atomic step is negative**
(B1 LW=0.5 to B2 LW=1.0: ARI -0.028, noise +2.27pp). Its real
contribution **only materializes through Queue interaction**
(B2 to B3: ARI +0.023, noise -4.89pp). NeCo's isolated effect (B4 to
B5) is **-0.004 ARI**, within same-seed run-to-run variance; B4
(no NeCo) is in fact the best single-step Real Baseline
configuration (ARI 0.8605, Comp 0.9852, noise 0.524%). The B0
Global-only baseline already achieves ARI 0.8231, indicating that
the TAPT ConvNeXtV2 backbone alone delivers 94.6% of the iter-37 ARI;
the contrastive head plus HDBSCAN tuning is a 5%-of-ARI polish on
top. This Component Interaction lesson (paper N6) sharpens the
contribution map and exposes a hazard of lever-isolated atomic
ablation that the original Iter A0 baseline (Local + Queue + NEG
already active) hides. (Section 7.9 / RESULTS table 13 /
ABLATION_PLAN.md)

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
mechanism reinterpretation plus the saturation lock-in plus the
**Real Baseline component-interaction map (N6, 2026-05-11)** are the
**six contributions (N1-N6)** of this work. The **Cluster-Aware
Synthesis Loop** (8.3 F2) is the paradigm-level next step and the
subject of a follow-up paper approved by the project user on 9 May
2026. The negative result for Zone-Aware NeCo (section 7.3) is
preserved as a conditional positive — it could carry signal in a
synthesis-spec-deficient regime that the Cluster-Aware Synthesis
Loop would itself create in early iterations, with the multi-seed
protocol applied at every re-test.

A fourth lesson sits alongside the three above: **Component
Interaction matters more than lever isolation**. The Real Baseline
B0-to-B5 path shows that the LW lever's reported "noise -50%" headline
(Iter A0 to Iter 1) is in fact a conditional improvement that flips
sign without Queue support, and NeCo's headline contribution
("noise -70%") is within same-seed run-to-run variance once isolated.
Future fab-porting work must replicate the four baseline components
(Local + LW + Queue + NEG) together, not in isolation, and should
expect that getting a good TAPT backbone (which delivers 94.6% of the
iter-37 ARI alone) is the dominant practitioner-facing lever.

## 8.6 N7 — Component Dependency Hierarchy and an alternative NEW cfg (NEW, 2026-05-12; ★ revised 2026-05-12)

A fifth lesson, added after the section 7.10 four-component lattice
exploration (iter 67-77, 2026-05-12), is significant but more nuanced than
originally claimed. The iter-37 five-component configuration (Global + Local
DenseCL + MoCo Queue + NEG filter + NeCo) is not the only configuration
producing the headline metrics. The NEW configuration **drops Local DenseCL
entirely**, keeps NeCo as the sole patch-neighbor consistency mechanism, and
produces (★ corrected 2026-05-12):

- 3-seed mean ARI 0.859 +/- 0.018 (vs B5 0.856 +/- 0.012, marginal +0.003
  within multi-seed std)
- Silhouette under apples-to-apples HDBSCAN protocol (eom + mcs=12 + ms=3,
  defect-only): NEW seed=42 0.7860 vs B5 seed=42 0.7988 — **equivalent
  within seed variance (−0.013)**, **not +30% better**
- 4 components instead of 5
- defect-noise +0.52pp slight regression (NEW 1.48% vs B5 0.96%)

The headline single-seed (seed=42) iter-70 reading is ARI 0.8797 with
Silhouette 0.7860. On the multi-seed protocol (section 7.1 / 7.7) the ARI
gain over B5 (+0.003) is within standard deviation and we report ARI as
equivalent.

★ **CORRECTION 2026-05-12 (paper N8 protocol mismatch retraction)**: The
previously-claimed "Silhouette +0.184 = +30% over B5, robust across all
three seeds" headline was the result of a cross-protocol comparison — B5
Silhouette had been measured under leaf + ms=4 HDBSCAN protocol (Sil 0.6104),
while NEW was under eom + ms=3 (Sil 0.7860). Apples-to-apples re-measurement
shows the two are equivalent. The "Sil +30%" claim is retracted.

The **genuine NEW-vs-B5 differentiator** is Normal/defect boundary stability
(paper N1 v5): under full-set clustering (with Normal class), NEW gives
Completeness 0.917 vs B5 0.851 and full-set ARI 0.83 vs B5 0.69, via
Normal-cluster consolidation (Normal noise 77.7% → 14.1%, 859 of 1000
Normals merge into 1 dense cluster). The defect-cluster intra_p95 actually
widens +26% under NeCo — NeCo's wafer-domain mechanism is Normal/defect
boundary stability, not defect-cluster compactness.

### Component Dependency Hierarchy (paper N7, ★ N1 v6 refined 2026-05-12)

The four-component lattice exposes a dependency structure that lever-isolated
ablation cannot detect:

```
Required:      Global InfoNCE + {NeCo alone covers BOTH HDBSCAN (Normal-stream)
                                  AND oracle-K Agglomerative Ward frontiers on
                                  multi-seed avg — single-cfg recommendation}
Significant:   MoCo Queue (+0.029 lift with NeCo, +0.023 with Local LW=1.0)
Conditional:   NEG filter <- requires Queue (no-Queue NEG produces zero effect)
Complementary single-seed only: Local DenseCL ↔ NeCo (★ N1 v6 single-seed
                observation; ★ N1 v7 multi-seed correction) — aggregate ARI
                identical under HDBSCAN; at single-seed=42 per-class K=42
                Agglomerative Ward purity shows complementary inductive
                biases (Local integrates fork/scratch sub-pattern variants,
                NeCo consolidates uniform-pattern classes). Combining both
                (B5) reaches single-seed=42 ARI 0.9358 under linkage clustering
                — ★ v7 retracted: seed=1 reproduce = 0.8482, B5 2-seed avg
                0.8920 ± 0.062 < NEW 3-seed avg 0.9014 ± 0.022. The v5
                "substitutable" framing applies only to aggregate HDBSCAN
                unknown-K ARI; the v6 "complementary" framing applies only
                to single-seed=42 observations.
```

Three pieces of evidence are load-bearing here:

1. **NeCo aggregate-identical to Local DenseCL on HDBSCAN (iter 69 vs B1)**:
   ARI 0.8514 = 0.8514 to four decimals, identical noise (3.93%), identical
   n_cluster (37). Under HDBSCAN unknown-K density clustering, the two
   mechanisms produce indistinguishable aggregate partitions.

2. **★ N1 v6 NEW (2026-05-12) Complementary per-class inductive biases (B5 vs
   NEW under Agglomerative Ward K=42)**: per-GT-class dominant cluster purity
   shows class-by-class winner flips that aggregate ARI hides:
   - B5 (Local + NeCo combined) wins on sub-pattern variant classes:
     `Edge-Ring_fork` 100% (B5) vs 64.5% (NEW), `Center_scratch` 95% vs 75%,
     `Donut_fork` 100% vs 81.1%, `Edge-Top_scratch` 100% vs 84.2%.
   - NEW (NeCo alone) wins on uniform-pattern classes: `CenterCircle` 100%
     (NEW) vs 54.8% (B5), `Edge-Top_fork` 100% vs 90%.
   - Net B5 marginal win at micro-aggregate (97.0% vs 96.2%, Δ −0.83pp).
   - Absolute SOTA: B5 single-seed ARI 0.9358 on Agglomerative Ward K=42,
     strictly above NEW 0.9200 (Δ +0.0158) — the complementarity yields the
     known-K SOTA only when both mechanisms are kept active.
   Local DenseCL is therefore **not deprecated**; the v5 "substitutable"
   framing is refined to "complementary at per-class scope".

3. **NEG filter requires Queue (iter 74 vs iter 69)**: ARI 0.8514 = 0.8514,
   noise 3.93% = 3.93%, Sil 0.7071 = 0.7071, n_cluster 37 = 37 — **all four
   metrics identical to four decimal places**. NEG filter's false-negative
   protection has no statistical mass to operate over when the in-batch
   negatives are 8 wafers only. With a Queue of 4096, NEG provides
   meaningful protection. The dependency is binary, not gradient.

### Practitioner consequences (★ revised 2026-05-12 — N1 v6 complementary)

For practitioners porting the contrastive head to a new fab, the eight
contribution map (N1-N8) leads to a more nuanced recipe than iter 37 alone:

1. **NeCo and Local DenseCL are aggregate-substitutable on HDBSCAN, single-seed
   complementary on per-class Agglo K=42** (N7 (i), ★ N1 v6 single-seed
   observation; ★ N1 v7 multi-seed correction). Aggregate HDBSCAN ARI shows
   the two mechanisms as identical (iter 69 vs B1 4-decimal equality). On
   single-seed=42 Agglomerative Ward K=42, per-class purity shows complementary
   inductive biases — Local DenseCL integrates fork/scratch sub-pattern variants
   (4 classes go 100% B5 vs 64-84% NEW), NeCo consolidates uniform-pattern
   classes (CenterCircle 100% NEW vs 54.8% B5), and B5 reaches single-seed
   ARI 0.9358 vs NEW 0.9200. **However, on multi-seed average across all three
   benchmarked clustering methods (HDBSCAN, Agglo Ward K=42, KMeans K=42),
   NEW (NeCo only, no Local) > B5 (Local + NeCo)**: HDBSCAN +0.0245, Agglo
   +0.0094, KMeans +0.0138, with NEW std 1.7-2.8× lower than B5. The v6
   "B5 absolute SOTA at known-K Agglo ARI 0.9358" claim is retracted in v7
   (single-seed cherry-picked outlier; seed=1 reproduce = 0.8482, Δ −0.088).
   Local DenseCL is **operationally optional, not required for SOTA** —
   use NEW for both frontiers (RESULTS §17, v7 FINAL). The v6 per-class
   purity flips are preserved as single-seed observations only. The
   previously-claimed "Adopt NeCo, drop Local DenseCL — strictly superior
   on Sil" is retracted (paper N8 protocol mismatch); v5 "substitutable"
   refined to "single-seed complementary, multi-seed redundant".
2. **Keep MoCo Queue** at size 4096 (N7 Significant). It carries the largest
   isolated lift across both NEW and B5 paths.
3. **NEG filter at 0.72 is conditional** on Queue presence (N7 Conditional).
   It buys noise floor at the cost of ARI; effect is zero without Queue.
4. **NeCo weight 0.2** is the ARI sweet spot (inverse-U with peak at 0.2).
   The previously-claimed "Sil monotonic-increasing → push to 0.4 for
   geometry" is retracted; under apples-to-apples protocol Sil is not
   monotonic and there is no meaningful Sil gain from weight 0.4.
5. **Re-tune TEMP after component swap** (N6 extension): TEMP 0.05 is
   correct on the Local-based B5 path, TEMP 0.07 is correct on the NEW
   no-Local path. Single-component sweeps find local optima; cross-cfg
   transfers require re-sweep.
6. **★ NEW (N8)**: when reporting cross-cfg Silhouette / ARI / noise
   comparisons, **explicitly fix the HDBSCAN configuration AND the metric
   scope (full-set vs defect-only)**. Multi-seed robustness within a
   fixed protocol does not detect cross-protocol artefacts. Our own
   v0.5 abstract was retracted as the worked example.

### Eight contributions summarized (★ revised 2026-05-12)

The cycle now produces eight paper contributions:

- **(N1 v5 final) 43-class open-set wafer benchmark + NeCo mechanism on this
  domain = Normal/defect boundary stability**, NOT defect-cluster compactness.
  NeCo's effect on defect-only metrics is functionally equivalent to DenseCL
  (ARI ±0.003 multi-seed avg, Sil ±0.013 apples). Its real gain channel is
  Normal-cluster consolidation (Normal noise 77.7% → 14.1%, 859 of 1000
  Normals merge into 1 dense cluster, full-set Completeness 0.917 vs 0.851,
  full-set ARI 0.83 vs 0.69). The defect-cluster intra_p95 actually widens
  +26% under NeCo.
- **(N2) Multi-seed honesty across two cfg families**: iter-37 B5 at
  0.866 +/- 0.014, NEW at 0.859 +/- 0.018; single-seed +0.010 lucky
  variance reproduces on three independent axes.
- **(N3) Five atomic encoder levers + three HDBSCAN-side axes + 14 dead axes**
  — a complete ablation map. HDBSCAN-side configuration alone reduces
  defect noise -91% at fixed encoder.
- **(N4) NeCo mechanism reinterpreted** as Normal-defect boundary repulsion
  (Normal-cluster consolidation evidence above).
- **(N5) Comprehensive saturation point** at iter 37 across six hparam axes
  and two Spatial-NeCo variants.
- **(N6) Component Interaction Matters** — Real Baseline isolation B0 to B5
  reveals LW's isolated atomic step is negative (-0.028 ARI), and its real
  effect is interaction with Queue (+0.023 ARI in B2 to B3).
- **(N7) Component Dependency Hierarchy (★ N1 v7 FINAL 2026-05-12)** —
  NeCo and DenseCL are aggregate-identical on HDBSCAN (ARI/noise/n_cl
  4-decimal at iter 69 vs B1). At per-class single-seed scope under
  Agglomerative Ward K=42, the two mechanisms show complementary winners
  (B5 wins fork/scratch sub-pattern variants at 100% vs NEW 64-84% on 4
  classes; NEW wins CenterCircle 100% vs B5 54.8%) — preserved as
  single-seed observation (RESULTS §16). **However, on multi-seed average
  ARI, NEW (NeCo only, no Local) > B5 (Local + NeCo) on all three
  clustering methods**: HDBSCAN +0.0245, Agglomerative Ward K=42 +0.0094,
  KMeans +0.0138, with B5 std 2.8× higher than NEW on Agglo (RESULTS §17b).
  The v6 "B5 is the absolute SOTA at known-K Agglo Ward K=42 with ARI
  0.9358" claim is retracted in v7 — single-seed=42 0.9358 was a
  cherry-picked outlier (seed=1 reproduces at 0.8482, Δ −0.088).
  NEG requires Queue (iter 74 vs iter 69 4-decimal identical). TEMP flips
  sign across cfg families. **Practitioner recipe: single-cfg (NEW) +
  two-clustering-target** (HDBSCAN for unknown-K, Agglomerative Ward K=42
  for known-K). Local DenseCL is operationally optional, not required for
  SOTA. (Previously-asserted "+30% Sil multi-seed robust" retracted on N8;
  "Local DenseCL substitutable" refined to "complementary single-seed
  only" on N1 v6; "B5 absolute SOTA on Agglo K=42 ARI 0.9358" retracted
  on N1 v7 / N2 reproducibility evidence.)
- **★ (N8 NEW) HDBSCAN Protocol Mismatch Methodology** — cross-cfg
  Silhouette / ARI / noise comparisons require explicit unification of
  `cluster_selection_method`, `mcs`, `ms`, `epsilon`, and metric scope
  (full-set vs defect-only). Multi-seed robustness within a fixed protocol
  does not detect cross-protocol artefacts. Our v0.5 ABSTRACT "+30% Sil
  robust" was a worked example of the artefact and is retracted in v0.6,
  preserved as documented methodology evidence.

Future work (8.3 F1 / F2 / F3) is unchanged. The Cluster-Aware Synthesis
Loop (F2) remains the paradigm-level next step; lever stacking (F1) gains
a new candidate (NeCo weight push to 0.4 for geometry-sensitive deployments,
section 8.6 practitioner consequence 4); production deployment (F3) gains
a NEW-cfg vs B5-cfg per-downstream selection rubric.

## 8.7 N9 — Clustering Algorithm Dependency and the dual-frontier framework (NEW, 2026-05-12)

The Real Baseline ablation (N6, §8.6 / DISCUSSION 7.9), the four-component lattice
(N7, §8.6 / DISCUSSION 7.10), and the HDBSCAN protocol mismatch retraction (N8,
§8.6 / DISCUSSION 7.11) all assumed a single clustering algorithm (HDBSCAN with
eom + mcs=12 + ms=3, defect-only). A five-method clustering benchmark (iter 82-83,
2026-05-12; RESULTS §15) measures B4, B5, and iter 70 NEW under HDBSCAN, DP-GMM,
KMeans K=42, Agglomerative Ward K=42, and Spectral K=42 on defect-only scope.

The findings (DISCUSSION 7.12):

1. **Cfg ranking flips across clustering method families**. Density-based methods
   (HDBSCAN, DP-GMM) rank iter 70 NEW > B5; centroid/linkage-based methods with
   oracle K (KMeans, Agglomerative) rank B5 > iter 70 NEW.

2. **ARI magnitude shifts +0.04 to +0.10 across methods at fixed embedding**. On
   B5, ARI moves from 0.8564 (HDBSCAN) to 0.9358 (Agglomerative Ward K=42), a
   +0.079 lift purely from clustering algorithm choice.

3. **Spectral K=42 is unstable** (ARI 0.23-0.79 spread across cfg families with
   graph-not-fully-connected warnings) and is excluded from the recommendation.

This leads to a **dual-frontier framework** for SOTA reporting (DISCUSSION 7.12.4):

- **Unknown-K real-world frontier (HDBSCAN + iter 70 NEW)**: 3-seed mean ARI
  0.859 ± 0.018, with Normal-cluster consolidation (paper N1 v5) and full-set
  ARI 0.83 vs B5 0.69. Recommended for open-set production deployment with
  unknown new defect modes.
- **Known-K oracle benchmark frontier (Agglomerative Ward K=42 + iter 70 NEW)
  ★ v7 revised**: NEW 3-seed mean ARI **0.9014 ± 0.022** (multi-seed
  authoritative). B5 / iter 37 cfg 2-seed avg = 0.8920 ± 0.062 (BELOW NEW,
  std 2.8× higher; B5 seed=42 0.9358 was cherry-picked, seed=1 = 0.8482,
  Δ −0.0876). Recommended encoder cfg: **same as Frontier 1 (iter 70 NEW)**.
  Local DenseCL operationally optional. NMI 0.9704 single-seed=42 only;
  multi-seed NMI not measured. Reference: §8.8, RESULTS §17.

The dual-frontier framework replaces a single SOTA number with two operating
points indexed by the K-discovery regime. This is, to our knowledge, not standard
in the contrastive-clustering literature, which typically reports a single ARI
number under an implicit clustering algorithm choice. The N9 deliverable is the
methodology disclosure obligation: **any ARI claim on a contrastive-cluster
pipeline must specify the clustering algorithm and the K-discovery regime**.
HDBSCAN-only papers may have headline numbers that an oracle-K Agglomerative
implementation would push 0.04 to 0.10 higher on the same embedding — or vice
versa for cfg combinations that linkage methods penalize.

The practitioner choice tree (DISCUSSION 7.13) is the operational deliverable of
N1 v5 + N6 + N7 + N8 + N9 combined. Step 1 asks whether K is known; Step 2 asks
whether the operating stream is Normal-dominant; Step 3 enumerates the
multi-seed and protocol disclosure requirements. We recommend porting the choice
tree to a new fab rather than a single SOTA hyperparameter row.

## 8.8 Closing remark — when to use which (★ revised 2026-05-12 — N1 v7 FINAL single-cfg recommendation)

The 84-iteration cycle plus the Real Baseline B0-B5 ablation plus the
four-component lattice plus the five-method clustering benchmark plus the
per-class Agglomerative Ward K=42 breakdown plus the B5 seed=1 reproducibility
test (iter 84) together exhaust the encoder-side, component-composition-side,
HDBSCAN-side, clustering-algorithm-side, per-class-purity-side, and
cross-cfg multi-seed reproducibility-side ablation spaces under our
atomic-change policy. Nine contributions (N1 v7 final through N9) emerge.

The honest practitioner-facing answer to "what is the SOTA?" is no longer a
single ARI number. It is a **single encoder cfg with two clustering targets**:

| Regime | Encoder cfg | Clustering | Multi-seed ARI ± std | Why |
|---|---|---|---:|---|
| Real-world, unknown-K, open-set | iter 70 NEW (4-component, no Local) | HDBSCAN eom mcs=12 ms=3 | **0.859 ± 0.018** (3-seed) | Normal/defect boundary stability (N1 v5), Normal noise 77.7% → 14.1%, open-set deployment |
| Lab benchmark, known-K, oracle | **iter 70 NEW (same cfg)** | Agglomerative Ward K=42 | **0.9014 ± 0.022** (3-seed) | Linkage-based fine-structure recovery on multi-seed avg |

★ **The N1 v7 final correction (2026-05-12, iter 84)** retracts the v6 recipe
that recommended **B5 (Local + NeCo combined)** for the known-K Agglomerative
Ward row. The seed=42 reading of B5 Agglomerative Ward K=42 = ARI 0.9358 was a
cherry-picked single-seed outlier — seed=1 reproduction of the **same cfg**
under the **same protocol** gave ARI 0.8482, a Δ −0.0876 drop. Across all three
benchmarked clustering methods (HDBSCAN, Agglo Ward K=42, KMeans K=42), **NEW
(NeCo only, no Local) > B5 (Local + NeCo combined) on multi-seed average**:
HDBSCAN +0.0245, Agglo +0.0094, KMeans +0.0138, with B5 std 1.7–2.8× higher
than NEW. The v6 "complementary inductive bias" per-class purity observations
(CenterCircle 100% NEW vs 54.8% B5; Edge-Ring_fork 100% B5 vs 64.5% NEW) are
preserved as single-seed observations in RESULTS §16 but **do not propagate
to multi-seed averages**. Local DenseCL is therefore **operationally optional,
not required for SOTA**; the dual-cfg recipe collapses to **single-cfg
recommendation (NEW)** with two clustering frontier targets.

The earlier headline "iter 37 = ARI 0.870 SOTA" survives only as a single-seed
single-clustering-algorithm reading. The multi-seed mean (0.866 ± 0.014) is the
honest headline on HDBSCAN, and the **dual-frontier framework** with **dual-cfg
recipe** is the honest reporting discipline. Future fab deployment should
choose between the two frontiers based on operational K-discovery regime and
Normal-stream proportion, not based on which row produces the largest decimal
number.

We list five forward-looking lessons for the contrastive-clustering literature in
this domain (consolidating across all sections):

1. **HDBSCAN configuration matters as much as encoder hyperparameters** (7.4).
2. **Single-seed reading is unreliable** (7.1 / 7.7 / N2).
3. **iter 37 is a multi-axis saturation point**, not a single SOTA hyperparameter
   combination (7.6 / N5). Further encoder gains require a paradigm shift
   (Cluster-Aware Synthesis Loop, F2).
4. **Component Interaction matters more than lever isolation** (7.9 / 7.10 /
   N6 / N7). Real Baseline ablation and component-lattice mapping are required.
5. **Clustering algorithm choice matters more than several encoder levers**
   (7.12 / N9). Dual-frontier reporting and explicit algorithm disclosure are
   required.

The Cluster-Aware Synthesis Loop (8.3 F2) remains the paradigm-level next step.
With saturation (7.6 / N5) plus component-lattice exhaustion (7.10 / N7) plus
clustering-algorithm exhaustion (7.12 / N9) plus protocol-mismatch hygiene
(7.11 / N8) plus Component Interaction discipline (7.9 / N6) all locked in, the
remaining axis for further-significant improvement is the synthesis specification
itself.

> Iteration history: `ITERATIONS.md` (append-only, 77 iterations + 82-83 clustering
> benchmark through 2026-05-12). Tier 1+2 metrics: `RESULTS.md` (now §15 5-method
> matrix). Decision history: `docs/contrastive-eval/DECISIONS.md`.
