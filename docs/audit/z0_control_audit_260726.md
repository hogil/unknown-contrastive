# z0 (random-head) control audit — 260726

Read-only audit. No result files modified or deleted. No GPU used — missing z0
measurements are recorded as unmeasured with the exact command to fill them in
later, not computed here.

## Trigger

Anchor pool, paired-equivalence cell, dial mcs6/ms3/leaf/eps0.06:

| arm | P1 | noise% | Comp | Hom | ARI |
|---|---|--:|--:|--:|--:|
| frozen | 30/37 | 32.25 | 0.9435 | 0.9663 | 0.8867 |
| z0 (random head) | 32/37 | 35.23 | 0.9544 | 0.9841 | 0.9351 |

z0 beats frozen by ARI +0.048, Hom +0.018, P1 +2 classes — all outside the
measured noise band (ARI ±0.019, Hom ±0.005, P1 spread 0, from the clean546
4-cell repeat, see below). The sign is pool-dependent: on severstal frozen
beats z0 (ARI 0.331 vs 0.228); on anchor z0 wins. z0 is not a negligible null,
and "beats frozen" does not by itself demonstrate that training helped.

## Noise band used below (measured, clean546 4-cell repeat, mcs6/ms3/leaf)

Source: `docs/archive/root_results_260726/_crossds_leaderboard.md` lines 548-557
(t20/n78/n82/n86, same TEMP0.20·NEG0.72·QUEUE16384·SEED42 config, 4 independent
runs/epoch-selections).

| metric | spread |
|---|---|
| P1 | 0 (exact) |
| noise% (P2) | ±2.28pp |
| ARI | ±0.019 |
| Hom (P4) | ±0.005 |
| Comp (P3) | ±0.033 |

A claim only "survives" if the adapted-vs-z0 delta exceeds this band in the
favorable direction; a delta inside the band is a **tie**, not a win.

## Claim-by-claim

| # | claim | pool | frozen | z0 | adapted | z0 control exists? | beats z0? |
|---|---|---|---|---|---|---|---|
| 1 | clean546 champion P1 7/7 | clean546 (mcs6/ms3/leaf) | single 6/7 noise63.4 Comp.816 Hom.829 ARI.757 | single (n=10) mean noise48.2 Comp.879 Hom.802 ARI.714; ensemble (n=10) mean noise48.8 Comp.872 Hom.789 ARI.708 | single P1 7/7 noise28.6 Comp.765 Hom.855 ARI.692; **ensemble** P1 7/7 noise37.2 Comp.838 **Hom.923** ARI.858 | **YES** (measured 260726) | **MIXED** — noise & Hom robust wins (ensemble, outside full z0 range), ARI 90th-pct lean win, **Comp robust/lean loss**, P1 not differentiated (z0 hits 7/7 in 1/10 draws) |
| 2 | severstal adapt ep17 (Rule C validation) | severstal_pilot260726 (mcs6/ms3/leaf) | P1 4/4, noise 77.7, Comp 0.416, Hom 0.876, ARI 0.331 | P1 4/4, noise 79.8, Comp 0.382, Hom 0.822, ARI 0.228 | P1 4/4, noise 69.4, Comp 0.364, Hom 0.771, ARI 0.226 | **YES** | **MIXED** (see below) |
| 3 | 무라벨 grouping 성공 (260724): P1 4/7→5.7/7, noise 60→29 | clean546, same champion track as #1 | not reported | n/a | superseded by the current 7/7/7 champion (claim 1) | **SUPERSEDED** | claim 1 audits the state that replaced this number |
| 4 | cycle-1 운영점 (ens[s42+s1]+soft-reassign conf≥0.90): noise 29.1→5.1, ARI 0.692→0.707, P4 0.855→0.890 | clean546, built directly on #1's champion embeddings | n/a (post-hoc reassignment on top of champion, not vs frozen) | pre-reassign: noise 48.8±4.2 (claim1 ensemble); **post-reassign(5 seeds): noise 5.13-6.23 (mean 5.71)**, P4 0.755-0.894, ARI 0.528-0.768 | k16/7, P1 7/7, noise 5.1, Comp 0.757, Hom 0.890, ARI 0.707 | **YES** (measured 260726) | **MIXED, primary axis effectively ties z0** — champion's post-reassign noise (5.1) sits at the z0-ensemble's own floor (5.13); the reassignment mechanism, not the encoder, does most of the work. Hom/ARI ~80th pct (modest real edge) |
| 5 | temporal FAR champion arm (`fcmae_ad1_t010_s1_ep4`): champion FAR=0/4 vs frozen FAR=5/4 at P10/K1 | temporal novelty sim (result_grouping/temporal_novelty_260726/) | FAR(P10/25pct/K1)=1.25/batch | FAR(P10/25pct/K1)=**3.75-4.0/batch (mean of 3 seeds, worse than frozen)** | FAR(P10/25pct/K1)=**0/batch** | **YES** (measured 260726) | **STANDS — strengthened.** z0 is worse than frozen here, not a plausible alternative explanation for champion reaching FAR 0 |

### Claim 1 — clean546 champion P1 7/7 (flagship)

Searched every clean546-track result artifact: `runs/clean546/ablation_summary.json`,
all 97 `runs/clean546/eval_*.json` files that use `_grouping_eval.py` output
format, `_score_may_repro.py`, and the full cycle-1/2/3 sections of
`docs/archive/root_results_260726/_crossds_leaderboard.md` (lines 436-632,
including the 16-cell sweep and matched-capacity oracle tables). None of them
contain a `random_z0`/`z0` row. `_score_may_repro.py` — the score script
recovered alongside the trainer earlier today — has no z0/random logic in it
at all (confirmed via grep, 0 matches).

This is a **systemic gap for the entire clean546 track**, not just the single
champion number: none of the 16-cell temp/queue/NEG sweep, the matched-capacity
oracle table, or the deliverable v2 export (`result_grouping/deliverable_clean546_v2/`)
has ever been compared against a random head. Every clean546 P1/noise/Comp/Hom/ARI
number in this repo inherits this gap.

**Contrast with the anchor-pool trigger finding**: `_grouping_eval.py` (used for
anchor and severstal) computes z0 automatically and always includes it in its
output — this is why claims 2 has a z0 control and claims 1/3/4/5 do not: they
were scored with a different tool (`_score_may_repro.py` for the training-time
in-line HDBSCAN dump, or ad hoc scripts for temporal FAR) that was never wired
to include the z0 arm.

**MEASURED (260726, follow-up).** Two rounds, since the first round exposed a
capacity mismatch team-lead caught: the flagship deliverable is a **2-head
concat+L2 ensemble** (`abl_sw_t20_B4_260724_102757/proj_ep20.pt` +
`abl_best_s1_B4_260724_111053/proj_ep18.pt`, per `grouping_deploy.py`'s
champion default), not the single ep20 head. Comparing a 2-head ensemble
against a 1-head random control confounds exactly the effect under test (much
of z0's apparent effect is suspected to be the geometry of dim-change+BN+L2).
So both a single-head and a capacity-matched 2-head-ensemble z0 control were
run, each over 10 seeds (not 3 — the first 3-seed pass showed spreads up to
33x the noise band, too unstable for point comparisons). All CPU-only,
mcs6/ms3/leaf/eps0.06 (matches the original claim's dial exactly), clean546
pool. Verdict is now **percentile-of-champion-within-the-z0-distribution**,
not a single-draw comparison.

| metric | frozen | z0 single-head (n=10, mean±sd, range) | champion single (ep20) | champion percentile in z0-single | z0 ensemble (n=10, mean±sd, range) | champion ensemble (measured) | champion percentile in z0-ensemble |
|---|--:|---|--:|--:|---|--:|--:|
| noise% (lower=better) | 63.4 | 48.2±6.1 [39.4, 61.0] | 28.6 | **below full range — robust win** | 48.8±4.2 [40.5, 55.0] | 37.2 | **below full range — robust win** |
| Comp (P3) | 0.816 | 0.879±0.050 [0.799, 0.955] | 0.765 | **below full range — robust LOSS** (0th pct) | 0.872±0.042 [0.803, 0.939] | 0.838 | 30th pct — loses to majority, not robust |
| Hom (P4) | 0.829 | 0.802±0.057 [0.735, 0.901] | 0.855 | 70th pct — lean win, inside range | 0.789±0.054 [0.724, 0.870] | 0.923 | **above full range — robust win** |
| ARI | 0.757 | 0.714±0.084 [0.598, 0.840] | 0.692 | 60th pct — roughly average, inside range | 0.708±0.099 [0.586, 0.878] | 0.858 | 90th pct — strong lean win, inside range |
| P1 | 6/7 | mostly 5/7-6/7, hits 7/7 in 1/10 draws | 7/7 | matches z0's rare ceiling, exceeds its mode/median | mostly 5/7-6/7, hits 7/7 in 1/10 draws | 7/7 | matches z0-ensemble's rare ceiling |

**Capacity-matching changes the verdict materially, in the flagship's favor**:
at the single-head level Hom and ARI looked ambiguous/seed-flipping and Comp
looked like a robust loss; once compared against the *correct* (capacity-matched)
ensemble control, **Hom flips to a robust win** (champion 0.923 exceeds every
one of the 10 z0-ensemble draws) and **ARI strengthens to a 90th-percentile
lean win**. Comp stays unfavorable at both capacities (majority of z0 draws
beat champion), though it softens from "loses to all 10" (single) to "loses to
7/10" (ensemble).

**Restated claim** (supersedes the single-head-only version below it does not
delete): contrastive adaptation, measured against a capacity-matched random
ensemble, **robustly reduces noise** (~20pp below the entire z0-ensemble
range) **and robustly improves Homogeneity** (above the entire z0-ensemble
range). ARI is a strong lean-win (90th percentile) but not fully robust.
**Completeness is not established** — champion sits in roughly the bottom
third of the z0-ensemble distribution. **P1="7/7" is not on its own evidence
of learning** — an untrained random ensemble reaches 7/7 in 1 of 10 draws;
what is a discriminating result is that champion reaches it *every* time,
where z0 only does so by chance ~10% of the time.

**Separate finding worth tracking on its own**: a bare, untrained random
projection head beats *frozen* by ~14pp of noise on this pool (63.4 → ~48-49)
with zero training — team-lead flagged this as a candidate "free win, no
training needed" for deployment. It is pool-dependent (on severstal frozen
beats z0 instead, 77.7 vs 79.8), so it must be re-measured per pool, not
assumed.

Reproduction:
```
python _grouping_eval.py --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \
  --pool data/pools/mwm38_clean546.json --proj-dir runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints \
  --mcs 6 --ms 3 --method leaf --eps 0.06 --tag clean546_z0check
```
(single-head z0 only, seed 42; the 10-seed single/ensemble sweep above was run
via a scratch script that imports `_grouping_eval.py`'s functions unmodified —
`build_proj`, `load_backbone`, `extract_backbone_features`,
`embedding_from_features`, `label_free`, `offline` — looping the seed instead
of the tool's hardcoded `torch.manual_seed(42)`; not committed to the repo,
available on request.)

### Claim 2 — severstal adapt ep17

Per-metric verdict against the band above:

| metric | adapted | z0 | Δ (adapted − z0) | band | verdict |
|---|--:|--:|--:|--:|---|
| P1 | 4/4 | 4/4 | 0 | 0 | tie (both at ceiling — pool is 4-class, non-discriminating) |
| noise% (P2, lower=better) | 69.4 | 79.8 | −10.4pp | ±2.28pp | **beats z0** |
| Comp (P3) | 0.364 | 0.382 | −0.018 | ±0.033 | tie (inside band, adapted nominally lower) |
| Hom (P4) | 0.771 | 0.822 | −0.051 | ±0.005 | **loses to z0**, clearly outside band |
| ARI | 0.226 | 0.228 | −0.002 | ±0.019 | tie |

**Restated claim**: adaptation (ep17, Rule C-selected) reduces noise% far
beyond what an untrained random head achieves (this is the metric the Rule C
external-validation claim was actually about, and it survives). It does **not**
demonstrably improve Completeness or ARI over a random head — both are ties —
and it is **measurably worse than a random head on Homogeneity**, well outside
the noise band. The original severstal leaderboard entry's P1-gate-then-P2
framing is unaffected by this (P1 is tied at ceiling either way, so the
existing P2-noise-led verdict logic already used the metric that survives),
but any future write-up that cites Comp/Hom/ARI gains for this run needs to
drop that framing — those axes do not distinguish this run from an untrained
random projection.

**OPEN ITEM (not resolved here, flagged per team-lead 260726)**: this z0
comparison was run at **mcs6/ms3**, and that dial has since been shown
(`runs/severstal/dial_sweep/REPORT.md`) to be geometrically wrong for the
severstal pool (n/k ≈ 199, mcs6 ≈ 3.0% of n/k vs the ≈10% band that works
elsewhere; the correct dial is closer to mcs20/ms5). The frozen-vs-adapted
severstal conclusion has already been shown to flip under the correct dial
(adapted wins on every axis at mcs20/ms5 instead of losing on Comp/Hom/frag at
mcs6). **The z0 comparison above has not been re-run at mcs20/ms5 and should
not be treated as final** — it is left as-is (MIXED, per the table above) as
a placeholder pending that re-run, not because it is confirmed correct.

### Claim 3 — 무라벨 grouping 성공 (260724)

**SUPERSEDED (team-lead determination, 260726).** The "P1 5.7/7, noise≈29"
figure is a pre-optimization-loop snapshot of the same s42/s1/s2 champion
training track — the later "finer HDBSCAN mcs6/ms3/leaf" optimization loop
replaced it, and all three seeds now individually reach 7/7 (the exact
checkpoints claim 1 audits: `abl_sw_t20_B4_260724_102757/proj_ep20.pt`,
`abl_best_s1_B4_260724_111053/proj_ep18.pt`,
`abl_best_s2_B4_260724_111604/proj_ep17.pt`). Since claim 1 already audits the
current/final state that superseded this number, no separate z0 measurement
is needed for claim 3 — closing it as superseded rather than leaving it
UNVERIFIED indefinitely. (The exact pre-optimization checkpoint set was not
identified in the leaderboard archive during this audit; if that historical
number needs its own z0 check for some other reason, it would need the same
`_grouping_eval.py --mcs 6 --ms 3` treatment claim 1 used, once the run_dir is
located.)

### Claim 4 — cycle-1 operating point (soft-reassign + ensemble)

Built directly on top of the (z0-unverified) champion embeddings from claim 1,
so it inherits that gap. There is an additional wrinkle worth flagging: this
claim's mechanism is a **post-hoc reassignment procedure**, not additional
training. A fair z0 comparison therefore needs two separate checks, not one:

1. Bare z0 embeddings scored at mcs6/ms3/leaf (same as claim 1's remediation) —
   establishes whether the champion's base row (k18/7, noise 29.1, P1 7/7)
   itself beats a random head.
2. The **same** ensemble-concat + soft-reassign(conf≥0.90) pipeline run on
   z0 embeddings instead of champion embeddings — because it is plausible that
   the reassignment mechanism (nearest-centroid confidence gating) recovers
   noise regardless of whether the underlying embedding was trained, in which
   case the 29.1→5.1 gain would be a property of the reassignment step, not
   of the encoder.

**MEASURED (260726, follow-up, check 2 of 2).** Check 1 (bare z0-ensemble at
mcs6/ms3, no reassignment) is covered by claim 1's ensemble row above — same
embeddings, same pool, champion ensemble base noise=37.2 vs z0-ensemble base
noise=48.8±4.2 (robust win, below full z0 range), so the *pre-reassignment*
starting point is a genuine encoder effect. Check 2 (does the reassignment
mechanism itself, run on z0 instead of champion, reproduce the same recovery)
is the one that matters for this specific claim, and the answer is
uncomfortable: **mostly yes.**

Ran `scripts/predict_grouping_prod.py::reassign_noise_to_nearest_cluster`
(imported unmodified, the exact function `grouping_deploy.py` uses for the
deployed operating point) with `mode="nearest_q90"` on 5 z0-ensemble seeds:

| arm | seed noise (pre-reassign) | noise after conf>=0.90 | P1 | Comp | Hom | ARI |
|---|--:|--:|---|--:|--:|--:|
| champion ens[s42+s1] | 37.2 | **5.1** | 7/7 | 0.757 | 0.890 | 0.707 |
| z0-ensemble (5 seeds) | 56.4/61.0/61.7/50.4/51.6 (mean 56.2) | **5.13, 6.23, 6.23, 5.13, 5.32 (mean 5.71, sd 0.45)** | 7/7,5/7,7/7,6/7,6/7 | 0.649-0.814 | 0.755-0.894 | 0.528-0.768 |

**Champion's post-reassign noise (5.1) sits essentially at the z0-ensemble's
own minimum (5.13), not meaningfully below it** — a margin of 0.03 percentage
points against a starting gap (pre-reassign) of ~19pp (37.2 vs 56.2). The
soft-reassign confidence-gating mechanism recovers noise from an untrained
random ensemble almost as effectively as from the trained champion. **The
noise 29.1(or 37.2)->5.1% drop this operating point is famous for is mostly a
property of the reassignment post-processing step, not of what the encoder
learned.** P4(Hom) and ARI after reassignment are less ambiguous but still not
robust wins: champion (0.890, 0.707) sits near the top of the z0-ensemble
range (0.755-0.894 and 0.528-0.768 respectively — roughly 80th percentile on
both, one seed exceeds champion on each), so there is a real but modest
encoder contribution to purity/agreement after reassignment, just not to the
noise number the operating point's headline is built around.

**Restated claim**: the pre-reassignment encoder gap (claim 1's ensemble
result: champion base noise 37.2 vs z0-ensemble base ~48.8, robust win) is
real. But the specific, widely-cited "noise 29->5.1%" cycle-1 number is not
mostly attributable to that encoder gap — an untrained random ensemble run
through the identical reassignment pipeline lands within a fraction of a
point of the same result. This claim should not be cited as "the trained
encoder achieves 5.1% noise" without that caveat; it is closer to "the
reassignment heuristic achieves ~5-6% noise regardless of whether the
embedding was trained, and the trained encoder adds a modest purity/ARI edge
on top of that floor.

### Claim 5 — temporal FAR champion arm

`result_grouping/temporal_novelty_260726/` contains only `f0_frozen.npy` and
`f_champion.npy` (confirmed via directory listing) — no z0 embedding file, and
`scripts/run_temporal_novelty_embeddings.py` has no z0/random logic (0 grep
matches). The FAR/lag comparison (champion FAR 0/4 vs frozen FAR 5/4 at P10/K1)
has never been checked against a random head. Note this claim already carries
its own documented scope caveat (in-domain adaptation, not zero-shot — see the
leaderboard's "★★ 범위 한계" note above the FAR table) — the z0 gap is a
second, independent limitation on top of that one.

**MEASURED (260726, follow-up).** Champion here is a single residual adapter
(`f_champion = f0 + gamma * adapter(f0)`, not an ensemble), so capacity
already matched with no fix needed: z0 = same architecture
(Linear(1024,128)->GELU->Linear(128,1024)) with **freshly random-initialized
weights** and the **same gamma magnitude** as the real trained adapter (only
the residual *direction* is randomized, not its scale) — isolating "does an
untrained residual of the same size do this" from "does a bigger residual
help". 3 seeds; unlike claim 1's static clean546 measurement, all 3 seeds
landed within ~1 alarm of each other (FAR aggregates over many
batches/thresholds, which averages out single-draw noise much more than a
one-shot static clustering does). Recomputed frozen and champion on the
current `run_arm()` schema too (the original `temporal_novelty_report.json`
predates a schema restructure, so it couldn't be compared to the new z0 output
directly) — read-only reads of `f0_frozen.npy`/`f_champion.npy`, output to a
new file, nothing existing touched.

FAR at P=10 (the documented operating point), alarms per 4 held-out background
batches, at the two m_min floors that matter (m_min_pct=25 is the originally
reported "baseline" operating point; m_min_pct=0 is the loosest/most
permissive floor):

| arm | m_min_pct=0, K=1 | K=2 | K=3 | m_min_pct=25, K=1 | K=2 | K=3 |
|---|--:|--:|--:|--:|--:|--:|
| frozen | 23 (5.75/batch) | 10 (2.50) | 4 (1.00) | 5 (**1.25/batch**) | 3 (0.75) | 0 |
| z0 (mean of 3 seeds) | ~21 (5.25/batch) | ~11.3 (2.83) | ~3.7 (0.92) | ~15 (**3.75-4.0/batch**) | 8 (2.00) | 3 (0.75) |
| champion | 9 (2.25/batch) | 5 (1.25) | 3 (0.75) | 0 (**0/batch**) | 0 | 0 |

**z0 does not explain away this claim — if anything it strengthens it.** At
the documented operating point (P10/m_min-25th-pct/K1), z0's false-alarm rate
(3.75-4.0/batch) is *worse than frozen's* (1.25/batch), and champion is the
only arm that reaches 0. Unlike claims 1/3/4, a random projection is not a
plausible alternative explanation for champion's FAR advantage here — an
untrained random residual of the same magnitude produces *more* false alarms
than doing nothing (frozen), not fewer. This is the one originally-unverified
claim where filling in the z0 control supports rather than weakens the
original result. The claim's other documented scope limits (in-domain
adaptation, not zero-shot; FAR is computed once per arm and shared across the
size-variant rows, not independently re-derived per variant) still apply
unchanged.

## Summary

Final disposition, as confirmed by team-lead (260726):

```
MIXED      3  #1 flagship (capacity-matched: noise/Hom robust wins, ARI 90th-pct, Comp minority loss)
              #2 severstal (noise wins / Hom loses -- but needs re-evaluation after the dial
                 correction, see open item above)
              #4 cycle-1  (pre-reassign gap is real; post-reassign headline number ties z0)
STANDS     1  #5 temporal FAR (z0 is worse than frozen -- the one claim z0 supports)
SUPERSEDED 1  #3 무라벨 5.7/7 (replaced by the optimization loop; final state audited in #1)
RETRACTED  0
```

| verdict | count | claims |
|---|---|---|
| SUPERSEDED (no longer the current claim, folded into #1) | 1 | #3 |
| MIXED (z0 control exists, per-metric split) | 3 | #1 (flagship, capacity-matched ensemble), #2 (severstal — dial re-check pending, see open item), #4 (cycle-1 reassign — primary axis effectively ties z0) |
| STANDS / strengthened (z0 does not explain the result away) | 1 | #5 (temporal FAR — z0 is worse than frozen here) |
| Retracted (loses to z0 on the claim's own headline axis) | 0 | — (none loses outright, but #4's headline number is a near-tie, the closest to this bucket) |

**Audit complete: all 5 originally-listed claims now have a final disposition**
(2 MIXED-but-net-positive, 1 MIXED-but-net-negative-on-its-headline-number
(#4), 1 STANDS, 1 SUPERSEDED). No claim needs further z0 measurement unless
new headline numbers are produced.

Follow-up work (260726, same day): claim 1's first pass used a capacity
mismatch (single random head vs the flagship's actual 2-head ensemble) and
only 3 z0 seeds (spread up to 33x the noise band on Hom/ARI — too unstable for
point comparisons). Re-run with a capacity-matched 2-head z0-ensemble and 10
seeds, judged by percentile-within-distribution rather than point comparison,
changed the verdict materially: Hom flipped from "loses to z0" to "robust
win," ARI strengthened from "tie" to "90th-percentile lean win," and only
Comp remains unfavorable. Claim 5 (temporal FAR) was also filled in and is the
one claim where the z0 control *supports* the original result. Claims 3/4
remain genuinely unmeasured (same clean546 tooling gap as claim 1, not yet
re-run) and are correctly still UNVERIFIED, not retracted.

**Headline finding (updated after the capacity-matched follow-up)**: the
z0 gap was real but the news is mixed, not uniformly bad. Where it has now
been measured, adaptation genuinely beats a matched random control on noise
(claims 1, 2, 5, robustly) and, once capacity is matched correctly, on
Homogeneity too (claim 1). It does NOT beat z0 on Completeness (claims 1, 2)
and the flagship's "P1 7/7" headline is not, by itself, evidence of learning
(z0 reaches it too, just less reliably). Claims 3 and 4 are still completely
unmeasured — same tooling gap as claim 1 before its own re-run — and should
not be cited without the same treatment. The corrected methodology (capacity-
matched ensemble, >=10 seeds, percentile-in-distribution rather than a single
point) should be the standard for any future z0 check, not the original
single-seed/single-head pass this audit started with.


## Addendum — deployment-default question (team-lead, 260726)

**Superseded by its own follow-up below — read the "corrected" section, not
the first pass.** Left both in place to show why the correction mattered.

### First pass (WRONG comparison point — kept for the record)

Comparing the two champion arms already computed for claim 1 raised a
separate, actionable question: `grouping_deploy.py`'s champion default is the
**2-head ensemble**, but under this project's own stated priority order
(P1 > P2 noise > P3 Comp > P4 Hom), does the ensemble actually win?

Checked *before reassignment* (CPU, single script run, same pool/dial by
construction — clean546, mcs6/ms3/leaf/eps0.06; both arms' raw numbers
independently matched the archived leaderboard values):

| arm | k | noise% (pre-reassign) | P1 | captured defect classes | Comp | Hom | ARI |
|---|--:|--:|---|---|--:|--:|--:|
| single (ep20) | 18 | 29.12 | 7/7 | {C, D, EL, ER, L, NF, S} | 0.765 | 0.855 | 0.692 |
| ensemble (ep20 + s1_ep18) | 16 | 37.18 | 7/7 | {C, D, EL, ER, L, NF, S} — **identical set** | 0.838 | 0.923 | 0.858 |

P1 tied exactly (same 7 classes), noise gap 8.06pp (outside the 2.28pp band)
→ concluded "single head wins the tie-break." **This was the wrong operating
point.** `grouping_deploy.py`'s actual champion path applies
`--reassign nearest_q90`, not the bare pre-reassign clustering — and claim 4
(above) had just shown that reassignment collapses noise gaps of this size.
Team-lead caught this before it became a default-changing decision: "네가
방금 발견한 원리를 우리 자신의 비교에 적용하지 않으면 같은 실수를 반복하는
것이다" (if you don't apply the principle you just found to our own
comparison, we repeat the same mistake).

### Corrected comparison — at the actual deployment operating point (post-reassign, conf>=0.90)

Same two checkpoints, same pool/dial, `reassign_noise_to_nearest_cluster`
(imported unmodified) applied identically to both arms — this is what
`grouping_deploy.py` actually runs. z0 (single and ensemble) included in the
same table for reference.

| arm | seed_noise (pre) | **noise (post-reassign)** | P1 | captured classes | Comp | Hom | ARI |
|---|--:|--:|---|---|--:|--:|--:|
| single (ep20) | 29.12 | **2.93** | 7/7 | {C,D,EL,ER,L,NF,S} | 0.703 | 0.845 | 0.623 |
| **ensemble (deployed default)** | 37.18 | **3.85** | 7/7 | {C,D,EL,ER,L,NF,S} (identical) | **0.742** | **0.885** | **0.697** |
| z0-single (3 seeds) | 44.3-47.3 | 4.58-4.76 | 5-6/7 | subset, missing EL and/or S | 0.837-0.872 | 0.727-0.835 | 0.601-0.722 |
| z0-ensemble (5 seeds) | 50.4-61.7 | 5.13-6.23 | 5-7/7 | mostly missing 1 class | 0.649-0.814 | 0.755-0.894 | 0.528-0.768 |

- **The noise gap that decided the first pass is gone**: post-reassign, single
  vs ensemble is 2.93 vs 3.85 — a 0.92pp gap, *inside* the 2.28pp band. P2
  now ties.
- P1 already tied exactly. With P1 and P2 both tied, the lexicographic
  comparison falls through to **P3 (Comp) and P4 (Hom)**, where the ensemble
  wins clearly: Comp 0.742 vs 0.703 (+0.039), Hom 0.885 vs 0.845 (+0.040), and
  ARI 0.697 vs 0.623 (+0.074) agrees.
- **Verdict: at the deployment operating point, the ensemble wins the
  lexicographic comparison. `grouping_deploy.py`'s current default (ensemble)
  is correct — no change warranted.** The first-pass conclusion is retracted;
  it was measured at the wrong point in the pipeline.
- Both real-champion arms still beat the full z0 range on noise even after
  reassignment (single 2.93, ensemble 3.85, both below every z0 draw's
  post-reassign noise of 5.13+), so this correction only changes the
  single-vs-ensemble question, not the champion-vs-z0 question already
  settled in claim 1/4.

Not done: each arm's own best-fitting HDBSCAN dial (lower priority than this
correction per team-lead; clean546 mcs6 already independently confirmed
geometrically appropriate for this pool, unlike severstal/v2).
