# Severstal HDBSCAN dial sweep — diagnostic report

Authored by `perf-dial` (260726). Written to file by team-lead (the agent's harness blocks
report-file writes; content is verbatim from its report plus a team-lead addendum at the end).

**Purpose:** test whether `mcs6/ms3/leaf/eps0.06` — tuned on clean546 (546 imgs / 7 classes) —
mismeasured severstal adaptation (995 imgs / 5 classes).

## Method
No training, no re-embedding. Both arms reuse the saved raw GAP cache (`frozen_raw_gap.npy`
verified byte-identical to `adapt_raw_gap.npy` — same backbone/pool/order, one cache serves both).
Frozen = L2-norm(raw). Adapted = L2-norm(proj_ep17(raw)), `proj_ep17.pt` from
`runs/severstal/may_repro/abl_B4_260726_100437/checkpoints/`. Clustering/metrics = unmodified
`may_hdbscan` / `label_free` / `offline` imported directly from `_grouping_eval.py`.

**Sanity check passed:** the `mcs6/ms3/leaf/eps0.06` cell reproduced the recorded baseline to the
last decimal (frozen ARI 0.3314 / sn 76.68; adapted ARI 0.2259 / sn 66.83).

**Grid:** mcs{6,10,15,20,30,40,50} × ms{3,5,10} × method{leaf,eom} × eps{0.0,0.06}
= 84 dials × 2 arms = **168 cells**. `eps` had zero effect anywhere in this pool — not a live
variable here. No existing files modified or deleted; no defaults changed.

## Q1 — Any dial where adapted beats frozen on Comp+Hom+ARI while keeping its noise lead?
**Yes.** `mcs20/ms5/leaf` and neighbours: adapted wins all three simultaneously and keeps the
seed_noise lead. **Hypothesis H supported.** (22 of 84 dials satisfy the conjunction.)

## Q2 — Winner at each arm's own best (honest) dial?
Raw ARI-max cheats via merging — frozen best-ARI cell is `mcs40/ms10` with ARI 0.712 but **k=2**;
adapted best-ARI is `mcs40/ms3` with ARI 0.883 but **k=3**. Restricting to k∈[4,6] (honest zone,
≈ true class count 5):

| dial | arm | P1 | seed_noise | Comp | Hom | ARI | k |
|---|---|--:|--:|--:|--:|--:|--:|
| mcs20/ms5/leaf | frozen | 2/4 | 70.15 | 0.505 | 0.573 | 0.522 | 4 |
| mcs20/ms5/leaf | **adapted** | **3/4** | **59.50** | **0.859** | **0.758** | **0.840** | **5** |

**Adapted wins qualitatively, not just numerically.** Across all 30 frozen k=4–6 cells, frozen's
P1 **never exceeds 2/4** — it structurally cannot resolve a 3rd defect class into its own cluster
without k ballooning to 7–17, where noise/Comp/Hom all degrade. Adapted reaches P1=3/4 at three
different dials inside the honest zone, and P1=4/4 at six dials once k reaches 15–24. Frozen only
reaches 4/4 at the original baseline (k=17, noise 76.7%) — buying the 4th class at near-total noise.

Note `mcs30/ms3/leaf` gives adapted ARI 0.874 at k=4 but **P1 drops to 2/4** — ARI-only selection
would pick a worse dial by the metric that matters most.

## Q3 — Predictable from pool geometry without labels?  [the important one]

### 3a. Size heuristic `mcs ≈ n/(k·10)`
| pool | n | k (true) | n/k | winning mcs | n/(mcs·k) |
|---|--:|--:|--:|--:|--:|
| severstal | 995 | 5 | 199 | ~20 | 9.95 |
| clean546 | 546 | 7 | 78 | 6 | 13.0 |

Ratio lands in a 10–13 band on both pools — same order of magnitude, not exact. Usable as a coarse
**search radius** (e.g. mcs from n/(15k) to n/(8k)) *if* an engineering estimate of k̂ exists.
**Without any k estimate the rule is unusable** — that is the real limitation for no-label deployment.

### 3b. Unlabeled proxy metrics (Spearman ρ vs ARI)
| metric | frozen (84) | frozen (leaf,42) | adapted (84) | adapted (leaf,42) |
|---|--:|--:|--:|--:|
| stability | +0.51 | +0.32 | +0.41 | +0.21 ns |
| coherence | +0.46 | **−0.42** | +0.48 | **−0.39** |
| Sil (cosine) | +0.22 ns | −0.18 ns | **+0.93** | **+0.80** |
| over_merge==0 gate | ρ=−0.74 (!) | — | +0.16 ns | +0.73 |
| k | −0.07 ns | **−0.96** | +0.57 | −0.30 border |

**No universal rule found.** Two concrete failures:
1. `over_merge==0` does **not** catch frozen's k=2 merge-cheat — the best-ARI-among-gated cell is
   still the degenerate `mcs40/ms10` k=2 cell. The 20%-of-n threshold never fires when noise
   absorbs most points and only two small clean clusters survive. It guards one failure mode
   (giant blob), not this one (over-conservative + lucky purity).
2. For frozen-leaf, **k anti-correlates with ARI almost perfectly (ρ=−0.96)** — selecting frozen's
   dial by max-ARI / min-noise walks straight into the merge-cheat.

`Sil` is a strong lead **only for adapted** (ρ 0.80–0.93), useless or negative for frozen — and you
cannot know in advance which arm you are scoring.

**Deployment constraint: without labels, no rule tested here reliably picks the dial.**
`mcs ≈ n/(10k)` gives a coarse range if k̂ is guessable; beyond that it needs a size-informed sweep
plus a sanity spot-check, not a single unlabeled statistic.

## Q4 — Does the ranking ever invert?
**Yes, repeatedly and non-monotonically.** Head-to-head over 84 shared dials (±0.019 ARI tie band):
**frozen wins 34, adapted wins 38, 12 ties.** Not monotonic in mcs — both arms win at low mcs
(6–15) and high mcs (40–50) depending on ms/method. `mcs6/ms5/leaf` favours adapted while
`mcs6/ms3/leaf` (the original scoring dial) favours frozen; `mcs40/ms3/eom` favours adapted while
`mcs40/ms10/eom` favours frozen at the same mcs. **The severstal conclusion is dial-dependent and
must carry that caveat.**

## eom + eps=0 collapse corner (warning)
`mcs6/ms3/eom/eps0.0` and low-mcs/ms eom neighbours collapse **both arms** to seed_noise 0%, k=2,
ARI 0, P1 0/4 — one or two giant blobs, zero noise. This corner **is** caught by the existing
`over_merge` gate (a cluster ≥20% of n fires correctly). But the frozen k=2 cheat above is the
*opposite* degenerate corner (few small clean clusters, most points noise) and slips past the same
gate because no single cluster reaches 20%.

**Recommendation (not applied — policy call):** add a k-floor near an expected/engineering
class-count estimate, or a `non_noise_pct` floor, alongside the existing
over_merge/stability/coherence gate, so a noise-minimising selector cannot be pulled into either corner.

## Summary
| Q | Answer |
|---|---|
| Q1 | Yes — H supported |
| Q2 | Adapted, qualitatively (frozen P1 ceiling 2/4 vs adapted 3–4/4 in the k=4–6 zone) |
| Q3 | Coarse geometry rule works but needs k̂; no unlabeled proxy tracks ARI for both arms — deployment constraint |
| Q4 | Yes — 34/38/12 split, non-monotonic |

Data: `runs/severstal/dial_sweep/sweep_results.json`, `sweep_results.csv` (168 rows, all columns).

---

## Addendum (team-lead)

**1. Rule C already carries part of the recommended guard.** Rule C is
`gate-pass → keep k ≥ own-run 75th percentile → argmin noise`. That k-percentile filter is exactly
the k-floor recommended above, so Rule C is *not* as exposed to the merge-cheat as the analysis
assumes. Two caveats remain: the floor is **relative to the run's own epochs**, so it gives no
protection if every epoch collapses together; and it guards **epoch** selection only — **dial**
selection has no such guard, which is where the k=2 cheat actually bites.

**2. The geometry rule extends to four more pools** (team-lead computation, `mcs6` expressed as a
percentage of n/k):

| pool | n | k | n/k | mcs6 as % | mcs @10% |
|---|--:|--:|--:|--:|--:|
| mwm38_clean546 | 546 | 9 | 60.7 | 9.9% | 6 ✅ mcs6 works |
| anchor_avg30_repro | 2260 | 43 | 52.6 | 11.4% | 5 ✅ mcs6 works |
| unknown_eval100 | 4149 | 42 | 98.8 | 6.1% | 10 |
| severstal_pilot | 995 | 5 | 199.0 | **3.0%** ❌ | **20** (=10.1%) ← sweep winner |
| **v2 strict_novel_train** | 12647 | 22 | 574.9 | **1.0%** ❌❌ | **57** |
| **v2 strict_novel_val** | 4196 | 10 | 419.6 | **1.4%** ❌❌ | **42** |

Every pool where mcs6 demonstrably works sits at ~10% of n/k, and the sweep-discovered severstal
winner (mcs20) is 10.1% — independent confirmation of the rule. **The v2 pools are off by 7×**,
worse than severstal was, so v2 results must be scored at mcs≈42 as well as mcs6.

**3. Scope limit.** This applies to *pool-level* grouping evaluation. The temporal-novelty track
clusters **per batch**, where n per clustering call is small, so mcs6 may well be correct there.
Do not re-score the temporal track on the strength of this table.
