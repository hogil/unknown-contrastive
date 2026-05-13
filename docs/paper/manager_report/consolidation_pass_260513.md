# Consolidation Pass — Multi-seed Compliance Audit (2026-05-13)

> **Scope**: Read-only audit of 11 paper section files for multi-seed compliance,
> per-claim numeric consistency, and policy adherence (Tier 1+2 official metrics
> only, no custom metrics). Direct edits NOT applied — patches recommended for
> user review.
>
> **Reference policy**:
> - CLAUDE.md feedback: "Multi-seed avg ± std required for paper claims (no
>   single-seed conclusions)".
> - `docs/contrastive-eval/METRICS.md`: Tier 1+2 official metrics only; custom
>   metrics (weighted_isolation, pure_rate, binary_*, etc.) forbidden.
> - ABSTRACT v0.9 (CURRENT): "NEW cfg unified multi-seed SOTA. v6 B5 absolute
>   SOTA ARI 0.9358 retracted on multi-seed reproducibility grounds."
> - README ★ 2026-05-12 N1 v7 FINAL section: v6 retraction is the CURRENT state.

---

## 1. Audit summary

| Section file | Multi-seed compliant? | Outdated v6 claims? | Tier 1+2 only? | Status |
|---|:-:|:-:|:-:|---|
| `ABSTRACT.md` | ★ v0.9 yes (v0.8/earlier inconsistent) | yes (in v0.8 body, deprecated header missing) | yes | partial — older versions need explicit deprecation tags |
| `INTRODUCTION.md` | partial (C7 v7 retraction present, but C4/C7 inline v6 claims unresolved) | **YES** (lines 198-228 v6 content still active) | yes | needs patch |
| `METHOD.md` | partial (3.6 v7 final recipe at end, but 3.6 body still v6) | **YES** (§3.6 body lines 176-310, RESULTS §16 / §3.6 conclusion block retained complementary framing) | yes | needs patch |
| `RESULTS.md` | **YES** §17 v7 retraction is authoritative | residual single-seed claims in §15a/§15b (table headers), §16 (per-class, marked single-seed=42 only) | yes | mostly OK (§15a/§15d still recommends B5 for known-K) |
| `DISCUSSION.md` | partial (7.10.7 + 7.12.4 v7 corrections present) | **YES** (§7.13 Practitioner choice tree Step 1 YES → "Use B5 / iter 37 cfg", line 958) | yes | **critical** patch needed at 7.13 |
| `CONCLUSION.md` | partial (8.8 v7 final at end) | **YES** (§8.6 N7 v6 block lines 282-329, "absolute SOTA" claim at 297, practitioner consequences 1 at 336-348 still recommends B5 for known-K, §8.7 dual-frontier still recommends B5) | yes | needs patch |
| `ABLATION_PLAN.md` | N/A (pre-v6 history doc, scoped to B0-B5 isolation) | residual ("paper recommendation: deprecate Local, use NeCo + Queue + NEG only" at line 162 — pre-v6 framing, OK as-is if historical) | yes | OK if marked historical |
| `ITERATIONS.md` | **append-only** — iter 84 entry contains v7 retraction (lines 1972-2062) | iter 82/83 entries contain v6 absolute SOTA framing — **append-only policy prohibits edits** | yes | OK (append-only) |
| `FIGURES.md` | partial (F-N7-multiseed-Sil deprecated) | **YES** F-N7-lattice caption line 91 still says "Frontier 2 cfg (B5) keeps both for known-K Agglo Ward absolute SOTA ARI 0.9358" | yes | needs caption patch |
| `README.md` | mixed — ★ N1 v7 FINAL section is CURRENT, but the v6 section below it ("dual-cfg recipe") still active without explicit "superseded" tag at section header | **YES** (line 119 "superseded by v7 on multi-seed" tag is good but Frontier 2 recommendation at 138 still says B5) | yes | needs patch (section 119+ needs body to reflect retraction) |
| `manager_report/SUMMARY.md` | §0.7 v7 FINAL is current, §0.6 has retract notes | **YES** §10 Configuration A vs B at 825-865 still recommends "K known + linkage → Config A (B5, iter 37) absolute SOTA 0.9358" (lines 858-864) | yes | needs patch |
| `manager_report/REPORT.md` | partial (Phase 2 결론 has v7 final correction note at top of §3, but body still has v6 dual-cfg) | **YES** Phase 3 결론 lines 442-449 still recommends "Frontier 2: B5 + Agglomerative = 0.9358" | yes | needs patch |

**Bottom-line**: ABSTRACT v0.9 (CURRENT) and RESULTS §17 + ITERATIONS iter 84 are
the authoritative v7 sources. All other section files are **partially patched** —
each contains v7 correction notes but retains v6 dual-cfg / B5-absolute-SOTA
narrative in body or downstream sections. **The single-cfg recipe (NEW only) is
not yet fully propagated to practitioner-facing recommendation tables.**

---

## 2. Critical inconsistencies (file:line)

### 2.1 v6 "B5 absolute SOTA Agglo K=42 ARI 0.9358" still appears as a recommendation (not as retracted claim)

These instances treat **0.9358 as the current SOTA** rather than as a retracted
single-seed cherry-picked outlier:

| Location | Issue | v7 Authoritative source |
|---|---|---|
| `DISCUSSION.md:958` | "→ Frontier 2. Use B5 / iter 37 cfg + Agglomerative Ward K=K_gt. → Expected: ARI 0.90-0.93 (single-seed), 0.89-0.92 (3-seed)." | should recommend NEW + Agglomerative; v7 (RESULTS §17b): B5 2-seed avg = 0.8920 ± 0.062 (BELOW NEW 0.9014 ± 0.022) |
| `DISCUSSION.md:612` | "linkage-cluster + known-K → B5 (highest absolute SOTA ARI 0.9358 via complementary per-class integration)" | §7.10.7 retracts; should mark as single-seed-only observation |
| `DISCUSSION.md:684` | "B5 (Local + NeCo combined) is the absolute SOTA for known-K oracle clustering (ARI 0.9358 single-seed)" | should append: "★ v7 retracted — seed=1 = 0.8482; 2-seed avg 0.8920 ± 0.062 < NEW 3-seed 0.9014 ± 0.022" |
| `CONCLUSION.md:297` | "Combining both (B5) recovers absolute SOTA ARI 0.9358 single-seed under linkage clustering." | §8.8 (line 493-506) retracts; this line is in §8.6 N7 v6 block — needs explicit deprecation banner |
| `CONCLUSION.md:342-345` | "For absolute SOTA under known-K linkage clustering, **keep both** (B5 ARI 0.9358 single-seed, +0.0158 above NEW 0.9200)." | §8.8 v7 recipe = NEW for BOTH frontiers |
| `CONCLUSION.md:455-457` | "**Known-K oracle benchmark frontier (Agglomerative Ward K=42 + B5 / iter 37)**: single-seed ARI 0.9358, NMI 0.9704; NEW 3-seed mean on same Agglo K=42 = 0.9014 ± 0.022. Recommended for closed-taxonomy lab benchmarks." | recommendation contradicts §8.8; should be `Agglomerative Ward K=42 + NEW` |
| `INTRODUCTION.md:222-228` | "The absolute SOTA single-seed ARI is **B5 (Local + NeCo combined) Agglomerative Ward K=42 = 0.9358**, strictly above iter 70 NEW 0.9200 (Δ +0.0158). ... **Local DenseCL is NOT deprecated**" | C7 head at lines 178-198 has v7 correction notice but this body content still active. v7 final: "Local DenseCL is operationally optional, not required for SOTA" |
| `RESULTS.md:603` | "**Absolute SOTA (single-seed)**: B5 + Agglomerative Ward K=42 = **ARI 0.9358** (oracle K required)." | should append: "★ v7 retracted — single-seed cherry-picked outlier; multi-seed B5 2-seed avg = 0.8920 ± 0.062 < NEW 3-seed avg 0.9014 ± 0.022 (RESULTS §17b)" |
| `RESULTS.md:613` | "NMI ranking matches ARI ranking direction across all 5 methods. ★ B5 + Agglomerative = NMI 0.9704 (absolute SOTA on NMI)." | NMI single-seed claim — needs the same v7 retraction; multi-seed NMI not measured |
| `RESULTS.md:755` | "→ absolute SOTA Agglo K=42 ARI 0.9358 (single-seed=42)" (in §16d mechanism interpretation) | §17 retracts; mark as single-seed observation |
| `README.md:128-143` | "evidence" table shows 0.9358 as "★ ★" current SOTA and dual-cfg recipe recommends B5 for Frontier 2 | this v6 section at line 119 has "superseded by v7" tag but recommendation body not updated |
| `FIGURES.md:91` | F-N7-lattice caption: "Frontier 2 cfg (B5) keeps both for known-K Agglo Ward absolute SOTA ARI 0.9358." | replace with: "Per-class complementarity observed at single-seed=42 (RESULTS §16) does NOT propagate to multi-seed avg — see RESULTS §17b. v7 recommendation: NEW for both frontiers." |
| `manager_report/SUMMARY.md:786` | "B5 + Agglomerative K=42 = 0.9358 (single) / NEW 0.9014 ± 0.022 (3-seed)" recommendation | §0.7 v7 final retracts |
| `manager_report/SUMMARY.md:858-861` | "**K known + linkage clustering (oracle benchmark)** → **Config A (B5, iter 37)** = absolute SOTA on Agglomerative Ward K=42 (ARI **0.9358** single-seed, +0.0158 vs NEW)" | v7: NEW dominates B5 on multi-seed avg; this practitioner table contradicts §0.7 |
| `manager_report/REPORT.md:442-451` | "Frontier 2 (Known-K, oracle, lab benchmark): Encoder: B5 / iter 37 ... ARI: single-seed=42 0.9358 ... 추천: 알려진 defect taxonomy lab benchmark" | Phase 2 결론 (line 327-376) v7 correction note present but this Phase 3 결론 contradicts |
| `manager_report/REPORT.md:466-475` | "B5 (both combined) absolute SOTA Agglo K=42 ARI **0.9358** (Δ +0.0158 vs NEW)" + "Frontier 2 (known-K + Agglo Ward K=42) → B5 (Local + NeCo combined, absolute SOTA)" | same as above |

### 2.2 Single-seed-only conclusions without "multi-seed pending" marker

The following are **single-seed=42** numbers presented without an explicit
`single-seed only / multi-seed not measured` caveat. By the user policy
("Multi-seed avg ± std required for paper claims"), these need a marker:

| Location | Single-seed claim | Multi-seed status |
|---|---|---|
| `RESULTS.md:613` | "B5 + Agglomerative = NMI 0.9704 (absolute SOTA on NMI)" | NMI multi-seed NOT measured. Should append "(single-seed=42 only; multi-seed NMI not measured)" |
| `RESULTS.md:613` (B4/B5/NEW NMI row in 15b) | All NMI values are single-seed=42 | mark column header |
| `RESULTS.md:594-597` (§15a 5-method × 3-cfg ARI matrix) | All cells are single-seed=42 | column header already says "Single-seed ARI (seed=42)" — OK |
| `RESULTS.md` §16 per-class purity tables | All entries are single-seed=42 | §16 header marks "(seed=42, oracle K)" — OK. §17d adds "v7 caveat that complementary inductive biases is at most a single-seed observation, not a multi-seed claim" — adequate |
| `RESULTS.md:606` Spectral K=42 row | Single-seed ARI 0.4046 / 0.7898 / 0.2289 | observation 3 marks it as "unstable across cfg" — OK |
| `manager_report/SUMMARY.md:756-758` (Phase 3 ARI matrix) | single-seed=42 | section header says "(single-seed=42)" — OK |

### 2.3 Inconsistent NEW 3-seed avg notation across sections

The NEW HDBSCAN 3-seed avg value appears in **two different forms**:

- `0.8588 ± 0.018` (more precise — 4-decimal form): ABSTRACT v0.9, RESULTS §17b,
  README ★ section, SUMMARY §0.7, REPORT Phase 2 결론
- `0.859 ± 0.018` (3-decimal abbreviated): INTRODUCTION C4 line 295, CONCLUSION
  §8.6 line 380, ABSTRACT v0.7 line 296

Both round to the same value (0.8588 ≈ 0.859), so this is a **cosmetic
inconsistency only**, not a multi-seed compliance violation. Recommend
**standardize on `0.859 ± 0.018`** (3 decimals matches `0.870` and the
README "현재 SOTA" table style — `0.866 ± 0.014` is also 3-decimal).

### 2.4 README "현재 SOTA" table (lines 57-67) is **stale** w.r.t. v7

```
| ARI (single-seed best) | 0.870 |
| ARI (3-seed mean) | 0.866 ± 0.014 |
```

These are **iter 37 (B5 family)** numbers from the v3/v4 era. v0.9 CURRENT
headline = NEW cfg ARI 0.8588 ± 0.018 (HDBSCAN) / 0.9014 ± 0.022 (Agglo K=42).
The README "현재 SOTA" table should either:

- (a) Replace numbers with NEW cfg multi-seed values (recommended);
- (b) Re-title as "iter 37 (legacy B5 family) reference numbers"; the actual
  current SOTA goes into the ★ N1 v7 FINAL section.

Currently the v7 section is appended below without contradicting this table,
so a reader who stops at line 67 gets the iter 37 / B5 numbers as if current.

### 2.5 Tier 1+2 metric compliance — clean across all sections

**No custom metric outputs detected** (weighted_isolation, pure_rate, mixed_rate,
binary_*, classifier-style precision/recall/F1). All reported metrics are:

- Tier 1: Completeness, AMI, noise_pct, class_capture_rate, class_fragmentation_summary
- Tier 2: Homogeneity, Silhouette (cosine), ARI
- Tier 2 NMI (RESULTS §15b) reported as "supplementary" — acceptable
- Tier 3 (NMI/V-measure/FMI/DB/CH) — only NMI shown in §15b, marked supplementary

**No D-1 / D-2 policy violations found**.

### 2.6 Multi-seed compliance summary

| Claim category | v7 compliance | Action |
|---|:-:|---|
| ABSTRACT v0.9 headline (NEW unified SOTA, multi-seed) | ★ compliant | none |
| RESULTS §17 (NEW vs B5 multi-seed avg table) | ★ compliant | none |
| ITERATIONS iter 84 entry (B5 seed=1 retraction) | ★ compliant | none (append-only) |
| INTRODUCTION C7 retraction notice | partial | body of C7 (lines 198-228) still v6 — append "see §1.4 ★ v7 FINAL" inline pointer at each B5 / 0.9358 mention |
| METHOD §3.6 Conclusion block (lines 312-326) | ★ compliant (v7 final block at end) | body of §3.6 still v6 framing — patch §3.6 body OR add deprecation header |
| DISCUSSION §7.10.7 + §7.12.4 v7 corrections | ★ compliant | §7.13 practitioner choice tree (lines 954-970) Step 1 YES branch contradicts §7.10.7 |
| CONCLUSION §8.8 v7 final | ★ compliant | §8.6 + §8.7 still v6 framing; §8.6 N7 block (lines 282-329, 336-348) recommends B5 for known-K |
| README ★ N1 v7 FINAL section | ★ compliant | "현재 SOTA" table at 57-67 + v6 section at 119-143 are stale |
| SUMMARY §0.7 + Phase 2 결론 | ★ compliant | §10 Configuration A vs B table at 825-865 still recommends Config A (B5) for K-known |
| REPORT Phase 2 결론 v7 final block | ★ compliant | Phase 3 결론 at 437-475 still recommends B5 for Frontier 2 |
| FIGURES F-N7-lattice caption | not compliant | caption line 91 references B5 known-K absolute SOTA |
| ABLATION_PLAN | N/A (pre-v6 doc) | OK |
| ITERATIONS pre-iter-84 entries (v6 claims) | append-only | OK — explicit append-only policy preserves |

---

## 3. Recommended patches (for user review — direct edits NOT applied)

### Patch P-1 — DISCUSSION.md §7.13 Practitioner choice tree (critical)

`DISCUSSION.md:954-970` (Step 1 YES branch):

**OLD**:
```
  YES (closed taxonomy, lab benchmark):
    → Frontier 2. Use B5 / iter 37 cfg + Agglomerative Ward K=K_gt.
    → Expected: ARI 0.90-0.93 (single-seed), 0.89-0.92 (3-seed).
```

**NEW** (proposed):
```
  YES (closed taxonomy, lab benchmark):
    → Frontier 2 (v7). Use iter 70 NEW cfg + Agglomerative Ward K=K_gt.
    → Expected: ARI 0.9014 ± 0.022 (3-seed avg, multi-seed compliant).
    → ★ v7 note: v6 recipe (B5 / iter 37 cfg + Agglo K=42 single-seed 0.9358)
      retracted on seed=1 reproducibility (iter 84) — B5 2-seed avg
      0.8920 ± 0.062 is BELOW NEW 3-seed avg 0.9014 ± 0.022. NEW std 2.8× lower.
      Reference: §7.10.7, §7.12.4, RESULTS §17.
```

### Patch P-2 — CONCLUSION.md §8.6 N7 block + §8.7 Frontier 2

`CONCLUSION.md:336-348` (N7 (i) practitioner consequence):

**OLD**:
```
1. **NeCo and Local DenseCL are complementary at per-class scope** (N7 (i),
   ... For absolute SOTA under known-K linkage clustering, **keep both**
   (B5 ARI 0.9358 single-seed, +0.0158 above NEW 0.9200). ...
```

**NEW** (proposed):
```
1. **NeCo and Local DenseCL are aggregate-substitutable on HDBSCAN, single-seed
   complementary on per-class Agglo K=42** (N7 (i), ★ N1 v6 single-seed
   observation; ★ N1 v7 multi-seed correction). On single-seed=42 Agglomerative
   Ward K=42, B5 (both mechanisms) reaches ARI 0.9358 vs NEW 0.9200 with
   per-class winner flips on both sides. **However, on multi-seed average across
   all three clustering methods (HDBSCAN, Agglo Ward K=42, KMeans K=42), NEW
   (NeCo only, no Local) > B5 (Local + NeCo)**: HDBSCAN +0.0245, Agglo +0.0094,
   KMeans +0.0138, with NEW std 1.7-2.8× lower than B5. The v6 "B5 absolute
   SOTA at known-K Agglo ARI 0.9358" claim is retracted (single-seed
   cherry-picked outlier; seed=1 reproduce = 0.8482, Δ −0.088). Local DenseCL
   is **operationally optional, not required for SOTA** (RESULTS §17, v7
   FINAL). The v6 per-class purity flips are preserved as single-seed
   observations only.
```

`CONCLUSION.md:455-457` (§8.7 Frontier 2 recommendation):

**OLD**:
```
- **Known-K oracle benchmark frontier (Agglomerative Ward K=42 + B5 / iter 37)**:
  single-seed ARI 0.9358, NMI 0.9704; NEW 3-seed mean on same Agglo K=42 = 0.9014
  ± 0.022. Recommended for closed-taxonomy lab benchmarks.
```

**NEW** (proposed):
```
- **Known-K oracle benchmark frontier (Agglomerative Ward K=42 + iter 70 NEW)
  ★ v7 revised**: NEW 3-seed mean ARI 0.9014 ± 0.022; B5 / iter 37 cfg 2-seed
  avg = 0.8920 ± 0.062 (BELOW NEW, std 2.8× higher; B5 seed=42 0.9358 was
  cherry-picked, seed=1 = 0.8482). Recommended encoder cfg: same as Frontier 1
  (iter 70 NEW). Local DenseCL operationally optional. Reference: §8.8, RESULTS §17.
```

### Patch P-3 — INTRODUCTION.md C7 body

`INTRODUCTION.md:203-228` (C7 v6 body that contradicts the v7 retraction header at 178-198):

Add a deprecation banner at the start of the v6 block (line 203) and inline
"★ v7 retracted" markers at lines 222-228 where 0.9358 is mentioned.

### Patch P-4 — METHOD.md §3.6 body

`METHOD.md:176-310` (v6 framing body):

The §3.6 "Conclusion (METHOD-level, v6 final)" block at lines 286-326 already
contains the **v7 FINAL recommendation** ("Use iter 70 NEW for BOTH frontiers").
However, the §3.6 body above (lines 184-285) describes the v6 dual-cfg complementarity
without flagging the multi-seed retraction inline. Recommend prepending the
section with:

```
★ **v7 FINAL CORRECTION 2026-05-12 (iter 84 B5 seed=1 reproducibility)**:
The v6 "B5 absolute SOTA Agglo K=42 ARI 0.9358" claim is retracted on multi-seed
grounds — B5 2-seed avg 0.8920 ± 0.062 < NEW 3-seed avg 0.9014 ± 0.022. The
single-seed per-class purity flips below are preserved as observations, but the
dual-cfg recipe collapses to single-cfg (NEW) recommendation. See METHOD §3.6
Conclusion block (line 312+) and RESULTS §17 for the v7 final recipe.
```

### Patch P-5 — README.md "현재 SOTA" table (lines 57-73)

Either replace numbers with v7 multi-seed values OR add a "(legacy reference,
see ★ N1 v7 FINAL below for current)" tag at the section header.

`README.md:138-143` (v6 dual-cfg recipe summary):

This v6 section already has "superseded by v7 on multi-seed" in its header
(line 119), but the bullet point at 138 still recommends "Frontier 2 (known-K +
Agglo Ward) → B5 (Local + NeCo combined) = absolute SOTA". The header tag
exists; recommend replacing bullets 137-143 with a 1-line cross-reference:

```
→ **dual-cfg recipe RETRACTED in v7**: see ★ N1 v7 FINAL section above (line 98).
  v6 per-class purity observations (RESULTS §16) preserved as single-seed only.
```

### Patch P-6 — FIGURES.md F-N7-lattice caption (line 91)

Replace:
```
Frontier 2 cfg (B5) keeps both for known-K Agglo Ward absolute SOTA ARI 0.9358.
```
With:
```
Per-class purity flips at single-seed=42 (RESULTS §16) do NOT propagate to
multi-seed avg (RESULTS §17b): NEW (NeCo only) > B5 (Local + NeCo) on Agglo Ward
K=42 multi-seed (0.9014 ± 0.022 vs 0.8920 ± 0.062). v7 single-cfg recipe = NEW
for both frontiers.
```

### Patch P-7 — manager_report/SUMMARY.md §10 + manager_report/REPORT.md Phase 3 결론

Both "Configuration A vs B" decision tables and "Phase 3 결론" recommendations
still steer Frontier 2 to B5. Update to single-cfg (NEW) recipe consistent with
§0.7 and Phase 2 결론 v7 final corrections.

### Patch P-8 — RESULTS.md §15 (5-method benchmark) single-seed disclaimers

`RESULTS.md:603` (observation 4):

**OLD**:
```
4. **Absolute SOTA (single-seed)**: B5 + Agglomerative Ward K=42 = **ARI 0.9358** (oracle K required).
```

**NEW** (proposed):
```
4. **Single-seed=42 max**: B5 + Agglomerative Ward K=42 = ARI 0.9358 (oracle K). ★ v7
   retracted as multi-seed SOTA — seed=1 = 0.8482, 2-seed avg = 0.8920 ± 0.062, BELOW
   NEW 3-seed avg 0.9014 ± 0.022 (RESULTS §17b). Use NEW + Agglomerative Ward K=42
   as the multi-seed-compliant known-K SOTA.
```

`RESULTS.md:613` (NMI claim):

**OLD**:
```
NMI ranking matches ARI ranking direction across all 5 methods. ★ B5 + Agglomerative = NMI 0.9704 (absolute SOTA on NMI).
```

**NEW** (proposed):
```
NMI ranking matches ARI ranking direction across all 5 methods. B5 + Agglomerative
= NMI 0.9704 (single-seed=42 only; multi-seed NMI not measured — see §17 for
multi-seed ARI). ★ v7 caveat: same B5 seed-flip hazard applies (Δ ARI −0.088
seed=42 → seed=1 on Agglo K=42, RESULTS §17a); single-seed NMI values likely
share the same lucky-outlier hazard.
```

---

## 4. Append-only ITERATIONS.md status

`ITERATIONS.md:1972-2062` (iter 84 entry) is the **authoritative v7 source**.
Pre-iter-84 entries (especially iter 82/83 at lines 1830-1965) contain v6
"absolute SOTA" framing. Per append-only policy, these are **NOT to be edited**.

The v6→v7 retraction is correctly tracked **in the iter 84 entry** (lines 2010,
2037-2062), which is the proper append-only mechanism. No action required on
ITERATIONS.md.

---

## 5. Other findings (informational only)

- **Inconsistent NEW HDBSCAN 3-seed value notation** (cosmetic): `0.8588 ± 0.018`
  vs `0.859 ± 0.018`. Both correct; recommend standardize. Not a policy violation.
- **B5 multi-seed std is computed from n=2 samples** (iter 83 seed=42, iter 84
  seed=1). The std value 0.062 on Agglo K=42 is range-based for n=2; the v7
  claim "B5 2-seed avg ± std" is **mathematically meaningful but
  high-variance**. Recommend a footnote: "B5 std computed from n=2; 3rd seed
  measurement (iter 85+) pending — std value may tighten with more samples."
- **DP-GMM K_discovered (46-47)** is reported (RESULTS §15a). This is K_discovered
  not K_gt — appropriate Tier 2 supplementary metric, no policy issue.
- **n_cluster column** in RESULTS §15 table (single-seed=42): B0 37 / B3 36 /
  most others 37 / iter 70 NEW 37 / iter 75 38. Minor variation; not a
  multi-seed compliance issue.

---

## 6. Action priority

| Priority | Patch | Justification |
|:-:|---|---|
| P0 (critical) | P-1 (DISCUSSION §7.13 practitioner choice tree) | Practitioner-facing — directly contradicts v7 final in same file |
| P0 (critical) | P-2 (CONCLUSION §8.6 / §8.7 Frontier 2) | Practitioner-facing recommendation table |
| P0 (critical) | P-7 (SUMMARY §10 + REPORT Phase 3) | Manager-facing — Configuration A vs B table is the first thing reviewers see |
| P1 (high) | P-8 (RESULTS §15 absolute SOTA / NMI disclaimers) | RESULTS is the load-bearing evidence section |
| P1 (high) | P-5 (README v7 propagation + 현재 SOTA table) | First contact entry-point doc |
| P1 (high) | P-6 (FIGURES F-N7-lattice caption) | Figure caption — directly visible in paper |
| P2 (medium) | P-3 (INTRODUCTION C7 body) | Already has retraction header; body inconsistency softer |
| P2 (medium) | P-4 (METHOD §3.6 body) | Conclusion block at end already v7 final; body inconsistency softer |

P0 patches (P-1, P-2, P-7) close the **practitioner-facing recommendation
contradiction**: with these applied, a reader cannot find a section that
recommends B5 for the known-K frontier. P1/P2 are inline narrative cleanup
that improves coherence.

---

## 7. Summary

- **Multi-seed avg ± std compliance**: ★ achieved at the **paper-headline
  level** (ABSTRACT v0.9, RESULTS §17, ITERATIONS iter 84). However, **9 of 11
  paper section files** still contain residual v6 "B5 absolute SOTA 0.9358"
  recommendations in body content, practitioner choice trees, or figure
  captions.
- **Tier 1+2 metric policy**: ★ clean. No custom metrics detected. NMI used
  appropriately as Tier 2 supplementary.
- **B5 absolute SOTA retraction**: tracked in iter 84 append entry per
  append-only policy. **The retraction has not been propagated** to ~6
  practitioner-facing recommendation tables — these still recommend B5 for
  known-K oracle benchmarks.
- **Oracle K=42 cheating warning**: present in DISCUSSION §7.12.4 / §7.13 and
  CONCLUSION §8.7 ("any ARI claim on a contrastive-cluster pipeline must
  specify the clustering algorithm and the K-discovery regime"). RESULTS §15
  + §17 also disclose K-discovery regime. **OK** at the disclosure level, but
  the practitioner recommendation needs the same v7 multi-seed correction.
- **Append-only ITERATIONS**: ★ preserved correctly.

**Recommendation**: User-supervised application of P0 patches first (3 files —
DISCUSSION, CONCLUSION, SUMMARY+REPORT), then P1 patches (RESULTS, README,
FIGURES). P2 (INTRODUCTION, METHOD) can be done in a follow-up pass.

---

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/consolidation_pass_260513.md
