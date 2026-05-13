# Patches Applied — Consolidation Pass v7 Propagation + P-0a Headline Provenance Disclosure (2026-05-13)

> **Source**: `consolidation_pass_260513.md` (8 patches) + new P-0a (headline ARI
> protocol provenance disclosure).
>
> **Outcome**: All v6 "B5 absolute SOTA Agglo K=42 ARI 0.9358" recommendations in
> practitioner-facing sections are now retracted with explicit v7 multi-seed
> correction. New METHOD.md §7.1 discloses headline ARI provenance (defect-only
> post-hoc HDBSCAN, NOT default `eval_summary.json`). Cosmetic 4-decimal vs
> 3-decimal NEW HDBSCAN avg notation standardized to **`0.859 ± 0.018`** across
> paper section files.
>
> ★ **ITERATIONS.md NOT modified** (append-only policy enforced).

---

## P-0a — METHOD.md §7.1 Headline ARI provenance (NEW, critical reviewer disclosure)

**Status**: applied to `METHOD.md` (immediately after §7 Evaluation metrics, before §8 monitoring).

**Rationale**: paper headline NEW + HDBSCAN ARI `0.859 ± 0.018` (≡ 0.8588) is sourced
from 3 separate `tier1_*.json` files produced by a **defect-only post-hoc
re-cluster** with `eom + mcs=12 + ms=3 + NO cluster_selection_epsilon`. The
default `_eval_contrastive_unknown_n50.py` pipeline uses full-set scope + eps=0.06,
producing a different (lower) ARI ≈ 0.74. Reviewers reproducing via the published
eval pipeline alone would see a 12 pp gap from the headline. This is a critical
methodology disclosure obligation (paper N8 reinforcement).

**Source evidence** (3 tier1 JSON files, defect-only re-cluster):

| seed | run_dir | tier1 file | ARI | Comp | AMI | noise_pct | n_cl |
|---|---|---|---:|---:|---:|---:|---:|
| 42 | `outputs_contrastive_260512_001719/` | `tier1_neco_replaces_local.json` | 0.8797 | 0.9872 | 0.9594 | 0.87% | 37 |
| 1 | `outputs_contrastive_260512_010113/` | `tier1_sota_seed1.json` | 0.8491 | 0.9856 | 0.9488 | 1.05% | 35 |
| 2 | `outputs_contrastive_260512_014507/` | `tier1_sota_seed2.json` | 0.8475 | 0.9747 | 0.9428 | 2.53% | 36 |
| **avg** | — | — | **0.8588** | 0.9825 | 0.9503 | 1.48% | — |
| **std (ddof=1)** | — | — | **0.0181** | 0.0070 | 0.0091 | 0.84pp | — |

**Diff snippet** (METHOD.md §7 → new §7.1):

```diff
 ### Tier 2 (보조)
 - Homogeneity (Rosenberg 2007)
 - Silhouette cosine (Rousseeuw 1987)
 - ARI (Hubert & Arabie 1985) — over-cluster 페널티 inherent
+
+### 7.1 Headline ARI provenance — defect-only post-hoc HDBSCAN (★ NEW 2026-05-13, paper N8 reinforcement)
+
+★ **Critical reviewer-facing disclosure**: Paper headline ARI claim **NEW + HDBSCAN
+ARI 0.859 ± 0.018** (3-seed mean ± ddof=1 std, equivalently `0.8588 ± 0.018`) is
+**NOT** the number a reviewer would obtain by re-running the default eval pipeline
+`_eval_contrastive_unknown_n50.py` on the published checkpoints. The headline comes
+from a **separate post-hoc HDBSCAN re-cluster** with the following explicit protocol:
+
+| axis | headline (paper) | default `eval_summary.json` |
+|---|---|---|
+| scope | **defect-only** (n ≈ 1146, Normal class excluded) | full-set (n = 2146 inc. Normal) |
+| `cluster_selection_method` | `eom` | `eom` |
+| `min_cluster_size` (mcs) | **12** | 5 |
+| `min_samples` (ms) | **3** | 5 |
+| `cluster_selection_epsilon` (eps) | **NOT set (none)** | 0.06 |
+| metric | cosine (L2-normalized 128-d projection) | cosine |
+| K-discovery | density (unknown-K) | density (unknown-K) |
+
+[... source 3-seed table, default-eval vs headline gap explanation, B5 comparison
+protocol parity, provenance trace cross-link ...]
+
+★ **All ARI / Sil / NMI / Hom / Comp / noise / capture numbers reported in this
+paper headline tables (RESULTS §15-17, ABSTRACT v0.9, README current SOTA,
+DISCUSSION §7.10.7 / §7.12.4) are on this defect-only protocol unless explicitly
+marked otherwise (e.g., "full-set" qualifier).**
```

---

## P0-1 — DISCUSSION.md:958 Practitioner choice tree Step 1 YES branch

**Status**: applied to `DISCUSSION.md` §7.13.

**Diff**:

```diff
   YES (closed taxonomy, lab benchmark):
-    → Frontier 2. Use B5 / iter 37 cfg + Agglomerative Ward K=K_gt.
-    → Expected: ARI 0.90-0.93 (single-seed), 0.89-0.92 (3-seed).
+    → Frontier 2 (★ v7 revised). Use iter 70 NEW cfg + Agglomerative Ward K=K_gt.
+    → Expected: ARI 0.9014 ± 0.022 (3-seed avg, multi-seed compliant).
+    → ★ v7 note: v6 recipe (B5 / iter 37 cfg + Agglo K=42 single-seed 0.9358)
+      retracted on seed=1 reproducibility (iter 84) — B5 2-seed avg
+      0.8920 ± 0.062 is BELOW NEW 3-seed avg 0.9014 ± 0.022. NEW std 2.8× lower.
+      Reference: §7.10.7, §7.12.4, RESULTS §17.
     → Watch out for: linkage-based clustering is sensitive to global scaling;
                      verify cosine-normalization before fitting.
```

---

## P0-2 — CONCLUSION.md §8.6 + §8.7

**Status**: 3 edits applied (line 297 N7 dependency block; line 336-348 N7 practitioner consequence 1; line 455-457 §8.7 Frontier 2 recommendation).

### P0-2a — §8.6 N7 dependency hierarchy block (line 287-300)

```diff
-Required:      Global InfoNCE + {Local DenseCL && NeCo combined for SOTA,
-                                  or NeCo alone for density-cluster + Normal stream}
+Required:      Global InfoNCE + {NeCo alone covers BOTH HDBSCAN (Normal-stream)
+                                  AND oracle-K Agglomerative Ward frontiers on
+                                  multi-seed avg — single-cfg recommendation}
 Significant:   MoCo Queue (...)
 Conditional:   NEG filter <- requires Queue (...)
-Complementary: Local DenseCL ↔ NeCo (★ N1 v6) — aggregate ARI identical
-                under HDBSCAN, but per-class K=42 Agglomerative Ward purity
-                shows complementary inductive biases (...). Combining both
-                (B5) recovers absolute SOTA ARI 0.9358 single-seed under
-                linkage clustering. The v5 "substitutable" framing applies
-                only to aggregate HDBSCAN unknown-K ARI.
+Complementary single-seed only: Local DenseCL ↔ NeCo (★ N1 v6 single-seed
+                observation; ★ N1 v7 multi-seed correction) — aggregate ARI
+                identical under HDBSCAN; at single-seed=42 per-class K=42
+                Agglomerative Ward purity shows complementary inductive
+                biases (...). Combining both (B5) reaches single-seed=42
+                ARI 0.9358 under linkage clustering — ★ v7 retracted:
+                seed=1 reproduce = 0.8482, B5 2-seed avg 0.8920 ± 0.062 <
+                NEW 3-seed avg 0.9014 ± 0.022. The v6 "complementary" framing
+                applies only to single-seed=42 observations.
```

### P0-2b — §8.6 practitioner consequence 1 (line 336-348)

```diff
-1. **NeCo and Local DenseCL are complementary at per-class scope** (N7 (i),
-   ★ N1 v6 NEW). ... For absolute SOTA under known-K linkage clustering,
-   **keep both** (B5 ARI 0.9358 single-seed, +0.0158 above NEW 0.9200).
+1. **NeCo and Local DenseCL are aggregate-substitutable on HDBSCAN, single-seed
+   complementary on per-class Agglo K=42** (N7 (i), ★ N1 v6 single-seed
+   observation; ★ N1 v7 multi-seed correction). ... **However, on multi-seed
+   average across all three benchmarked clustering methods (HDBSCAN, Agglo Ward
+   K=42, KMeans K=42), NEW (NeCo only, no Local) > B5 (Local + NeCo)**: HDBSCAN
+   +0.0245, Agglo +0.0094, KMeans +0.0138, with NEW std 1.7-2.8× lower than B5.
+   The v6 "B5 absolute SOTA at known-K Agglo ARI 0.9358" claim is retracted
+   (single-seed cherry-picked outlier; seed=1 reproduce = 0.8482, Δ −0.088).
+   Local DenseCL is **operationally optional, not required for SOTA** — use NEW
+   for both frontiers.
```

### P0-2c — §8.7 Frontier 2 recommendation (line 455-457)

```diff
-- **Known-K oracle benchmark frontier (Agglomerative Ward K=42 + B5 / iter 37)**:
-  single-seed ARI 0.9358, NMI 0.9704; NEW 3-seed mean on same Agglo K=42 = 0.9014
-  ± 0.022. Recommended for closed-taxonomy lab benchmarks.
+- **Known-K oracle benchmark frontier (Agglomerative Ward K=42 + iter 70 NEW)
+  ★ v7 revised**: NEW 3-seed mean ARI **0.9014 ± 0.022** (multi-seed
+  authoritative). B5 / iter 37 cfg 2-seed avg = 0.8920 ± 0.062 (BELOW NEW,
+  std 2.8× higher; B5 seed=42 0.9358 was cherry-picked, seed=1 = 0.8482,
+  Δ −0.0876). Recommended encoder cfg: **same as Frontier 1 (iter 70 NEW)**.
+  Local DenseCL operationally optional. NMI 0.9704 single-seed=42 only;
+  multi-seed NMI not measured. Reference: §8.8, RESULTS §17.
```

---

## P0-3a — SUMMARY.md §3 Phase 3 결론 + §10 Configuration A vs B

**Status**: 2 edits applied to `manager_report/SUMMARY.md`.

### Phase 3 결론 (line 783-787)

```diff
-3. ★★★ Dual-frontier framework (paper-grade deliverable)
+3. ★★★ Dual-frontier framework (paper-grade deliverable, ★ v7 revised 2026-05-13)
    Unknown-K (real-world):
      iter 70 NEW + HDBSCAN = 0.859 ± 0.018 (3-seed)
    Known-K (oracle benchmark):
-     B5 + Agglomerative K=42 = 0.9358 (single) / NEW 0.9014 ± 0.022 (3-seed)
-     rationale: linkage-based recovers fine sub-structure
+     iter 70 NEW + Agglomerative K=42 = 0.9014 ± 0.022 (3-seed) ★ multi-seed SOTA
+     rationale: linkage-based recovers fine sub-structure
+     ★ v7 retracted: v6 recommended "B5 + Agglo K=42 = 0.9358 single-seed".
+     B5 seed=1 reproduce = 0.8482 (Δ −0.0876), 2-seed avg 0.8920 ± 0.062
+     BELOW NEW 3-seed 0.9014 ± 0.022 with std 2.8× higher. RESULTS §17, iter 84.
```

### §10 Configuration A vs B 운영자 선택지 table (line 857-865)

```diff
-→ **운영자 선택지** (★ N1 v6 dual-cfg recipe 2026-05-12):
-- **K known + linkage clustering (oracle benchmark)** → **Config A (B5, iter 37)**
-  = absolute SOTA on Agglomerative Ward K=42 (ARI **0.9358** single-seed, +0.0158 vs NEW).
+→ **운영자 선택지** (★ N1 v7 FINAL single-cfg recipe, 2026-05-13 revised; supersedes v6 dual-cfg):
+- **K known + linkage clustering (oracle benchmark)** → **Config B (NEW, iter 70) + Agglomerative Ward K=42**
+  = ARI **0.9014 ± 0.022** (3-seed avg, multi-seed SOTA). Local DenseCL operationally optional.
+  ★ v7 retracted: v6 recommended Config A (B5, iter 37) = 0.9358 — single-seed cherry-picked
+  outlier. B5 seed=1 reproduce = 0.8482 (Δ −0.0876), B5 2-seed avg 0.8920 ± 0.062 BELOW NEW
+  3-seed 0.9014 ± 0.022 with std 2.8× higher. RESULTS §17.
 - **K unknown + density clustering** → **Config B (NEW, iter 70)**
```

---

## P0-3b — REPORT.md Phase 3 결론 + paper contributions

**Status**: 2 edits applied to `manager_report/REPORT.md`.

### Phase 3 결론 (line 437-458)

```diff
-### Phase 3 결론
-
-paper Methods 권장 cfg — **dual frontier**, single SOTA 아님:
+### Phase 3 결론 (★ v7 revised 2026-05-13 — single-cfg recipe, supersedes v6 dual-cfg)
+
+paper Methods 권장 cfg — **single-cfg + dual clustering target** (iter 70 NEW covers both frontiers on multi-seed avg):

 Frontier 2 (Known-K, oracle, lab benchmark):
-  Encoder: B5 / iter 37 (5-component: Global + Local LW=1.0 + Queue 4096 + NEG 0.72 + NeCo 0.2)
+  Encoder: iter 70 NEW (SAME cfg as Frontier 1) ★ v7 revised
   Clustering: Agglomerative Ward with K=42 (defect-only scope)
-  ARI: single-seed=42 0.9358, NMI 0.9704 (NEW 3-seed 0.9014 ± 0.022)
+  ARI: 3-seed 0.9014 ± 0.022 (multi-seed authoritative SOTA)
+  ★ v7 retracted: v6 recommended B5 / iter 37 + Agglo K=42 = 0.9358 single-seed.
+    iter 84 (seed=1) reproduce = 0.8482 (Δ −0.0876); B5 2-seed avg 0.8920 ± 0.062
+    BELOW NEW 3-seed 0.9014 ± 0.022. Local DenseCL operationally optional.
```

### paper contributions N1 / N7 / N9 (line 460-475)

```diff
-paper contributions 최종 (★ N1 v6 FINAL + N6 + N7 v6 + N8 + N9, 2026-05-12):
-- **★ N1 v6 FINAL**: ... B5 (both combined) absolute SOTA Agglo K=42 ARI **0.9358**
-- N7 v6: ... Local DenseCL ↔ NeCo **complementary** per-class
-- N9: ... Frontier 2 (known-K + Agglo Ward K=42) → B5 (Local + NeCo combined, absolute SOTA)
+paper contributions 최종 (★ N1 v7 FINAL + N6 + N7 v7 + N8 + N9, 2026-05-13 revised):
+- **★ N1 v7 FINAL**: ... Aggregate HDBSCAN ARI: 4-decimal identity. Single-seed=42
+  per-class: complementary winners (RESULTS §16). **Multi-seed avg Agglo K=42**:
+  NEW 3-seed 0.9014 ± 0.022 > B5 2-seed 0.8920 ± 0.062 (Δ +0.0094, std 2.8× lower).
+  **Local DenseCL operationally optional, not required for SOTA**. v6 "0.9358 absolute
+  SOTA" retracted (cherry-picked seed=42; seed=1 reproduce = 0.8482).
+- N7 v7: Local DenseCL ↔ NeCo **aggregate-substitutable on HDBSCAN, single-seed
+  complementary on Agglo K=42, multi-seed redundant**
+- N9: ... Frontier 2 (known-K + Agglo Ward K=42) → NEW (same cfg, multi-seed SOTA 0.9014 ± 0.022)
```

---

## P1-1 — RESULTS.md §15 (5-method benchmark) + §16d retractions

**Status**: 3 edits applied.

### §15a observation 4 (line 603)

```diff
-4. **Absolute SOTA (single-seed)**: B5 + Agglomerative Ward K=42 = **ARI 0.9358** (oracle K required).
+4. **Single-seed=42 max**: B5 + Agglomerative Ward K=42 = ARI 0.9358 (oracle K). ★ v7
+   retracted as multi-seed SOTA — seed=1 reproduce = 0.8482, 2-seed avg = 0.8920 ± 0.062,
+   BELOW NEW 3-seed avg 0.9014 ± 0.022 (RESULTS §17b). Use NEW + Agglomerative Ward K=42
+   as the multi-seed-compliant known-K SOTA.
```

### §15b NMI claim (line 613)

```diff
-NMI ranking matches ARI ranking direction across all 5 methods. ★ B5 + Agglomerative = NMI 0.9704 (absolute SOTA on NMI).
+NMI ranking matches ARI ranking direction across all 5 methods. B5 + Agglomerative = NMI 0.9704
+(single-seed=42 only; multi-seed NMI not measured — see §17 for multi-seed ARI). ★ v7 caveat:
+same B5 seed-flip hazard applies (Δ ARI −0.088 seed=42 → seed=1 on Agglo K=42, RESULTS §17a);
+single-seed NMI values likely share the same lucky-outlier hazard.
```

### §16d Mechanism interpretation code block (line 755)

```diff
 Combined (B5 = Local + NeCo):
    → both per-class strength axes active
-   → absolute SOTA Agglo K=42 ARI 0.9358 (single-seed=42)
-   → marginal win on micro-aggregate purity (97.0% vs 96.2%)
+   → single-seed=42 Agglo K=42 ARI 0.9358 ★ v7 retracted as multi-seed SOTA
+     (seed=1 reproduce = 0.8482; B5 2-seed avg 0.8920 ± 0.062 < NEW 3-seed
+     0.9014 ± 0.022 on same Agglo K=42 — RESULTS §17b)
+   → marginal win on single-seed=42 micro-aggregate purity (97.0% vs 96.2%)
```

---

## P1-2 — README.md "현재 SOTA" table + v6 dual-cfg recipe bullets

**Status**: 2 edits applied.

### "현재 SOTA" table (line 57-67) — replaced iter-37/B5 numbers with v7 NEW multi-seed values

```diff
-## 현재 SOTA (iter 37, new anchor `avg30_new_260508_123037`)
+## 현재 SOTA (★ N1 v7 FINAL — iter 70 NEW, anchor `avg30_new_260508_123037`, ★ revised 2026-05-13)

 | metric | value |
 |---|---|
-| Completeness | 0.991 |
-| AMI | 0.960 |
-| ARI (single-seed best) | 0.870 |
-| ARI (3-seed mean) | 0.866 ± 0.014 |
-| noise_pct (defect) | 0.61% |
+| Completeness | 0.983 ± 0.007 |
+| AMI | 0.950 ± 0.009 |
+| ARI HDBSCAN (3-seed mean) | **0.859 ± 0.018** |
+| ARI Agglo Ward K=42 (3-seed mean) | **0.9014 ± 0.022** |
+| noise_pct (defect, HDBSCAN) | 1.48% (3-seed mean) |
 | class_capture_rate | 43/43 = 1.000 |
-| HDBSCAN cfg | eom mcs=12 ms=3 |
+| HDBSCAN cfg | eom mcs=12 ms=3, defect-only, no eps |

+★ v6 reference numbers (iter 37 / B5 family, **superseded by v7 on multi-seed**):
+Completeness 0.991, AMI 0.960, ARI 0.870 single-seed / 0.866 ± 0.014 3-seed, noise 0.61%.
+B5 multi-seed reproducibility (RESULTS §17b): 2-seed avg 0.8343 ± 0.031 (HDBSCAN) /
+0.8920 ± 0.062 (Agglo K=42) — std 1.7-2.8× higher than NEW; B5 seed=42 0.9358 Agglo claim
+retracted as cherry-picked outlier (seed=1 reproduce = 0.8482, Δ −0.088).
```

### v6 dual-cfg recipe bullets (line 137-143)

```diff
-→ **dual-cfg recipe** (paper v0.8):
-- **Frontier 2 (known-K + Agglo Ward)** → **B5 (Local + NeCo combined)** = absolute SOTA
-- **Frontier 1 (unknown-K + HDBSCAN + Normal-stream)** → NEW (NeCo only)
-→ **Local DenseCL는 NOT deprecated**. v5 "substitutable" framing 은 HDBSCAN ...
+→ **dual-cfg recipe RETRACTED in v7** (★ 2026-05-13): see ★ N1 v7 FINAL section above
+(line 98). v6 per-class purity observations (RESULTS §16) preserved as single-seed only.
+Multi-seed Agglo K=42: NEW 0.9014 ± 0.022 > B5 0.8920 ± 0.062 (RESULTS §17b).
+
+→ **Single-cfg recipe (NEW)** for both frontiers (multi-seed authoritative):
+- Frontier 1 (unknown-K + HDBSCAN) → NEW
+- Frontier 2 (known-K + Agglo Ward K=42) → NEW (same cfg)
+
+→ v6 framing "Local DenseCL NOT deprecated" superseded by v7: Local DenseCL is
+**operationally optional**, not required for SOTA.
```

---

## P1-3 — FIGURES.md F-N7-lattice caption (line 82-91)

**Status**: applied — caption now matches v7 single-cfg recipe.

```diff
-caption 후보 (★ N1 v6 refined 2026-05-12): "Four-component lattice ...
-Per-class Agglomerative Ward K=42 purity (RESULTS §16) refines this aggregate
-equivalence to **complementary per-class inductive bias** (N1 v6) ...
-Frontier 1 cfg (iter 70, top row) drops Local for unknown-K HDBSCAN deployment;
-Frontier 2 cfg (B5) keeps both for known-K Agglo Ward absolute SOTA ARI 0.9358."
+caption 후보 (★ N1 v7 FINAL revised 2026-05-13): "Four-component lattice ...
+Per-class purity flips at single-seed=42 Agglomerative Ward K=42 (RESULTS §16)
+do NOT propagate to multi-seed avg (RESULTS §17b): NEW (NeCo only) > B5 (Local +
+NeCo) on Agglo Ward K=42 multi-seed (0.9014 ± 0.022 vs 0.8920 ± 0.062, B5 std
+2.8× higher). **v7 single-cfg recipe = iter 70 NEW for both Frontier 1 and
+Frontier 2.** Per-class complementarity preserved as single-seed observation
+only; v6 dual-cfg recipe retracted."
```

---

## P2-1 — INTRODUCTION.md C7 body (line 223-228)

**Status**: applied — single-seed 0.9358 claim now has explicit v7 multi-seed retraction inline.

```diff
-(+10pp). Net average per-class purity: B5 = 97.0%, NEW = 96.2%, Δ −0.83pp.
-The absolute SOTA single-seed ARI is **B5 (Local + NeCo combined) Agglomerative
-Ward K=42 = 0.9358**, strictly above iter 70 NEW 0.9200 (Δ +0.0158). The v5
-"substitutable" framing is therefore refined in v6 to **complementary**, and
-**Local DenseCL is NOT deprecated** — it carries sub-pattern variant integration
-that NeCo cannot recover alone.
+(+10pp). Net average per-class purity (single-seed=42): B5 = 97.0%, NEW = 96.2%,
+Δ −0.83pp. The single-seed=42 max ARI is **B5 (Local + NeCo combined) Agglomerative
+Ward K=42 = 0.9358**, above iter 70 NEW 0.9200 (Δ +0.0158) — ★ **v7 retracted as
+multi-seed SOTA**: seed=1 reproduce = 0.8482 (Δ −0.0876), B5 2-seed avg 0.8920 ±
+0.062 BELOW NEW 3-seed avg 0.9014 ± 0.022 (std 2.8× higher). The v5 "substitutable"
+framing is refined in v6 to **single-seed complementary**; the v7 multi-seed
+correction collapses to **NEW alone covers both frontiers** — Local DenseCL is
+**operationally optional, not required for SOTA**.
```

---

## P2-2 — METHOD.md §3.6 body deprecation banner

**Status**: applied — prepended v7 FINAL CORRECTION banner at §3.6 header.

```diff
 ### 3.6 NeCo vs DenseCL Local InfoNCE — complementary, not substitutable
-(★ N1 v6 FINAL, iter 67-77 + per-class K=42 breakdown 2026-05-12)
+(★ N1 v6 FINAL, iter 67-77 + per-class K=42 breakdown 2026-05-12; ★ N1 v7 FINAL multi-seed correction 2026-05-13)
+
+★ **v7 FINAL CORRECTION 2026-05-13 (iter 84 B5 seed=1 reproducibility)**:
+The v6 "B5 absolute SOTA Agglo K=42 ARI 0.9358" claim is retracted on multi-seed
+grounds — B5 seed=1 reproduce = 0.8482 (Δ −0.0876), B5 2-seed avg 0.8920 ± 0.062
+< NEW 3-seed avg 0.9014 ± 0.022 (std 2.8× lower). The single-seed=42 per-class
+purity flips below are preserved as observations, but the dual-cfg recipe collapses
+to single-cfg (NEW) recommendation for **both** frontiers. Local DenseCL is
+**operationally optional, not required for SOTA**. See §3.6 Conclusion block
+and RESULTS §17 for the v7 final recipe.

 The 11-iteration four-component lattice exploration ...
```

---

## Cosmetic standardization — `0.8588 ± 0.018` → `0.859 ± 0.018` (paper section files)

**Status**: `replace_all` applied to 8 paper section files (and SUMMARY/REPORT). ITERATIONS.md preserved (append-only). manager_report audit files (`claim_0859_origin_trace.md`, `consolidation_pass_260513.md`, `performance_data_260513.md`, `tier1_protocol_fair_4run.md`) preserved at 4-decimal — these are provenance/audit traces that intentionally reference the precise 0.8588 value as the audit anchor.

**Affected files** (replace_all `0.8588 ± 0.018` → `0.859 ± 0.018` and `0.8588 +/- 0.018` → `0.859 +/- 0.018`):

| file | before count | after count |
|---|---:|---:|
| `ABSTRACT.md` | 5 (mixed ± / +/-) | 0 → all 0.859 |
| `INTRODUCTION.md` | 1 (+/-) | 0 → 0.859 |
| `METHOD.md` | 2 (±) | 0 → 0.859 |
| `RESULTS.md` | 5 (±) | 0 → 0.859 |
| `DISCUSSION.md` | 5 (mixed) | 0 → 0.859 |
| `CONCLUSION.md` | 3 (mixed) | 0 → 0.859 |
| `README.md` | 1 (±) | 0 → 0.859 |
| `manager_report/SUMMARY.md` | 7 (mixed) | 0 → 0.859 |
| `manager_report/REPORT.md` | 1 (±) | 0 → 0.859 |

**Bare 0.8588** (without ± paired form) retained where it appears inside multi-seed seed-by-seed reproduction tables (e.g., `| HDBSCAN | 0.8797 | 0.8491 | 0.8475 | 0.8588 | 0.018 |`) to preserve column alignment with the 4-decimal seed values 0.8797 / 0.8491 / 0.8475.

---

## Files NOT modified

| file | reason |
|---|---|
| `ITERATIONS.md` | append-only policy — past iter entries (82, 83, pre-84) contain v6 "absolute SOTA" framing; correctly retracted in iter 84 entry per append-only mechanism |
| `manager_report/claim_0859_origin_trace.md` | audit/provenance trace — explicitly tracks 4-decimal 0.8588 as anchor |
| `manager_report/consolidation_pass_260513.md` | the audit driving this patch run; preserving original audit form |
| `manager_report/performance_data_260513.md` | performance baseline reference |
| `manager_report/tier1_protocol_fair_4run.md` | tier1 protocol audit; intentional 4-decimal precision |
| `manager_report/apples_to_apples_7run_3cfg.md` | apples comparison audit |
| `manager_report/b5_recluster_eom_ms3.md` | B5 reclustering audit |
| `manager_report/cluster_analysis_b4_noNeCo_3seed.md` | B4 cluster analysis |
| `manager_report/sota_tangents_final_consolidation.md` | sota consolidation |
| `ABLATION_PLAN.md` | pre-v6 historical doc (scoped to B0-B5 isolation) |

---

## Verification summary

After all patches:

| Section | v6 "B5 0.9358 absolute SOTA" residual? | v7 retraction inline? |
|---|:-:|:-:|
| ABSTRACT.md | no | yes (v0.9 CURRENT) |
| INTRODUCTION.md | no | yes (C7 body now has ★ v7 retracted inline marker) |
| METHOD.md | no | yes (§3.6 v7 banner + §7.1 protocol provenance) |
| RESULTS.md | no | yes (§15 obs 4 + §15b NMI + §16d + §17 retraction table) |
| DISCUSSION.md | no | yes (§7.10.7 + §7.12.4 + §7.13 practitioner) |
| CONCLUSION.md | no | yes (§8.6 N7 + §8.7 Frontier 2 + §8.8 v7 closing) |
| FIGURES.md | no | yes (F-N7-lattice caption v7 revised) |
| README.md | no | yes (현재 SOTA table now iter 70 NEW; v6 section retracted) |
| ITERATIONS.md | iter 82/83 v6 entries preserved | append-only iter 84 carries v7 retraction |
| SUMMARY.md | no | yes (Phase 3 결론 + §10 운영자 선택지) |
| REPORT.md | no | yes (Phase 3 결론 + contributions block) |

**Multi-seed compliance**: ★ achieved at all practitioner-facing recommendation tables. A reader cannot find a paper section that currently recommends B5 for the known-K frontier.

**Headline provenance**: ★ disclosed at METHOD.md §7.1 (new). Reviewers reproducing via default `_eval_contrastive_unknown_n50.py` now have an explicit map to the defect-only post-hoc HDBSCAN protocol that produces the headline 0.859 ± 0.018.

**Tier 1+2 metric policy**: ★ unchanged — no custom metrics introduced.

---

## Total stats

- **8 patches applied** (P-0a, P0-1, P0-2 [×3 edits], P0-3a [×2 edits], P0-3b [×2 edits], P1-1 [×3 edits], P1-2 [×2 edits], P1-3, P2-1, P2-2).
- **9 paper section files modified**: ABSTRACT, INTRODUCTION, METHOD, RESULTS, DISCUSSION, CONCLUSION, FIGURES, README, SUMMARY+REPORT (manager_report).
- **0 ITERATIONS edits** (append-only preserved).
- **Cosmetic 0.8588 ± 0.018 → 0.859 ± 0.018** applied across all paper section files (4-decimal form preserved in audit/provenance manager_report traces).

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/patches_applied_260513.md
