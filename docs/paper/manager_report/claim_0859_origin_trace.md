# Claim "NEW + HDBSCAN ARI 0.859 ± 0.018" — origin trace (2026-05-13)

> **Scope**: read-only audit. Trace the exact data source behind the paper claim
> "NEW + HDBSCAN ARI 0.859 ± 0.018" (or `0.8588 ± 0.018`) appearing across
> ABSTRACT / INTRODUCTION / METHOD / RESULTS / DISCUSSION / CONCLUSION /
> ITERATIONS / FIGURES / README / SUMMARY / REPORT. No paper section edits.
>
> **Trigger**: user observation that `eval_summary.json` 3-seed NEW HDBSCAN
> (eom mcs=12 ms=3 eps=0.06) ARI = 0.7375 ± 0.0634, in disagreement with paper
> headline 0.8588 ± 0.018.

---

## 1. Verdict (one-liner)

The paper claim **`0.8588 ± 0.018`** (≈ `0.859 ± 0.018`) is **NOT** an
`eval_summary.json` figure. It is the 3-seed mean of a **post-hoc HDBSCAN
re-cluster on the defect-only subset of the embedding**, saved as

- `outputs_contrastive_260512_001719/tier1_neco_replaces_local.json` (seed=42)
- `outputs_contrastive_260512_010113/tier1_sota_seed1.json` (seed=1)
- `outputs_contrastive_260512_014507/tier1_sota_seed2.json` (seed=2)

with HDBSCAN cfg `eom + mcs=12 + ms=3` (note: **no `cluster_selection_epsilon`**
in the tier1 jsons; the eval_summary applies `eps=0.06` which the tier1
post-hoc deliberately drops).

The `eval_summary.json` 0.7375 ± 0.0634 is a different measurement — it
includes Normal samples in the same HDBSCAN run (eom + mcs=12 + ms=3 + **eps=0.06**)
and reports `with_normal` ARI. The `without_normal` block in the JSON is a
**duplicate of `with_normal`** for these 3 runs (identical values down to the
last decimal — likely a bug where Normal filtering was not applied to the
"without_normal" computation), so it is **not** a usable defect-only number
from eval_summary.

## 2. Numeric reconciliation

### 2.1 tier1 post-hoc (defect-only re-cluster) — paper headline source

| seed | run_dir | ARI | AMI | Comp | Hom | noise | n_cl | Sil | source file |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 260512_001719 | 0.8797 | 0.9594 | 0.9872 | 0.9479 | 0.87% | 37 | 0.7860 | `tier1_neco_replaces_local.json` |
| 1  | 260512_010113 | 0.8491 | 0.9488 | 0.9856 | 0.9323 | 1.05% | 35 | 0.7832 | `tier1_sota_seed1.json` |
| 2  | 260512_014507 | 0.8475 | 0.9428 | 0.9747 | 0.9335 | 2.53% | 36 | 0.8130 | `tier1_sota_seed2.json` |
| **avg** | — | **0.8588** | **0.9503** | **0.9825** | **0.9379** | **1.48%** | — | **0.7941** | — |
| **std (ddof=1)** | — | **0.0181** | **0.0084** | **0.0068** | **0.0087** | **0.91pp** | — | **0.0165** | — |

★ This **exactly** reproduces paper claims:
- ARI 0.8588 ± 0.018 (ABSTRACT v0.9, DISCUSSION §7.10/§7.12, ITERATIONS iter 72)
- AMI 0.9503 ± 0.008 (ITERATIONS line 1413)
- Comp 0.9825 ± 0.007 (ITERATIONS line 1412)
- Sil 0.7941 ± 0.017 (ITERATIONS line 1414, used as the "+30% over B5 0.6104" headline)
- noise_avg 1.48% (ABSTRACT v0.9 / ITERATIONS line 1415)

DISCUSSION.md line 892 shows the per-seed table that matches exactly:
> `| HDBSCAN | 0.8797 | 0.8491 | 0.8475 | 0.8588 | 0.018 |`

### 2.2 eval_summary `with_normal` scope (default) — user-observed source

| seed | run_dir | ARI | AMI | Comp | Hom | noise | n_cl |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | 260512_001719 | 0.8269 | 0.9257 | 0.9173 | 0.9531 | 7.13% | 39 |
| 1  | 260512_010113 | 0.6865 | 0.8943 | 0.8754 | 0.9407 | 11.37% | 37 |
| 2  | 260512_014507 | 0.6990 | 0.8942 | 0.8801 | 0.9355 | 13.00% | 38 |
| **avg** | — | **0.7375** | **0.9048** | **0.8909** | **0.9431** | 10.50% | — |
| **std (ddof=0)** | — | **0.0634** | **0.0148** | **0.0188** | **0.0091** | 3.03pp | — |
| **std (ddof=1)** | — | **0.0777** | **0.0181** | **0.0230** | **0.0091** | 3.71pp | — |

★ The user-hinted "AMI 0.9048 ± 0.0148 / Completeness 0.8909 ± 0.0188" exactly
matches **eval_summary `with_normal` 3-seed mean with population std (ddof=0)**.
The ARI in this scope is 0.7375 ± 0.0634 (also ddof=0), confirming the user's
observation that eval_summary ARI differs sharply from the paper claim.

### 2.3 eval_summary `without_normal` scope — IDENTICAL to `with_normal` (likely bug)

All three `eval_summary.json` files have `with_normal` and `without_normal`
blocks with **bit-identical** ARI, AMI, Comp, Hom, noise_pct, n_clusters, and
even per-class noise dictionaries (including `Normal: {n: 1000, ...}`).

Example seed=42 (260512_001719/eval/eval_summary.json):

```
"with_normal":    { "ari": 0.8269278656037636, "noise_pct": 7.13, "n_clusters": 39, "n": 2146, ... }
"without_normal": { "ari": 0.8269278656037636, "noise_pct": 7.13, "n_clusters": 39, "n": 2146, ... }
```

`n` is 2146 in both — the Normal=1000 wafer were **not** filtered out for
`without_normal`. The duplicate suggests an evaluator bug where the "without
Normal" code path was wired to the "with Normal" computation; it does **not**
represent a defect-only HDBSCAN.

## 3. HDBSCAN cfg drift between two sources

| field | eval_summary cfg | tier1 post-hoc cfg | implication |
|---|---|---|---|
| `MIN_CLUSTER_SIZE` | 12 | 12 | match |
| `MIN_SAMPLES` | 3 | 3 | match |
| `CLUSTER_SELECTION_METHOD` | eom | eom | match |
| `CLUSTER_SELECTION_EPSILON` | **0.06** | **0** (not in JSON, default) | ★ different |
| input embedding scope | full 2146 (incl. 1000 Normal) | defect-only (≈1146) | ★ different |
| L2 normalization | yes (eval pipeline default) | yes (post-hoc default) | match |

→ The tier1 post-hoc uses a **different HDBSCAN protocol** (no eps, defect-only).
The +0.012 to +0.150 ARI gap per seed (0.8797 vs 0.8269, 0.8491 vs 0.6865,
0.8475 vs 0.6990) is driven mostly by **scope** (Normal in/out) and partly by
**eps** (0.06 vs none). Per DISCUSSION.md / RESULTS.md the paper authors
deliberately moved to the no-eps + defect-only protocol for "apples-to-apples"
comparison vs B5 (which was also run under the same defect-only no-eps
protocol in iter 82-83).

## 4. Provenance map — `0.859 / 0.8588` occurrences across paper docs

### 4.1 `0.8588 ± 0.018` (4-decimal form) — tier1 post-hoc source

| file:line | scope | direct quote |
|---|---|---|
| ABSTRACT.md:182 | v0.5 headline | "3-seed mean ARI 0.8588 +/- 0.018, Silhouette 0.7941 +/- 0.017" |
| ABSTRACT.md:242 | v0.6 dual-cfg | "iter-70 NEW ... ARI 0.8588 +/- 0.018, Silhouette 0.7860, defect-noise 1.48%" |
| ABSTRACT.md:297 | v0.7 dual-frontier | "single-seed ARI 0.8797 / 3-seed 0.8588 ± 0.018" |
| ABSTRACT.md:349 | v0.8 | same |
| ABSTRACT.md:416 | v0.9 (CURRENT) | "Frontier 1 ... 3-seed mean ARI 0.8588 ± 0.018, defect-noise 1.48% mean" |
| CONCLUSION.md:254 | §8.6 v7 corrected block | "3-seed mean ARI 0.8588 +/- 0.018 (vs B5 0.856 +/- 0.012)" |
| CONCLUSION.md:452 | §8.7 dual-frontier | "0.8588 ± 0.018, with Normal-cluster consolidation" |
| CONCLUSION.md:490 | §8.8 v7 recipe table | Regime row "0.8588 ± 0.018 (3-seed)" |
| DISCUSSION.md:508 | §7.10.4 corrected | "NEW (NeCo + Queue + NEG, no Local) 0.8588 +/- 0.018" |
| DISCUSSION.md:726 | §7.10.7 v7 table | HDBSCAN cell "0.8588 ± 0.018" |
| DISCUSSION.md:760 | §7.13 practitioner | Frontier 1 row "0.8588 ± 0.018 (3-seed)" |
| DISCUSSION.md:892 | §7.15 multi-seed table | Per-seed: 0.8797 / 0.8491 / 0.8475 → 0.8588 std 0.018 |
| DISCUSSION.md:916 | §7.16 Frontier 1 | "3-seed mean ARI: 0.8588 ± 0.018" |
| METHOD.md:314 | §3.6 conclusion v7 | "Unknown-K real-world (HDBSCAN): multi-seed avg ARI 0.8588 ± 0.018" |
| METHOD.md:376 | §3.6.2 dual-frontier | "3-seed ARI 0.8588 ± 0.018 (HDBSCAN, defect-only)" |
| ITERATIONS.md:1411 | iter 72 final claim | "ARI avg 0.8588 ± 0.018" |
| ITERATIONS.md:1601 | iter 72 cfg table | "NEW 4-comp ... 0.8588 ± 0.018" |
| ITERATIONS.md:1825 | iter 82-83 multi-seed | HDBSCAN row "0.8797 0.8491 0.8475 0.8588 0.018" |
| ITERATIONS.md:1999 | iter 84 retraction | "HDBSCAN 0.8588 ± 0.018" |
| ITERATIONS.md:2025 | iter 84 v7 final | "Unknown-K HDBSCAN (0.8588 ± 0.018) AND Oracle-K Agglomerative Ward (0.9014 ± 0.022)" |
| consolidation_pass_260513.md:89-91 | audit note | flags 0.8588 vs 0.859 notation drift |

### 4.2 `0.859 ± 0.018` (3-decimal form) — same source, abbreviated

| file:line | direct quote |
|---|---|
| ABSTRACT.md:191 | "iter-37 B5 (Local-based) at ARI 0.866 +/- 0.014, NEW (NeCo-replaces-Local) at ARI 0.859 +/- 0.018" |
| INTRODUCTION.md:231 | "3-seed mean ARI 0.8588 +/- 0.018 over B5 0.856 +/- 0.012 is marginal +0.003" |
| INTRODUCTION.md:295 | "iter 70 NEW (ARI 0.880 single-seed, 0.859 ± 0.018 3-seed" |
| CONCLUSION.md:380 | "(N2) Multi-seed honesty: iter-37 B5 at 0.866 +/- 0.014, NEW at 0.859 +/- 0.018" |
| ITERATIONS.md:1837 | "Unknown-K (real-world): iter 70 NEW + HDBSCAN = 0.859 ± 0.018 (3-seed)" |
| RESULTS.md:635 | "iter 70 NEW + HDBSCAN = ARI 0.880 (single) / 0.859 ± 0.018 (3-seed)" |
| RESULTS.md:664 | "ARI 0.880 single / 0.859 ± 0.018 3-seed" |
| sota_tangents_final_consolidation.md:6 | "ARI 0.859 ± 0.018 (3-seed)" |
| sota_tangents_final_consolidation.md:57, 116, 119, 131, 204, 212, 228, 236 | reference baseline "0.859" |
| SUMMARY.md:783 | "iter 70 NEW + HDBSCAN = 0.859 ± 0.018 (3-seed)" |
| REPORT.md:426 | "iter 70 NEW + HDBSCAN = 0.859 ± 0.018 (3-seed)" |

Both notations refer to **the same 3 tier1 post-hoc jsons**. The 4-decimal form
is the precise mean of the underlying values (0.8588); the 3-decimal form is
just rounded.

### 4.3 Stale README "현재 SOTA" table — **different** number 0.866

`README.md:57-67` reports `ARI (3-seed mean) | 0.866 ± 0.014` — this is **iter
37 B5 family** numbers from a pre-v6 era, not the iter 70 NEW headline. The v0.9
ABSTRACT and v7 final headline is 0.8588 ± 0.018. This staleness is already
flagged in `consolidation_pass_260513.md §2.4`.

## 5. Was the post-hoc re-cluster a custom Python eval?

Yes. Evidence:

- `tier1_neco_replaces_local.json` / `tier1_sota_seed1.json` / `tier1_sota_seed2.json`
  are **standalone single-row JSON arrays** with hdbscan output (ms, mcs, noise_pct,
  n_clusters, ARI, AMI, Comp, Hom, Sil, NMI, cap). The schema differs from
  `eval_summary.json` (no `with_normal`/`without_normal` block, no per-class noise,
  no `hdbscan_cfg` / `hdbscan_overrides` keys, no `stress_test`).
- The schema matches a **custom sweep wrapper** that runs HDBSCAN directly on a
  pre-loaded embedding numpy array, with mcs/ms grid (here only `ms=3, mcs=12`).
- The Sil value (0.7860 for seed=42) is computed on **defect-only embedding** —
  this is consistent with the "apples-to-apples HDBSCAN protocol (eom + mcs=12 +
  ms=3, defect-only)" framing in DISCUSSION.md §7.10.7.
- The same protocol is the source of `tier1_clustering_benchmark.json` at the
  repo root (B4 / B5 / iter 70 NEW × 5 clustering methods, single-seed=42),
  which gives HDBSCAN_ARI = 0.8797093031666471 for "iter 70 NEW SOTA" — identical
  to `tier1_neco_replaces_local.json[0].ARI`.

→ The paper claim ARI 0.8588 ± 0.018 is **mathematically correct** and traceable
to 3 deterministic post-hoc re-clusters. It is **not** a manual calculation
error. The disagreement with eval_summary is **methodological** (scope: defect-only
vs full 2146; eps: 0 vs 0.06).

## 6. Manual calculation check

```
mean(0.8797093, 0.8491196, 0.8475059) = 0.858778
std(ddof=1)                            = 0.0181  → "± 0.018"
std(ddof=0)                            = 0.0148
```

→ The paper's ± 0.018 std uses **ddof=1 (sample std)**. The same convention is
used for B5 ± 0.012 and ± 0.014 elsewhere. Consistent within the paper.

Note: in §2.2 above the user-hinted AMI ± 0.0148 / Comp ± 0.0188 use **ddof=0**.
Those eval_summary-derived stds use a different convention than the tier1
post-hoc stds. The paper headline uses tier1 ddof=1 throughout.

## 7. Recommendation for `consolidation_pass_260513.md` P0/P1 priorities

### 7.1 P0/P1 do NOT need to change priority order — claim is reproducible

The 8 patches (P-1 through P-8) target **v6→v7 retraction propagation**, not the
0.8588 number itself. Since the 0.8588 ± 0.018 claim is reproducible from the 3
tier1 post-hoc jsons, no patch needs to "correct" the number. The priorities
stay as in the consolidation pass.

### 7.2 ★ ADD a new pre-P0 patch — "P-0a Evaluation Protocol Disclosure"

The paper currently has a hidden methodological asymmetry: headline ARI 0.8588
uses defect-only HDBSCAN with **no `eps`**, while the canonical `eval_summary.json`
saves a full-2146 HDBSCAN **with `eps=0.06`**. A reviewer reproducing the work
from the eval pipeline alone would see 0.7375, not 0.8588 — a 12 pp gap that
looks like a fundamental disagreement.

**Recommended new patch** (insert before P-1 as `P-0a`):

> In METHOD §3.5 (or equivalent eval pipeline section), add an explicit
> "**Evaluation HDBSCAN protocol**" subsection stating:
>
> 1. The headline ARI/Sil/AMI/Comp numbers (RESULTS §17, ABSTRACT v0.9, etc.)
>    are measured under **defect-only HDBSCAN with `metric=euclidean,
>    cluster_selection_method=eom, min_cluster_size=12, min_samples=3, no
>    cluster_selection_epsilon`**, applied to the L2-normalized 128-d defect
>    embedding (Normal samples excluded).
> 2. The `eval_summary.json` files include a `with_normal` block (full 2146
>    samples, including Normal=1000, with `eps=0.06`) — this is **not** the
>    headline metric and should not be quoted as the paper SOTA.
> 3. The `eval_summary.json` `without_normal` block in iter 70/71/72 runs is
>    **bit-identical to `with_normal`** (a likely evaluator bug where Normal
>    filtering was not applied). The headline defect-only numbers are
>    therefore computed by the **post-hoc tier1 wrapper** (`tier1_*.json`),
>    not by the eval_summary `without_normal` field.
> 4. The full per-seed defect-only numbers are tabulated in DISCUSSION.md
>    §7.15 / ITERATIONS.md iter 82-83 / tier1_clustering_benchmark.json.

This patch is **information-only** (no number changes) and prevents a
reproducibility-audit failure. Priority: **P0** alongside P-1 / P-2 / P-7.

### 7.3 P1 secondary recommendation — fix `eval_summary` `without_normal` bug

Out of scope for the paper-recorder agent (code change), but worth surfacing to
the user: the `without_normal` block in eval_summary.json should actually
exclude Normal samples. The current bug makes `with_normal` and `without_normal`
identical, which loses an important diagnostic. Filed as **out-of-scope code
bug** — recommend a separate non-paper issue.

### 7.4 P1 cosmetic — standardize 0.8588 vs 0.859 notation

Already noted in `consolidation_pass_260513.md §2.3` / §5 (informational). No
change to priority — keep as P1 cosmetic cleanup.

## 8. Summary table — claim vs source

| paper claim | mathematically derived from | scope | confidence |
|---|---|---|:-:|
| `NEW + HDBSCAN ARI 0.8588 ± 0.018` (3-seed) | `tier1_neco_replaces_local.json` + `tier1_sota_seed1.json` + `tier1_sota_seed2.json` (defect-only re-cluster, eom mcs=12 ms=3, no eps) | defect-only (n ≈ 1146) | ★ verified, reproducible |
| `NEW + HDBSCAN AMI 0.9503 ± 0.008` (3-seed) | same 3 tier1 jsons | defect-only | ★ verified |
| `NEW + HDBSCAN Comp 0.9825 ± 0.007` (3-seed) | same 3 tier1 jsons | defect-only | ★ verified |
| `NEW + HDBSCAN Sil 0.7941 ± 0.017` (3-seed) | same 3 tier1 jsons | defect-only | ★ verified |
| `NEW + HDBSCAN noise 1.48% mean` (3-seed) | same 3 tier1 jsons | defect-only | ★ verified |
| `NEW + HDBSCAN ARI 0.7375 ± 0.063` (user obs) | `eval_summary.json` `with_normal` (or duplicated `without_normal`) | full 2146 incl. Normal=1000, eps=0.06 | ★ verified — NOT a paper headline |

The two number sets are **not contradictory** — they are different protocols
on the same embeddings. The paper headline uses the defect-only protocol
consistently; the user observation was reading `eval_summary` which uses the
with-Normal full-scope protocol.

---

## 9. Files inspected

- `D:/project/unknown-contrastive/outputs_contrastive_260512_001719/tier1_neco_replaces_local.json`
- `D:/project/unknown-contrastive/outputs_contrastive_260512_001719/eval/eval_summary.json`
- `D:/project/unknown-contrastive/outputs_contrastive_260512_010113/tier1_sota_seed1.json`
- `D:/project/unknown-contrastive/outputs_contrastive_260512_010113/eval/eval_summary.json`
- `D:/project/unknown-contrastive/outputs_contrastive_260512_014507/tier1_sota_seed2.json`
- `D:/project/unknown-contrastive/outputs_contrastive_260512_014507/eval/eval_summary.json`
- `D:/project/unknown-contrastive/tier1_clustering_benchmark.json`
- `D:/project/unknown-contrastive/docs/paper/ABSTRACT.md` (lines 182, 191, 242, 297, 349, 416)
- `D:/project/unknown-contrastive/docs/paper/INTRODUCTION.md` (lines 231, 295)
- `D:/project/unknown-contrastive/docs/paper/METHOD.md` (lines 314, 376)
- `D:/project/unknown-contrastive/docs/paper/RESULTS.md` (lines 521, 635, 664)
- `D:/project/unknown-contrastive/docs/paper/DISCUSSION.md` (lines 508, 726, 760, 892, 916)
- `D:/project/unknown-contrastive/docs/paper/CONCLUSION.md` (lines 254, 380, 452, 490)
- `D:/project/unknown-contrastive/docs/paper/ITERATIONS.md` (lines 1411, 1601, 1825, 1837, 1999, 2025)
- `D:/project/unknown-contrastive/docs/paper/README.md` (lines 57-67 stale, 119-143 v6 block)
- `D:/project/unknown-contrastive/docs/paper/manager_report/SUMMARY.md` (line 783)
- `D:/project/unknown-contrastive/docs/paper/manager_report/REPORT.md` (line 426)
- `D:/project/unknown-contrastive/docs/paper/manager_report/consolidation_pass_260513.md` (lines 86-97, 331-332)
- `D:/project/unknown-contrastive/docs/paper/manager_report/sota_tangents_final_consolidation.md` (lines 6, 57, 116, 119, 131, 204, 212, 228, 236)

No paper section files were modified. Only this trace report was written.

---

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/claim_0859_origin_trace.md
