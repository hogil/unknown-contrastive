# Step 1 (eval-only) — Paper addition summary (2026-05-13)

**Plan reference**: `C:\Users\hgcho\.claude\plans\floating-splashing-key.md` Roadmap Step 1.
**Single source-of-truth**: `docs/paper/manager_report/step1_eval_only_summary_260513.md`.
**Trigger**: user directive — "RESULTS.md 와 ABLATION_PLAN.md 에 Step 1 (eval-only) 결과를 추가하라."

This file is the **edit summary** for the paper additions made on 260513. All numbers in
the inserts are direct copies from the source-of-truth (no derivation); RESULTS / METHOD /
ITERATIONS now reflect Step 1a / 1b / 1c outcomes.

---

## 1. Files modified

| file | section affected | type | size delta |
|---|---|---|---|
| `docs/paper/RESULTS.md` | §19 (NEW) — Step-by-step performance improvement (Step 1 eval-only) | append after §18.8 | +109 lines |
| `docs/paper/METHOD.md` | §4c (NEW) — Post-process refinement (soft τ-reassignment) | insert between §4b.7 and §5 | +84 lines |
| `docs/paper/ABLATION_PLAN.md` | NEW completion record at file end | append after 정책 정합 section | +50 lines |
| `docs/paper/ITERATIONS.md` | NEW iter 86 entry | append after iter 84 (append-only) | +89 lines |

Total: +332 lines, 4 file edits, **no overwrites** (append/insert only — ITERATIONS.md
strictly append-only respected).

---

## 2. Step 1 result table (single source-of-truth copy)

P1 class capture rate = 1.000 across all steps. Baseline = NEW 3-seed (iter 70/71/72),
defect-only Tier1 HDBSCAN (eom mcs=12 ms=3).

| Step | Method addition | P2 noise % | P3 Comp | P4 Hom | AMI | ARI | std | RankMe |
|:-:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | NEW (Global+NeCo+Queue+NEG) baseline | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | **23.44 ± 1.80** |
| 1a | + RankMe representation column | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | 23.44 |
| 1b | + HDBSCAN ε ∈ [0.00, 0.15] sweep | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | — |
| 1c τ = 0.9 | + soft KNN-softmax τ-reassign | **0.49** | 0.9952 | 0.9436 | 0.9616 | 0.8709 | 0.0132 | — |
| 1c τ = 0.7 | + soft τ-reassign | **0.15** | 0.9944 | 0.9430 | 0.9607 | 0.8696 | **0.0123** | — |
| 1c τ = 0.5 ★ | + soft τ-reassign | **0.00** ★★ | 0.9938 | 0.9424 | 0.9600 | 0.8681 | 0.0125 | — |

---

## 3. Paper claims locked-in by Step 1

| # | claim | paper location |
|:-:|---|---|
| **N9 reinforcement** | "On strong contrastive embeddings (NEW recipe), HDBSCAN `cluster_selection_epsilon` parameter is **redundant** — the cluster tree is determined by the `(method, mcs, ms)` triple alone (here: `eom, 12, 3`). 21-cell sweep (3 seeds × 7 ε values) all converge to identical Tier 1+2 metrics." | RESULTS §19.2, METHOD §4c.5 |
| **N9 extension** | "Soft KNN-softmax reassignment of HDBSCAN noise points achieves **0.00 % noise rate** at τ = 0.5 with marginal ARI cost (Δ = −0.005, well within seed std 0.014)." | RESULTS §19.3, METHOD §4c.4 |
| **N10 (new)** | "RankMe (Garrido et al. 2023) is informative for **cross-seed stability of representation quality** (NEW std 1.80 vs B5 std 4.99 → NEW 64 % more reproducible), **NOT for ARI ranking** (Spearman ρ = −0.429, n = 7 runs)." | RESULTS §19.1 |

---

## 4. Production deployment recommendation (METHOD §4c.4)

```yaml
post_process:
  type:                 knn_softmax_tau_reassign
  k:                    10
  similarity:           cosine
  softmax_temperature:  0.1
  tau:                  0.5         # production lock — every wafer receives a cluster label
  source:               defect-only HDBSCAN (eom, mcs=12, ms=3) baseline
```

- Throughput overhead: ≈ 1 ms / wafer (10-NN against pre-fitted FAISS index on 1146
  embedding points) — negligible vs 14 ms encoder forward (METHOD §4b.7).
- Reassigns ≈ 17 / 1146 wafers (3-seed avg) initially flagged as noise.
- Paper research evaluation retains **∞ baseline** (no reassign) for apples-to-apples
  comparison with prior HDBSCAN-only literature.

---

## 5. ARI historical-vs-inline reconciliation

| context | value | source |
|---|---:|---|
| paper headline / ABSTRACT v0.9 / README | **0.859 ± 0.018** (3-seed HDBSCAN) | prior run-time measurement |
| Step 1 inline (§19, this addition) | **0.8731 ± 0.0140** | re-measured this iter, same protocol |
| Δ | **0.014** | within HDBSCAN tree non-determinism (sklearn build-order variance) |

**Both values are valid measurements**, recorded together in RESULTS §19.6.
ABSTRACT / INTRODUCTION / CONCLUSION historical inline citations of 0.859 are
**preserved**; the inline-measured 0.8731 is used only as Step 1c Δ-reassignment
baseline. Per user directive: "0.8588 (paper claim historical) vs 0.8731 (inline)
의 0.014 차이는 HDBSCAN tree 비결정성 — 두 값 모두 valid, paper 는 historical
인용 유지".

---

## 6. Step 1 → next step decision matrix

| outcome | downstream decision |
|---|---|
| Step 1a (RankMe ρ = −0.429, n = 7) | **report only** as cross-seed stability column, NOT as SOTA arbiter |
| Step 1b (ε zero-effect across 21 cells) | **epsilon deprecated** for NEW production cfg |
| Step 1c (noise 0 % at τ = 0.5, ARI Δ = −0.005) | **production cfg lock: τ = 0.5** for every-wafer labeling |
| Step 2 (EMA target encoder) | requires training dispatch — **pending user approval** |

---

## 7. Append-only compliance + data integrity

- **ITERATIONS.md append-only** respected: iter 86 added after iter 84 entry, no past
  iteration body modified. The "다음 (iter 85+)" line of iter 84 is preserved
  unchanged; iter 86 entry starts after it under a new `---` separator.
- **Tier 1+2 official metrics only** — no custom metrics introduced (RankMe and NESum
  are paper-grade representation-quality metrics from Garrido et al. 2023, not the
  banned `weighted_isolation`/`pure_rate`/classifier metric family).
- **Multi-seed avg ± std obligatory** — all Step 1 numbers are NEW 3-seed (iter 70/71/72).
- **Eval-only protocol** — no encoder retraining, reusing iter 70/71/72 embeddings.
- **Source-of-truth fidelity** — all numbers copied verbatim from
  `step1_eval_only_summary_260513.md` (no derivation, no rounding changes).

---

## 8. Cross-references

- Source-of-truth: `docs/paper/manager_report/step1_eval_only_summary_260513.md`
- Raw measurements:
  - `docs/paper/manager_report/_step1b_hdbscan_eps_sweep.json` (21 cells)
  - `docs/paper/manager_report/_step1c_soft_tau_reassign.json` (12 cells)
- Plan: `C:\Users\hgcho\.claude\plans\floating-splashing-key.md` Roadmap Step 1
- Related paper sections:
  - RESULTS.md §17 (multi-seed avg ARI NEW vs B5 — Step 1 baseline anchor)
  - RESULTS.md §18 (computational performance — Step 1 measurements share the
    same NEW 3-seed runs)
  - METHOD.md §4b (Computational Requirements — Step 1c τ-reassignment slots
    immediately after as §4c)
  - DISCUSSION.md §7.x (paper N9 / N10 framing — pending DISCUSSION update if
    user requests narrative integration)

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/step1_paper_addition_260513.md
