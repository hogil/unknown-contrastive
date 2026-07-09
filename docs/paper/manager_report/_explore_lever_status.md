# Lever Status Report — Attempted vs Untried SOTA Techniques
**Date**: 2026-05-13 (paper-finalization phase, read-only analysis)
**Scope**: Distinguish rejected/accept levers from ITERATIONS.md (iter 60-85) + SOTA tangents consolidation
**Purpose**: Roadmap step-by-step performance improvement with clear negative results + candidate techniques

---

## 1. LEVERS ATTEMPTED & REJECTED (Negative Results)

### 1.1 Temperature (NCE_TEMP) Lever

| Technique | Config | Result | ARI Δ | Status |
|---|---|---|---|---|
| **TEMP 0.05 (lower sharpen)** | NEW (NeCo+Queue, no Local) + TEMP 0.07→0.05 | **NEGATIVE** | **−0.024** | ★ REJECTED (iter 73) |
| TEMP 0.05 in B5 context | B5 (Local+Queue+NEG+NeCo) | Positive gain +0.014 | +0.014 | **CONTEXT-DEPENDENT** |

**Finding**: TEMP interaction **asymmetric** — lower TEMP (0.05) benefits Local-based cfg (B5) but harms NeCo-only cfg (NEW). Mechanism: Local stabilizes, TEMP sharpens (synergistic); NeCo alone over-sharpens neighbor signals → noise increase. **Paper N6 evidence #5**.

**Implication**: TEMP sweep locked at **0.07** for NEW cfg. Lower TEMP not explored further.

---

### 1.2 Queue Size Scaling

| Queue Size | Config | ARI | Δ vs 4096 | Status |
|---|---|---|---|---|
| **8192** | NEW (NeCo+Queue+NEG, no Local) | 0.8674 | **−0.0122** | **REJECTED** (iter 79) |
| 4096 (baseline) | NEW | 0.8797 | (base) | **SWEET SPOT** |
| 2048 (planned lower) | — | — | — | Not tested (4096 confirmed) |

**Finding**: Queue 8192 regression. 4096 is reproducible sweet spot (iter 70, 71, 72 consistency).

**Implication**: Queue size parameter **locked at 4096**. No gain from scaling up.

---

### 1.3 NeCo Weight Sweep (Boundary Cases)

| NeCo Weight | ARI (seed=42) | Sil (defect-only) | Δ vs weight=0.2 | Status |
|---|---|---|---|---|
| 0.0 (no NeCo, B4) | 0.8605 | 0.8012 | (base) | Baseline |
| **0.1** (lower bound) | **0.8605** | 0.8012 | **−0.0192 (ARI)** | **REJECTED** (iter 78) |
| **0.2** (narrow peak) | **0.8797** | 0.7860 | (base) | **SWEET SPOT (iter 70)** |
| **0.4** (Pariza upper) | **0.8605** | 0.8012 | **−0.0192** | **REJECTED** (iter 77) |

**Finding**: NeCo weight forms **narrow inverse-U peak at 0.2** (step function, not monotonic). Both 0.1 and 0.4 regress to B4-level ARI (0.8605). Weights ≠ 0.2 are functionally equivalent to removing NeCo.

**Paper claim retraction**: Sil +30% claim retracted (post-iter 78 HDBSCAN protocol mismatch correction) — defect-only Sil actually marginal (−0.013 vs B5, within noise).

**Implication**: NeCo weight **locked at 0.2** with no alternatives viable.

---

### 1.4 Local LW (Local Weight) in Isolation

| Lever | Context | ARI Δ | Status |
|---|---|---|---|
| **LW 0.5→1.0** | **No Queue, no NEG** (B1→B2) | **−0.0278** | **REJECTED (iter 62)** — atomic regression |
| LW 1.0 w/ Queue (B2→B3) | With Queue active | +0.0231 | **POSITIVE via interaction** |

**Finding**: Local weight exhibits **component interaction dependency** — harmful in isolation but benefits emerge only when Queue is present (absorption of over-emphasis). Paper N6 evidence #1-2.

**Implication**: LW tuning **not viable as standalone lever**. Locked at 1.0 (existing practice).

---

## 2. LEVERS ACCEPTED & LOCKED (Current NEW SOTA)

### 2.1 NEW Configuration (iter 70-72, 3-seed final)

```
L = L_global(weight 1.0)
  + L_NeCo(weight 0.2) [replaces Local DenseCL]
  + L_NEG_filter(sim < 0.72)
  + MoCo Queue(size 4096)

HDBSCAN: eom, mcs=12, ms=3, cluster_selection_epsilon not used
```

**Multi-seed (3-seed: 42, 1, 2) Final Metrics** (iter 70/71/72):
- **ARI**: 0.8588 ± 0.018
- **Noise (defect)**: 1.48%
- **Completeness**: 0.9825 ± 0.007
- **Silhouette (defect-only, eom)**: 0.7860~0.8130 (protocol corrected)

**Path to Lock-in**:
1. iter 60: B0 baseline (Global only, ARI 0.823)
2. iter 61-65: component isolation (B0→B1→B2→B3→B4→B5)
3. iter 65: **NeCo isolated effect ≈0** (paper N1 v1 claim retracted)
4. iter 69: **NeCo ≡ Local DenseCL functionally** (4-decimal ARI identity, iter 69 vs B1)
5. iter 70: **Replace Local with NeCo → NEW SOTA** (ARI +0.019 vs B4, Local removed)
6. iter 73: TEMP 0.05 rejected (interaction harm demonstrated)
7. iter 74-78: Queue/NEG/NeCo weight isolation complete (all locked)
8. **iter 82-84**: Multi-seed reproducibility + clustering algorithm benchmark (final validation)

**Key Trade-off Resolved**: iter 84 multi-seed reproducibility test revealed:
- B5 (Local+NeCo+Queue+NEG) seed=42: ARI 0.9358 (lucky outlier)
- B5 seed=1: ARI 0.8482 (−0.088 drop, 2.8× higher std than NEW)
- **NEW is the TRUE SOTA on multi-seed average** (0.8588 ± 0.018 vs B5 0.8343 ± 0.031 on HDBSCAN)

---

### 2.2 NEG Filter (NV-Retriever Style) Lock-in

| Config | NEG Threshold | ARI | Noise (defect) | Status |
|---|---|---|---|---|
| iter 75 (no NEG) | — | 0.8822 | 1.31% | baseline |
| iter 70/NEW (NEG 0.72) | 0.72 | 0.8797 | 0.87% | **TRADE-OFF: ARI −0.003, noise −0.44pp** |
| iter 74 (NEG, no Queue) | 0.72 | 0.8514 | 3.93% | **NEG effect = 0 sans Queue** |

**Finding**: NEG filter **depends on Queue for activation** (iter 74 proof: effect disappears without Queue). Trade-off is noise reduction only, marginal ARI cost. Paper N7 (Component Dependency Hierarchy).

**Implication**: NEG threshold **locked at 0.72** (NV-Retriever Moreira 2024 empirical sweet spot, iter 58 / iter 64-76 confirmation).

---

## 3. USER EXPLICITLY REJECTED TECHNIQUES (D-series DECISIONS)

From `docs/contrastive-eval/DECISIONS.md`:

### 3.1 Multi-Crop (SwAV Style) — D-4 ABSOLUTE VETO

| Technique | Reason |
|---|---|
| **Multi-crop (global 384 + local 128 random crop)** | **REJECTED** — wafer location identity critical (Edge-Top vs Edge-Bottom as class feature). Random crop destroys positional info. |
| Alternative adopted | **Grid-based spatial contrast (USE_LOCAL=true)** — preserves location via grid cells |

**Status**: ★ FORBIDDEN. No SwAV, no random crop.

---

### 3.2 SupCon (Supervised Contrastive) — D-5 REJECTION

| Technique | Reason |
|---|---|
| **SupCon + label dependency** | **REJECTED** — unknown defect generalization damage. Labels sharpen known-class manifold only → unknown defects absorbed into nearest known class. Production risk. |
| Alternative adopted | **SSL InfoNCE only** (current contrastive.py) |

**Status**: ★ FORBIDDEN. SupCon as primary method disallowed.

**Note**: `feedback_no_multicrop_no_supcon.md` cited in ITERATIONS D-series.

---

### 3.3 Hard Negative Mining (Robinson 2021 Importance Weighting) — D-6 DEPRECATED

| Technique | Status | Replacement |
|---|---|---|
| **Robinson 2021 β-weighted hard negatives** | Superseded (D-14) | **NV-Retriever false-negative filter** (iter 64+) |

**Finding (D-14)**: NV-Retriever (arxiv 2407.15831) is filter-based (label-aware false-neg detection), not pure hard-mining. More suitable than Robinson for SSL-only domain.

**Status**: Robinson approach **NOT in iter 60-85 chain** (baseline already using NV-Retriever from iter 60).

---

### 3.4 Data Augmentation Changes — D-8 LOCK-IN

| Aspect | Constraint |
|---|---|
| **Synthetic data modification** | FORBIDDEN (合成 데이터 변경 금지) |
| **Augmentation type/strength changes** | FORBIDDEN |
| **Data sampling ratios** | FROZEN (D-13: method ablation must use same-data anchor) |

**Status**: ★ IMAGE PIPELINE FROZEN. Only encoder/loss/hparam changes allowed.

---

### 3.5 Bigger Anchor Data — DEFERRED

| Change | Status | Reason |
|---|---|---|
| **Defect class size 30→5000+** | Deferred (line 380 ITERATIONS) | User currently rejects; data spec lock-in principle (D-13). |

**Status**: Can be explored in separate **data ablation track (Iter D1, D2, ...)** but NOT in method iteration.

---

## 4. SOTA CANDIDATE TECHNIQUES — NOT YET TRIED

### 4.1 RankMe (Effective Rank Post-Hoc) — ★★★ HIGH PRIORITY

| Paper | Metric | Applicability | Effort | Paper Impact |
|---|---|---|---|---|
| **arxiv 2210.02885** (Garrido et al., ICML 2023) | Effective rank (Shannon entropy of singular values) | ★★★ post-hoc, label-free | 0 (append column) | **High** (representation quality standard) |

**Recommendation (A1 in sota_tangents)**: 
- Add RankMe + NESum columns to 4-variant embedding quality table
- Both are post-hoc, no training needed
- Reviewer-expected for representation analysis

**Plan**: Compute on (NEW, B5, B4, FUSION) embeddings; add to paper table.

---

### 4.2 NESum / α-ReQ / Stable Rank — ★★★ COMPLEMENTARY

| Paper | Metrics | Applicability | Effort | Advantage |
|---|---|---|---|---|
| **arxiv 2305.16562** (Tsitsulin et al., PMLR 2023) | NESum, α-ReQ, stable rank, coherence, condition number | ★★★ post-hoc | 0 | More stable downstream correlation than RankMe alone |

**Recommendation (A1 extended)**:
- Compute NESum + stable rank together with RankMe (paper claims NESum more robust)
- Trio coverage: RankMe + NESum on defect embedding; augment alignment+uniformity (already used)

**Plan**: Multi-metric panel (4 metrics: RankMe, NESum, alignment, uniformity).

---

### 4.3 Iterative Cluster Harvesting (ICH) — ★★ MODERATE PRIORITY

| Paper | Method | Applicability | Effort | Wafer Fit |
|---|---|---|---|---|
| **arxiv 2404.15436** (Pleli et al., April 2024) | PCA→AC→silhouette loop, partial assignment | ★★★ post-hoc, eval-only | 10-20 lines wrapper | **Direct wafer domain** (WM1K/WM811K cited) |

**Recommendation (A4 in sota_tangents)**:
- Post-hoc alternative to HDBSCAN on NEW embedding
- Outputs partial assignment (confident samples only) — may lower P1 (capture_rate)
- Risk: partial assignment incompatible with our P1=1.000 requirement

**Plan**: Ablation 1 row ("ICH on NEW embedding") — compare vs Oracle Ward K=42. Document trade-off (partial vs full assignment).

---

### 4.4 Soft HDBSCAN (Probabilistic Reassignment) — ★★★ HIGH PRIORITY

| Method | Applicability | Effort | Expected Gain |
|---|---|---|---|
| **hdbscan library `all_points_membership_vectors()`** | ★★★ config-only | 5 lines | Noise(def) 6%→? via soft τ-reassignment |

**Recommendation (A3 in sota_tangents)**:
- Reassign HDBSCAN noise points to nearest cluster (soft threshold τ=0.3~0.5)
- Eval-only, no retraining
- May improve P2 (noise_pct) from 1.48%→1.2% + robustness

**Plan**: Ablation 1-2 rows ("soft HDBSCAN τ=0.3", "τ=0.5") — measure P1/P2 impact.

---

### 4.5 HDBSCAN Hybrid eps (DBSCAN*/HDBSCAN blend) — ★★★ MODERATE

| Paper | Method | Applicability | Effort | Benefit |
|---|---|---|---|---|
| **arxiv 1911.02282** (Malzer & Baum) | `cluster_selection_epsilon` param (threshold ε sweep) | ★★★ config sweep | 5-line grid search | Suppress micro-clusters, noise -0.5~1.0pp |

**Recommendation (A2 in sota_tangents)**:
- Current: HDBSCAN eom, mcs=12, ms=3 (no epsilon)
- Sweep: eps ∈ {0.05, 0.10, 0.15} on NEW embedding
- May reduce noise floor + Homogeneity trade-off

**Plan**: Ablation 1-2 rows ("HDBSCAN eps=0.05", "eps=0.10") — no retraining needed.

---

### 4.6 k-NN Label Propagation on HDBSCAN Noise — ★★★ SIMPLE HEURISTIC

| Method | Applicability | Effort | Trade-off |
|---|---|---|---|
| **k-NN voting (k=5~15) on noise points** | ★★★ post-hoc, graph-propagation | 10 lines | Assign noise to nearest cluster (may increase P2 slightly but recover P1?) |

**Recommendation (A3 variant in sota_tangents)**:
- Standard BERTopic workflow complement
- Sanity-check ablation: noise reduction via simple graph propagation
- Risk: may contaminate clusters with true defects currently labeled as noise

**Plan**: Optional ablation row if soft HDBSCAN insufficient.

---

### 4.7 DECOR (D4-Equivariant CAE + DeepDPM) — ★ RELATED WORK ONLY

| Paper | Method | SOTA (their dataset) | Wafer Fit | Recommendation |
|---|---|---|---|---|
| **arxiv 2510.03328** (Jothiraj et al., AAAI 2026) | R2Conv (D4-equivariant) + DeepDPM clustering | ARI 0.296 (MixedWM38, multi-label) | ★ (different dataset) | Related work citation only |

**Finding**: 
- Our NEW ARI 0.859 ≫ DECOR 0.296 (but dataset MixedWM38 ≠ our 43-class synthetic)
- **D4 equivariance inappropriate for us** — scratch_rot angle IS class identity, not rotation-invariant feature
- per-comparison unfair (different data, different task)

**Recommendation (A5 in sota_tangents)**: 
- Cite DECOR in RELATED WORK
- Contrast: "DECOR's D4-equivariance assumes angle-invariance; our domain treats angle as class-identity (scratch_rot ≠ scratch)" 
- ARI 0.859 vs 0.296 comparison explicitly flags dataset difference

---

### 4.8 Other Clustering Algorithms (SNC, TANGO, DPC) — ★ OPTIONAL

| Technique | Applicability | Recommendation |
|---|---|---|
| **SNC (Selective Neighbor Clustering, arxiv 2304.06928)** | ★★ K-discovery alternative to HDBSCAN | K-discovery ablation (known-K comparison only); GCD assumptions conflict |
| **TANGO (DPC variant, arxiv 2408.10084)** | ★ DPC-family alternative | Related work only; HDBSCAN already sufficient (paper N9 benchmark confirmed) |
| **Spectral K=42** | ★ Oracle-K baseline | Explicitly NOT recommended (iter 82-83: 0.23~0.79 instability, graph disconnection) |

**Recommendation**: Keep as related-work citations; no ablation rows (not paper-grade deliverables).

---

## 5. USER-SPECIFIED REJECTION CONSTRAINTS (Locked Decisions)

### 5.1 Metric Constraints (D-1, D-2, D-3)

| Decision | Locked Choice | Forbidden |
|---|---|---|
| **D-1: Official metrics only** | Tier 1 (Completeness, AMI, noise_pct, class_capture_rate) + Tier 2 (Homogeneity, Silhouette, ARI) | Custom metrics (weighted_isolation, contamination_rate, binary_homogeneity) |
| **D-2: B-Cubed drop** | Completeness covers B-Cubed Recall | B-Cubed Precision/Recall/F1 as primary |
| **D-3: Priority lock** | P1=capture_rate > P2=noise > P3=Completeness > P4=Homogeneity | Arbitrary weighting |

---

### 5.2 Methodology Lock-Ins (D-13, D-15)

| Constraint | Implication |
|---|---|
| **Same-data atomic method ablation (D-13)** | Normal sampling ratios FROZEN. Data ablation in separate track (Iter D1, D2, ...). Method Iter 60-85 uses fixed anchor (avg30_new_260508_123037). |
| **Anchor spec (D-15)** | Defect avg 30 random distrib + Normal 100% source. All iter 60-85 use identical anchor. |

**Impact on roadmap**: Cannot try bigger data in method iteration; must branch to data ablation track.

---

## 6. SUMMARY TABLE: Rejected vs Accepted vs Unexplored

| # | Lever / Technique | Category | ARI Δ | Status | Locked? |
|---|---|---|---|---|---|
| 1 | TEMP 0.05 (lower) | Rejected | −0.024 | ✗ NEGATIVE | ★ LOCKED (0.07) |
| 2 | Queue 8192 | Rejected | −0.012 | ✗ NEGATIVE | ★ LOCKED (4096) |
| 3 | NeCo weight 0.1 | Rejected | −0.019 | ✗ NEGATIVE | ★ LOCKED (0.2) |
| 4 | NeCo weight 0.4 | Rejected | −0.019 | ✗ NEGATIVE | ★ LOCKED (0.2) |
| 5 | LW 1.0 (no Queue) | Rejected | −0.028 | ✗ NEGATIVE (interaction) | ★ LOCKED (1.0) |
| 6 | Multi-crop (SwAV) | User veto | — | ★ FORBIDDEN | **VETO** |
| 7 | SupCon | User veto | — | ★ FORBIDDEN | **VETO** |
| 8 | Bigger anchor (5000+) | Deferred | — | ⏸ Data ablation track | Separate Iter D1/D2 |
| — | — | — | — | — | — |
| A1 | RankMe + NESum | Not tried | — | ✓ Post-hoc metric | **0 effort, HIGH impact** |
| A2 | HDBSCAN eps sweep | Not tried | +0.5~1.0pp noise | ✓ Config sweep | **~10 lines, eval-only** |
| A3 | Soft HDBSCAN τ | Not tried | noise −0.5~1pp | ✓ Reassignment | **~5 lines, eval-only** |
| A4 | ICH (post-hoc) | Not tried | ? (paper: +0.13 Homog) | ✓ Post-hoc, wafer-domain | **eval-only, risk P1 tradeoff** |
| A5 | DECOR D4-equiv | Not tried | 0.296 ARI (their data) | ✗ Dataset mismatch | **Related work only** |
| A6-A7 | ViT-Tiny, Mean Teacher | Not tried | 98.4% F1 / 83.4% F1 (labels) | ✗ Supervised paradigm | **Related work ceiling** |

---

## 7. RECOMMENDED PAPER-GRADE LEVER ROADMAP

### Phase 1: Immediate (no training, eval-only)
- **A1 (RankMe+NESum)**: Add to embedding quality table (5 min)
- **A2 (HDBSCAN eps)**: Grid search 0.05, 0.10, 0.15 on existing embedding
- **A3 (Soft HDBSCAN)**: Reassignment τ=0.3~0.5 on existing embedding

### Phase 2: Medium (post-hoc, 30 min each)
- **A4 (ICH ablation)**: One row PCA→AC→Sil loop, compare vs Oracle K=42 (document partial assignment trade-off)
- **k-NN label propagation**: Optional if soft τ insufficient

### Phase 3: Related work (citation + 1 paragraph each)
- **A5 (DECOR)**: 2025-2026 wafer SOTA, different dataset, D4 unsuitability contrast
- **A6 (Mean Teacher+SupCon)**: Semi-supervised upper bound
- **A7 (ViT-Tiny)**: Supervised ceiling

### Phase 4: NOT recommended
- Temperature re-tuning (interaction mismatch confirmed)
- Queue scaling (4096 sweet spot final)
- NeCo weight variants (0.2 narrow peak final)
- SwAV / SupCon (user veto)
- D4-equivariance (angle-identity conflict)

---

## 8. PAPER FINALIZATION CHECKLIST

From sota_tangents_final_consolidation.md "Recommended action items":

| Item | Type | Effort | Paper Gain | Priority |
|---|---|---|---|---|
| **RankMe + NESum post-hoc** | A1 | ✓✓✓ none | High (reviewer expectation) | **P0** |
| **HDBSCAN eps sweep** | A2 | ✓✓✓ 15 min | Medium (noise alternative) | **P1** |
| **Soft τ-reassignment** | A3 | ✓✓✓ 10 min | Medium (robustness) | **P1** |
| **ICH ablation row** | A4 | ✓✓ 30 min | High (wafer-domain citation) | **P2** |
| **DECOR related work** | A5 | ✓ citation | High (2025 SOTA compare) | **P2** |
| **Mean Teacher related work** | A6 | ✓ citation | Medium (semi-sup contrast) | **P3** |
| **ViT-Tiny upper bound** | A7 | ✓ citation | Medium (fairness) | **P3** |

---

## CONCLUSION

**Green-light for roadmap**: ARI 0.8588 ± 0.018 (NEW cfg multi-seed) is TRUE SOTA after iter 84 multi-seed validation. No further contrastive training needed (saturated). All post-hoc levers (A1-A7) are **zero-retraining improvements** — apply in order A1 → A2 → A3 → A4 for paper robustness + reviewer credibility.

**Hard gates**: 
- ★ Multi-crop / SupCon / D4-equivariance: **FORBIDDEN**
- ★ TEMP/Queue/NeCo_weight: **IMMUTABLE** (locked via iter 60-85 evidence)
- Bigger anchor: Separate data ablation track (iter D1+)

**Next step**: RankMe post-hoc metric (A1) → HDBSCAN eps ablation (A2) → paper revision Phase 3.

---

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/_explore_lever_status.md
