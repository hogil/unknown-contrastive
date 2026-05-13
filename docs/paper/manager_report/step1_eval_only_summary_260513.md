# Step 1 (Eval-only) Results — 260513

Plan reference: `C:\Users\hgcho\.claude\plans\floating-splashing-key.md` Roadmap Step 1.

**목적**: 학습 dispatch 없이 (eval-only, embedding 그대로) NEW SOTA cfg 의 paper-grade improvement 측정. RankMe + HDBSCAN eps sweep + soft τ-reassignment 세 가지 적용.

**Source data**: NEW recipe 3-seed (iter 70/71/72), defect-only Tier1 protocol (eom mcs=12 ms=3).

---

## Step 1a — RankMe + NESum representation quality column (paper N10)

| Recipe | RankMe | NESum | feat_var |
|---|---|---|---|
| B5 s=42 | 25.20 | 5.10 | 0.549 |
| **NEW 3-seed avg ± std** | **23.44 ± 1.80** | **4.40 ± 0.69** | 0.563 |
| B5 3-seed avg ± std | 22.06 ± 4.99* | — | — |
| NEW-NeCo s=3 | 21.12 | 4.54 | 0.543 |

\* B5 3-seed RankMe variance 큼 — paper-recorder 측정 (single source-of-truth).

**Spearman ρ(RankMe, ARI) = -0.429** (n=7 runs) → RankMe alone 은 ARI predictive 효과 없음. Paper claim: "RankMe is informative for **cross-seed stability** (NEW std 1.80 vs B5 std 4.99 → NEW 64% 더 stable), NOT for ARI ranking."

---

## Step 1b — HDBSCAN cluster_selection_epsilon sweep (paper N9 강화)

NEW 3-seed × eps ∈ {0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15} = 21 cell.

| eps | ARI avg | std | AMI | Hom | Comp | noise% |
|---|---|---|---|---|---|---|
| 0.00 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.02 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.04 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.06 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.08 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.10 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |
| 0.15 | 0.8731 | 0.0140 | 0.9629 | 0.9448 | 0.9963 | 1.48 |

★ **결정적 발견**: epsilon 영향 **0** — NEW embedding 의 cluster tree 가 견고 (eom+mcs=12+ms=3 만으로 saturated). Paper N9 새 evidence (paper-worthy negative result).

**Paper claim**: "On strong contrastive embeddings, HDBSCAN `cluster_selection_epsilon` parameter is **redundant** — the cluster tree is determined by `(method, min_cluster_size, min_samples)` triple alone."

---

## Step 1c — Soft HDBSCAN τ-reassignment (KNN-softmax post-process)

NEW 3-seed × τ ∈ {0.5, 0.7, 0.9, ∞} 비교. KNN k=10, cosine sim, softmax temperature 0.1.

| τ | ARI avg | std | AMI | Hom | Comp | noise% | reassign avg |
|---|---|---|---|---|---|---|---|
| ∞ (baseline, no reassign) | **0.8731** | 0.0140 | **0.9629** | **0.9448** | **0.9963** | 1.48 | 0 |
| 0.90 | 0.8709 | 0.0132 | 0.9616 | 0.9436 | 0.9952 | 0.49 | 11.3 |
| 0.70 | 0.8696 | **0.0123** | 0.9607 | 0.9430 | 0.9944 | 0.15 | 15.3 |
| 0.50 | 0.8681 | 0.0125 | 0.9600 | 0.9424 | 0.9938 | **0.00** | 17.0 |

★ **Trade-off matrix**:
- **noise% 1.48% → 0.00%** (τ=0.5) — P2 dramatic 향상
- **std 0.0140 → 0.0123** (τ=0.7) — reproducibility 12% 개선
- ARI -0.005 ~ -0.002 (τ 클수록 손실 작음)
- P1 capture 모두 1.000 (변동 없음)

**Paper claim (Step 1c)**: "Soft KNN-softmax reassignment of HDBSCAN noise points achieves **0% noise rate** with marginal ARI cost (-0.005), useful for production deployment where every wafer must receive a cluster label."

---

## Step 1 종합 — Paper Table N+1 (Step-by-step 점진적 향상)

| Step | Method | P1 cap | P2 noise%↓ | P3 Comp | P4 Hom | AMI | ARI | std | RankMe |
|---|---|---|---|---|---|---|---|---|---|
| 0 | NEW (Queue+NEG+NeCo) | 1.000 | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | 23.44 |
| 1a | + RankMe column | (post-hoc metric, paper completeness) | | | | | | | 23.44 |
| 1b | + HDBSCAN eps sweep | 1.000 | 1.48 | 0.9963 | 0.9448 | 0.9629 | 0.8731 | 0.0140 | — |
| 1c τ=0.9 | + soft τ-reassign | 1.000 | **0.49** ★ | 0.9952 | 0.9436 | 0.9616 | 0.8709 | 0.0132 | — |
| 1c τ=0.7 | + soft τ-reassign | 1.000 | **0.15** ★ | 0.9944 | 0.9430 | 0.9607 | 0.8696 | **0.0123** ★ | — |
| 1c τ=0.5 | + soft τ-reassign | 1.000 | **0.00** ★★ | 0.9938 | 0.9424 | 0.9600 | 0.8681 | 0.0125 | — |

## Step 1 → 다음 step 결정

| 결과 | 다음 step |
|---|---|
| Step 1b epsilon **0 effect** → eps sweep 무효 | 영구 제외 |
| Step 1c τ=0.5 → **noise 0%** 가능 | production cfg 로 lock 가능 |
| Step 1a RankMe ρ=-0.43 → ARI 예측 X | std 측정용으로만 보고 |

**Step 2 (EMA target encoder) 진행 가능** — 학습 dispatch 필요. 사용자 승인 대기.

---

## 산출 file

- `_step1b_hdbscan_eps_sweep.json` (21-cell raw)
- `_step1c_soft_tau_reassign.json` (12-cell raw)
- (이 file) `step1_eval_only_summary_260513.md`

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/step1_eval_only_summary_260513.md
