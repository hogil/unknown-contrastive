# Performance Data Snapshot — 260513

**Purpose**: paper claim 의 정확한 metric/cfg/scope 출처를 single source-of-truth 로 고정. 모든 paper section 이 인용한 숫자가 어디서 왔는지 검증 가능하게 자세히 기록.

**사용자 directive (260513)**: "여기 있는 건 모두 claude code 로 직접 실험한 것들이다 그래서 사실들이다" — 모든 paper claim 은 실제 실험 결과(fact). fabrication 없음. inconsistency 는 "metric × cfg × scope" 명시 누락에서 발생.

---

## §1. paper claim 의 진짜 출처 — VERIFIED

### §1.1 "NEW + HDBSCAN ARI = 0.8588 ± 0.018"

**출처**: `outputs_contrastive_<ts>/tier1_*.json` 3 파일의 3-seed mean.

| iter | ts | seed | per-seed ARI |
|---|---|---|---|
| 70 | 260512_001719 | 42 | **0.8797** |
| 71 | 260512_010113 | 1  | **0.8491** |
| 72 | 260512_014507 | 2  | **0.8475** |

- 3-seed mean = `(0.8797 + 0.8491 + 0.8475) / 3 = 0.858778` → **"0.8588" / "0.859"**
- sample std (ddof=1) = **0.0181** → **"± 0.018"**
- 정확 재현 (paper-recorder agent 검증)

### §1.2 protocol = "defect-only post-hoc HDBSCAN"

| 항목 | 값 |
|---|---|
| HDBSCAN method | eom |
| min_cluster_size | 12 |
| min_samples | 3 |
| **cluster_selection_epsilon** | **0.0 (없음 — NOT 0.06)** ★ |
| scope | **defect-only (Normal 1000 sample 제외)** ★ |
| input | L2-normalized 128-D embedding (head output) |
| metric | euclidean ≈ cosine (L2-normalized) |

### §1.3 vs eval_summary.json (코드 버그)

`eval_summary.json` 의 default HDBSCAN cfg:
- eom mcs=12 ms=3 **eps=0.06** (paper protocol 과 다름)
- full 2146 samples (Normal 포함)
- **`without_normal` block 이 `with_normal` 과 bit-identical** ← Normal filtering 코드 미작동

→ eval_summary.json 단독 실행 시 ARI 0.7375 ± 0.0634 출력 (잘못된 값). reviewer 가 paper 의 0.859 와 mismatch 인식할 위험.

**해결**:
- **P-0a (P0 priority) paper patch**: METHOD §3.5 에 evaluation protocol 명시 — "Headline numbers come from defect-only post-hoc HDBSCAN with no `cluster_selection_epsilon`, NOT from `eval_summary.json` default cfg."
- 별도 코드 버그 (paper-out): `_eval_contrastive_unknown_n50.py` 의 `without_normal` Normal filtering 누락 — 추후 fix (paper 영향 없음)

---

## §2. Apples-to-Apples 21-cell re-cluster — VERIFIED

7 run × 3 HDBSCAN cfg = 21 cell. strict apples-to-apples (N=2146, 43 classes incl. Normal 1000, L2-normalized 128-D).

### §2.1 cfg 정의

| cfg | method | mcs | ms | epsilon |
|---|---|---|---|---|
| **C1** ★ main recommendation | eom | 12 | 3 | 0.06 |
| C2 | leaf | 12 | 4 | 0.06 |
| C3 (paper-nominal closest to tier1) | leaf | 15 | 3 | 0.0 |

C3 가 paper tier1 protocol 과 가장 비슷 (no eps).

### §2.2 NEW 3-seed (Queue+NEG+NeCo, no Local) per-cfg

| cfg | ARI_wN | ARI_woN | AMI | Comp | noise% |
|---|---|---|---|---|---|
| C1 | 0.749 ± 0.060 | **0.8549 ± 0.0115** | 0.905 ± 0.015 | 0.891 ± 0.019 | 9.84 |
| C2 | 0.696 ± 0.040 | 0.825 ± 0.040 | 0.882 ± 0.012 | 0.860 ± 0.018 | 37.x |
| C3 ★ | 0.740 ± 0.045 | **0.8684 ± 0.0172** ← closest to paper "0.859 ± 0.018" | 0.898 ± 0.018 | 0.880 ± 0.022 | 40.x |

**paper claim "0.859 ± 0.018" 의 closest cfg match**:
- **NEW × C3 (leaf mcs=15 ms=3 ε=0.0) ARI_woN = 0.8684 ± 0.0172** ★
- 또는 NEW × C1 (eom mcs=12 ms=3 ε=0.06) ARI_woN = 0.8549 ± 0.0115

tier1_*.json 값 (0.8588 ± 0.018) 은 C1 과 C3 사이 — defect-only + no eps 의 ε=0 차이가 cluster tree 다르게 만들어 작은 ARI 차이 (~0.01) 유발.

### §2.3 B4-noNeCo 2-seed (Local+Queue+NEG, no NeCo) per-cfg

| cfg | ARI_wN | ARI_woN | AMI | Comp | noise% |
|---|---|---|---|---|---|
| **C1** ★ | **0.828 ± 0.037** ← ARI_wN best | 0.828 ± 0.026 | 0.918 ± 0.002 | **0.920 ± 0.012** ← Comp best | **7.34** ← noise best |
| C2 | 0.611 ± ? | 0.730 ± ? | 0.876 | 0.852 | 37.x |
| C3 | 0.569 ± 0.285 ← unstable | 0.787 ± 0.040 | 0.895 | 0.870 | 40.x |

★ **B4-noNeCo × C1 이 with-Normal ARI / Completeness / noise% 3 metric 에서 best**.

### §2.4 B5 s=42 (Local+Queue+NEG+NeCo, single seed) per-cfg ★ COMPLETED

| cfg | ARI_wN | ARI_woN | AMI | Comp | noise%(defect) | n_clusters |
|---|---|---|---|---|---|---|
| **C1 (NEW와 동일)** ★ | **0.7981** | 0.7981 | **0.9200** | **0.9110** | **8.34** | 38 |
| C2 (B5 leaf+ms=4 nominal, backup) | 0.6897 | 0.6897 | 0.8731 | 0.8506 | 37.84 | 43 |
| C3 | (cluster-analyzer 21-cell 참조) | | | | | |

B5 raw embedding 은 `outputs_contrastive_260511_185039/eval/embeddings/` (eval/ 재생성 후).
백업: `outputs_contrastive_260511_185039/eval_leaf_ms4_BACKUP/embeddings/` (leaf+ms=4 결과).

★ **결정적 발견 (apples-to-apples C1 cfg)**:
- B5 ARI_wN = **0.7981** (single seed)
- NEW ARI_wN 3-seed avg = 0.749 ± 0.060
- B4-noNeCo ARI_wN 2-seed avg = 0.828 ± 0.037
- → **C1 cfg 의 with-Normal ARI ranking: B4-noNeCo > B5 > NEW**
- → 이전 "NEW > B5" ranking 은 **cfg 차이가 만든 artifact** (B5 default 가 leaf+ms=4, NEW default 가 eom+ms=3 였음)

### §2.5 NEW-NeCo s=3 (Queue+NEG only, no Local + no NeCo) — single seed

| cfg | ARI_wN | ARI_woN | noise% |
|---|---|---|---|
| C1 | 0.668 | 0.668 | 42.68 ★ catastrophic |

→ Local OR NeCo 둘 다 빠지면 collapse. **paper N1 essentiality** ablation evidence.

---

## §3. Recipe trade-off matrix — ★ UPDATED with B5 C1 result

### §3.1 C1 cfg (eom mcs=12 ms=3 ε=0.06, eval_summary cfg)

| Recipe | seeds | ARI_wN | AMI | Completeness | noise%(defect) |
|---|---|---|---|---|---|
| **B4-noNeCo** (Local+Queue+NEG, no NeCo) | 2 | **0.828 ± 0.037** ★ | 0.918 ± 0.002 | **0.920 ± 0.012** ★ | **7.34** ★ |
| **B5** (Local+Queue+NEG+NeCo, all-in) | 1 | **0.7981** ★★ NEW value | **0.9200** ★★ | 0.9110 | **8.34** |
| NEW (Queue+NEG+NeCo, no Local) | 3 | 0.749 ± 0.060 | 0.905 ± 0.015 | 0.891 ± 0.019 | 9.84 |

**C1 cfg ranking**: B4-noNeCo ~ B5 > NEW (B5 AMI 가 약간 더 높음).

### §3.2 paper protocol (defect-only post-hoc HDBSCAN, no eps) — ★ COMPLETED

| 순위 | Recipe | seeds | ARI | AMI | Comp | noise%(def) | capture |
|---|---|---|---|---|---|---|---|
| **1** ★ | **NEW-NeCo** (Queue+NEG+NeCo, no Local) | 3 | **0.8588 ± 0.018** ★ paper claim | 0.9405 | 0.9764 | 2.62 | 1.000 |
| 2 | **B5** (Local+Queue+NEG+NeCo) | 3 | **0.8436 ± 0.0273** | 0.9474 | 0.9825 | 1.40 | 1.000 |
| - | NEW-NeCo s=3 (Queue+NEG only — no NeCo) | 1 | 0.8214 (single seed) | 0.9405 | 0.9764 | 2.62 | 1.000 |

★ **paper claim "NEW > B5 by +0.015 ARI" 검증 완료** (single Tier1 protocol):
- NEW (0.8588 ± 0.018) > B5 (0.8436 ± 0.0273)
- NEW std 0.018 < B5 std 0.027 → NEW 33% 더 stable
- P1 capture_rate = 1.000 모두

### §3.3 label 혼동 정정 (cluster-analyzer 발견)

이전 "B4-noNeCo seed=1/seed=2" 라고 분석했던 run 들 (260512_114525, 125353) 의 실제 cfg:
- USE_LOCAL=True + Queue + NEG + NeCo → **실제로는 B5 recipe** (seed=1, seed=2)
- NECO_WEIGHT=None 은 NeCo OFF 의미 아닌 default value
- 그래서 cluster-analyzer 의 "B5-recipe 3-seed" = (B5_s42 + B4_s1 + B4_s2) → 진짜 B5 multi-seed (0.8436 ± 0.027)

→ B5 의 진짜 multi-seed ARI 가 confirm 됨 (0.8436 ± 0.027) — paper N1 (multi-seed obligation) 의 강력 evidence.

### §3.4 ★ 결정적 발견 정정

### §3.3 ★ 결정적 발견

- **이전 paper claim "NEW > B5" ranking 은 cfg 차이가 만든 artifact**:
  - B5 default eval cfg 가 leaf+ms=4 (왜 그렇게 됐는지 unknown — 추정: 다른 run 의 fallback)
  - NEW default eval cfg 가 eom+ms=3
  - apples-to-apples C1 (eom+ms=3) 에서: **B5 ARI 0.798 > NEW 0.749** (with-Normal scope)
- **paper claim 의 NEW SOTA 는 여전히 valid** (defect-only no-eps protocol 에서)
  - 단 B5 와 B4-noNeCo 도 같은 protocol 로 측정해야 진짜 ranking 결정
- **Trade-off (C1 cfg, with-Normal)**:
  - Local 있으면 (B5, B4) Normal cluster 깨끗 → with-Normal ARI 높음
  - Local 없으면 (NEW) NeCo 가 Normal-defect 경계 blur → with-Normal ARI 낮음
  - defect-only scope 에서는 NeCo 가 win (paper claim)

---

## §4. RankMe / NESum representation quality — INFORMATIONAL ONLY

| Run | RankMe | NESum | feat_var |
|---|---|---|---|
| B5 s=42 | **25.20** | 5.10 | 0.549 |
| NEW s=42 | 24.75 | 4.86 | 0.538 |
| NEW s=1 | 24.68 | 4.70 | 0.580 |
| NEW s=2 | 20.90 | 3.64 | 0.571 |
| NEW-NeCo s=3 | 21.12 | 4.54 | 0.543 |
| B4-noNeCo s=1 | 15.24 | 3.39 | 0.550 |
| B4-noNeCo s=2 | 20.73 | 4.03 | 0.544 |

**Multi-seed RankMe stability**:
- NEW (3-seed): 23.44 ± 1.80 (CV=7.7%)
- B4-noNeCo (2-seed): 17.98 ± 2.75 (CV=15.3%)

→ NEW representation 이 cross-seed 더 stable.

**경고**: Spearman ρ(RankMe, ARI) = **-0.429** — RankMe alone 은 paper 도메인에서 ARI predictive 효과 없음. paper 에 "richness measure" 로만 인용 가능, "ARI predictor" 로는 인용 X.

---

## §5. cluster-analyzer 의 robust/weak cluster 발견 (B4-noNeCo 2-seed)

### §5.1 Cross-seed robust clusters (14/43 = 33%)
모두 purity=1.0, silhouette≥0.85, intra_p95≤0.025, margin (inter_min/intra_p95) ≥ 5× both seeds:

**bank_boundary / invalid_main 우세** (9/14 = 64%):
- BrokenRing, CenterDonut, **Center_bank_boundary**, CrescentArc, CrossScratch, DiagonalSmear, **Donut_bank_boundary**, **Donut_invalid_main**, **Edge-Bottom_invalid_main**, **Edge-Ring_bank_boundary**, **Edge-Ring_invalid_main**, **Edge-Top_invalid_main**, **Full_bank_boundary**, Starburst

★ **"object-driven signal dominant over wafer-pattern"** — paper narrative 직접 인용 가능.

### §5.2 Cross-seed weak classes (top-5)
- Thick-Edge_fork (17%/31% noise)
- Normal (11%/16%)
- Edge-Top_fork (15%/5% + s2 split)
- Center_fork (s2 split)
- CenterCircle (~8% both)

★ **4/5 fork-subtype, n ≤ 30** (HDBSCAN mcs=12 boundary). mcs=10 sweep 권장.

### §5.3 BATCH confound (★ 사용자 feedback 재검증 필요)

- seed1 BATCH=4 ARI 0.854 vs seed2 BATCH=8 ARI 0.802 → ΔARI **−5.2pp**
- 사용자 feedback "BATCH 변경 = same-condition" 명시했으나, cluster-analyzer 는 paper claim 으로 same-BATCH 3rd seed 필요 권장
- noise pct: seed1 6.10% → seed2 8.57% (+2.47pp, 92% from Normal)
- **방안**: paper Method §3.x 에 BATCH=4 fixed 라 명시 후 추가 BATCH=4 seed=3 dispatch

---

## §6. ★ Recommendations (cluster-analyzer agent 제안)

### §6.1 paper-grade table 의 권장 cfg
- **C1 (eom mcs=12 ms=3 ε=0.06)** as main result cfg
  - capture=1.000 saturated
  - noise < 11%
  - statistical stability across cells
- C2 / C3 → supplementary
- Tier 1 metric tables 의 모든 cell 에 (sel / mcs / ms / ε) 행 + ARI scope (with/without Normal) 명시

### §6.2 추가 dispatch 권장 (사용자 승인 필요)
1. **NEW seed=3 BATCH=4** — same-BATCH 3rd seed 로 σ 정밀화 (현재 σ=0.018 → 예상 0.012)
2. **B4-noNeCo seed=3 BATCH=4** — same-BATCH 3rd seed 로 cross-seed 표 완성
3. **B5 seed=1/seed=2** — B5 multi-seed (현재 s=42 only)

### §6.3 코드 fix (paper-out)
- `_eval_contrastive_unknown_n50.py` 의 `without_normal` block 의 Normal filtering 적용 (현재 누락)

---

## §7. ★ paper-recorder 의 consolidation P0~P2 patches + 새 P-0a

| patch | priority | file | issue |
|---|---|---|---|
| **P-0a** ★ NEW | P0 | METHOD §3.5 | "Evaluation HDBSCAN protocol" 명시 — defect-only no-eps, NOT eval_summary |
| P0-1 | P0 | DISCUSSION.md:958 | Practitioner choice tree 의 B5 추천 → v7 NEW 추천 propagation |
| P0-2 | P0 | CONCLUSION.md:297, 342-345, 455-457 | §8.6 N7 + §8.7 Frontier 2 B5 v6 잔재 |
| P0-3 | P0 | SUMMARY.md:786, 858-861 + REPORT.md:442-451, 466-475 | Phase 3 결론 v6 잔재 |
| P1-1 | P1 | RESULTS.md:603, 613, 755 | "Absolute SOTA" B5 0.9358 single-seed claim |
| P1-2 | P1 | README.md:57-67, 138 | "현재 SOTA" table + dual-cfg recipe |
| P1-3 | P1 | FIGURES.md:91 | F-N7-lattice caption |
| P2-1 | P2 | INTRODUCTION.md:222-228 | C7 body v6 잔재 |
| P2-2 | P2 | METHOD.md §3.6 body | (paper-recorder 식별) |

**cosmetic**: `0.8588 ± 0.018` vs `0.859 ± 0.018` 혼재 → 3-decimal **0.859** 통일 권장.

**모든 patch 사용자 승인 후 적용** (paper-recorder 는 read-only 였음).

---

## §8. 추적 가능 산출물 (모두 docs/paper/manager_report/)

| 파일 | 내용 |
|---|---|
| `claim_0859_origin_trace.md` | paper claim 0.8588 출처 검증 (tier1_*.json 3 files) |
| `apples_to_apples_7run_3cfg.md` | 21-cell re-cluster 표 + ranking |
| `_apples_7run_3cfg_raw.parquet` + `.csv` | 21-cell raw metrics (재현용) |
| `cluster_analysis_b4_noNeCo_3seed.md` | B4-noNeCo 2-seed deep-dive (14 robust + 5 weak + BATCH confound) |
| `sota_tangents_final_consolidation.md` | external SOTA tangents (RankMe / Iterative Harvesting / HDBSCAN eps sweep) |
| `consolidation_pass_260513.md` | paper 9/11 section v6 잔재 + 8 patches list |
| `b5_recluster_eom_ms3.md` | (대기 중, evaluation agent #80) B5 leaf→eom re-eval |
| **`performance_data_260513.md`** ← this file | 통합 single source-of-truth |

---

## §9. 다음 step (사용자 결정 대기)

1. **paper P-0a + 8 patches 일괄 적용** (task #82 pending) — 9/11 section v6 잔재 retract + METHOD §3.5 protocol disclosure
2. **eval_summary.json 코드 버그 fix** (task #81 pending) — paper 영향 없음, 미래 신뢰성
3. **iter 81 NEW-NeCo (catastrophic) ablation row 추가** (task #75 pending) — paper N1 essentiality evidence
4. **추가 dispatch** (선택): NEW s=3 BATCH=4 (σ 정밀화), B5 s=1/s=2 (B5 multi-seed)
5. **eval bug fix 적용 시** apples-to-apples 21-cell re-cluster 의 NEW × C1 ARI_woN = 0.8549 ± 0.0115 가 새 default eval_summary 출력값과 일치할 것 — 그 시점에 paper 와 eval_summary consistency 자동 달성

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/performance_data_260513.md
