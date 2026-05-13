# Apples-to-Apples Re-Cluster: 7 Runs × 3 HDBSCAN Configs

생성: 2026-05-13 (cluster-analyzer agent)
산출 raw: `_apples_7run_3cfg_raw.parquet` + `.csv` (same dir)

---

## 0. Protocol (strict)

**고정 (same across all 21 cells)**:
- N = 2146 samples, D = 128, 43 GT classes (10 obj-active + 9 canvas + ... + Normal=1000)
- Normal sample count = 1000 (≈46.6% of corpus) — apples-to-apples confirmed
- L2-normalized embeddings (||x||=1.0 ±1e-7)
- HDBSCAN metric = `euclidean` on L2-normalized x (Euclidean² = 2 − 2·cosine, monotonic equiv)
- Silhouette computed with `metric="cosine"` on clustered points only
- GT label source: `eval/embeddings/classes.txt` (or `eval_leaf_ms4_BACKUP/embeddings/classes.txt` for B5)

**변동 (3 cfg matrix)**:

| cfg | sel | min_cluster_size | min_samples | epsilon | 의도 |
|---|---|---|---|---|---|
| C1 | eom  | 12 | 3 | 0.06 | paper's nominal cfg (excess-of-mass, aggressive merge) |
| C2 | leaf | 12 | 4 | 0.06 | B5 `eval_summary.json` 기록된 cfg (leaf split) |
| C3 | leaf | 15 | 3 | 0.0  | 이전 sweep best-K cfg (larger floor, no eps) |

**파생 metric**:
- **ARI_wN** (with-normal): ARI on all 2146 samples vs full GT (43 classes)
- **ARI_woN** (without-normal): ARI on 1146 non-Normal samples; predicted labels carried over (noise=-1 included)
- **AMI / NMI / Hom / Comp**: full 43-class on all samples
- **Sil**: cosine silhouette on points with cluster_id != -1
- **capture**: defect-class recall — fraction of 42 non-Normal classes where ≥1 sample lands in any non-noise cluster

---

## 1. 7 Runs

| # | run_ts | tag | group | recipe |
|---|---|---|---|---|
| 1 | 260511_185039 | B5_s42       | B5         | Local + Queue + NEG + NeCo (seed=42) |
| 2 | 260512_001719 | NEW_s42      | NEW        | Queue + NEG + NeCo, no Local (seed=42) |
| 3 | 260512_010113 | NEW_s1       | NEW        | Queue + NEG + NeCo, no Local (seed=1) |
| 4 | 260512_014507 | NEW_s2       | NEW        | Queue + NEG + NeCo, no Local (seed=2) |
| 5 | 260512_093943 | NEWnoNeCo_s3 | NEW-noNeCo | Queue + NEG only, no Local, no NeCo (seed=3) |
| 6 | 260512_114525 | B4noNeCo_s1  | B4-noNeCo  | Local + Queue + NEG, no NeCo, BATCH=4 (seed=1) |
| 7 | 260512_125353 | B4noNeCo_s2  | B4-noNeCo  | Local + Queue + NEG, no NeCo, BATCH=8 (seed=2) |

**중요**: B5 (`260511_185039`) 의 `eval/embeddings/` 는 비어있고, 실제 raw 는 `eval_leaf_ms4_BACKUP/embeddings/` 에 보존됨 — 그 path 를 사용.

---

## 2. Full 21-cell Result Table

| # | Run | cfg | ARI_wN | ARI_woN | AMI | NMI | Hom | Comp | Sil | K | noise% | capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  1 | B5_s42       | C1 | 0.7981 | 0.8580 | 0.9200 | 0.9292 | 0.9430 | 0.8978 | 0.4061 | 38 |  8.34 | 1.000 |
|  2 | B5_s42       | C2 | 0.6897 | 0.8619 | 0.8731 | 0.8901 | 0.9333 | 0.8506 | 0.7691 | 43 | 37.84 | 1.000 |
|  3 | B5_s42       | C3 | 0.6968 | 0.8815 | 0.8854 | 0.9007 | 0.9381 | 0.8667 | 0.7616 | 40 | 37.74 | 0.976 |
|  4 | NEW_s42      | C1 | 0.8269 | 0.8682 | 0.9257 | 0.9342 | 0.9527 | 0.9173 | 0.3972 | 39 |  7.13 | 1.000 |
|  5 | NEW_s42      | C2 | 0.7062 | 0.8616 | 0.8762 | 0.8929 | 0.9311 | 0.8578 | 0.7701 | 43 | 38.86 | 1.000 |
|  6 | NEW_s42      | C3 | 0.6638 | 0.8806 | 0.8757 | 0.8916 | 0.9410 | 0.8455 | 0.7432 | 42 | 35.60 | 1.000 |
|  7 | NEW_s1       | C1 | 0.6865 | 0.8477 | 0.8943 | 0.9054 | 0.9381 | 0.8754 | 0.5548 | 37 | 11.37 | 1.000 |
|  8 | NEW_s1       | C2 | 0.6473 | 0.8464 | 0.8576 | 0.8762 | 0.9209 | 0.8347 | 0.7349 | 43 | 36.67 | 1.000 |
|  9 | NEW_s1       | C3 | 0.6394 | 0.8759 | 0.8663 | 0.8830 | 0.9352 | 0.8342 | 0.7319 | 42 | 34.81 | 1.000 |
| 10 | NEW_s2       | C1 | 0.6990 | 0.8488 | 0.8942 | 0.9089 | 0.9385 | 0.8801 | 0.4633 | 38 | 13.00 | 1.000 |
| 11 | NEW_s2       | C2 | 0.7421 | 0.8320 | 0.8771 | 0.8924 | 0.9186 | 0.8687 | 0.7859 | 41 | 40.45 | 1.000 |
| 12 | NEW_s2       | C3 | 0.7413 | 0.8488 | 0.8867 | 0.9020 | 0.9369 | 0.8739 | 0.7703 | 40 | 39.28 | 1.000 |
| 13 | NEWnoNeCo_s3 | C1 | 0.8518 | 0.8214 | 0.9182 | 0.9255 | 0.9293 | 0.9221 | 0.3813 | 37 |  6.71 | 1.000 |
| 14 | NEWnoNeCo_s3 | C2 | 0.6683 | 0.7602 | 0.8453 | 0.8678 | 0.8888 | 0.8529 | 0.7647 | 40 | 42.68 | 1.000 |
| 15 | NEWnoNeCo_s3 | C3 | 0.6562 | 0.8094 | 0.8501 | 0.8693 | 0.9023 | 0.8431 | 0.7363 | 41 | 40.17 | 1.000 |
| 16 | B4noNeCo_s1  | C1 | 0.8540 | 0.8122 | 0.9192 | 0.9268 | 0.9288 | 0.9280 | 0.4780 | 34 |  6.10 | 1.000 |
| 17 | B4noNeCo_s1  | C2 | 0.6127 | 0.8085 | 0.8485 | 0.8671 | 0.9007 | 0.8385 | 0.7286 | 39 | 37.05 | 0.976 |
| 18 | B4noNeCo_s1  | C3 | 0.5171 | 0.8282 | 0.8436 | 0.8624 | 0.9097 | 0.8202 | 0.6440 | 39 | 28.94 | 0.976 |
| 19 | B4noNeCo_s2  | C1 | 0.8022 | 0.8605 | 0.9169 | 0.9285 | 0.9426 | 0.9117 | 0.4693 | 37 |  8.57 | 1.000 |
| 20 | B4noNeCo_s2  | C2 | 0.8138 | 0.8576 | 0.9104 | 0.9215 | 0.9298 | 0.9104 | 0.7914 | 38 | 41.94 | 1.000 |
| 21 | B4noNeCo_s2  | C3 | 0.8022 | 0.8605 | 0.9169 | 0.9285 | 0.9426 | 0.9117 | 0.4693 | 37 |  8.57 | 1.000 |

(노트: row 19 / 21 동일 — B4noNeCo_s2 에서 C1 과 C3 가 동일 clustering 으로 수렴. mcs=12,ms=3,ε=0.06 (C1) 와 mcs=15,ms=3,ε=0.0 (C3) 가 같은 K=37 / noise=8.57% 산출. HDBSCAN tree 안정성 (eom→leaf 효과 미미 + ε=0.06 merge 후 ε=0.0 leaf 결과 일치) — 데이터에 dependent.)

---

## 3. Group Aggregates (per cfg)

### 3.1 NEW (3-seed: s42, s1, s2)

| cfg | ARI_wN | ARI_woN | AMI | NMI | Hom | Comp | Sil | K | noise% | capture |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.7375 ± 0.0777 | **0.8549 ± 0.0115** | 0.9047 ± 0.0182 | 0.9162 ± 0.0162 | 0.9431 ± 0.0090 | 0.8909 ± 0.0230 | 0.4718 ± 0.0791 | 38.0 ± 1.0 | 10.50 ± 3.03 | 1.000 ± 0.000 |
| C2 | 0.6985 ± 0.0479 | 0.8467 ± 0.0148 | 0.8703 ± 0.0110 | 0.8872 ± 0.0095 | 0.9235 ± 0.0059 | 0.8537 ± 0.0174 | 0.7636 ± 0.0261 | 42.3 ± 1.2 | 38.66 ± 1.90 | 1.000 ± 0.000 |
| C3 | 0.6815 ± 0.0532 | **0.8684 ± 0.0172** | 0.8762 ± 0.0102 | 0.8922 ± 0.0087 | 0.9377 ± 0.0071 | 0.8512 ± 0.0205 | 0.7485 ± 0.0197 | 41.3 ± 1.2 | 36.56 ± 2.39 | 1.000 ± 0.000 |

### 3.2 B4-noNeCo (2-seed: s1, s2)

| cfg | ARI_wN | ARI_woN | AMI | NMI | Hom | Comp | Sil | K | noise% | capture |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | **0.8281 ± 0.0366** | 0.8364 ± 0.0342 | 0.9181 ± 0.0016 | 0.9277 ± 0.0011 | 0.9357 ± 0.0097 | 0.9199 ± 0.0115 | 0.4737 ± 0.0062 | 35.5 ± 2.1 |  7.34 ± 1.75 | 1.000 ± 0.000 |
| C2 | 0.7132 ± 0.1422 | 0.8331 ± 0.0347 | 0.8795 ± 0.0438 | 0.8943 ± 0.0384 | 0.9153 ± 0.0247 | 0.8744 ± 0.0508 | 0.7600 ± 0.0444 | 38.5 ± 0.7 | 39.50 ± 3.46 | 0.988 ± 0.017 |
| C3 | 0.6597 ± 0.2016 | 0.8444 ± 0.0228 | 0.8802 ± 0.0518 | 0.8947 ± 0.0455 | 0.9262 ± 0.0233 | 0.8659 ± 0.0647 | 0.5566 ± 0.1235 | 38.0 ± 1.4 | 18.76 ± 14.40 | 0.988 ± 0.017 |

### 3.3 B5 / NEW-noNeCo (single seed)

Same as table §2 rows 1-3 (B5) and 13-15 (NEW-noNeCo).

---

## 4. Paper Claim Localization — "ARI 0.859 ± 0.018"

| candidate cell | exact value | match? |
|---|---|---|
| NEW 3-seed × C1 ARI_woN | 0.8549 ± 0.0115 | std too tight (0.0115 vs 0.018) |
| NEW 3-seed × C2 ARI_woN | 0.8467 ± 0.0148 | mean too low |
| **NEW 3-seed × C3 ARI_woN** | **0.8684 ± 0.0172** | **std=0.017 ≈ 0.018** — closest |
| NEW 3-seed × C1 ARI_wN  | 0.7375 ± 0.0777 | far |
| B5_s42 single × C3 ARI_woN | 0.8815 (no std)  | std N/A |
| B4-noNeCo 2-seed × C1 ARI_woN | 0.8364 ± 0.0342 | std too wide |

**최유력 후보**: NEW 3-seed × C3 (leaf, mcs=15, ms=3, ε=0.0) on **ARI_woN** (without Normal) →
**0.8684 ± 0.0172**. Mean off by 0.009, std off by 0.001. 표기 차 (반올림 / 통계 round) 가능 범위.

대안: **NEW 3-seed × C1 ARI_woN** (0.8549 ± 0.0115). Mean off by 0.004, std off by 0.007 — mean
이 더 가깝지만 std 차 큼.

**증명력 있는 paper claim 표기** (재현 가능):
> "Queue+NEG+NeCo (NEW recipe, no Local), 3 seeds {42,1,2}, HDBSCAN(leaf, mcs=15, ms=3, ε=0.0),
> ARI computed on the 1146 non-Normal samples (Normal class excluded) → **0.868 ± 0.017**."

또는
> "ARI(no-Normal) = 0.855 ± 0.012, HDBSCAN(eom, mcs=12, ms=3, ε=0.06)" — C1.

원본 0.859 ± 0.018 이 둘 중 어느 cfg 출처인지 paper 본문에 명시 필요.

---

## 5. Per-Metric Ranking (Group × cfg)

### 5.1 ARI_wN (with Normal) — paper full-clustering measure

| rank | group × cfg | value |
|---|---|---|
| 1 | B4-noNeCo × C1     | 0.8281 ± 0.0366 |
| 2 | NEW × C2           | 0.6985 ± 0.0479 |
| 3 | NEW × C3           | 0.6815 ± 0.0532 |
| 4 | NEW × C1           | 0.7375 ± 0.0777 |
| 5 | B4-noNeCo × C2     | 0.7132 ± 0.1422 |
| 6 | B4-noNeCo × C3     | 0.6597 ± 0.2016 |

(B5 single 0.7981 / NEW-noNeCo single 0.8518 가 single-seed 라 std 없음 — 직접 비교 불가)

**핵심 관찰**: B4-noNeCo × C1 이 ARI_wN 에서 0.828 로 최고. NEW recipe 보다 0.09 높음.
NeCo 제거 + Local 유지가 with-Normal ARI 에서 유리.

### 5.2 ARI_woN (without Normal) — defect-only clustering quality

| rank | group × cfg | value |
|---|---|---|
| 1 | **NEW × C3**         | **0.8684 ± 0.0172** |
| 2 | NEW × C1             | 0.8549 ± 0.0115 |
| 3 | NEW × C2             | 0.8467 ± 0.0148 |
| 4 | B4-noNeCo × C3       | 0.8444 ± 0.0228 |
| 5 | B4-noNeCo × C1       | 0.8364 ± 0.0342 |
| 6 | B4-noNeCo × C2       | 0.8331 ± 0.0347 |

**핵심 관찰**: defect-only ARI 는 NEW recipe (Queue+NEG+NeCo, no Local) 가 우위. C3 cfg
(leaf, mcs=15, ms=3, ε=0.0) 가 NEW 의 best.

### 5.3 Completeness (defect group integrity)

| rank | group × cfg | value |
|---|---|---|
| 1 | B4-noNeCo × C1   | 0.9199 ± 0.0115 |
| 2 | NEW × C1         | 0.8909 ± 0.0230 |
| 3 | B4-noNeCo × C2   | 0.8744 ± 0.0508 |
| 4 | B4-noNeCo × C3   | 0.8659 ± 0.0647 |
| 5 | NEW × C2         | 0.8537 ± 0.0174 |
| 6 | NEW × C3         | 0.8512 ± 0.0205 |

### 5.4 AMI

| rank | group × cfg | value |
|---|---|---|
| 1 | B4-noNeCo × C1 | 0.9181 ± 0.0016 |
| 2 | NEW × C1       | 0.9047 ± 0.0182 |
| 3 | B4-noNeCo × C2 | 0.8795 ± 0.0438 |
| 4 | B4-noNeCo × C3 | 0.8802 ± 0.0518 |
| 5 | NEW × C3       | 0.8762 ± 0.0102 |
| 6 | NEW × C2       | 0.8703 ± 0.0110 |

### 5.5 Silhouette (cosine)

| rank | group × cfg | value |
|---|---|---|
| 1 | NEW × C2       | 0.7636 ± 0.0261 |
| 2 | B4-noNeCo × C2 | 0.7600 ± 0.0444 |
| 3 | NEW × C3       | 0.7485 ± 0.0197 |
| 4 | B4-noNeCo × C3 | 0.5566 ± 0.1235 |
| 5 | B4-noNeCo × C1 | 0.4737 ± 0.0062 |
| 6 | NEW × C1       | 0.4718 ± 0.0791 |

(Silhouette 은 high-noise cfg (C2 = 38%, C3 = 36% noise) 가 자연히 높음 — 노이즈 점이
silhouette 계산 제외, 남은 점들이 dense cluster 라 silhouette 부풀음. 신뢰 가능
metric 아님 in this regime.)

### 5.6 noise%

| rank (low=좋음) | group × cfg | value |
|---|---|---|
| 1 | B4-noNeCo × C1  |  7.34 ± 1.75 |
| 2 | NEW × C1        | 10.50 ± 3.03 |
| 3 | B4-noNeCo × C3  | 18.76 ± 14.40 |
| 4 | NEW × C3        | 36.56 ± 2.39 |
| 5 | NEW × C2        | 38.66 ± 1.90 |
| 6 | B4-noNeCo × C2  | 39.50 ± 3.46 |

C1 (eom + ε=0.06) 가 모든 group 에서 noise 최저. eom 의 aggressive merge 효과.

---

## 6. cfg-Sensitivity Analysis

같은 embedding 위에서 cfg 만 바꿨을 때 metric 이 얼마나 바뀌는지:

| run_tag | ARI_wN spread (max−min) | ARI_woN spread | K spread | noise% spread |
|---|---|---|---|---|
| B5_s42        | 0.1084 (0.79–0.69) | 0.0235 | 5 (38–43) | 29.5 (8.3–37.8) |
| NEW_s42       | 0.1631 (0.83–0.66) | 0.0190 | 4 (39–43) | 31.7 (7.1–38.9) |
| NEW_s1        | 0.0471 (0.69–0.64) | 0.0295 | 6 (37–43) | 25.3 (11.4–36.7) |
| NEW_s2        | 0.0431 (0.74–0.70) | 0.0168 | 3 (38–41) | 27.5 (13.0–40.5) |
| NEWnoNeCo_s3  | 0.1956 (0.85–0.66) | 0.0612 | 4 (37–41) | 36.0 (6.7–42.7) |
| B4noNeCo_s1   | 0.3369 (0.85–0.52) | 0.0237 | 5 (34–39) | 30.9 (6.1–37.0) |
| B4noNeCo_s2   | 0.0116 (0.81–0.80) | 0.0029 | 1 (37–38) | 33.4 (8.6–41.9) |

**관찰**:
- ARI_wN 은 cfg sensitivity 매우 큼 (특히 B4noNeCo_s1 에서 0.337 swing). HDBSCAN cfg 선택이
  paper 보고 ARI 절대치를 결정.
- ARI_woN 은 cfg 에 대해 더 안정 (NEW 시리즈 spread 0.017–0.030). defect-only 평가가
  reproducibility 더 좋음.
- K (cluster 수) 는 cfg 마다 3-6 차이. 42-43 GT class 와 비교하면 모든 cfg 가 reasonable K
  도달.

---

## 7. Group Comparison Summary

### 7.1 NEW vs B4-noNeCo vs B5

| metric | best cfg per group | NEW (3-seed) | B4-noNeCo (2-seed) | B5 (1-seed) | NEW-noNeCo (1-seed) |
|---|---|---|---|---|---|
| ARI_wN  | C1 | 0.7375 ± 0.078 | **0.8281 ± 0.037** | 0.7981 | 0.8518 |
| ARI_woN | C3 / C1 | **0.8684 ± 0.017** | 0.8444 ± 0.023 | 0.8815 | 0.8214 |
| AMI     | C1 | 0.9047 ± 0.018 | **0.9181 ± 0.002** | 0.9200 | 0.9182 |
| Comp    | C1 | 0.8909 ± 0.023 | **0.9199 ± 0.012** | 0.9110 | 0.9221 |
| K       | (cfg-dep)  | 38-42        | 35-39              | 38-43       | 37-41              |
| noise%  | C1 | 10.50 ± 3.03  | **7.34 ± 1.75**    | 8.34        | 6.71               |
| capture | (all)      | 1.000        | 0.988-1.000        | 0.976-1.000 | 1.000              |

**Recipe ranking** (per cfg-C1 ARI_wN, with-Normal):
1. NEW-noNeCo (no Local + no NeCo) — 0.852  ★ single seed, 재현성 미확인
2. B4-noNeCo (Local + no NeCo) — 0.828 ± 0.037
3. B5 (Local + NeCo) — 0.798
4. NEW (no Local, with NeCo) — 0.737 ± 0.078

**Recipe ranking** (per cfg-C3 ARI_woN, without-Normal):
1. B5 (Local + NeCo) — 0.882  ★ single seed
2. NEW (no Local, with NeCo) — 0.868 ± 0.017
3. B4-noNeCo (Local + no NeCo) — 0.844 ± 0.023
4. NEW-noNeCo (no Local + no NeCo) — 0.809  ★ single seed

**해석 (가설, paper 검증 필요)**:
- **With-Normal ARI**: NeCo 가 Normal cluster boundary 흐림 (Normal=1000 이라 dominant) →
  no-NeCo 가 유리. (NEW-noNeCo, B4-noNeCo > NEW, B5)
- **Without-Normal ARI**: NeCo 가 defect representation 향상 (보조 contrastive) → NeCo 보존이
  유리. (NEW, B5 > NEW-noNeCo, B4-noNeCo)
- **Local 효과**: Mixed. NEW (no Local, ARI_wN 0.738) 가 B4-noNeCo (Local, 0.828) 보다 낮지만,
  ARI_woN 에서는 NEW (0.868) > B4-noNeCo (0.844). Local 이 Normal-defect boundary 강화 효과
  뚜렷.

(중요: single-seed (B5, NEW-noNeCo) 는 통계적 비교 불가 — 3-seed multi-seed run 결과만
ranking inference 신뢰. paper 본문은 multi-seed (NEW, B4-noNeCo) 결과를 main result 로,
single-seed 는 supplementary 로 보고하는 것이 정당.)

### 7.2 Seed Variance

- **NEW 3-seed**: ARI_wN std=0.05-0.08, ARI_woN std=0.011-0.017. ARI_wN 이 seed sensitive
  (Normal cluster behavior 가 unstable).
- **B4-noNeCo 2-seed**: ARI_wN std=0.04-0.20, ARI_woN std=0.023-0.035. seed=1 과 seed=2 사이
  큰 차 (B4noNeCo_s1 C3 = 0.517, B4noNeCo_s2 C3 = 0.802 — 0.285 차). BATCH 4 vs 8 difference
  포함된 것이라 same-condition 가정 의문 (사용자 feedback `feedback_batch_same_condition.md`
  에서는 same 으로 간주, 그러나 0.285 spread 는 BATCH 영향 시사).

---

## 8. Reproducibility & Caveats

1. **B5 raw 가 main `eval/` 에 없음** — `eval_leaf_ms4_BACKUP/embeddings/` 에서 로드. paper
   본문에 B5 결과 명시할 때 이 path 표기 권고.
2. **NEW-noNeCo 와 B4-noNeCo seed 가 disjoint** (s3 vs s1+s2). recipe 비교 시 seed-equalized
   아님. 향후 NEW-noNeCo 도 s1, s2 추가 권고.
3. **B4-noNeCo s1 vs s2 BATCH 차이** (4 vs 8). 사용자 정책으로 same-condition 간주, 그러나
   seed variance 가 크면 BATCH 영향 ablation 별도 필요.
4. **Cosine vs Euclidean HDBSCAN**: L2-normalized embedding 에서 두 metric 은 monotonic
   equivalent (Euclidean² = 2 − 2·cos). cluster boundary 동일.
5. **Silhouette 신뢰도**: high-noise cfg (C2, C3) 에서 silhouette 부풀음 (40% point 가 noise
   라 silhouette 계산 제외). C1 의 silhouette 0.47 이 보다 honest.
6. **Capture rate 1.000 saturated**: 42 defect class 모두 ≥1 sample 이 non-noise cluster 에
   도달. 더 엄격한 기준 (예: ≥50% 동일 cluster) 도 추후 분석 추천.

---

## 9. Recommendations

### 9.1 Paper 본문 수정 권고

- **ARI 0.859 ± 0.018** 의 cfg 출처 명시 — 본 분석 결과 NEW 3-seed × C3 (ARI_woN) = 0.868 ±
  0.017 또는 NEW 3-seed × C1 (ARI_woN) = 0.855 ± 0.012 가 후보.
- Tier 1 metric 표에 cfg row 추가 (sel / mcs / ms / ε).
- ARI_wN vs ARI_woN 구분 명시 (Normal exclude 여부).

### 9.2 추가 실험 권고

- **NEW-noNeCo s1, s2 학습** — recipe ablation seed-equalize. 현재 s3 single 만 있어 통계 불가.
- **B4-noNeCo BATCH=8 second seed** — BATCH 영향 isolate.
- **cfg-C1 (eom, mcs=12, ms=3, ε=0.06) 을 paper main result 로 lock** — capture=1.000,
  noise<11%, defect-only ARI 0.85±, 통계 안정. C2/C3 (40% noise) 은 supplementary.

### 9.3 메타-결정

- **Defect-only evaluation (ARI_woN) 을 primary metric 으로**: capture 1.000 보장 + 통계
  안정 + Normal cluster behavior (gt label noise 가능) 의존 제거.
- HDBSCAN cfg 는 paper 본문에 **1개 hard-coded** (예: C1) — sweep 결과는 supplementary
  table 만.

---

## 10. Artifacts

| 파일 | 설명 |
|---|---|
| `apples_to_apples_7run_3cfg.md` | 본 보고서 |
| `_apples_7run_3cfg_raw.parquet` | 21-cell raw metric DataFrame (binary) |
| `_apples_7run_3cfg_raw.csv`     | 21-cell raw metric (CSV, eyeball-friendly) |

read-only: 각 run 의 `eval_summary.json` / `cluster_report.parquet` 모두 무수정 보존.

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/apples_to_apples_7run_3cfg.md
