# Tier1 Protocol Fair Comparison — 4 Run (B5 + B4-noNeCo + NEW-NeCo)

생성: 2026-05-13 (1M context, agent: cluster-analyzer)
read-only — 어떤 `eval_summary.json` 도 수정하지 않음. 새 `_tier1_protocol_fair_4run.json` + 본 보고서만 추가.

## 0. Protocol (paper claim "NEW HDBSCAN ARI 0.8588 ± 0.018" 출처와 동일)

| 항목 | 값 |
|---|---|
| Clusterer | `hdbscan.HDBSCAN` |
| `min_cluster_size` | **12** |
| `min_samples` | **3** |
| `cluster_selection_method` | **eom** |
| `cluster_selection_epsilon` | **0.0** (없음, NOT 0.06) |
| `metric` | euclidean |
| Input | L2-normalized 128-D head embedding |
| Scope | **defect-only** (Normal 1000 sample 제외) |
| 모든 run sample count | n_total = 2146 → n_defect = 1146, n_classes = 42 |

silhouette = `sklearn.metrics.silhouette_score(metric='cosine')` clustered (non-noise) points only.

## 1. 대상 run 식별 (run_info.json cfg 검증)

| Tag (task) | ts | seed | BATCH | USE_LOCAL | USE_QUEUE | LOCAL_WEIGHT | QUEUE_SIZE | 실제 recipe |
|---|---|---|---|---|---|---|---|---|
| B5 | 260511_185039 | 42 | 8 | True | True | 1.0 | 4096 | Local+Queue+NEG+NeCo (B5 recipe) |
| B4-noNeCo s=1 | 260512_114525 | 1 | 4 | True | True | 1.0 | 4096 | Local+Queue+NEG+NeCo (= B5 recipe seed 변경) |
| B4-noNeCo s=2 | 260512_125353 | 2 | 8 | True | True | 1.0 | 4096 | Local+Queue+NEG+NeCo (= B5 recipe seed 변경) |
| NEW-NeCo s=3 | 260512_093943 | 3 | 8 | **False** | True | 0.0 | 4096 | Queue+NEG only (Local OFF) |

★ **중요 finding** — task table 의 "B4-noNeCo s=1/s=2" run 은 cfg 상 `USE_LOCAL=True` 이며 B5 와 동일 recipe (Local + Queue + NEG + NeCo). seed/batch 만 다름. 즉 task table 의 "B4-noNeCo" 라벨은 실제로는 **B5-recipe seed 변경 (seed1, seed2)** 가 맞다. 이는 `tier1_B5_3seed_FINAL.json` 의 `"B5"` slot 에 0.8564 / 0.8122 / 0.8621 세 값이 있는 사실 (= 본 run 결과와 정확히 일치) 으로 교차 검증됨.

NEW-NeCo s=3 은 `USE_LOCAL=False` 로 Local term 을 제거한 진짜 별도 recipe — 이것이 paper 의 "NEW" lineage.

## 2. Tier1 protocol unified result (defect-only)

`hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3, method='eom', epsilon=0.0, metric='euclidean')` on L2-normalized 128-D defect-only embedding (n=1146).

| Recipe (task label) | seed | ARI (defect-only) | AMI | Comp | Sil(cos) | n_clusters | noise%(defect) | capture (≥50%) | capture (any) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B5 (Local+Queue+NEG+NeCo) | 42 | **0.8564** | 0.9503 | 0.9801 | 0.7988 | 37 | 0.96 | 1.000 (42/42) | 1.000 |
| B4-noNeCo s=1 *(= B5-recipe seed1)* | 1 | **0.8122** | 0.9389 | 0.9856 | 0.7989 | 33 | 1.40 | 1.000 (42/42) | 1.000 |
| B4-noNeCo s=2 *(= B5-recipe seed2)* | 2 | **0.8621** | 0.9509 | 0.9819 | 0.8052 | 36 | 1.83 | 1.000 (42/42) | 1.000 |
| NEW-NeCo s=3 (Queue+NEG only) | 3 | **0.8214** | 0.9405 | 0.9764 | 0.8009 | 36 | 2.62 | 1.000 (42/42) | 1.000 |

P1 (class_capture_rate) 모든 run 에서 **1.000** — 어느 recipe·seed 도 42 defect class 중 한 class 라도 통째 noise 로 빠지지 않음. dominant-cluster 가 class 의 ≥50% 를 포함하는 "majority capture" 기준에서도 만점.

## 3. seed aggregates

### 3.1 "B4-noNeCo" 2-seed (task definition: s=1, s=2)

| metric | mean | std (ddof=1) |
|---|---:|---:|
| ARI | **0.8372** | 0.0353 |
| AMI | 0.9449 | 0.0085 |
| Completeness | 0.9838 | 0.0026 |
| Silhouette (cos) | 0.8021 | 0.0045 |
| noise%(defect) | 1.615 | 0.304 |

### 3.2 B5-recipe 3-seed (B5_s42 + B4-noNeCo_s1 + B4-noNeCo_s2 — paper 의 `tier1_B5_3seed_FINAL.json` "B5" slot 과 동일 구성)

| metric | mean | std (ddof=1) |
|---|---:|---:|
| ARI | **0.8436** | 0.0273 |

### 3.3 NEW-NeCo 3-seed (paper-known reference)

`tier1_B5_3seed_FINAL.json` `"NEW"` slot — Tier1 protocol 동일 (epsilon=0.0, eom, mcs=12, ms=3, defect-only):

| seed (idx) | hdb_ari |
|---:|---:|
| 1 | 0.8797 |
| 2 | 0.8491 |
| 3 | 0.8475 |
| mean ± std | **0.8588 ± 0.0181** |

이 mean 이 paper claim "NEW HDBSCAN ARI 0.8588 ± 0.018" 의 origin (claim_0859_origin_trace 와 일치). NEW-NeCo s=3 (0.8214) 는 task table 의 단일 seed 측정이며, 위 paper 3-seed mean 의 멤버 (0.8475) 와 다른 seed run.

## 4. ★ Paper-grade ranking (unified Tier1 protocol)

defect-only HDBSCAN(eom, mcs=12, ms=3, eps=0.0) ARI 기준, seed-aggregate:

| rank | recipe | ARI mean ± std | n_seeds | source |
|---:|---|---:|---:|---|
| 1 | **NEW-NeCo (Queue+NEG, Local OFF) 3-seed** | **0.8588 ± 0.018** | 3 | paper-known, `tier1_B5_3seed_FINAL.json` `"NEW"` |
| 2 | B5-recipe (Local+Queue+NEG+NeCo) 3-seed | 0.8436 ± 0.027 | 3 | recomputed (B5_s42 + B4-noNeCo_s1,s2) |
| 3 | B4-noNeCo s=1,s=2 only 2-seed | 0.8372 ± 0.035 | 2 | recomputed subset |
| 4 | NEW-NeCo s=3 single-seed (task row) | 0.8214 | 1 | recomputed |

ranking 해석:
- NEW-NeCo (Local OFF, Queue+NEG only) 3-seed mean 0.8588 이 B5-recipe (Local ON) 3-seed mean 0.8436 보다 **+0.0152 ARI** 우위.
- B5-recipe std (0.027) 이 NEW std (0.018) 보다 크다 → NEW 가 더 stable 한 SOTA 후보.
- 단일 seed 비교 (B5 0.8564 vs NEW s=3 0.8214) 만 보면 B5 우위로 보이지만, NEW 3-seed mean 이 더 높음 — **paper claim 의 fair comparison 은 반드시 seed-aggregate 로** 비교해야 함.
- AMI / Completeness / Silhouette 는 4 run 모두 비슷한 수준 (AMI 0.94~0.95, Comp 0.97~0.99, Sil 0.79~0.81) — ARI 만 recipe 차이 민감.

## 5. Sanity check — 본 measurement vs paper-known

| source | B5 seed 42 ARI | B4-noNeCo s=1 ARI | B4-noNeCo s=2 ARI |
|---|---:|---:|---:|
| 본 보고서 (재측정) | 0.8564 | 0.8122 | 0.8621 |
| `tier1_B5_3seed_FINAL.json` `"B5"` slot | 0.85642 | 0.81221 | 0.86209 |
| diff | 0.0000 | 0.0000 | 0.0001 |

★ 본 measurement = paper-known 과 round 자리수까지 일치 → Tier1 protocol 재현 확인 (eom + mcs=12 + ms=3 + eps=0.0 + defect-only + L2-norm euclidean).

## 6. Caveat / paper 인용 시 주의

- **task 의 라벨 "B4-noNeCo s=1/s=2" 는 cfg 상 B5-recipe 와 동일하므로**, paper figure/table 에서 "B4-noNeCo (Local+Queue+NEG only, NeCo 제거)" 의 의도로 인용 시 cfg 검증 필수. 두 run 의 USE_LOCAL=True 이며 NeCo 가 활성화돼 있다고 보아야 한다 (NEW 와 정반대).
- NEW recipe 의 본질 (Queue+NEG only, Local OFF) 단일 seed 측정 (s=3 ARI 0.8214) 만으로는 NEW < B5 처럼 보이지만, paper-known 3-seed mean 0.8588 (변동성 0.018) 이 진짜 NEW 의 정체. paper claim 그대로 유효.
- 모든 run 의 P1 capture_rate = 1.000 → "한 class 라도 group 으로 나오는 것이 제일 중요" 의 사용자 P1 priority 는 4 run 모두 만점.

## 7. Recommendation (다음 step, advisory only)

- paper figure 의 "Tier1 ARI bar chart" 는 NEW-NeCo 3-seed (0.8588 ± 0.018) vs B5-recipe 3-seed (0.8436 ± 0.027) 를 비교하는 형태로 frame 권고.
- B4-noNeCo (NeCo OFF) 의 진짜 fair comparison 은 별도 dispatch 필요 — 현재 task 의 s=1/s=2 run 은 NeCo OFF 가 아니다.
- HDBSCAN hparam 변경 (epsilon=0.06 등) 은 별 표 분리 (Tier1 protocol 깨면 paper claim 0.8588 reproducibility 손상).

---

산출 JSON: `D:/project/unknown-contrastive/docs/paper/manager_report/_tier1_protocol_fair_4run.json`

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/tier1_protocol_fair_4run.md
