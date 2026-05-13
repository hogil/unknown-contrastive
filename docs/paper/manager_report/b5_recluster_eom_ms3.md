# B5 Recluster — eom mcs=12 ms=3 eps=0.06 (apples-to-apples vs NEW)

**Date**: 2026-05-13
**Run dir**: `D:/project/unknown-contrastive/outputs_contrastive_260511_185039`
**Checkpoint**: `checkpoints/final_infer.pt` (B5 / iter 37, `260511_185039`)
**Purpose**: B5 의 eval 을 NEW (iter 70/71/72) 와 **동일 HDBSCAN cfg** 로 재실행하여 apples-to-apples 비교 확보.

---

## 1. 문제 정의

- 기존 B5 eval (`eval_leaf_ms4_BACKUP/`) 은 cfg `leaf + ms=4 + mcs=12 + eps=0.06` 로 실행됨.
  - 이 cfg 는 학습 종료 시 `run_info.json::cfg` 에 박힌 run-time fallback 으로 추정 (CFG override 없이 default 진행).
  - 산출: ARI 0.6897 / Comp 0.8506 / noise 37.84% / silhouette 0.7691.
- 반면 NEW iter 70/71/72 는 모두 cfg `eom + ms=3 + mcs=12 + eps=0.06` 로 평가됨.
  - 3-run 평균: ARI 약 0.7375.
- 같은 encoder embedding 도 cluster cfg 가 다르면 ARI/AMI/noise 가 크게 흔들리므로
  **NEW vs B5 ranking 비교 불가**.

→ B5 embedding 을 NEW cfg 로 재clustering 후 metric 재계산.

## 2. 실행

### 2.1 백업 (CLAUDE.md 규칙: 결과 폴더 삭제 금지 → rename only)

```
mv outputs_contrastive_260511_185039/eval \
   outputs_contrastive_260511_185039/eval_leaf_ms4_BACKUP
```

### 2.2 새 eval (eom ms=3 mcs=12 eps=0.06)

```bash
python _eval_contrastive_unknown_n50.py \
  --run-dir outputs_contrastive_260511_185039 \
  --cluster-selection-method eom \
  --min-samples 3 \
  --min-cluster-size 12 \
  --cluster-selection-epsilon 0.06 \
  --no-rename \
  --no-overall
```

- `--no-rename` : run-dir 자동 rename 차단 (`<run>_ari{:.2f}` 형식 변경 방지).
- `--no-overall` : `_overall_meta.json` (overall/ best mirror) 갱신 차단.
- final_infer.pt 로딩 후 2,146 image embedding 재계산 (약 9.5 분 GPU).
- 새 HDBSCAN cfg 적용 후 metric 재계산.

산출 로그 발췌:
```
[hdbscan_full] params={'min_cluster_size': 12, 'min_samples': 3, 'metric': 'euclidean',
                       'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.06,
                       'allow_single_cluster': False}
[hdbscan_full] 0.6s, clusters=38, noise=179
[Tier 1] Completeness=0.9110 AMI=0.9200 noise_def=8.34% capture=43/43(1.0000)
[Class frag] coverage=0.9166 single_cluster=40/43(0.9302) mean_n_clusters=1.07
```

## 3. 결과 — Tier 1+2 공식 metric

### 3.1 Tier 1

| metric | leaf+ms4 (기존) | eom+ms3 (새, NEW와 동일) | Δ |
|---|---|---|---|
| **Completeness** (P3)            | 0.8506 | **0.9110** | +0.0604 |
| **AMI** (보조)                   | 0.8731 | **0.9200** | +0.0469 |
| **noise_pct (defect)** (P2)      | 37.84% | **8.34%**  | -29.50 pp |
| **class_capture_rate** (P1)      | 43/43 (1.0000) | 43/43 (1.0000) | 0 |

### 3.2 Tier 2 보조

| metric | leaf+ms4 (기존) | eom+ms3 (새) | Δ |
|---|---|---|---|
| Homogeneity (P4)                 | 0.9326 | 0.9493 | +0.0167 |
| ARI (over-cluster 페널티)         | 0.6897 | **0.7981** | +0.1084 |
| Silhouette (cosine, noise 제외)   | 0.7691 | 0.4061 | -0.3630 |
| n_clusters                        | 43     | 38     | -5 |
| n_noise                           | 812    | 179    | -633 |

Silhouette 가 크게 감소한 것은 noise(-1) 가 큰 폭 감소하면서 cluster 가 더 많은 boundary
sample 까지 포함했기 때문 — intrinsic geometry quality 하락이 아니라 cluster 정의 자체가
"보수적 leaf+ms4 → tight cluster + 큰 noise" 에서 "관대한 eom+ms3 → 더 큰 cluster + 작은 noise"
로 바뀐 결과. P1 (capture) + P2 (noise) + P3 (completeness) 모두 개선.

### 3.3 class_fragmentation_summary

| field | leaf+ms4 | eom+ms3 | 해석 |
|---|---|---|---|
| n_defect_classes              | 43     | 43     | 동일 |
| n_classes_captured            | 43     | 43     | 동일 (capture=1.0000) |
| class_capture_rate (P1)       | 1.0000 | 1.0000 | 동일 |
| mean_cluster_coverage         | 0.9520 | 0.9863 | +0.0343 |
| **weighted_cluster_coverage** | 0.6216 | **0.9166** | +0.2950 (noise 가 weight 에 강하게 반영됨) |
| mean_n_clusters_per_class     | 1.1163 | 1.0698 | -0.0465 |
| n_classes_single_cluster      | 41     | 40     | -1 |
| n_classes_split_2             | 1      | 3      | +2 |
| n_classes_split_3plus         | 1      | 0      | -1 |
| frac_single_cluster           | 0.9535 | 0.9302 | -0.0233 |

`weighted_cluster_coverage` 의 0.6216 → 0.9166 점프는 leaf+ms4 가 Normal (n=1000) 의 777
sample 을 noise 로 격리시킨 영향이 큼. eom+ms3 에서는 Normal noise 가 큰 폭 감소하여 weighted
coverage 가 정상화.

## 4. NEW vs B5 비교 (apples-to-apples)

같은 cfg (eom ms=3 mcs=12 eps=0.06) 하에서:

| run | ARI | AMI | Completeness | noise_pct (defect) | capture | weighted_coverage |
|---|---|---|---|---|---|---|
| **B5 (iter 37, 260511_185039)** — recomputed | **0.7981** | **0.9200** | **0.9110** | **8.34%** | 43/43 | 0.9166 |
| NEW iter 70/71/72 (avg) — 기존 보고          | ~0.7375    | (별표 참조) | (별표 참조) | (별표 참조) | (별표 참조) | (별표 참조) |

→ **같은 HDBSCAN cfg 하에서 B5 가 NEW 3-run 평균 (ARI 0.7375) 보다 높음 (ARI 0.7981)**.
이전 leaf+ms4 vs eom+ms3 비교 (B5 0.6897 < NEW 0.7375) 는 cfg 차이가 만든 artifact 였음.

> 별표: NEW 의 AMI / Completeness / noise_pct / capture / weighted_coverage 값은
> `iter70/71/72 eval_summary.json` 들에서 직접 확인 필요 (이 보고서 범위 밖).
> NEW vs B5 종합 ranking 결정은 본 보고서 + NEW 측 같은 4 metric aggregate 후 별도 정리.

## 5. 출력 위치 + 무결성

- 새 eval (eom+ms3): `D:/project/unknown-contrastive/outputs_contrastive_260511_185039/eval/`
  - `eval_summary.json` (Tier 1+2 새 cfg)
  - `cluster_report.parquet` (39 rows = 38 cluster + 1 noise)
  - `class_fragmentation.parquet` (43 rows)
  - `retrieval_report.parquet` (44 rows)
  - `file_list.parquet` (2146 rows)
  - `embeddings/embedding.npy` (2146, 128)
  - `groups/` 이미지 view
  - `per_class_report.txt`, `plots/`
- 기존 eval (leaf+ms4) 백업: `D:/project/unknown-contrastive/outputs_contrastive_260511_185039/eval_leaf_ms4_BACKUP/`
  - 원본 그대로 보존 (rename only, 파일 변경 0).
- `_overall_meta.json` 변경 없음 (`--no-overall` 적용).
- run-dir 명 변경 없음 (`--no-rename` 적용).

## 6. 결론

1. B5 의 cfg 차이 (leaf+ms4 vs eom+ms3) 가 **ARI 0.11 / AMI 0.05 / noise 30pp** 의 큰 차이를
   만들었음. NEW 와 비교 시 반드시 같은 cfg 로 통일해야 fair.
2. **같은 cfg 하 B5 가 NEW 3-run 평균보다 ARI 높음 (0.7981 vs 0.7375)**. iter 37 (B5)
   embedding quality 가 iter 70/71/72 NEW 보다 cluster 친화적.
3. Tier 1 4 metric 모두 P1 capture=1.0000 (lock), P2 noise 8.34% (low), P3 Completeness 0.9110,
   AMI 0.9200 — production-ready level.
4. NEW 측 같은 cfg metric 4종을 aggregate 한 후 ranking 정식 확정 필요.

---

[OUT] D:/project/unknown-contrastive/outputs_contrastive_260511_185039/eval/
[OUT] D:/project/unknown-contrastive/outputs_contrastive_260511_185039/eval_leaf_ms4_BACKUP/
[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/b5_recluster_eom_ms3.md
