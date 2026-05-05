---
name: evaluation
description: val 이미지 + 학습된 모델 → 공식 clustering metric 계산 → eval_summary.json 산출. Tier 1 (Completeness, AMI, noise_pct, class_capture_rate) + Tier 2 (Homogeneity, Silhouette cosine, ARI) + class_fragmentation_summary. 커스텀 metric 출력 절대 금지.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# evaluation agent

val set 에 대해 **공식 clustering metric** 만 계산. JSON 저장 + 콘솔 1-2줄 보고.

## 가장 먼저 할 일

1. `.claude/skills/contrastive-eval/SKILL.md` 읽기 — Tier 1/2/3 + 금지 metric
2. `docs/contrastive-eval/METRICS.md` — 출처 / 정의 / 우선순위 P1-P4
3. `docs/contrastive-eval/DECISIONS.md` — 거부된 옵션 사유

## 사전 조건

- `data/wm811k_val/` 존재
- `outputs_<preset>_<ts>/checkpoints/final_infer.pt` 존재
- `outputs_<preset>_<ts>/eval/embeddings/embedding.npy` (학습 후 산출)
- `outputs_<preset>_<ts>/eval/cluster_report.parquet` (기존 함수 산출)
- `outputs_<preset>_<ts>/eval/class_fragmentation.parquet` (기존 함수 산출)

## 실행 단계

1. val 이미지 embedding 계산 (eval mode).
2. HDBSCAN `approximate_predict` 또는 centroid 기반 cluster 할당.
3. **Tier 1 metric 계산**:
   - **Completeness** (sklearn `homogeneity_completeness_v_measure`) — P3
   - **AMI** (sklearn `adjusted_mutual_info_score`) — 보조
   - **noise_pct (defect only)** = (cluster_id == -1 ∩ defect) / total_defect × 100 — P2
   - **class_capture_rate** = (n_clusters ≥ 1 인 class) / total_class — P1
4. **Tier 2 metric 계산** (보조):
   - Homogeneity (P4) — 같은 sklearn 함수
   - Silhouette (cosine, sklearn `silhouette_score`, noise -1 제외)
   - ARI (sklearn `adjusted_rand_score`) — over-cluster 페널티 inherent 명시
5. **class_fragmentation_summary** (eval/class_fragmentation.parquet 의 aggregate, 7 필드):
   - `n_defect_classes`, `n_classes_captured`, `class_capture_rate`
   - `mean_cluster_coverage`, `weighted_cluster_coverage`
   - `mean_n_clusters_per_class`, `n_classes_single_cluster`, `n_classes_split_2`, `n_classes_split_3plus`, `frac_single_cluster`
6. `eval_summary.json` 저장 (기존 키 보존, 새 키 `class_fragmentation_summary` 추가).
7. **콘솔 1-2 줄 보고** (고정 형식):
   ```
   [Tier 1] Completeness=0.947 AMI=0.929 noise_def=0.71% capture=38/38(1.000)
   [Class frag] coverage=0.993 single_cluster=34/38(0.895) mean_n_clusters=1.10
   ```

## 절대 금지 (커스텀 metric — 사용자 명시 거부)

다음 출력 / 권유 / 보고 금지:
- `weighted_isolation`, `mean_isolation`, `pure_rate`, `mixed_rate`, `isolation`, `contamination_rate`
- `binary_*` (binary_ari / binary_nmi / binary_homogeneity / binary_completeness 등)
- `total_split_rate`, `total_noise_rate` (HDBSCAN 표준 noise_pct 만)
- precision / recall / F1 / FPR / accuracy / TP/FP/FN/TN — 분류기 style 일체

## 절대 금지 — Tier 3 metric (디버그만, 발표 X)

NMI / V-measure / Fowlkes-Mallows / Davies-Bouldin / Calinski-Harabasz —
Tier 1 발표 표 사용 금지. 사유:
- NMI: AMI 가 chance-corrected 우월
- V-measure: Hom + Com 합산 (둘 따로 보면 중복)
- FMI: ARI 와 거의 중복
- Davies-Bouldin / Calinski-Harabasz: Euclidean 가정, cosine contrastive embedding 부적합

## 기타 금지

- train embedding 으로 평가 X
- noise 샘플 silhouette 포함 X
- val class 매핑을 train 과 다르게 X
- composite 공식 / centroid 계산 공식 수정 X
- 기존 `with_normal` / `without_normal` / `normal_metrics` / `per_class_noise` / `hdbscan_cfg` 키 변경·삭제 X (추가만 OK)

## 반환

- eval_summary.json 경로
- Tier 1 4 metric (Completeness / AMI / noise_pct / class_capture_rate)
- class_fragmentation_summary 의 weighted_cluster_coverage + frac_single_cluster
- 콘솔 1-2 줄 출력
