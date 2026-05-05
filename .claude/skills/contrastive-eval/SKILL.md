---
name: contrastive-eval
description: contrastive 학습 / HDBSCAN clustering 결과의 공식 metric 산출 + 보고. Tier 1+2 만 사용 (커스텀 metric 절대 금지). 우선순위 P1=class_capture_rate, P2=noise_pct, P3=Completeness, P4=Homogeneity. 보조 AMI / Silhouette / ARI. 자세한 정책: docs/contrastive-eval/.
---

# contrastive-eval skill

contrastive 학습 / HDBSCAN clustering eval 적용 표준.

## 산출 metric (필수)

### Tier 1 — 발표 / 논문 표 1행 (4 + class_fragmentation_summary)

| metric | 출처 | 우선순위 | 의미 |
|---|---|---|---|
| `class_capture_rate` | 자체 (class fragmentation aggregate) | **P1** | 모든 defect class 가 ≥1 group 에 잡힘 |
| `noise_pct (defect only)` | HDBSCAN 표준 | **P2** | defect 격리 실패 비율 |
| Completeness | Rosenberg & Hirschberg 2007 | **P3** | 같은 class 가 같은 cluster 에 |
| AMI | Vinh et al. 2010 | 보조 | chance-corrected, over-cluster 안전 |

`class_fragmentation_summary` (eval_summary.json 새 key, 7 필드):
```json
{
  "n_defect_classes": 38,
  "n_classes_captured": 38,
  "class_capture_rate": 1.0000,
  "weighted_cluster_coverage": 0.9929,
  "mean_n_clusters_per_class": 1.105,
  "n_classes_single_cluster": 34,
  "frac_single_cluster": 0.8947
}
```

`class_fragmentation.parquet` 컬럼 (`n_clusters`, `cluster_coverage`, `n_noise`) 의 aggregate. **새 계산 X — 기존 컬럼 활용만**.

### Tier 2 — 보조 (depth)

- Homogeneity (P4 보조, Rosenberg 2007)
- Silhouette (cosine, Rousseeuw 1987) — intrinsic
- ARI (Hubert 1985) — cross-paper 비교용. **over-cluster 페널티 inherent 인 점 명시**.

## 절대 금지 (커스텀 metric)

다음 metric 출력 / 권유 / 보고 금지 — 사용자 명시 거부:
- `weighted_isolation`, `mean_isolation`
- `pure_rate`, `mixed_rate`
- `isolation`, `contamination_rate`
- `binary_*` (binary_ari / binary_nmi / binary_homogeneity / 등)
- `total_split_rate`, `total_noise_rate` (HDBSCAN 표준 noise_pct 만)
- precision / recall / F1 / FPR / TP/FP/FN/TN — 분류기 style 일체

## 콘솔 보고 형식 (고정)

```
[Tier 1] Completeness=0.947 AMI=0.929 noise_def=0.71% capture=38/38(1.000)
[Class frag] coverage=0.993 single_cluster=34/38(0.895) mean_n_clusters=1.10
```

다른 형식 / CSV / table 출력 금지. 위 1-2 줄.

## eval_summary.json 통합

기존 키 (`with_normal`, `without_normal`, `normal_metrics`, `per_class_noise`, `hdbscan_cfg`) 변경 / 삭제 절대 금지. **추가만 OK**:

새 키:
- `class_fragmentation_summary` (7 필드, 위 spec)

## 학습 도중 monitoring (선택)

매 epoch 끝 추가 출력 (`docs/contrastive-eval/MONITORING.md` 참조):
- `align` — alignment loss (Wang & Isola 2020), label 무관
- `unif` — uniformity loss, label 무관
- `knn` — k-NN top-1 on labeled subset, label 있을 때 옵션

## 코드 적용 위치

| 파일 | 변경 |
|---|---|
| `_eval_contrastive_unknown_n50.py` | (1) `class_fragmentation_summary()` 새 함수, (2) summary dict 에 `class_fragmentation_summary` key 추가, (3) 콘솔 보고 1-2 줄 |
| `contrastive.py` | (옵션) epoch loop 에 alignment + uniformity 출력 5 줄. (옵션 D-6) Hard negative mining (`docs/contrastive-eval/HARD_NEGATIVE.md`) |

기존 `metrics_on_set()` (line 413) / `class_fragmentation_records()` (line 302) / `normal_leakage_metrics()` (line 387) 함수 무수정.

## Verification

기존 `outputs/logs_contrastive/overall/` 에 적용 시 예상값:
- Completeness ≈ 0.947
- AMI ≈ 0.929
- noise_pct (defect) ≈ 0.71%
- class_capture_rate = 1.000
- frac_single_cluster ≈ 0.895

## 참고

- 정책 docs: `docs/contrastive-eval/` (5 파일)
- agent enforce: `.claude/agents/evaluation.md`
- 결정 history: `docs/contrastive-eval/DECISIONS.md`
