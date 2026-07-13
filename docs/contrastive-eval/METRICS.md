# METRICS — 공식 metric 정의 + 사용 정책

## 우선순위 lock-in (사용자 합의)

| 순위 | metric | 의미 | 현재 (overall) |
|---|---|---|---|
| **P1** | `class_capture_rate` | defect class 가 cluster의 dominant class로 ≥1회 등장 | run-dependent |
| **P2** | `noise_pct (defect only)` | defect 가 noise (-1) 로 격리 실패 비율 — 낮을수록 좋음 | **0.71%** |
| **P3** | Completeness | 같은 class 가 같은 group 에 모이는 정도 | 0.9466 |
| **P4** | Homogeneity | 한 group 안에 한 class 만 있는 정도 | 0.9154 |
| 보조 | AMI | chance-corrected 일반 quality | 0.9288 |

P1-P2 = production 직결 (recall / false-alarm 느낌). P3-P4 = clustering quality. AMI = cross-paper 비교 보조.

> Historical migration: the May-2026 evaluator counted a class when **any** sample was non-noise
> (`n_clusters >= 1`). That weaker value is now reported only as `legacy_presence_capture`; it is
> not P1 and cannot be compared directly with the dominant-class P1 above.
> The `38/38` JSON/log examples later in this document are historical legacy-presence examples,
> not canonical P1 reference values.

## Tier 1 — 발표 / 논문 표 1행 (4 + class_fragmentation_summary)

| metric | 공식 출처 | 정의 |
|---|---|---|
| **Completeness** | Rosenberg & Hirschberg 2007 | 정보이론 기반 partition-level. "각 GT class 가 단일 cluster 에 모이는 정도". 1.0 = 모든 class 가 정확히 1 cluster |
| **AMI** (Adjusted MI) | Vinh et al. 2010 | NMI 의 chance-corrected. cluster 수 많아도 inflate 안 함 — 우리 over-cluster 환경에 안전 |
| **noise_pct** (defect only) | HDBSCAN 보고 표준 | (cluster_id == -1 ∩ defect) / 전체 defect. % |
| **class_capture_rate** | 자체 정의 | (cluster의 **unique dominant/main** class로 등장한 defect class 수) / 전체 defect class 수. 동률 cluster는 대표 class 없음으로 처리. 1.0 = 모든 class 가 대표 group을 가짐 |

`class_fragmentation_summary` (criteria A/B/C aggregate, 7 키) — `eval/class_fragmentation.parquet` 의 컬럼들 (`n_clusters`, `cluster_coverage`, `n_noise`) 을 한 번 aggregate:

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

## Tier 2 — 보조 (depth + cross-check)

| metric | 출처 | 역할 |
|---|---|---|
| Homogeneity | Rosenberg & Hirschberg 2007 | Completeness 의 짝 (cluster 입장 — group 안에 한 class만) |
| Silhouette (cosine) | Rousseeuw 1987 | intrinsic embedding quality — GT 무관, label 없을 때 학습 monitoring 가능 |
| ARI | Hubert & Arabie 1985 | 다른 논문과 1:1 비교용. **주의: over-cluster 페널티 강해서 우리 setup 에서 inherent 낮음** |

## Tier 3 — skip (이유)

| metric | skip 사유 |
|---|---|
| NMI | AMI 가 chance-corrected 우월 대체. 두 개 같이 보고 X |
| V-measure | Hom + Com 의 harmonic mean. 둘 따로 보면 중복 |
| Fowlkes-Mallows (FMI) | ARI 와 거의 중복. 하나면 충분 |
| Davies-Bouldin | Euclidean 거리 가정. cosine contrastive embedding 부적합 |
| Calinski-Harabasz | Euclidean + over-cluster 선호 — 객관 평가 어려움 |
| **B-Cubed Precision/Recall** | 사용자 drop. Tier 1 에서 제외. |

→ Tier 3 metric 은 **발표 표 1차 사용 금지**. 디버그 부록만.

## 절대 금지 — 커스텀 metric

이전 라운드에서 산출했던 다음 metric 은 **공식 아님 (학술적 신뢰성 X)** — 출력 / 권유 / 보고 금지:

- `weighted_isolation`, `mean_isolation`
- `pure_rate`, `mixed_rate`
- `isolation`, `contamination_rate`
- `binary_homogeneity`, `binary_completeness`, `binary_v_measure`, `binary_ari`, `binary_nmi`, `binary_fmi`, `binary_homogeneity` (binary GT 변형 모두 — 공식 metric 의 binary 변형은 정보 손실, 의미 모호)
- `total_split_rate`, `total_noise_rate` (사용자 노이즈 표기는 `noise_pct` 표준만)

사용자 명시 거부: "전부 cnn 성능지표잖아" + "공식 지표 써야 논문/사내 발표 신뢰성".

## 산출 위치

| 산출 | 위치 | 누가 |
|---|---|---|
| `eval_summary.json` 의 `with_normal` / `without_normal` / `normal_metrics` | 기존 | `_eval_contrastive_unknown_n50.py::main` (변경 X) |
| `eval_summary.json` 의 새 `class_fragmentation_summary` | 신규 추가 | 같은 함수 안 새 helper |
| `eval/class_fragmentation.parquet` | 기존 | `class_fragmentation_records()` (변경 X) |
| 콘솔 1행 보고 | run.log 끝 | print 문 추가 |

## 콘솔 1행 보고 형식 (고정)

```
[Tier 1] Completeness=0.947 AMI=0.929 noise_def=0.71% capture=38/38(1.000)
[Class frag] coverage=0.993  single_cluster=34/38(0.895)  mean_n_clusters=1.10
```

다른 형식 출력 금지. CSV / table / markdown 풍 출력 X — 위 1-2 줄 정도.

## 참고

- 학습 도중 monitoring metric (alignment, uniformity, k-NN) 은 별도. `MONITORING.md` 참조
- Hard negative mining 도입 시 InfoNCE 변경. `HARD_NEGATIVE.md` 참조
- 거부한 옵션 (Multi-crop / SupCon) 의 사유. `DECISIONS.md` 참조
