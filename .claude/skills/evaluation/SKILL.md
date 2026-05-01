---
name: evaluation
description: val 이미지 + 학습된 모델 → ARI/NMI/purity/silhouette 계산 → eval_summary.json. silhouette는 cosine, val embedding 기반.
---

# evaluation — Cluster 품질 지표 계산

## 목적

stage 2가 만든 `data/wm811k_val/`과 stage 3이 만든 모델·centroid를 사용해
clustering 품질을 정량 평가한다.

## 입력

- `data/wm811k_val/<class>/*.png` (ground truth class 라벨 = 폴더명)
- `outputs_<preset>_<ts>/checkpoints/final_infer.pt`
- `outputs_<preset>_<ts>/centroids/centroids.npy`
- `outputs_<preset>_<ts>/centroids/clusterer.pkl`

## 산출물

KEEP cascade: `centroids.npy`의 post-filter `cluster_ids`는 eval path에 cascade되지 **않는다**. `hdbscan.approximate_predict`가 성공하면 pre-filter `clusterer.pkl`로 soft-predict하므로 KEEP 임계(`contrastive.py`의 `KEEP_PERSIST_*`)가 걸러낸 cluster도 val/test label로 다시 나타날 수 있다. 설계 의도: KEEP은 `predict.py` / `cluster_composite.py` 전용. eval meta 필드 `filter_cascaded`는 현재 항상 `false`이고 `approximate_predict_used`, `kept_cluster_count`로 상태를 기록한다.

`outputs_<preset>_<ts>/eval_summary.json`:

```json
{
  "n_val": 2000,
  "n_clusters": 12,
  "n_noise": 34,
  "ari": 0.73,
  "nmi": 0.81,
  "cluster_purity": 0.88,
  "silhouette": 0.42,
  "per_class_assignment": {
    "Center": {"cluster_3": 95, "cluster_-1": 5},
    "Donut": {"cluster_7": 100},
    ...
  },
  "cluster_assignment_source": "approximate_predict",
  "approximate_predict_used": true,
  "filter_cascaded": false,
  "kept_cluster_count": 8
}
```

(선택) `outputs_<preset>_<ts>/plots/`:
- `confusion_class_vs_cluster.png`
- `silhouette_distribution.png`

## 수행 절차

1. **val 데이터 로드**:
   - `data/wm811k_val/<class>/*.png` 전체 목록.
   - class → int 라벨 매핑은 `train` set과 동일하게 재사용 (알파벳 순).
2. **Embedding 계산**:
   - `final_infer.pt` 로드 → backbone + projector (stage 3과 동일 방식).
   - `tfm(train=False)` 적용 후 forward → L2-normalized embedding `(N, D)`.
3. **Cluster 할당**:
   - `hdbscan.approximate_predict(clusterer, emb)` 또는 centroid 거리 기반.
   - noise(-1) 샘플은 별도 카운트.
4. **지표 계산**:
   - `sklearn.metrics.adjusted_rand_score` → ARI.
   - `sklearn.metrics.normalized_mutual_info_score` → NMI.
   - cluster_purity: 각 cluster의 majority class 비율 weighted mean.
   - `sklearn.metrics.silhouette_score(emb, labels, metric='cosine')`:
     - noise 제외.
     - N > 5000이면 `sample_size=5000` subsample.
5. `eval_summary.json` 저장.

## 규칙 (금기)

- **noise(-1) 샘플을 silhouette 계산에 포함 금지** (label -1은 정의되지 않은 cluster).
- **val class 라벨 순서 변경 금지**. train과 동일 매핑 유지.
- **train embedding으로 평가 금지**. val 전용.
- **fail-map/mapviewer의 composite 공식을 수정하지 말 것** — 이 stage는 metric
  계산만 담당.

## 코드 위치

구현 후보:
- `experiments/eval_metrics.py`에 `compute_silhouette(emb, labels)` 추가.
- 신규 `experiments/eval_val.py`를 만들어 val 전용 entry point 제공.

## 검증 기준

- `eval_summary.json` 필수 키: `ari`, `nmi`, `cluster_purity`, `silhouette`,
  `per_class_assignment`, `n_val`, `n_clusters`, `n_noise`.
- ARI/NMI ∈ [-1, 1] (ARI는 음수 가능), purity ∈ [0, 1], silhouette ∈ [-1, 1].

## 다음 stage

composite-map agent가 `outputs_*/clusters/`를 소비해 시각화. evaluation은
평가값만 기록.
