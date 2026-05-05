---
name: cluster-analyzer
description: contrastive run 의 cluster group 통계 진단 — size 분포 / intra·inter cosine / silhouette / purity / leakage / HDBSCAN hparam 추천. read-only inline python.
---

# cluster-analyzer skill

`cluster-analyzer` agent 의 작업 표준.

## 분석 항목 정의

| 항목 | 정의 | 출력 |
|---|---|---|
| size 분포 | cluster 별 멤버 수 | n / mean / median / p10 / p90 / max / min |
| intra cosine | centroid (mean of members) ↔ 각 멤버 의 cosine distance | mean / std / p95 |
| inter cosine | cluster centroid ↔ 가장 가까운 다른 cluster centroid 의 cosine distance | top-3 nearest |
| silhouette per cluster | `sklearn.metrics.silhouette_samples(X, labels, metric="cosine")` 의 cluster 평균 | 1 float |
| purity | `cluster_report.parquet` 의 `purity` 컬럼 | 1 float |
| normal leakage | `normal_metrics.normal_noise_pct` + normal sample 이 defect cluster 에 들어간 count | 2 값 |
| hdbscan rec | `hdbscan_sweep_small.json` 에서 ARI 최대 row | (min_cluster_size, min_samples, ε) |

## weak cluster 임계값

다음 중 **하나라도** 해당 시 weak:

| 기준 | 값 | reason 라벨 |
|---|---|---|
| silhouette | `< 0.3` | low_silhouette |
| intra_p95 | `> 2 × intra_median` | wide_spread |
| purity | `< 0.6` | low_purity |
| inter_min | `< intra_p95` | boundary_blur |

multi-reason 인 경우 list 로 모두 기록.

## 출력 markdown 섹션 헤더 (고정)

```
## 1. 전체 요약
## 2. 클러스터 표 (size 내림차순 top-N + weak 전체)
## 3. weak cluster 리스트 + 사유
## 4. 추천 다음 step
```

## 사용 column 매핑

`cluster_report.parquet` 컬럼 (실측 기준):
- `cluster_id` (int) — HDBSCAN cluster label
- `dom_class` (str) — 가장 많이 차지하는 GT class name
- `purity` (float) — dom_class 비율
- `size` (int) — 멤버 수

`embeddings.npy` shape: `(N, D)` where D=128 or 256 (contrastive head dim)
`files.txt`: N 행, 각 행은 image basename
`classes.txt`: N 행, 각 행은 GT class label

## inline python 패턴 (Bash 호출)

```bash
python -c "
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_samples

run = Path(r'<run_dir>')
emb = np.load(run/'eval/embeddings/embedding.npy')
files = (run/'eval/embeddings/files.txt').read_text().splitlines()
classes = (run/'eval/embeddings/classes.txt').read_text().splitlines()
es = json.loads((run/'eval/eval_summary.json').read_text())
cr = pd.read_parquet(run/'eval/cluster_report.parquet')
# ... 통계 계산 ...
# write analyze_clusters.json
"
```

## 절대 금지

- 새 `.py` 파일 생성 (모두 inline python via Bash `python -c '...'`)
- `outputs/logs_contrastive/<run>/` 기존 파일 수정·삭제
- 학습 재시작 / hparam 자동 변경 — 권고만 (`hdbscan_recommendation` field)
- 외부 dependency 추가 — numpy / pandas / sklearn / scipy 만 사용
