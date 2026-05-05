---
name: image-analyzer
description: cluster 멤버 이미지 outlier 검출 — centroid distance(주) + pixel statistics z-score(보조). PIL + numpy + scipy.ndimage 만 사용. read-only.
---

# image-analyzer skill

`image-analyzer` agent 의 작업 표준.

## centroid distance 계산

```python
# cluster centroid
c = emb[labels == c_id].mean(axis=0)
# member cosine distance
def cosine_dist(x, y):
    return 1 - (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
dists = np.array([cosine_dist(e, c) for e in emb[labels == c_id]])
# z-score within cluster
z = (dists - dists.mean()) / (dists.std() + 1e-12)
# outlier criteria
out_mask = (dists > np.percentile(dists, 95)) | (dists > 2 * np.median(dists))
```

## pixel statistics 6 종

PIL → grayscale uint8 numpy array `arr`:

| 통계 | 식 |
|---|---|
| `mean` | `arr.mean()` |
| `std` | `arr.std()` |
| `fg_ratio` | `(arr > otsu(arr)).mean()` |
| `edge_density` | `np.hypot(sobel_x(arr), sobel_y(arr)).mean()` |
| `centroid_x` | fg mask 의 x mean |
| `centroid_y` | fg mask 의 y mean |

Otsu fallback (skimage 없으면):
```python
def otsu(a):
    hist, _ = np.histogram(a, bins=256, range=(0, 256))
    total = a.size
    sum_total = (np.arange(256) * hist).sum()
    best, best_t = 0, 128
    sum_b, w_b = 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best:
            best, best_t = var_between, t
    return best_t
```

scipy.ndimage 가 일반적으로 깔려있음 — `scipy.ndimage.sobel` 사용. 없으면 numpy `np.gradient` 로 대체.

## z-score threshold 표

| 통계 | secondary 기준 |
|---|---|
| `|mean_z|` | > 2.5 |
| `|std_z|` | > 2.5 |
| `|fg_ratio_z|` | > 2.5 |
| `|edge_z|` | > 2.5 |
| `|centroid_x_z|` | > 3.0 |
| `|centroid_y_z|` | > 3.0 |

centroid 위치는 noise 가 많아 임계 더 보수적.

## 의존성

- numpy (필수)
- PIL / Pillow (필수)
- scipy (sobel 필요 시 — 없으면 np.gradient fallback)
- pandas (cluster_report.parquet 읽기)

새 dependency (libvips / numba 등) 절대 추가 금지.

## 출력 markdown 섹션 (고정)

```
## 1. 요약
## 2. Primary outlier (centroid distance)
## 3. Secondary outlier (pixel stats)
## 4. Medoid dispersion top-10
## 5. 시각 리뷰 권고
```

## 절대 금지

- 새 `.py` 파일 생성
- 이미지 PNG 수정·생성·rename
- `cluster_summary/` 가 비어있을 때 자체 composite 생성 (composite-map 책임)
- weak cluster 가 아닌 전 cluster pixel-stat 분석 (비용 큼) — `--clusters` 명시 안 했을 때 weak 만
