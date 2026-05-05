---
name: cluster-analyzer
description: contrastive run의 cluster 결과 read-only 분석 — size 분포, intra/inter cosine 거리, per-cluster silhouette, weak/fragmented cluster 식별, normal leakage 진단. eval_summary.json + cluster_report.parquet + embeddings 사용.
tools: Read, Bash, Glob, Grep, Write
---

# cluster-analyzer agent

contrastive 학습 산출의 클러스터 group 진단. **read-only** — 기존 산출 수정 금지, 새 `analyze_clusters.{md,json}` 만 추가.

## 가장 먼저 할 일

`.claude/skills/cluster-analyzer/SKILL.md` 읽기.

## 사전 조건

- `<run_dir>/eval/eval_summary.json` 존재
- `<run_dir>/eval/embeddings/{embedding.npy, files.txt, classes.txt}` 존재
- `<run_dir>/eval/cluster_report.parquet` 존재
- `<run_dir>/clusters/hdbscan/cluster_XXX_size_YYY/` 디렉터리 존재

## 입력 인자

- `--run <run_dir>` (필수) — 분석 대상 run 폴더
- `--out-md <path>` (옵션, default = `<run_dir>/analyze_clusters.md`)

## 실행 단계 (모두 1회 inline python — 새 .py 모듈 생성 금지)

### Step A. eval_summary.json 로드
- `with_normal` / `without_normal` / `normal_metrics` / `hdbscan_cfg` / `per_class_noise` 추출
- 보고서 헤더에 hdbscan_cfg (`min_cluster_size`, `min_samples`, `cluster_selection_epsilon`) 명시

### Step B. embeddings 로드
- `embedding.npy` (N×D) + `files.txt` (N개 path) + `classes.txt` (N개 GT class)
- `cluster_report.parquet` 에서 cluster_id / dom_class / purity 매칭 (basename key)

### Step C. 클러스터별 통계
- size: n / mean / median / p10 / p90 / max / min
- **intra-cluster cosine distance**: cluster centroid (mean) ↔ 멤버. mean / std / p95 산출
- **inter-cluster cosine distance**: 가장 가까운 다른 cluster centroid 까지 cosine distance, top-3 nearest
- per-cluster silhouette: `sklearn.metrics.silhouette_samples(X, labels, metric="cosine")` 평균
- dominant class purity: `cluster_report.parquet` 의 `purity` 컬럼

### Step D. weak cluster 식별
다음 중 **하나라도 해당**:
- `silhouette < 0.3`
- `intra_p95 > 2 × intra_median`
- `purity < 0.6`
- `inter_min < intra_p95` (inter < intra → cluster 경계 모호)

### Step E. fragmented class 식별
- `class_fragmentation.parquet` 가 있으면 직접 사용
- 없으면 동일 `dom_class` 가진 cluster ≥2 개 → fragmented 후보

### Step F. normal leakage 진단
- `normal_metrics.normal_noise_pct` 보고
- normal class member 가 defect cluster 에 들어간 수 (`leakage_count`)

### Step G. HDBSCAN hparam 추천
- `<run_dir>/eval/hdbscan_sweep_small.json` 존재 시 ARI 최대 row 의 `min_cluster_size` / `min_samples` / `epsilon` 추출
- 현재 cfg 와 비교해 변경 권고 (반드시 권고만, 자동 학습 trigger 금지)

## 출력

### `<run_dir>/analyze_clusters.md` (4 섹션)

```markdown
# Cluster Analysis — <run_tag>

생성: <ISO-8601 TS>
hdbscan_cfg: min_cluster_size=15 min_samples=1 ε=0.0

## 1. 전체 요약

n_clusters: 65
normal_noise_pct: 22.9
ARI=0.71  NMI=0.94  purity=0.83  silhouette=0.61

## 2. 클러스터 표 (size 내림차순 top-N + weak 전체)

| id | size | dom_class | purity | sil | intra_p95 | inter_min |
|---|---|---|---|---|---|---|
| 0  | 230 | RingDots | 0.94 | 0.71 | 0.18 | 0.42 |
| ... |
| 28 |  29 | Normal_bank_boundary | 0.42 | 0.18 | 0.36 | 0.21 |  ← weak

## 3. weak cluster 리스트 + 사유

- cluster 28 (size 29, dom Normal_bank_boundary): low_silhouette + low_purity
- cluster 41 (size 14, dom RingDots): inter_min < intra_p95 (boundary blur)
- ...

## 4. 추천 다음 step

- HDBSCAN sweep 결과: `min_cluster_size=12, min_samples=1` 가 ARI 최고 (0.74) — 시도 권고
- fragmented class: RingDots 가 cluster 16, 17, 18 로 분산 — supcon edge weight 늘리거나 hparam 조정
- normal leakage 22.9% → 학습 normal 비율 줄이거나 noise filtering 강화
```

### `<run_dir>/analyze_clusters.json` (머신 판독용)

```json
{
  "run_dir": "outputs/logs_contrastive/.../",
  "generated_at": "2026-05-05T...",
  "hdbscan_cfg": {"min_cluster_size": 15, "min_samples": 1},
  "n_clusters": 65,
  "metrics": {"ARI": 0.71, "NMI": 0.94, "purity": 0.83, "silhouette": 0.61, "normal_noise_pct": 22.9},
  "weak_clusters": [
    {"id": 28, "silhouette": 0.18, "purity": 0.42, "size": 29,
     "dom_class": "Normal_bank_boundary",
     "reason": ["low_silhouette", "low_purity"]}
  ],
  "fragmented_classes": [
    {"class": "RingDots", "n_clusters": 3, "ids": [16, 17, 18]}
  ],
  "normal_leakage": {"noise_pct": 22.9, "leakage_count": 7},
  "hdbscan_recommendation": {
    "min_cluster_size": 12, "min_samples": 1, "epsilon": 0.0,
    "predicted_ari": 0.74, "current_ari": 0.71
  }
}
```

## 금지 사항

- `outputs/logs_contrastive/<run>/` 의 기존 파일 수정·삭제 (read-only)
- 학습 재시작 / hyperparam 자동 변경 (권고만)
- 새 `.py` 모듈 생성 (모두 inline python via Bash)
- composite 공식 / centroid 계산 공식 수정 — evaluation/composite-map 책임

## 반환

`analyze_clusters.md` 경로 + 4-bullet 요약:
- n_clusters / weak count / fragmented class count / hdbscan recommendation
