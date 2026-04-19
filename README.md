# unknown-contrastive

반도체 wafer 결함맵을 self-supervised contrastive learning으로 embedding하고 HDBSCAN으로 unknown 결함 유형을 자동 클러스터링하는 파이프라인. 학습된 모델로 새 제품을 cluster 폴더에 자동 분류하고, 각 cluster의 대표 패턴을 composite map(여러 wafer를 겹친 heatmap)으로 시각화.

연관 프로젝트:
- [`fail-map`](../fail-map): S3 raw wafer bin → palette-indexed PNG 생성
- [`mapviewer`](../mapviewer): 생성된 PNG을 UI에서 조회/composite 합성. 이 repo의 `common/composite.py`는 mapviewer의 `api/composite_map.py` 공식을 pure numpy로 포팅한 것.

## 핵심 파이프라인

```
                  ┌─────────────────────────────────────────────┐
                  │  WM-811K (LSWMD.pkl)                        │
                  │  data_prep/download_wm811k.py               │
                  │  data_prep/wm811k_to_palette.py             │
                  └───────────────────┬─────────────────────────┘
                                      ▼
                          data/wm811k/<class>/*.png  (palette PNG)
                                      │
              ┌───────────────────────┴────────────────────────┐
              ▼                                                ▼
    ┌──────────────────────┐                       ┌──────────────────────┐
    │ contrastive.py       │                       │ experiments/         │
    │   ConvNeXtV2 +       │                       │   15 variant preset  │
    │   InfoNCE + Queue +  │◀─ preset override ───▶│   run_experiment.py  │
    │   Local InfoNCE      │                       │   compare.py         │
    │   → HDBSCAN          │                       │   (ARI/NMI/purity)   │
    └──────────┬───────────┘                       └──────────────────────┘
               │ 학습 후 저장
               ▼
    run_dir/
    ├── checkpoints/final_infer.pt
    ├── centroids/{centroids.npy, centroids_meta.json, clusterer.pkl}
    ├── clusters/hdbscan/cluster_XXX_size_YYY/
    ├── cluster_summary/
    │   ├── <medoid 1장>              ← 기존
    │   └── composite/cluster_XXX_composite.png   ← 신규 (top-10 medoid 합성)
    └── eval_summary.json            ← experiments runner만 생성

               │
               ▼
    ┌──────────────────────┐
    │ predict.py           │   새 이미지 → 가장 가까운 cluster 폴더
    │   (cosine or HDBSCAN │   → <out>/cluster_XXX/, <out>/unknown/
    │    approximate)      │   + prediction_report.json
    └──────────────────────┘
```

## 파일/모듈 가이드

| 경로 | 역할 |
|---|---|
| `contrastive.py` | 메인 학습/클러스터링 파이프라인. `CFG`가 하드코딩된 경로(Linux 서버) 유지. |
| `cnn_yolo.py` | 기존 2-stage 결함 분류기 (ConvNeXtV2 + GradCAM + YOLO). **이 repo의 main 기능 아님**, 레거시로 보존. |
| `predict.py` | 학습된 체크포인트 + centroid 로드 → 새 이미지 분류 CLI. |
| `cluster_composite.py` | HDBSCAN 결과에서 각 cluster top-10 medoid를 골라 composite PNG 생성. `contrastive.py` 학습 종료 시 자동 호출되며 별도 CLI로도 실행 가능. |
| `common/palette_io.py` | palette PNG 로드, 14+→0 정규화, 8-13-only 특수 규칙 (mapviewer와 동일). |
| `common/composite.py` | `compute_grade_counts`, `compute_square_mean`, `render_composite_png` — pure numpy/PIL 구현. |
| `common/centroids.py` | cluster centroid + percentile threshold 저장/로드/할당. |
| `experiments/presets.py` | 15개 성능 variant preset (`baseline`, `idx_gray_x8`, `unfreeze_s3`, `hard_negatives`, …). |
| `experiments/eval_metrics.py` | ARI / NMI / cluster purity 계산, subset 선정 (largest/random/explicit). |
| `experiments/run_experiment.py` | preset 이름 받아 `CFG` 런타임 패치 후 `contrastive.main()` 호출 → `eval_summary.json` 저장. |
| `experiments/compare.py` | 여러 run의 `eval_summary.json`을 CSV/TSV로 집계. |
| `data_prep/download_wm811k.py` | Kaggle CLI → HF mirror → 수동안내 3단 폴백. |
| `data_prep/wm811k_to_palette.py` | LSWMD.pkl → palette PNG. 매핑: `{0→31 transparent, 1→0 normal, 2→7 severe}`. |
| `tests/` | pytest/unittest 호환. 현재 **26 passed**. |

## 15 Experiment Presets

| preset | 변경점 | 기대효과 |
|---|---|---|
| `baseline` | 변경 없음 | 기준선 |
| `idx_gray_x8` | 입력을 palette index × 8 (grayscale 3채널 복제) | 색 정보 배제 시 cluster 분리 어떻게 되는가 |
| `idx_1ch` | **TODO**: conv1 surgery + stage3 unfreeze. 현재 `NotImplementedError`. | 1채널 네이티브 입력 |
| `unfreeze_s3` | `stages.3` unfreeze, `LR_HEAD=5e-4` | backbone 일부 적응 — 가장 큰 지렛대 |
| `unfreeze_s3_gray` | `idx_gray_x8` + `stages.3` unfreeze | grayscale에서 backbone 적응 |
| `hard_negatives` | queue에서 top-512 hardest만 사용 | negative의 sharpness ↑ |
| `strong_augment` | RandomErasing, GaussianBlur, degrees=15 | augment invariance ↑ |
| `multicrop` | **TODO**: SwAV 2 global + 4 local. 현재 `NotImplementedError`. | 다중 스케일 view |
| `deeper_head` | 3-layer projection MLP | head capacity ↑ |
| `queue_64k` | `QUEUE_SIZE=65536` | negative bank 4배 |
| `local_global_search` | `LOCAL_SEARCH="global"` | local loss가 전체 feature map에서 positive 탐색 |
| `no_local_loss` | `USE_LOCAL=False` | local loss 기여도 ablation |
| `epochs_40` | `EPOCHS=40, WARMUP_EPOCHS=4` | 수렴 더 보기 |
| `temp_0_05` | `TEMP=0.05` | sharp softmax |
| `temp_0_15` | `TEMP=0.15` | smooth softmax |

## 서버에서 실행

### 1. WM-811K 준비 (최초 1회)

```bash
# Kaggle API 토큰 설정 (https://www.kaggle.com/settings → Create New Token)
# ~/.kaggle/kaggle.json 배치 후
python data_prep/download_wm811k.py
python data_prep/wm811k_to_palette.py --n-per-class 222
```

출력: `data/wm811k/{Center,Donut,Edge-Loc,Edge-Ring,Loc,Near-full,Random,Scratch,none}/*.png` 약 2000장.

### 2. 단일 학습 (baseline)

```bash
python contrastive.py
```

`CFG`의 `TRAIN_DIR`, `UNKNOWN_DIR`, `OUTPUT_DIR`를 직접 수정해서 돌림. 결과는 `OUTPUT_DIR_{YYMMDD_HHMMSS}/`.

### 3. Experiment sweep

```bash
for p in baseline unfreeze_s3 hard_negatives idx_gray_x8 unfreeze_s3_gray \
         strong_augment deeper_head queue_64k local_global_search no_local_loss \
         epochs_40 temp_0_05 temp_0_15; do
  python -m experiments.run_experiment --preset $p \
    --dataset-root /home/sr5/ho.choi/project/wafer-defect-clustering/data/unknown \
    --train-root   /home/sr5/ho.choi/project/wafer-defect-clustering/data/self \
    --output-base  /home/sr5/ho.choi/project/wafer-defect-clustering/outputs_exp \
    --use-subset
done
python -m experiments.compare /home/sr5/ho.choi/project/wafer-defect-clustering/outputs_exp > comparison.csv
```

`--use-subset` 플래그: `experiments/presets.py`의 `EVAL_SUBSET`에 따라 10 classes × 200 = 2000장 subset에서 빠르게 sweep.

`idx_1ch`, `multicrop`은 서버 런타임에 `NotImplementedError` 발생 — 실제 사용 전에 TODO 해결 필요 (README 하단 참고).

### 4. Prediction

```bash
python predict.py \
    --checkpoint /path/to/run_dir/checkpoints/final_infer.pt \
    --centroids-dir /path/to/run_dir/centroids \
    --images-dir /path/to/new_product_images \
    --out-dir /path/to/classified \
    --threshold-mode p95     # p95(기본) | p90 | p99 | hdbscan
```

출력: `<out>/cluster_000/`, `<out>/cluster_005/`, `<out>/unknown/`, `prediction_report.json`, `summary.txt`.

## 개발 (로컬 Windows)

```bash
# 테스트 실행
python -m pytest tests/ -v
```

현재 **26 테스트 모두 통과**. Windows에서는 `hdbscan` 패키지가 필요 없음 (학습은 서버에서만).

실제 학습은 CUDA GPU + 대용량 wafer 데이터셋이 필요하므로 로컬에서는 합성 smoke test만 수행. 서버 경로(`/home/sr5/ho.choi/...`)는 `CFG`에 그대로 유지돼있으므로 수정 불필요.

## 규칙 / 주의

- **학습 결과 폴더 삭제 금지**: `outputs_*/`, `logs/`, `clusters/`, `checkpoints/`, `centroids/` 등은 사용자 명시 요청 전 절대 삭제/덮어쓰기하지 않음. 새 실험은 항상 새 timestamp 폴더로. 이 규칙은 `CLAUDE.md`에도 박혀있음.
- **서버 경로 변경 금지**: `CFG`의 `TRAIN_DIR`, `UNKNOWN_DIR`, `OVERLAY_DIR`, `OUTPUT_DIR`, `LOCAL_BACKBONE_WEIGHTS`는 Linux 서버 경로. 로컬 Windows에서 편집할 때 이들 값을 바꾸지 말 것.

## 미구현 TODO

| 항목 | 위치 | 비고 |
|---|---|---|
| `idx_1ch` input mode | `contrastive.py`의 `CL.__init__` | timm ConvNeXtV2의 stem conv1 (3→out_c) weight를 1→out_c로 mean 축소. 현재 `NotImplementedError`. |
| `multicrop` preset | `contrastive.py`의 `tfm` / `PairDS` | SwAV식 2 global(384) + 4 local(96) view dataset class. 현재 `NotImplementedError`. |
| `cluster_composite.py` CLI | `cluster_composite.py` | 학습 종료 시 in-memory 호출만 지원. 별도 CLI로 run_dir 받아 재-embedding은 TODO. |
| Pillow 13 대응 | `common/composite.py`, `data_prep/wm811k_to_palette.py` | `Image.fromarray(mode=...)` deprecation. 현재 경고만, 동작 영향 없음. |
