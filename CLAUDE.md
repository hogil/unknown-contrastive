# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 이 repo가 하는 일

반도체 wafer palette-indexed PNG을 self-supervised contrastive learning으로 embedding → HDBSCAN으로 unknown 결함 패턴을 자동 클러스터링 → 학습 모델로 새 wafer를 폴더별 분류 + 각 cluster의 대표 패턴을 composite map(여러 장 겹친 heatmap)으로 시각화.

연관 프로젝트(읽기 전용, 이 repo에서 수정 금지):
- `D:\project\fail-map`: S3 raw → palette PNG + positions JSON 생성
- `D:\project\mapviewer`: palette PNG 조회/composite UI. `api/composite_map.py`의 공식이 `common/composite.py`로 포팅됨.

## 주요 파이프라인 파일

| 파일 | 역할 | 수정 주의점 |
|---|---|---|
| `contrastive.py` | 메인 학습/클러스터링. `CFG` dict가 하드코딩된 Linux 서버 경로 (`/home/sr5/ho.choi/...`) 유지 | 서버 경로를 Windows 경로로 "고치지" 말 것 — 의도적. CFG 내 `__INPUT_MODE__`, `__UNFREEZE_STAGES__`, `__HARD_NEGATIVES__`, `__STRONG_AUGMENT__`, `__MULTICROP__`, `__DEEPER_HEAD__` 6개 마커는 experiments runner가 런타임 주입 |
| `cnn_yolo.py` | **레거시**. 2-stage 결함 분류기(ConvNeXtV2+GradCAM+YOLO). `gradcam_utils` 외부 의존. 이 repo의 메인 기능 아님 | 건드리지 말 것 |
| `predict.py` | best 모델 + centroids 로드 → 새 이미지 분류 CLI | `contrastive.CL`, `tfm`, `strip_prefixes`를 import해서 재사용. `final_infer.pt`만으로는 CFG 복원 안 돼서 **`last_training.pt`의 cfg 키**로 override 후 state_dict 로드 |
| `cluster_composite.py` | HDBSCAN 결과에서 cluster별 top-10 medoid로 composite PNG 생성 | `contrastive.py` main() 끝에서 자동 호출. 별도 CLI 스켈레톤은 있지만 run_dir에 `emb.npy` 선저장 필요 (TODO) |
| `common/palette_io.py` | palette PNG 로드 + mapviewer 정규화(14+→0, 8-13-only→8, all-invalid mask) | mapviewer `create_sum_map`과 같은 규칙 유지. 공식 바꾸면 composite 호환성 깨짐 |
| `common/composite.py` | `compute_grade_counts`, `compute_square_mean = Σ(count_g · g²) / N`, `compute_base_indices`(mean), `render_composite_png`(palette LUT base + 4-stop gradient overlay, RGB PNG) | pure numpy+PIL. numba/vips/turbojpeg 의존 추가 금지 |
| `common/centroids.py` | L2-normalized cluster centroid + per-cluster cosine distance percentile(50/90/95/99) + optional `clusterer.pkl` | threshold mode: p95 기본, hdbscan은 `approximate_predict` 폴백 |
| `experiments/presets.py` | 15 variant preset. 각 preset은 `cfg_overrides` dict + 6개 마커 플래그(`input_mode`, `unfreeze_stages`, `hard_negatives`, `strong_augment`, `multicrop`, `deeper_head`) | `idx_1ch`와 `multicrop`은 런타임 `NotImplementedError` — README TODO 참고 |
| `experiments/run_experiment.py` | preset 이름 받아 `contrastive.CFG` 런타임 패치 → `contrastive.main()` 호출 → `eval_summary.json` 저장. 한 번에 1 preset (contrastive의 `RUN_TS`가 모듈 로딩 시 고정되므로) | sweep은 bash loop |
| `experiments/eval_metrics.py` | ARI/NMI/cluster_purity + subset 선정(largest/random/explicit). subset은 `<output_base>/_subsets/<preset>/`에 symlink 트리로 영속화 | |
| `data_prep/download_wm811k.py` | kagglehub → kaggle CLI → HF mirror 5개 → 수동 안내 3단 폴백 | HF mirror 후보 5개가 현재 전부 404 — Kaggle API 설정 필수 |
| `data_prep/wm811k_to_palette.py` | LSWMD.pkl → `data/wm811k/<class>/*.png`. 매핑 `{0→31 transparent, 1→0 normal, 2→7 severe}` | WM-811K pkl 컬럼명 `failureType`/`faliureType` 오타 둘 다 처리. 라벨이 nested array인 경우도 정규화 |
| `tests/` | pytest + `__main__ unittest.main()` 양쪽 호환. 현재 **26 passed** | `python -m pytest tests/ -v` |

## Run dir 구조 (contrastive.main() 출력)

```
<OUTPUT_DIR>_<RUN_TS>/
├── run.log, run_info.json
├── checkpoints/
│   ├── final_infer.pt          # {"state_dict": model.state_dict()}
│   └── last_training.pt        # {"epoch":..., "model":..., "cfg": CFG, "class_to_idx":...}
├── centroids/                  # ★ 신규 (predict.py가 읽음)
│   ├── centroids.npy           # (K, D) L2-normalized
│   ├── centroids_meta.json     # cluster_ids, sizes, distance percentiles
│   └── clusterer.pkl           # hdbscan fitted (approximate_predict용)
├── clusters/hdbscan/
│   └── cluster_XXX_size_YYY/   # per-cluster 이미지 (hardlink or copy)
├── cluster_summary/
│   ├── <medoid 1장>.png        # 기존 대표
│   └── composite/              # ★ 신규 (top-10 medoid 합성)
│       └── cluster_XXX_composite.png
├── ignored_samples/            # KEEP 임계 미달 클러스터의 sample (top-N prob)
├── clusters_summary.txt, clusters_global_list.txt
└── eval_summary.json           # experiments runner만 생성 (ARI/NMI/purity)
```

## 런타임 확장 포인트 6개 (experiments가 주입, contrastive.py가 읽어서 처리)

| 마커 | 처리 위치 | 동작 |
|---|---|---|
| `__INPUT_MODE__` | `tfm()`, `_load_as_input_mode()`, `PairDS`, `SingleView` | `palette_rgb`(기본, `.convert("RGB")`), `idx_gray_x8`(`common.palette_io.load_palette_indices(p) * 8` → 3채널 복제), `idx_1ch`(TODO: `NotImplementedError` in `CL.__init__`) |
| `__UNFREEZE_STAGES__` | `CL.__init__`, `CL.train()` | 전체 freeze 후 `name.startswith(prefix)`인 param만 `requires_grad=True`. `train()`에서 trainable param 있으면 `backbone.train(mode)` 호출 |
| `__HARD_NEGATIVES__` | `info_nce_with_queue` | queue negative에서 `topk(min(512, N))` 가장 유사한 것만 사용 후 ignore_sim 마스킹 |
| `__STRONG_AUGMENT__` | `tfm(train)` | `T.GaussianBlur(3, (0.1,0.8))`, `T.RandomErasing(p=0.3, scale=(0.02,0.1))`, `degrees=15` 추가 |
| `__MULTICROP__` | `tfm()` | **TODO**: `NotImplementedError("multicrop WIP")`. SwAV식 2 global + 4 local view dataset 필요 |
| `__DEEPER_HEAD__` | `CL.__init__` | `Proj(d→PROJ_DIM)` 대신 `Linear(d,d) BN ReLU Linear(d,d/2) BN ReLU Linear(d/2,PROJ_DIM)` |

## 규칙 (HARD RULES)

- **결과 폴더 삭제 금지** (글로벌 CLAUDE.md 규칙 연장): `outputs_*/`, `logs/`, `clusters/`, `checkpoints/`, `centroids/`, `cluster_summary/` 등 학습/평가 산출물은 사용자 명시 요청 전 삭제·덮어쓰기 금지. 새 실험은 새 timestamp 폴더 또는 새 preset 이름으로.
- **서버 경로 변경 금지**: `CFG`의 Linux 서버 경로는 의도적. Windows 경로로 "고치지" 말 것.
- **`fail-map`, `mapviewer` 수정 금지**: 읽기 전용 참고. `composite_map.py`의 공식 포팅은 `common/composite.py` 한쪽에서만 유지보수.
- **데이터 파일 커밋 금지**: `data_raw/`, `data/`, `weights/`, `*.pkl`, `*.pt`, `*.pth` 등은 `.gitignore`에 있음. 강제 추가 금지.

## 일반적인 수정 작업

- **새 variant preset 추가**: `experiments/presets.py`의 `PRESETS` dict에 엔트리 추가. 기존 마커로 안 되는 구조 변경은 `contrastive.py`의 해당 지점도 같이 수정해야 함.
- **composite map 공식 수정**: `common/composite.py`만 수정, `tests/test_composite.py`로 검증. mapviewer의 `api/composite_map.py`와 일치 유지 권장.
- **prediction threshold 로직 변경**: `common/centroids.py`의 `save_cluster_centroids`에서 `percentiles` 인자 조정. `predict.py`의 `--threshold-mode` 옵션과 동기화.
- **새 데이터셋 추가**: `data_prep/<dataset>_to_palette.py` 패턴으로. 출력은 `data/<dataset>/<class>/*.png`. palette는 항상 `common.palette_io.get_default_palette()` 사용.

## 테스트

```bash
cd D:\project\unknown-contrastive
python -m pytest tests/ -v
```

현재 26 passed. 새 기능 추가 시 반드시 테스트도 같이.
