# contrastive_init.py vs Current Pipeline

## 오류 원인

초기 스크립트의 실패 지점:

```python
sd = torch.load(CFG["LOCAL_BACKBONE_WEIGHTS"], map_location="cpu")
```

PyTorch 2.6부터 `torch.load()`의 기본값이 `weights_only=True`로 바뀌었다. 기존 checkpoint 안에 `numpy._core.multiarray._reconstruct` 같은 pickle 객체가 있으면 안전 allowlist에 없어서 `UnpicklingError`가 난다.

해결:

```python
torch.load(path, map_location="cpu", weights_only=False)
```

단, 이건 checkpoint를 신뢰할 때만 써야 한다. 지금 케이스는 로컬 backbone weight라서 이 방식이 맞다. `scripts/contrastive_init.py`에는 이 fix를 넣어두었다.

## 학습 차이

| 항목 | `scripts/contrastive_init.py` | 현재 `train_pipeline_ddp.py` / `train_contrastive_ddp.py` |
|---|---|---|
| 실행 구조 | 단일 스크립트, single GPU 기준 | pipeline + DDP, stage1 CNN 생략/사용 가능 |
| backbone | `LOCAL_BACKBONE_WEIGHTS` 직접 로드 | `--backbone` 또는 `--cnn-run-dir`의 CNN best 사용 |
| 입력 크기 | `IMAGE_SIZE=384` | contrastive/grouping `512` |
| epoch | `20` | contrastive 기본 `20`, sweep 일부 `30` |
| batch | global `256` | per-GPU `64` 기본 |
| sampling | 매 epoch `TRAIN_SAMPLING_RATIO=0.25` | 동일하게 매 epoch `0.25` 서브샘플 |
| global loss | InfoNCE + queue | InfoNCE + MoCo-style queue |
| queue size | `16384` | `4096` |
| ignore negative | `IGNORE_NEG_SIM=0.72` | `0.95` |
| temp | `0.07` | `0.05` |
| local loss | Local multi-positive InfoNCE ON, `LOCAL_WEIGHT=0.5` | `USE_LOCAL=False`, 대신 NeCo ON |
| NeCo | 없음 | `NECO_WEIGHT=0.2`, patch-neighbor consistency |
| label smoothing | 없음 | InfoNCE label smoothing `0.02` |
| DDP 안정화 | 없음 | concat forward, all-gather queue sync, DDP cleanup 처리 |
| checkpoint | `final_infer.pt`, `last_training.pt` | `contrastive/best_model.pt`, config/history/embeddings 저장 |

핵심 차이:

- 초기 코드는 local patch matching을 직접 loss로 쓴다.
- 현재 코드는 local InfoNCE 대신 NeCo를 쓴다. NeCo는 같은 이미지의 두 view에서 patch 간 이웃 구조가 유지되도록 하는 방식이다.
- 현재 코드는 hard negative를 더 강하게 남긴다. `IGNORE_NEG_SIM=0.95`라서 similarity 0.95 이하 negative는 loss에 들어간다.
- 현재 코드는 temp가 낮다. `0.05`라서 비슷한 embedding 차이를 더 민감하게 본다.

## 클러스터링 차이

| 항목 | `scripts/contrastive_init.py` | 현재 production grouping |
|---|---|---|
| 대상 | `UNKNOWN_DIR` 전수 | `--image-roots` 하위 이미지 재귀 수집 |
| HDBSCAN metric | `euclidean` | `euclidean` |
| min_cluster_size | `12` | 기본 `12`, sweep은 `12/20` |
| min_samples | `4` | 기본 `15`, sweep은 `10/15` |
| method | `leaf` | 기본 `leaf` |
| epsilon | `0.06` | 기본 `0.06`, sweep은 `0.03/0.06/0.10` |
| allow_single_cluster | `False` 명시 | HDBSCAN 기본값 사용, 사실상 False |
| 저장 방식 | cluster 폴더에 전체 이미지 hardlink/copy | `clusters.parquet/csv`, `summary.json`, representatives |
| 대표 이미지 | medoid 1장 | centroid 근접 representative 30장 기본 |
| composite map | 없음 | `representatives/composite/*linear2_weighted_average.png` |
| corrupt image | truncated 허용 정도 | corrupt/unidentified image skip 기록 |
| overlay | overlay 있으면 overlay 저장 | overlay 기능 없음 |

초기 HDBSCAN은 `min_samples=4`라 훨씬 느슨하다. 현재는 `min_samples=15`라 밀도 기준이 엄격하다. 다만 `leaf + epsilon 0.06`은 세분화 후 가까운 leaf cluster를 일부 다시 붙이는 쪽이라, 실제 결과는 embedding 품질에 크게 의존한다.

## 현재 문제가 "비슷한 결함을 못 나눔"일 때 해석

HDBSCAN이 아니라 embedding 단계에서 이미 붙어 있으면 clustering 파라미터로는 한계가 있다. 초기 스크립트가 현재보다 나을 수 있는 부분은 `USE_LOCAL=True` local InfoNCE다. wafer의 국소 산포 차이를 직접 patch loss로 보존한다.

반대로 현재 코드가 초기보다 나은 부분:

- DDP로 대량 학습 가능
- CNN best backbone을 명확히 사용
- production 폴더 다중 선택/재귀 수집
- representative 30장과 composite map 생성
- 하드 negative를 더 강하게 사용
- sweep으로 조건 비교 가능

## 비교 실험 제안

현재 sweep 10개를 돌린 뒤에도 비슷한 결함이 계속 섞이면, 다음 비교는 둘 중 하나다.

1. 현재 코드에 local InfoNCE를 다시 넣은 조건 추가
2. `scripts/contrastive_init.py`를 같은 train/eval 폴더로 돌려서 local InfoNCE가 실제로 cluster 분리를 개선하는지 비교

초기 스크립트 실행 시 가장 먼저 볼 로그:

- `G`: global InfoNCE
- `Q`: queue loss
- `L`: local patch loss
- `clusters_summary.txt`
- `cluster_summary/` medoid 이미지

