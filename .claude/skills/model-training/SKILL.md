---
name: model-training
description: contrastive.py/experiments/run_experiment.py를 래핑해 학습을 돌리고 outputs_*/ 폴더를 산출한다. 기존 CFG/preset 구조 존중, 결과 폴더 삭제 금지.
---

# model-training — Contrastive 학습 실행

## 목적

stage 2가 만든 `data/wm811k_train/`을 입력으로 contrastive learning을 수행하고
checkpoint + centroid를 산출한다. **새 모델 코드를 작성하지 않는다** — 기존
`contrastive.py` / `experiments/run_experiment.py`를 호출하는 wrapper.

## 입력

- `data/wm811k_train/<class>/*.png` (stage 2 산출)
- preset 이름 (기본 `baseline`) — `experiments/presets.py`에 정의된 것만 가능

## 백본 초기화 정책 (중요)

**`contrastive.py::CFG["LOCAL_BACKBONE_WEIGHTS"]`는 ImageNet FCMAE pth가 아니라
`cnn_train.py`로 wafer 33-class supervised 학습이 끝난 `log/<run>/best_model.pth`를
가리키도록 설정한다.**

이유 (TAPT, Task-Adaptive Pre-Training / sequential transfer):
- ImageNet FCMAE 가중치는 자연 이미지(컬러 + 자연 텍스처) 학습 — wafer
  fail-bit map(palette grayscale + 32×32 chip grid)과 도메인 갭 큼
- cnn_train.py 결과는 동일 wafer 데이터로 supervised 학습된 backbone이라 chip-level
  grade, edge geometry, spatial defect pattern 같은 mid-level feature가 이미 정렬됨
- contrastive 목적이 unknown 발견이지만, unknown도 결국 같은 모달리티의 변형/조합이므로
  supervised로 얻은 broad-coverage feature가 starting point로 더 적합
- "supervised collapse" 우려는 (a) 도메인이 다르거나 (b) supervised class 수가 매우 적을
  때 의미 — 본 케이스(33-class, 동일 데이터, ConvNeXtV2 base capacity)에서는 무시

`best_model.pth` 구조: `{"model": <full state dict>, "classes": [...], "ema_state": ...}`.
contrastive.py의 prefix-strip 로직은 `model.`/`backbone.`/`module.` 평탄 prefix만 처리하므로
nested `{"model": ...}` 키를 한 번 더 unwrap해야 한다. 작은 추출 스크립트 또는
contrastive.py 로더 한 줄 추가로 해결.

backbone LR은 head 대비 매우 낮게 (e.g. backbone 1e-6, head 1e-3) 또는 부분
unfreeze (마지막 stage만) 권장 — 이미 정렬된 mid-level feature 보존 + contrastive
objective로 fine-grain만 추가 학습.

## 산출물

`outputs_<preset>_<RUN_TS>/` (기존 `contrastive.py` 구조 그대로):

- `checkpoints/final_infer.pt`
- `checkpoints/last_training.pt`
- `centroids/centroids.npy`, `centroids_meta.json`, `clusterer.pkl`
- `clusters/hdbscan/cluster_XXX_size_YYY/`
- `cluster_summary/`, `ignored_samples/`
- `run.log`, `run_info.json`

## 수행 절차

1. **Pre-flight check**:
   - `data/wm811k_train/summary.json` 존재 확인.
   - 주어진 preset이 `experiments/presets.PRESETS`에 존재 확인.
   - 디스크 여유 확인 (최소 10GB 권장 — outputs 폴더 크기).
2. **실행**:
   ```bash
   python experiments/run_experiment.py \
       --preset <name> \
       --override 'INPUT_DIR=data/wm811k_train'
   ```
   (또는 등가의 모듈 import 방식)
3. **완료 후**:
   - 생성된 `outputs_<preset>_<ts>/` 경로 문자열을 반환 (evaluation, composite-map
     agent에 전달용).
   - `final_infer.pt`, `centroids.npy`, `clusterer.pkl` 존재 확인.

## 규칙 (금기)

- **새 모델 아키텍처 작성 금지**. 이 stage는 래퍼일 뿐.
- **`contrastive.py::CFG`의 Linux 서버 경로 수정 금지**. 의도적으로 유지됨.
- **`outputs_*/` 삭제 금지**. 글로벌 규칙. 재실행은 새 timestamp에 생성됨.
- **`skip` 우회 목적의 폴더 삭제 금지**. 새 preset 이름으로 돌릴 것.
- **`__MULTICROP__` preset 호출 시 현재 `NotImplementedError`** — pre-flight에서
  감지해서 명시적으로 fail.

## 환경 변수 / 마커

6개 런타임 마커 (`__INPUT_MODE__`, `__UNFREEZE_STAGES__`, `__HARD_NEGATIVES__`,
`__STRONG_AUGMENT__`, `__MULTICROP__`, `__DEEPER_HEAD__`)는 preset에 의해 주입.
wrapper는 이를 직접 건드리지 않음.

## 검증 기준

- `outputs_<preset>_<ts>/checkpoints/final_infer.pt` 존재.
- `outputs_<preset>_<ts>/centroids/centroids.npy` shape `(K, D)` (K >= 1).
- `outputs_<preset>_<ts>/centroids/clusterer.pkl` load 가능 (hdbscan 객체).

## 다음 stage

생성된 `outputs_<preset>_<ts>/` 경로를 evaluation agent와 composite-map agent에
전달.
