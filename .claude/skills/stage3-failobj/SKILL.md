---
name: stage3-failobj
description: 3-stage CNN pipeline orchestration — chip-object classifier (stage 1), per-wafer obj_id_map build (stage 2), 3-channel feature wafer CNN (stage 3). Use when training the failbit + object_type_id fusion model that combines wafer-level fail-bit map with chip classifier predictions.
---

# stage3-failobj skill

`cnn_train_failobj.py` 기반 wafer 분류기 학습 — failbit map 한 채널만 보던 기존
`cnn_train.py` 와 다르게, chip 단위 object 정보를 추가 채널(G)로 함께 보는 3-channel
feature CNN 의 전체 학습 파이프라인을 orchestrate.

## Why 3 stages

기존 `cnn_train.py` 는 wafer PNG 자체만 학습했다. 33-class wafer pattern 식별에는
충분하지만, chip-level 의 "어떤 object 인가" (scratch / blast / bank / 21deg / invalid)
정보가 backbone 입력에 들어가지 않아 모델이 chip pattern 까지 새로 학습해야 함.

3 stage fusion 의도:
1. **Stage 1** chip 분류기 — `_sample_gen.save_chip_crops` 로 만든 200×200 chip
   crop 으로 5-class 학습. chip pattern 자체는 매우 단순해서 작은 모델/적은 데이터
   로도 high accuracy.
2. **Stage 2** chip 분류기 inference 로 wafer 단위 32×32 `object_type_id` map 빌드.
   wafer 1장당 한 번 .npy 로 cache → 학습 중 chip CNN 추가 inference 불필요.
3. **Stage 3** wafer CNN 입력을 3-channel 로 stack:
     - R = palette idx / 31 (0=정상~7=worst defect, 31=invalid_fill 정규화)
     - G = obj_id / MAX_OBJECT_TYPE_ID (= /5) 정규화 후 BICUBIC resize
     - B = zeros
   wafer 33-class 분류 학습.

## Why BICUBIC for G (categorical)

`object_type_id` 가 categorical(0=none, 1=invalid_main, ..., 5=bank_boundary) 임에도
NEAREST 가 아닌 BICUBIC 사용:
- 32×32 → 384 NEAREST 는 12×12 픽셀 블록 → ConvNeXt 부적합
- BICUBIC 으로 chip 경계에서 smooth gradient → spatial cue 학습 가능
- categorical 의미 손실은 있지만 (id=1.5 같은 fractional value 발생) CNN 입장에서
  "boundary between invalid and scratch zones" 로 spatial 학습 가능
- 모든 채널을 [0, 1] 범위로 정규화 (R: /31, G: /5, B: 0) → scale 통일

대안 (현재 안 함):
- 6-channel one-hot + BICUBIC each: backbone first conv 교체 필요. 코드 변경 큼.
- chip softmax probability G 채널: 의미 불명확.

## OBJECT_TYPE_ID 매핑

```python
OBJECT_TYPE_ID = {
    "none": 0,           # 정상 chip / 외곽 (no object)
    "invalid_main": 1,
    "scratch": 2,
    "scratch_21deg": 3,
    "particle_blast": 4,
    "bank_boundary": 5,
}
MAX_OBJECT_TYPE_ID = 5
```

`_build_obj_id_maps.py` 와 `cnn_train_failobj.py` 가 동일 매핑 사용 — 변경 시 두 파일
동시 수정 필수.

## Input/Output Paths

| 단계 | 입력 | 출력 |
|---|---|---|
| Stage 1 | `D:/project/data/wm-811k/classification_chips/<obj>/*.png` (5 class) | `log/<chip_tag>_<TS>_<f1>_<f1>/best_model.pth` |
| Stage 2 | wafer PNG + JSON + Stage 1 best_model.pth | `D:/project/data/wm-811k/obj_id_maps/<wafer_class>/<basename>.npy` (32×32 uint8) |
| Stage 3 | wafer PNG + Stage 2 .npy | `log/<failobj_tag>_<TS>_<f1>_<f1>/` |

## Scripts

| 파일 | 단계 | 역할 |
|---|---|---|
| `cnn_train.py --data-dir classification_chips ...` | Stage 1 | chip 5-class 학습 |
| `_build_obj_id_maps.py --chip-model ...` | Stage 2 | chip CNN inference → 32×32 obj_id .npy 캐시 |
| `cnn_train_failobj.py --obj-id-dir ... [--init-from <wafer>]` | Stage 3 | 3-channel wafer 33-class 학습 |

## 명령어 예시

```bash
# Stage 1: chip 분류기 (5 class × 100 = 500 sample, ~7분 CPU / ~3분 GPU)
python cnn_train.py \
    --data-dir D:/project/data/wm-811k/classification_chips \
    --subset-config experiments/chip_object_n100.yaml \
    --epochs 20 --batch 16 --img-size 384 \
    --workers 0 --model-tag chip5_n100

# Stage 2: obj_id maps 빌드 (~30분 GPU batch 64 / ~46시간 CPU)
python _build_obj_id_maps.py \
    --chip-model log/chip5_n100_w0_*/best_model.pth \
    --batch 64 --overwrite

# Stage 3: 3-channel wafer 학습 (cnn_train.py 수준 시간)
python cnn_train_failobj.py \
    --obj-id-dir D:/project/data/wm-811k/obj_id_maps \
    --init-from log/<wafer_best>/best_model.pth \
    --epochs 30 --batch 16 --img-size 384 \
    --model-tag failobj_v1
```

## Resource Pattern

- Stage 2 는 GPU 의존도 높음 (chip inference 310k 회). 타 GPU 학습과 경합 시 CPU
  fallback 가능하지만 매우 느림 — 1024 wafer 학습 완료 대기 권장.
- Stage 3 학습 자체는 일반 cnn_train 과 동일 자원 (RAM 80% / GPU 90% 한계).
  `_resource_guard` watchdog 동일 적용.
- `_resource_guard.py` 인라인 가드 + (선택) `cnn-master` + `resource-monitor`
  외부 layer 둘 다 사용 가능.

## 검증 체크포인트

| 단계 | 검증 |
|---|---|
| Stage 1 후 | `log/<chip_tag>/best_history.txt` macro F1 ≥ 0.95 (chip pattern 단순해서 1 epoch 수렴 흔함) |
| Stage 2 후 | `obj_id_maps/<class>/<basename>.npy` 개수 = unknown wafer 개수, sample npy unique ⊆ {0,1,2,3,4,5}, defect chip 위치에 nonzero 분포 |
| Stage 3 후 | `log/<failobj_tag>/best_history.txt` BEST OVERALL test F1 비교 — `cnn_train.py` baseline (failbit only) 대비 향상 여부 |

## 금지

- OBJECT_TYPE_ID 매핑 무근거 변경 금지 — 두 스크립트 일관성 깨짐
- Stage 2 .npy 를 wafer PNG 와 다른 폴더 구조로 저장 금지 — 학습 코드가
  `obj_id_dir/<wafer_class>/<basename>.npy` 형태 가정
- positions JSON 에 `chips[].obj` 추가 금지 (기존 정책 — chip 분류기로 inference만)
- chip 분류기를 매 학습 step 에서 inline 호출 금지 — Stage 2 cache 사용으로 GPU
  경합 회피
