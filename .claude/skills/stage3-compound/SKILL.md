---
name: stage3-compound
description: 3-stage CNN pipeline orchestration — chip 5-class classifier (stage 1) → per-wafer obj_id_map .npy cache (stage 2) → 3-channel feature wafer compound CNN (stage 3, cnn_train_compound.py). Three log roots logs_chip / logs_wafer / logs_all, each with overall/ best-run mirror.
---

# stage3-compound skill

`cnn_train_compound.py` (이전 이름 `cnn_train_failobj.py`) 기반 wafer 분류기 학습 —
기존 `cnn_train.py` 가 failbit 한 채널만 보던 것과 달리, chip 단위 object 정보를
추가 채널 G 로 합쳐 보는 **3-channel compound CNN** 의 전체 파이프라인 orchestrate.

## Why 3 stages

기존 `cnn_train.py` 는 wafer PNG 자체만 학습 → wafer pattern + chip object pattern
두 차원 모두 backbone 이 새로 학습해야 함. 3-stage 분리:

1. **Stage 1** chip 5-class 분류기 — `classification_chips/<obj>/*.png` 입력 (5 폴더).
   chip pattern 자체는 시각적으로 매우 구별돼서 작은 데이터 (100/class) 로도 빠르게 수렴.
   결과: `logs_chip/<chip_tag>_<TS>_<f1>_<f1>/best_model.pth`
2. **Stage 2** chip 분류기 inference — wafer 1장씩 `b ≥ 200` defect chip 들 batch
   predict → 32×32 `object_type_id` uint8 map .npy 로 cache. wafer 1장당 한 번만.
   결과: `obj_id_maps/<wafer_class>/<basename>.npy` + `obj_id_maps/_meta.json`
3. **Stage 3** wafer CNN — 3-channel feature tensor stack:
     - R = palette_idx / 31  (0=정상~7=worst defect, 31=invalid_fill)
     - G = obj_id / N        (N = `_meta.json` 의 `n_chip_objects`, BICUBIC up)
     - B = zeros
   33-class wafer 분류. 결과: `logs_all/<compound_tag>_<TS>_<f1>_<f1>/`

## obj_id 매핑 (dict 제거됨, runtime derive)

```python
# _build_obj_id_maps.py 에서:
class_idx_to_obj_id = np.arange(1, n_chip_objects + 1, dtype=np.uint8)
# obj_id 0 = "none / 정상 chip / 외곽" 예약
# obj_id 1..N = chip CNN ImageFolder 알파벳 정렬 class index + 1
```

ImageFolder 가 폴더명 알파벳 순으로 class index 부여하는 것을 활용 — 별도
OBJECT_TYPE_ID dict 유지 안 해도 자동으로 알파벳 순 매핑.

`_meta.json` (Stage 2 산출):
```json
{
  "n_chip_objects": 5,
  "chip_classes": ["bank_boundary", "invalid_main", "particle_blast", "scratch", "scratch_21deg"],
  "obj_id_to_label": ["none", "bank_boundary", "invalid_main", "particle_blast", "scratch", "scratch_21deg"],
  "chip_model": "log/chip5_n100_w0_*/best_model.pth",
  "built_at": "..."
}
```

`cnn_train_compound.py` 가 학습 시작 시 `_meta.json` 읽어 `n_chip_objects` 가져옴.
부재 시 `n_chip_objects=5` fallback.

## 검증 print

학습/빌드 시작 시 ImageFolder 가 매긴 class 알파벳 순서를 print:
```
[ImageFolder class order] ['bank_boundary', 'invalid_main', 'particle_blast', 'scratch', 'scratch_21deg']
[obj_id mapping] 0=none, 1=bank_boundary, 2=invalid_main, 3=particle_blast, 4=scratch, 5=scratch_21deg
```

## Why BICUBIC for G (categorical)

categorical 임에도 NEAREST 안 함:
- 32×32 → 384 NEAREST 는 12×12 픽셀 블록 → ConvNeXt 부적합
- BICUBIC 으로 chip 경계 smooth gradient → spatial cue 학습 가능
- id=1.5 같은 fractional value 의미 손실 있지만 "boundary between class regions" 으로 학습됨
- 모든 채널 [0, 1] 정규화 (R: /31, G: /N, B: 0) → scale 통일

## Logs root 분리 + overall/ best-run mirror

세 종류의 학습이 각자 독립 폴더:
- `logs_wafer/` — `cnn_train.py --data-dir unknown/` (33-class wafer, R-only)
- `logs_chip/` — `cnn_train.py --data-dir classification_chips/` (5-class chip)
- `logs_all/` — `cnn_train_compound.py` (33-class wafer, R+G compound)

`cnn_train.py` 가 `args.data_dir` 검사하여 자동 결정:
```python
log_root = "logs_chip" if "classification_chips" in data_dir else "logs_wafer"
```

각 logs_*/ 안에 학습 종료 시 `update_overall_best()` 호출:
```
logs_wafer/
├─ <run1_TS_f1>/   ← 개별 run
├─ <run2_TS_f1>/
└─ overall/        ← 그 폴더 내 best val_f1 run 의 통째 복사
   ├─ best_model.pth
   ├─ best_history.txt
   ├─ ...
   └─ _overall_meta.json   ← {val_f1, source_run, updated_at}
```

`overall/` 의 의미: "이 학습 종류 (wafer / chip / compound) 의 현재 best
checkpoint + 모든 산출물 한 곳에서 즉시 조회 가능". 새 run 의 val_f1 이 더
높으면 통째 교체.

## Augmentation (Stage 3)

`cnn_train_compound.FailObjImageFolder._augment` — wafer-safe + 3-channel 통째 적용:
- ✅ ±15° rotation (R/G/B 모두)
- ✅ ±3% translate / scale (R/G/B 모두)
- ✅ Gaussian noise σ=0.01 — **R 채널만** (G categorical 의미 보존)
- ❌ HFlip (scratch_21deg 21° → -21° angle 정체성)
- ❌ VFlip / 180° (Edge-Top↔Bottom)

## Scripts

| 단계 | script | 출력 root |
|---|---|---|
| Stage 1 | `cnn_train.py --data-dir classification_chips ...` | `logs_chip/` |
| Stage 2 | `_build_obj_id_maps.py --chip-model ...` | `obj_id_maps/<class>/<basename>.npy` + `_meta.json` |
| Stage 3 | `cnn_train_compound.py --obj-id-dir ... [--init-from <wafer>]` | `logs_all/` |
| baseline | `cnn_train.py --data-dir unknown ...` | `logs_wafer/` |

## 명령어 예시

```bash
# Stage 1 — chip 5-class (logs_chip/)
python cnn_train.py \
    --data-dir D:/project/data/wm-811k/classification_chips \
    --subset-config experiments/chip_object_n100.yaml \
    --epochs 20 --batch 16 --img-size 384 --workers 0 --model-tag chip5_n100

# Stage 2 — obj_id maps (~30분 GPU)
python _build_obj_id_maps.py \
    --chip-model logs_chip/chip5_n100_*/best_model.pth \
    --batch 64 --overwrite

# Stage 3 — compound n=50 baseline (logs_all/)
python cnn_train_compound.py \
    --obj-id-dir D:/project/data/wm-811k/obj_id_maps \
    --subset-config experiments/compound_n50.yaml \
    --epochs 30 --batch 16 --img-size 384 --model-tag compound_n50

# Stage 3 — compound n=100 main (logs_all/)
python cnn_train_compound.py \
    --obj-id-dir D:/project/data/wm-811k/obj_id_maps \
    --subset-config experiments/compound_n100.yaml \
    --epochs 30 --batch 16 --img-size 384 --model-tag compound_n100
```

## 검증 체크포인트

| 단계 | 검증 |
|---|---|
| Stage 1 후 | `logs_chip/chip*/best_history.txt` macro F1 ≥ 0.95 (chip 단순) |
| Stage 2 후 | `obj_id_maps/<class>/<basename>.npy` 개수 = wafer 개수, sample shape (32,32), unique ⊆ {0..N}, `_meta.json` 존재 |
| Stage 3 후 | `logs_all/compound_*/best_history.txt` BEST OVERALL test F1 vs `logs_wafer/sz*/` (failbit only) 비교 |
| overall 갱신 | `logs_*/overall/_overall_meta.json` 의 val_f1 이 그 폴더 안 best 와 일치 |

## 금지

- `obj_id_maps/_meta.json` 손상 금지 — `n_chip_objects` 누락 시 학습 fallback 5 적용 (의도와 다른 결과 가능)
- chip 분류기를 매 학습 step 에서 inline 호출 금지 — Stage 2 cache 사용 의도
- positions JSON 에 `chips[].obj` 추가 금지 (사용자 정책)
- `logs_*/<run>/` `obj_id_maps/<class>/` 무단 삭제 금지 (사용자 명시 요청 전)
- Augmentation 에 HFlip 추가 금지 (scratch_21deg angle 정체성 변경)
