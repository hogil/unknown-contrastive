---
name: chip-object-dataset
description: chip-object crop dataset is generated inline by _sample_gen.py during wafer generation, using per-chip true labels (75% primary + 25% mixed). Use when verifying chip dataset, training chip-object classifier, or building wafer object_type_id maps.
---

# chip-object-dataset skill

## 핵심 정책 (변경됨)

`_make_chip_object_dataset.py` 와 같은 후처리(post-process) 방식으로 wafer
class 폴더 suffix 만으로 chip 라벨링하면 **wafer 안에서 75% primary + 25%
mixed object 가 섞여 있을 때 25% chip 들이 잘못된 라벨을 받음** (이미지 4장이
한 폴더에 들어가는 증상).

따라서 **chip crop 저장은 wafer generation 시점에 inline으로 수행**한다.
`_sample_gen.py` / `_sample_gen_gpu.py` 가 chip 별 true object (`chip_meta[(gy,gx)]['obj']`)를
이미 알고 있으므로, PNG 저장 직후 같은 `chip_meta` 로 200×200 crop을 정확한
object 폴더에 저장한다.

후처리 스크립트 `_make_chip_object_dataset.py` 는 폐기 대상.

## Contract

- Source PNG/JSON: `_sample_gen.py` 가 생성하면서 동시에 chip crop 저장.
- chip crop 출력 root: `D:/project/data/wm-811k/classification_chips/<obj>/`
- Object label 5종: `bank_boundary`, `particle_blast`, `scratch`, `scratch_21deg`, `invalid_main`.
- 라벨 결정 규칙 (`_sample_gen.save_chip_crops`):
  - `chip_meta[(gy,gx)]['kind'] == 'defect'` → `chip_meta['obj']` (75% primary + 25% mixed 포함)
  - `chip_meta['kind'] == 'invalid'` → `invalid_main`
  - 그 외(또는 obj 가 OBJECT_LABELS 밖) → skip
- Source positions JSON 은 read-only — `chips[].obj` 추가 금지.
- ImageFolder layout — 별도 manifest/summary 없음.

## Filename Contract

```text
D:/project/data/wm-811k/classification_chips/<object_label>/<wafer_basename_without_yield_sys>_x<x>_y<y>_b<bin>.png
```

wafer basename 9-token 중 yield(idx 5), sys(idx 6) 제거.

예:
```text
abj471_00C_23_20260501_010000_96.0_2_PT_NORMAL.png
-> classification_chips/scratch/abj471_00C_23_20260501_010000_PT_NORMAL_x12_y4_b290.png
```

## Outputs

```text
classification_chips/
|- bank_boundary/*.png
|- invalid_main/*.png
|- particle_blast/*.png
|- scratch/*.png
`- scratch_21deg/*.png
```

manifest.jsonl / summary.json 모두 폐지. ImageFolder 가 폴더명으로 라벨 자동 매핑:

```python
torchvision.datasets.ImageFolder("D:/project/data/wm-811k/classification_chips")
```

## Generation

새 wafer + chip crop 동시 생성:

```bash
# CPU multiprocess
python _sample_gen.py --n 20 --workers 4

# GPU 가속
python _sample_gen_gpu.py --n 20 --save-workers 8
```

기존 wafer/JSON 보존 (filename prefix 가 random 6자리라 거의 충돌 없음).
chip crop 도 누적 저장.

기존 잘못 라벨된 chip crop 정리 후 재생성하려면 사용자가 명시적으로:

```bash
# 5 label 폴더만 정리 (gitignored)
rm -r D:/project/data/wm-811k/classification_chips/{bank_boundary,particle_blast,scratch,scratch_21deg,invalid_main}
```

## Validation

- 폴더별 파일 수: `ls classification_chips/<label>/ | wc -l`
- spot-check 샘플 200×200, palette mode 'P'
- visual sanity: 같은 폴더 안 chip 들이 모두 같은 object 모양인지 (이전에 4종이 섞이던 버그 해소 확인)
- 5 OBJECT_LABELS 외 폴더 생성 안 됨

## Downstream Wafer Object Map (다음 stage)

1. `classification_chips/` 로 chip-object classifier 학습 (5 class).
2. wafer 1장씩 inference 시:
   - wafer.png + wafer.json 로드
   - `b >= 200` chip 들을 200×200 crop 해서 chip classifier 에 batch predict
   - 32×32 `object_type_id` map 생성 (0=no object, 1-5=class id)
3. wafer 모델 입력 = 3-channel data tensor (visual RGB 의미 아님):
   - `channel 0` = failbit map (BICUBIC resize for 384/512/1024 ablation)
   - `channel 1` = object_type_id map (categorical, BICUBIC 금지 — chip cell 단위 fill 또는 NEAREST)
   - `channel 2` = zero dummy

## 금기

- `_make_chip_object_dataset.py` 후처리 사용 금지 (folder-suffix weak label 때문에 25% mixed chip 잘못 분류됨).
- positions JSON 에 `chips[].obj` 추가 금지 (사용자 정책).
- chip crop 파일명에 object label 중복 표기 금지 (폴더가 라벨).
- 5 OBJECT_LABELS 외 폴더 생성 금지.
