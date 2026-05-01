---
name: chip-object-dataset
description: Verify and use the chip-object crop dataset that _sample_gen.py / _sample_gen_gpu.py write inline during wafer generation.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# chip-object-dataset agent

## Read first

1. `.claude/skills/chip-object-dataset/SKILL.md`
2. `_sample_gen.py` `save_chip_crops()` (라벨 결정 로직)
3. `docs/image-generation/OUTPUT.md` if needed

## Policy

- chip crop dataset 은 wafer generation 도중에 inline으로 만들어진다 (`_sample_gen.py` / `_sample_gen_gpu.py`).
- 폐기: 후처리용 `_make_chip_object_dataset.py` (folder-suffix weak label 이라 25% mixed chip 오라벨).
- chip crop 라벨은 `chip_meta[(gy,gx)]['obj']` 의 true object — 75% primary + 25% mixed 정확.
- Source positions JSON 은 read-only — `chips[].obj` 추가 금지.

## Output Contract

```text
D:/project/data/wm-811k/classification_chips/<object_label>/<wafer_basename_without_yield_sys>_x<x>_y<y>_b<bin>.png
```

5 object labels: `bank_boundary`, `particle_blast`, `scratch`, `scratch_21deg`, `invalid_main`.

`{yield}_{sys}` 토큰 제거된 wafer basename + chip 위치 + bin.

## Generation

```bash
# CPU multiprocess
python _sample_gen.py --n 20 --workers 4

# GPU 가속
python _sample_gen_gpu.py --n 20 --save-workers 8
```

새 wafer + chip crop 동시 생성. 기존 데이터 보존 (random prefix).

## Verification 단계

```bash
# 폴더별 chip crop 수
for d in D:/project/data/wm-811k/classification_chips/*/; do
  echo "$(basename $d): $(ls $d | wc -l)"
done

# spot-check sample 1장 (200x200 P mode)
python -c "from PIL import Image; im=Image.open(list_first); print(im.size, im.mode)"

# visual sanity — 같은 폴더 chip 들이 한 종류 object 인지 확인
```

## 금지

- `_make_chip_object_dataset.py` 사용/실행 금지 (라벨 25% 잘못됨).
- chip crop 파일명에 object 중복 표기 금지.
- 5 OBJECT_LABELS 외 임의 폴더 생성 금지.
- source positions JSON 수정 금지.

## Return

- 폴더별 chip crop 수
- 샘플 PNG 200×200 / palette mode 확인
- visual mixed-label sanity (같은 폴더 안 chip 들이 일관된 object 인지)
