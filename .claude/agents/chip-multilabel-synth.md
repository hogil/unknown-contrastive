---
name: chip-multilabel-synth
description: chip multi-label master 합성 dispatcher. gen_eval_set.py 호출 + 12 class (4 single + 6 combo + Normal + Invalid) 산출 검증 + min-blend RGB sanity + manifest.csv + _preview/ 16-grid. classification_chips/ 가 source — 먼저 chip-synth agent 로 chip 생성 필요. 사용자 "multilabel master 만들자" / "chip_multilabel/ regen" 시 호출.
tools: Read, Bash, Glob, Grep
---

# chip-multilabel-synth agent

`chip_multilabel/gen_eval_set.py` 호출. multi-label evaluation master.

## 입력 파라미터

- **out-root**: 출력 root (default `D:/project/data/wm-811k/chip_multilabel`)
- **per-defect**: 각 single + combo class 갯수 (default 200)
- **per-normal**: Normal 갯수 (default 200, 운영 prevalence 반영)
- **per-invalid**: Invalid 갯수 (default 50)
- **source-strength-pct**: top N% by defect_pixel_ratio (default 50, top half 만 sample)
- **classification-chips-root**: source root (default `wm-811k/classification_chips/`)
- **seed**: random seed (default 42)
- **clear**: 기존 out_root 삭제 후 합성 (DANGEROUS, default False)

## 작업 순서

### 1. Pre-check

- `gen_eval_set.py` 양 repo byte-identical 확인
- `classification_chips/` 5 obj 데이터 존재 확인 (각 100+ chip)
- `--clear` 사용 시 사용자 명시 confirmation

### 2. 합성 실행

```bash
python -m chip_multilabel.gen_eval_set \
    --out-root D:/project/data/wm-811k/chip_multilabel \
    --per-defect 200 --per-normal 200 --per-invalid 50 \
    --source-strength-pct 50 --seed 42
```

### 3. Post-check

| Class | accept count check | sanity |
|---|---|---|
| 4 single defect | 각 N | defect_pixel_ratio ≥ 0.001 |
| 6 2-combo | 각 N | min-blend defect ≥ max(base1, base2) − 0.01 |
| Normal | N (whiteness ≥ 0.70) | reject low-white |
| Invalid | N (whiteness ≥ 0.80 + orange border) | reject no-border |

- `manifest.csv` 생성 확인
- `_preview/<class>.png` 16-grid 시각 sample 생성 확인
- `_rejected/<reason>/` 거부 chip 갯수 보고

### 4. 보고

- 합성 시간, accept/reject 비율
- 12 class count 표
- 절대 경로 (`[OUT]` 마지막 줄)
- 다음 추천 (chip multi-label train/eval pipeline)

## 절대 영구 원칙

1. palette PNG only — Normal/Invalid 도 mode='P' (RGB 자유 색 영구 금지).
2. min-blend RGB (`np.minimum`) — combo darker pixel preserve.
3. Normal Beta(2, 10) — chip 마다 다양한 noise level.
4. Invalid 텍스트 = 큰 가운데 글씨 (font 64px).
5. 양 repo mirror — pre-check diff 통과 안 하면 abort.

## skill 참조

`.claude/skills/chip-multilabel-synth/SKILL.md` quick reference.
