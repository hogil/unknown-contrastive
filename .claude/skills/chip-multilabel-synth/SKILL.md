---
name: chip-multilabel-synth
description: chip multi-label master 합성 — 4 single defect + 6 2-combo + Normal + Invalid = 12 class. classification_chips/ source 사용. min-blend 로 combo 합성. evaluation set 전용.
---

# chip-multilabel-synth skill

chip multi-label evaluation master 합성. 200×200 palette PNG, 12 class.

## 핵심 사용

```bash
python -m chip_multilabel.gen_eval_set \
    --out-root D:/project/data/wm-811k/chip_multilabel \
    --per-defect 200 \
    --per-normal 200 \
    --per-invalid 50 \
    --source-strength-pct 50 \
    --seed 42
```

총 ~2,250 chip 합성, ~5-10 min.

## 산출 (12 class)

| class | count | 합성 방식 |
|---|---:|---|
| **4 single defect** | 4×200=800 | `classification_chips/<obj>/` 에서 top 50% strength 만 sample |
| - bank_boundary | 200 | direct copy |
| - fork | 200 | direct copy |
| - scratch | 200 | direct copy |
| - scratch_rot | 200 | direct copy |
| **6 2-combo** | 6×200=1200 | min-blend (np.minimum RGB) |
| - bank_boundary+fork / +scratch / +scratch_rot | 200 each | |
| - fork+scratch / +scratch_rot | 200 each | |
| - scratch+scratch_rot | 200 | |
| **Normal** | 200 | per-chip Beta(2,10) noise prob, palette grade 0/1/2 |
| **Invalid** | 50 | grade 0 (white) + 2px orange border (palette 11) + 큰 'B<bin>' 텍스트 (palette 9) |

## 옵션

| Flag | default | 설명 |
|---|---|---|
| `--out-root PATH` | required | 출력 root (`chip_multilabel/`) |
| `--per-defect N` | 50 | 각 single + combo class 갯수 |
| `--per-normal N` | 200 | Normal class 갯수 (운영 prevalence) |
| `--per-invalid N` | 50 | Invalid 갯수 |
| `--per-class N` | None | DEPRECATED — uniform N (overrides 위 3) |
| `--classification-chips-root` | `wm-811k/classification_chips` | source chip root |
| `--source-strength-pct F` | 100.0 | top N% by defect_pixel_ratio 만 sample (50 = top half) |
| `--seed N` | 42 | random seed |
| `--clear` | False | DANGEROUS — 기존 out_root 삭제 후 합성 |

## 합성 logic

### 4 single defect
- chip 1 장 sample (`source-strength-pct=50` → 결함 강한 top 50% 만)
- 그대로 복사 (no augmentation)

### 6 2-combo (min-blend)
```python
chip_a, chip_b = sample 각 1장
arr_blend = np.minimum(chip_a_rgb, chip_b_rgb)   # darker pixel preserve
# 두 결함 영역 OR 효과 + overlap pixel 더 어두움
```

### Normal (palette + Beta noise)
```python
p_noise = rng.beta(2, 10)   # per-chip random ~0.17 mean, ~0.02-0.50 range
is_noise = rng.random((200,200)) < p_noise
noise_grade = where(rng.random < 0.95, 1, 2)   # 95% grade 1 + 5% grade 2
grades = where(is_noise, noise_grade, 0)
img = palette_PNG(grades)
```

### Invalid (palette + border + text)
```python
grades = zeros((200,200))   # all grade 0 (white)
grades[:2,:] = grades[-2:,:] = grades[:,:2] = grades[:,-2:] = 11   # orange border
draw_text(img, f"B{bin_num}", center, font=64px, fill=9)            # black text
```

## Sanity check

| class | check | threshold |
|---|---|---|
| Normal | whiteness | ≥ 0.70 |
| Invalid | whiteness + border detect | ≥ 0.80 + orange border presence |
| single | defect_pixel_ratio | ≥ 0.001 |
| combo | defect_pixel_ratio (blend ≥ max(base1, base2) - 0.01) | preserved |

## 절대 영구 원칙

1. palette PNG only (mode='P', grade 0-7).
2. min-blend RGB (np.minimum) — 두 결함 darker 보존, 둘 다 white 위치는 white 유지.
3. Normal Beta(2,10) — chip 마다 다양한 noise level (200 chip → 200 가지 다른 brightness).
4. Invalid 텍스트 = 큰 가운데 글씨 (font 64px) — 식별성 명확.

## 관련 파일

- 합성기: `D:/project/known-cnn/chip_multilabel/gen_eval_set.py`
- spec doc: `docs/synthesis/CHIP_MULTILABEL_SYNTH.md`
- agent: `.claude/agents/chip-multilabel-synth.md`
- output 검증: chip_multilabel/manifest.csv + _preview/<class>.png 16-grid
