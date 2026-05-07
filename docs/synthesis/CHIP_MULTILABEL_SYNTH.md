# Chip Multi-label Master Synthesis — 합성 spec

`chip_multilabel/gen_eval_set.py` 가 chip multi-label evaluation 용 master 합성. classification_chips/ 가 source.

## 목적

multi-label 분류기 evaluation set:
- 4 single defect (학습 + 평가)
- 6 2-combo (평가 only, 학습 데이터에 없는 multi-label 시뮬)
- Normal (학습 + 평가, 정상영역 noise)
- Invalid (평가 only, 측정불능 chip = 운영에서 false alarm 후보)

## CLI

```bash
python -m chip_multilabel.gen_eval_set \
    --out-root D:/project/data/wm-811k/chip_multilabel \
    --per-defect 200          [4 single + 6 combo 각 N] \
    --per-normal 200          [Normal class N — 운영 prevalence] \
    --per-invalid 50          [Invalid class N] \
    --classification-chips-root  D:/project/data/wm-811k/classification_chips \
    --source-strength-pct 50  [top N% by defect_pixel_ratio] \
    --seed 42 \
    [--clear]                  [DANGEROUS - 기존 out_root 삭제]
```

총 4×N + 6×N + per_normal + per_invalid = 12 class. 기본 default = 2,250 chip.

## 합성 단계

### 1. source filter (defect strength)

```python
# classification_chips/<obj>/ 의 모든 chip 의 defect_pixel_ratio 계산
# top source-strength-pct% 만 retain (예: 50 → top half)
# 200 chip × top 50% = 100 strong chip 만 sample 풀
```

### 2. 4 single defect (각 N 장)

```python
for cls in ['bank_boundary', 'fork', 'scratch', 'scratch_rot']:
    for i in range(N):
        chip = sample_strong(cls)         # top 50% pool 에서 1 장
        sanity = defect_pixel_ratio(chip) ≥ 0.001
        if sanity: save(chip, out/cls/...)
```

### 3. 6 2-combo (min-blend, 각 N 장)

```python
for combo_key in ['bank_boundary+fork', 'bank_boundary+scratch', 'bank_boundary+scratch_rot',
                   'fork+scratch', 'fork+scratch_rot', 'scratch+scratch_rot']:
    obj_a, obj_b = combo_key.split('+')
    for i in range(N):
        chip_a = sample_strong(obj_a).convert('RGB')
        chip_b = sample_strong(obj_b).convert('RGB')
        blended = np.minimum(chip_a, chip_b)   # darker pixel preserve
        # sanity: defect_pixel_ratio(blended) ≥ max(d1, d2) - 0.01
        save(blended, out/combo_key/...)
```

**왜 min-blend?** RGB white (255,255,255) 은 max. 어떤 결함이든 적어도 한 채널 darker → MIN 으로 두 결함 union. 두 결함 같은 pixel 위치면 최대로 darker (mixed defect). 학습 안 한 multi-label 시뮬 효과.

### 4. Normal (per-chip Beta noise, palette PNG)

```python
def _make_normal_chip(rng):
    p_noise = float(rng.beta(2, 10))          # mean ~0.17, range ~0.02-0.50
    u = rng.random((200, 200))
    is_noise = u < p_noise
    u2 = rng.random((200, 200))
    noise_grade = where(u2 < 0.95, 1, 2)       # 95% grade 1 + 5% grade 2
    grades = where(is_noise, noise_grade, 0)   # else grade 0 (white)
    img = palette_PNG(grades, sg.PALETTE)
    return img
```

200 chip → 200 가지 다른 noise level (per-chip Beta 분포 매번 다름).

sanity: whiteness ≥ 0.70 (chip 본질적으로 흰색 dominant).

### 5. Invalid (palette + orange border + 큰 텍스트)

```python
def _make_invalid_chip(rng):
    grades = zeros((200, 200))                 # all grade 0 (white)
    BORDER_IDX = sg.KEY_TO_INDEX['border_inv']  # palette 11 (orange)
    grades[:2,:] = grades[-2:,:] = grades[:,:2] = grades[:,-2:] = BORDER_IDX
    img = palette_PNG(grades, sg.PALETTE)
    bin_num = rng.integers(200, 300)
    text = f"B{bin_num}"
    font = font(64px)                           # large
    text_pos = center(img, text)
    text_idx = sg.KEY_TO_INDEX['text']          # palette 9 (near-black)
    draw.text(text_pos, text, fill=text_idx, font=font)
    return img
```

sanity: whiteness ≥ 0.80 + orange border presence.

## 산출

```
chip_multilabel/
├── bank_boundary/         200 chip
├── fork/                  200 chip
├── scratch/               200 chip
├── scratch_rot/           200 chip
├── bank_boundary+fork/    200 chip
├── bank_boundary+scratch/ 200 chip
├── bank_boundary+scratch_rot/ 200 chip
├── fork+scratch/          200 chip
├── fork+scratch_rot/      200 chip
├── scratch+scratch_rot/   200 chip
├── Normal/                200 chip
├── Invalid/               50 chip
├── manifest.csv           (전체 list + per-chip metadata)
├── _preview/<class>.png   (16-grid 시각 sample)
└── _rejected/<reason>/    (sanity 거부 chip — debugging)
```

## v5.2 spec 기준값

- min-blend: `np.minimum(chip_a_rgb, chip_b_rgb)` per-pixel
- Normal: `Beta(2, 10)` noise prob
- Invalid: palette idx 11 (orange), 9 (text), font 64px
- source filter: top 50% by defect_pixel_ratio (strong-only)

## 절대 영구 원칙

1. palette PNG mode='P', grade 0-7 (Normal/Invalid 도 RGB 합성 영구 금지).
2. min-blend RGB 사용 — alpha-blend 또는 평균 사용 영구 금지 (defect 약화).
3. Normal 의 p_noise 는 chip 마다 random (Beta dist) — fixed value 사용 영구 금지.
4. Invalid 텍스트는 가운데 큰 글씨 (font 64px) — 작은 글씨 영구 금지 (식별성 손상).

## 관련 doc / agent / skill

- skill: `.claude/skills/chip-multilabel-synth/SKILL.md`
- agent: `.claude/agents/chip-multilabel-synth.md`
- 원본 chip source: [CHIP_SYNTH.md](CHIP_SYNTH.md)
- v5 history: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
- 학습 / 추론 pipeline: `chip-multilabel-pipeline` skill
