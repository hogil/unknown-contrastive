# Multi-class chip combo blend (260508)

> chip-level multi-class 합성 spec. 2-class + 3-class N-way pixel min-blend. eval set + 학습 augmentation 양쪽 활용.

## 목적

학습은 4 single class (bank_boundary, fork, scratch, scratch_rot) 만 — 평가는 6 2-class combo 까지만 있어 3+ class combo robustness 미검증. 사용자 directive 260508: "2 + 3-class combo 10 total" — eval spectrum 14 class 로 확장 + (옵션) 학습 통합.

## 메커니즘 — N-way pixel min-blend

```python
def _min_blend_n(arrs):
    return np.minimum.reduce(arrs).astype(np.uint8)
```

Chip palette PNG 의 흰색 grade 0 = (255,255,255). 결함 색은 더 어두운 RGB. 두/세 chip 의 같은 위치 픽셀을 element-wise min 으로 합치면 어느 한쪽에 결함이 있으면 그 픽셀이 결함색으로 살아남음 (둘다 흰색이면 흰색 유지).

```
chip_A (fork)         chip_B (scratch)        min-blend (fork+scratch)
□□□░░░░               □□██████□              □□██████□    ← scratch 살아남음
█████████             ░░░░░░░░░░             █████████    ← fork 살아남음
░░██░░░░░             ░░░░░░░░░░             ░░██░░░░░
```

3-class 도 같은 원리로 `np.minimum.reduce([a, b, c])`. 한 픽셀에 3 색이 겹치면 가장 어두운 색이 우선.

## 10 combo (총)

### 2-class (C(4,2) = 6) — `COMBO_KEYS`

| key | sources |
|---|---|
| bank_boundary+fork | bank_boundary 1 chip + fork 1 chip |
| bank_boundary+scratch | ... |
| bank_boundary+scratch_rot | ... |
| fork+scratch | ... |
| fork+scratch_rot | ... |
| scratch+scratch_rot | ... |

### 3-class (C(4,3) = 4) — `TRIPLE_COMBO_KEYS` (260508 신규)

| key | sources |
|---|---|
| bank_boundary+fork+scratch | 3 chip |
| bank_boundary+fork+scratch_rot | 3 chip |
| bank_boundary+scratch+scratch_rot | 3 chip |
| fork+scratch+scratch_rot | 3 chip |

## Sanity check

```python
d_blend = _defect_pixel_ratio(blended)
d_max = max(_defect_pixel_ratio(s) for s in sources)
ok = d_blend >= d_max - 0.01
```

3-class 는 sources 가 3 개 — 보통 d_blend > d_max (defect 더 많이 보임). 통과 확률 매우 높음. 통과 못 한 chip 은 `_rejected/<reason>/` 보존.

## Source filter

각 single class 폴더 (`classification_chips/<cls>/`) 에서 `_defect_strength` (= grade 2+ 픽셀 비율) top-50% 만 source 로 사용 (`--source-strength-pct 50`). weak defect chip 으로 합성하면 d_blend 가 sanity 떨어짐.

## Palette PNG mode='P' 정책

source chip 은 palette PNG (mode='P') 이지만 `_load_chip_rgb` 가 RGB 변환 → blend → 결과는 RGB ndarray. 저장 시:
- `_save_chip_rgb` 가 RGB ndarray 를 그대로 PNG 저장 (palette quantize 안 함)
- RGB 값은 palette 색상 set 의 부분집합 (min 결과라도 두 chip 모두 palette 색이면 결과도 palette 색)
- ★ 단, 두 chip 의 grade RGB 가 element-wise min 결과 새 색 (palette 외) 발생 가능 — `np.minimum([255,153,0], [50,50,50]) = [50,50,0]` 같은 경우. 실제로는 흔치 않음.

→ 정책 보강: 결과를 다시 palette quantize (PIL `convert("P", palette=PIL.Image.ADAPTIVE)`) 하면 안전. 현 spec: RGB 저장 (eval 모델 forward 시 model 이 RGB 변환해 사용) — 학습 데이터로 쓸 때 palette quantize 후 저장 권장 (후속 iter).

## 산출 layout

```
D:/project/data/wm-811k/chip_multilabel_synth/        ← master folder
├── manifest.csv         (chip_path, class_key, defect_pixel_ratio, base1/2/3_path, gen_method)
├── bank_boundary+fork/                  100 chip
├── bank_boundary+scratch/               100
├── bank_boundary+scratch_rot/           100
├── fork+scratch/                        100
├── fork+scratch_rot/                    100
├── scratch+scratch_rot/                 100
├── bank_boundary+fork+scratch/          100
├── bank_boundary+fork+scratch_rot/      100
├── bank_boundary+scratch+scratch_rot/   100
├── fork+scratch+scratch_rot/            100
├── _preview/                            10 × 16-grid preview PNG
└── _rejected/<reason>/                  sanity 위반 chip 보존
```

총 1000 chip. master-folder 정책 — runtime sample (`--n-per-class 30/50/100` 등).

## CLI 사용

### 1. Master folder bulk synth

```bash
python -m chip_multilabel.gen_multi_combo_synth \
    --out-root D:/project/data/wm-811k/chip_multilabel_synth \
    --per-combo 100 \
    --source-strength-pct 50.0 \
    --seed 42
```

(--include-triples default ON; 끄려면 --no-triples)

### 2. Eval set 14 class

```bash
python -m chip_multilabel.gen_eval_set \
    --out-root D:/project/data/wm-811k/chip_multilabel_v14class \
    --per-defect 50 --per-normal 200 --per-invalid 50 \
    --source-strength-pct 50 \
    --include-triples \
    --seed 42
```

산출 14 class (4 single + 10 combo + Normal + Invalid) = 4×50 + 10×50 + 200 + 50 = **950 chip**.

### 3. Eval (run_stage1)

```bash
python -m chip_multilabel.run_stage1 \
    --model outputs/iter16B_*/best_model.pth \
    --eval-set D:/project/data/wm-811k/chip_multilabel_v14class \
    --variants I3 \
    --n-per-class 50 --strength-min 0.5
```

## Stop criterion / 결과 분기

| metric | 기준 |
|---|---|
| Synth accept rate | ≥ 95% (sanity 위반 < 5%) |
| 3-combo defect_pixel_ratio mean | > 2-combo mean (3 source 합쳐졌으니) |
| 3-combo F1 zero-shot eval | ≥ 0.5 → robustness OK; < 0.5 → 학습 통합 권장 |
| 12-class macro_f1 (기존) | 1.0 ± 0.005 유지 (eval 변경이 기존 평가 손해 X) |

## 관련 spec / docs

- code: `chip_multilabel/gen_multi_combo_synth.py` (NEW), `chip_multilabel/gen_eval_set.py` (extend)
- constants: `chip_multilabel/constants.py:24 COMBO_KEYS` + `:34 TRIPLE_COMBO_KEYS` (260508)
- chip source: [CHIP_SYNTH.md](CHIP_SYNTH.md)
- chip multilabel master: [CHIP_MULTILABEL_SYNTH.md](CHIP_MULTILABEL_SYNTH.md)
- iter doc: `docs/chip-multilabel/iters/iter_17_multi_combo.md`
- plan: `~/.claude/plans/skills-memory-agent-starry-puzzle.md`

## 절대 영구 원칙

1. palette PNG mode='P' source — RGB blend 결과는 ndarray 저장 (RGB).
2. rotation/flip aug 영구 금지 (학습 통합 시).
3. TTA 영구 금지 (eval).
4. 1 atomic change/iter — 본 iter 17 은 eval set 12→14 class 만, 학습 데이터 추가는 별 iter.
5. master folder 만, 별 subset 폴더 안 만듦.
6. 합성 후 `[OUT]` 절대 경로 메시지 마지막 줄 (CLAUDE.md 강제).

## 양 repo mirror

| 파일 | 양 repo |
|---|---|
| `chip_multilabel/gen_multi_combo_synth.py` | known-cnn ↔ unknown-contrastive |
| `chip_multilabel/gen_eval_set.py` | byte-identical |
| `chip_multilabel/constants.py` | byte-identical |
| `D:/project/data/wm-811k/chip_multilabel_synth/` | shared data dir, single source of truth |
