# Chip Synthesis (chip-only) — 합성 spec

`dist_apply/_synth_chips_only.py` 가 wafer 무관 단일 chip 200×200 palette PNG 합성.

## 목적

chip-internal 결함 5종 (`bank_boundary`, `fork`, `scratch`, `scratch_rot`, `invalid_main`) 의 supervised classification 학습 데이터.

## CLI

```bash
python dist_apply/_synth_chips_only.py \
    --per-class N             [default 200] \
    --out PATH                [default wm-811k/classification_chips] \
    --classes ...             [default 5 obj] \
    --seed N                  [default 20260506] \
    --clean-first             [default false]
```

## 합성 단계 (1 chip 기준)

### 1. baseline canvas (200×200, grade 0/1/2)

```python
baseline_tier = pick_baseline_tier(rng)        # clean / normal / hazy (per-chip 자체 random)
cum_base = CUM_BASELINE_TIERS[baseline_tier]
u = rng.random((200, 200))
canvas = np.searchsorted(cum_base, u)          # grade 0 dominant + sparse 1
```

**chip-only mode** 는 wafer pink noise 없음 — tier searchsorted 로 baseline.

### 2. alpha map (결함 모양)

| obj | alpha 함수 | spec |
|---|---|---|
| `fork` | `alpha_fork(rng)` | top horizontal + 7-9 vertical legs (height 70-130 px) |
| `scratch` | `alpha_scratch(rng)` | 평행 vertical lines |
| `scratch_rot` | `alpha_scratch_rot(rng)` | 21° 대각선 lines |
| `bank_boundary` | `alpha_bank_boundary(rng)` | 3 vertical (cx 50/100/150) + 1 horizontal (cy 100) |

`pick_intensity_tier(rng)` (strong/mid/weak) 로 alpha scale 조정.

### 3. defect 영역 grade 결정 (v5.2)

#### fork — 2-stage smoothstep `0.53/0.90`
```python
is_defect = rng.random < alpha
t2 = clip((alpha-0.53)/(0.90-0.53))
p_2 = smoothstep(t2)
defect_grade = where(p_2, 2, where(0.95, 1, where(0.99, 3, 4)))
```

#### scratch / scratch_rot — 2-stage smoothstep `0.60/0.91`
동일 logic, 더 좁은 peak window — grade 1 dominant 강화.

#### bank_boundary — 3-way zone mix `split 0.45/0.55`
```python
t_low = clip(alpha/0.45)
t_high = clip((alpha-0.45)/0.55)
w_bg = mask_low * (1-s_low)
w_edge = mask_low * s_low + mask_high * (1-s_high)
w_center = mask_high * s_high
cum_mixed = w_bg * CUM_DEFECT_BG + w_edge * CUM_EDGE + w_center * CUM_OBJ['bank_boundary']
grades = sample(cum_mixed)
```

(chip-only 합성에서는 wafer pink slice 없음 → CUM_DEFECT_BG fixed 사용. wafer 합성 의 v5.2 bank seam fix 와 다름 — wafer 합성 시 pink_baseline 으로 대체.)

#### invalid_main — palette grade 31 fill + 큰 텍스트
```python
canvas = full(31)                              # white fill
border = palette[border_inv=11]                # orange 2px
text = "B<bin>"                                # font 64px center
fill = palette[text=9]                          # near-black
```

### 4. border 추가

| obj | border type |
|---|---|
| defect (fork/scratch/scratch_rot/bank_boundary) | 2px BIN-color (`BIN_TO_BORDER_IDX[bin]`) |
| invalid_main | 2px orange (palette 11) |

### 5. 파일명 + save

```python
prefix = rand_prefix(rng)                       # 3-letter UPPER + 3-digit
kind = rng.choice(['00P', '00C'])               # process type
w_idx = rng.randint(1, 24)                      # wafer index
yld, sys = compute_yield(...)                   # synthetic
TD = rng.choice(['EE','PT','PE'])               # tester
LT = rng.choice(['NORMAL','PWQ','ENGINEER'])    # lot type
x_abs = rng.randint(0, 32)                      # spatial coord
y_abs = rng.randint(0, 32)
bin_id = assign_defect_bin(kind, rng)           # 200-299
filename = f"{prefix}_{kind}_{w_idx:02d}_20260501_010000_{TD}_{LT}_X{x_abs}_Y{y_abs}_B{bin_id}.png"
img.save(filename, optimize=False, compress_level=1)
```

## 산출

`D:/project/data/wm-811k/classification_chips/<obj>/<filename>.png`

| obj | count | size | mode | grade range |
|---|---:|---|---|---|
| bank_boundary | 200 | 200×200 | P | 0-7 (peak 2 dominant via OBJECT_DISTS) |
| fork | 200 | 200×200 | P | 0-4 (peak 2 + halo 1, defect_other 1/3/4) |
| scratch | 200 | 200×200 | P | 0-4 (동일) |
| scratch_rot | 200 | 200×200 | P | 0-4 |
| invalid_main | 200 | 200×200 | P | 0 (white fill) + 11 (border) + 9 (text) |

## v5.2 spec 기준값

- fork smoothstep: `(alpha - 0.53) / (0.90 - 0.53)`
- scratch / scratch_rot smoothstep: `(alpha - 0.60) / (0.91 - 0.60)`
- bank_boundary 3-way split: `0.45 / 0.55`
- intensity tier: strong (1.0) / mid (0.96) / weak (0.93) — alpha scale
- baseline tier: clean (BG 95%) / normal (BG 83%) / hazy (BG 65%)

## 양 repo mirror

- known-cnn: `D:/project/known-cnn/dist_apply/_synth_chips_only.py`
- unknown-contrastive: `D:/project/unknown-contrastive/_synth_chips_only.py`
- byte-identical (`diff -q` 통과 필수)

## 관련 doc / agent / skill

- skill: `.claude/skills/chip-synth/SKILL.md` (quick reference)
- agent: `.claude/agents/chip-synth.md` (dispatcher orchestrator)
- v5 history: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
- next pipeline: [CHIP_MULTILABEL_SYNTH.md](CHIP_MULTILABEL_SYNTH.md) (multi-label master)
