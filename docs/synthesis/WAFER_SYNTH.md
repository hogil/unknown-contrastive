# Wafer Synthesis — 합성 spec

`dist_apply/_sample_gen.py` (CPU) + `_sample_gen_gpu.py` (GPU) + `_sample_canvas_gen.py` (CPU canvas) 가 6400×6400 wafer palette PNG + positions JSON 합성.

## 목적

WM-811K 분포 + chip-internal 결함 5종 + wafer-canvas 패턴 10종 → 30+ class wafer 학습 데이터.

## 두 갈래 path

| path | code | class 갯수 | 합성 방식 |
|---|---|---:|---|
| **obj-active** | `_sample_gen.py` (CPU multiproc) / `_sample_gen_gpu.py` (GPU) | 32 (Center×5 + Donut×5 + Edge-Ring×5 + Edge-Bottom×5 + Edge-Top×5 + Full×5 + Thick-Edge×2) | distribution heatmap → defect chip locations → chip 마다 alpha 함수 (chip-internal 결함) |
| **wafer-canvas** | `_sample_canvas_gen.py` (canonical CPU) / `_sample_gen_gpu.py::render_wafer_canvas` (GPU legacy) | 10 (BrokenRing/CenterDonut/CrescentArc/CrossScratch/DiagonalSmear/ParallelScratches/RingDots/Row/Starburst/CenterCircle) | wafer-level alpha 함수 (직선/곡선 wafer 가로지르기) → 통과 chip 만 BIN |

## 핵심 사용

### CPU canonical (권장)

```bash
# obj-active 32 class × 200 = 6,400 wafer
python dist_apply/_sample_gen.py --n 200 --workers 6

# wafer-canvas 10 class × 200 = 2,000 wafer
python dist_apply/_sample_canvas_gen.py --n 200
```

병렬 실행 가능 (다른 class subdir → 충돌 X). 총 ~25-30 min.

### GPU (alternative)

```bash
python dist_apply/_sample_gen_gpu.py --n 200 --save-workers 8
```

GPU 의 NEW_WAFER_CANVAS_CLASSES 는 active 27 spec 과 mismatch (RingDots/CenterDonut/Row 미지원, SpiralTrail/EdgeSmudge/BlobChain 만 지원). CPU canvas gen 추가 필요.

## v5.2 합성 단계

### 1. wafer pink baseline (uniform [0.22, 0.42] linear)

```python
wafer_pink = pink_noise_field(SIZE=6400, exponent=1.5)   # 1/f^1.5 FFT, 자연 sensor noise
P_FLOOR, P_CAP = 0.22, 0.42                              # 가장 밝음, 가장 어두움
t_wafer = rng.uniform(0.0, 1.0)                          # uniform per-wafer (Beta clip 폐기 v5.2)
p_bg_field = floor + (cap - floor) * clip(t_wafer + 0.3*(pink - 0.5), 0, 1)
# per-pixel:
is_noise = rng.random < p_bg_field
noise_grade = where(rng.random < 0.92, 1, 2)             # 92% grade 1 + 8% grade 2
canvas = where(is_noise, noise_grade, 0)                 # else grade 0
```

**floor 0.22** = 가장 밝은 wafer 도 22% noise (78% white).
**cap 0.42** = 가장 어두운 wafer 는 42% noise (58% white).
spread 14.9pp linear distribution (uniform t_wafer).

### 2. distribution chip placement (obj-active 32 class)

```python
inside = wafer_inside_mask()                             # 32×32 bool, 안쪽 chip
defect_mask = select_distribution_chips(class, rng, inside)   # heatmap 기반
budget = DEFECT_BUDGET[class.split('_')[0]]              # Center=18, Donut=30, Edge-Ring=70, Edge-Bottom=20, Edge-Top=20, Full=250, Thick-Edge=400
# random invalid chip (within wafer)
invalid_pct = pick_invalid_pct(rng)                      # 0.05 / 0.10 / 0.15 / 0.20
invalid_inside_mask = select_random_invalid(rng, defect_mask, inside, n=budget*invalid_pct)
```

**v5.2 변경**: Edge-Top / Edge-Bottom DEFECT_BUDGET 6 → 20 (cluster 명확).

### 3. defect chip rendering (chip-level)

#### fork / scratch / scratch_rot — 2-stage smoothstep
```python
chip_p_bg = p_bg_field[gy*200:(gy+1)*200, gx*200:(gx+1)*200]
pink_baseline = sample(chip_p_bg)                        # wafer slice → chip seam 없음

is_defect = rng.random < alpha
t2 = clip((alpha - lo)/(hi - lo))   # fork: 0.53/0.90, scratch/scratch_rot: 0.60/0.91
p_2 = smoothstep(t2)
defect_grade = where(p_2, 2, where(0.95, 1, where(0.99, 3, 4)))
grades = where(is_defect, defect_grade, pink_baseline)
```

#### bank_boundary 등 — independent sample + per-pixel choice (v5.2 unified)
```python
edge_g = sample(CUM_EDGE)                                # halo grade
center_g = sample(cum_obj['bank_boundary'])              # peak grade (center 85% grade 2)
choice = rng.random
grades = where(choice < w_bg, pink_baseline,             # ★ wafer 와 매끄러움
        where(choice < w_bg+w_edge, edge_g, center_g))
```

→ chip 본체가 wafer pink slice → chip 경계 seam 영구 제거 (v5.2 fix).

### 4. invalid chip = palette 31 white + orange border + 큰 bin 텍스트

```python
canvas[y0:y1, x0:x1] = 31                                # white fill
draw_border(orange, 2px)                                 # palette 11
draw_text(f"{bin}", center, font_64px, text_idx)         # palette 9
```

### 5. chip border 추가 (모든 inside chip)

| chip type | border |
|---|---|
| normal (inside, no defect) | 1px gray (palette 25) |
| defect | 2px BIN-color (`BIN_TO_BORDER_IDX[bin]`) |
| invalid | 2px orange (palette 11) |

### 6. positions JSON 작성

`positions/unknown/<class>/<filename>.json`:
```json
{
    "wafer_meta": {"synth_round": "29_v15", "intensity_tier": "...", "invalid_pct": 0.10},
    "coord": {"tiles_w": 32, "tiles_h": 32, ...},
    "chips": [{"x_abs": 0, "y_abs": 0, "b": "201", "x_cal": -16, "y_cal": -16, "rect": {...}}, ...],
    "ftn_keys": [...], "qtn_keys": [...],
    "yield": "97.6", "sys": "0", ...
}
```

### 7. wafer-canvas (10 class — `_sample_canvas_gen.py`)

```python
alpha_full = ALPHA_FNS[class](rng)                       # 6400×6400 alpha (line / arc / ring / ...)
# 위 wafer pink baseline 위에 alpha 적용
mask = alpha_full > 0.05
grades_line = mix(CUM_DEFECT_BG, CUM_EDGE, CUM_LINE)     # 라인 통과 영역만
canvas[mask] = grades_line[mask]
```

| class | alpha 함수 | spec |
|---|---|---|
| DiagonalSmear | `alpha_diagonal_smear` | 대각선 한 줄 |
| CrossScratch | `alpha_cross_scratch` | + 모양 |
| CrescentArc | `alpha_crescent_arc` | 초승달 |
| ParallelScratches | `alpha_parallel_scratches` | 평행 줄 3-5 |
| BrokenRing | `alpha_broken_ring` | 끊긴 링 |
| **RingDots** | `alpha_ring_dots` | **FIXED**: 18 dots @ R×0.55 (v5.2) |
| CenterDonut | `alpha_center_donut` | 가운데 도넛 |
| Row | `alpha_row` | 가로 한 줄 (PIL Draw line) |
| Starburst | `alpha_starburst` | 별빛 폭발 |
| CenterCircle | `alpha_center_circle` | 가운데 solid disk |

## 산출

```
unknown/                                  positions/unknown/
├── Center_bank_boundary/    200 wafer    ├── Center_bank_boundary/    200 JSON
├── Center_fork/             200          ├── Center_fork/             200
├── ...                                    ├── ...
├── Edge-Top_scratch_rot/    200          ├── Edge-Top_scratch_rot/    200
├── Thick-Edge_fork/         50  (특수)   ├── Thick-Edge_fork/         50
├── Thick-Edge_invalid_main/ 200          ├── Thick-Edge_invalid_main/ 200
├── Normal_bank_boundary/    200          ├── ...
├── BrokenRing/              200 (canvas) ├── BrokenRing/              200
├── CenterDonut/             200          ├── ...
├── ...
├── CenterCircle/            200
└── (총 약 30 + 10 = 40 class, ~10,000 wafer @ 200/class)
```

## v5.2 spec 기준값

| 항목 | 값 |
|---|---|
| wafer pink floor | 0.22 (가장 밝은 wafer 22% noise) |
| wafer pink cap | 0.42 (가장 어두운 wafer 42% noise) |
| pink mapping | `floor + (cap-floor) * clip(uniform[0,1] + 0.3*(pink-0.5), 0, 1)` |
| pink noise FFT exponent | 1.5 |
| fork smoothstep | 0.53 / 0.90 |
| scratch / scratch_rot smoothstep | 0.60 / 0.91 |
| bank_boundary 3-way zone mix split | 0.45 / 0.55 |
| bank_boundary bg source | **independent sample, bg=pink_baseline (chip seam 제거 v5.2)** |
| Edge-Top / Edge-Bottom DEFECT_BUDGET | 20 (이전 6) |
| RingDots r_center | R × 0.55 (FIXED) |
| RingDots n_dots | 18 (FIXED) |
| RingDots th_off | 0 (no rotation) |
| RingDots peak | uniform(0.40, 0.60) |
| chip-object 5 obj | bank_boundary, fork, scratch, scratch_rot, invalid_main |
| chip noise grade noise mix | 92% grade 1 + 8% grade 2 |

## 환경 변수 (output 경로 override)

```bash
export WAFER_PNG_OUT_DIR=D:/project/data/wm-811k/unknown_v6
export WAFER_JSON_OUT_DIR=D:/project/data/positions/unknown_v6
python dist_apply/_sample_gen.py --n 200 --workers 6
```

## 양 repo mirror (5 wafer-gen 파일)

- known-cnn: `D:/project/known-cnn/dist_apply/`
- unknown-contrastive: `D:/project/unknown-contrastive/` (root)
- 5 파일 byte-identical: `_sample_gen.py`, `_sample_gen_gpu.py`, `_sample_canvas_gen.py`, `_synth_chips_only.py`, `_fq_metadata.py`

## 절대 영구 원칙

1. palette PNG mode='P' 6400×6400. PNG 외 다른 format 금지.
2. 32×32 chip grid, 200×200 chip 고정.
3. RingDots 위치 영구 fixed (R×0.55, 18 dots, th_off=0).
4. wafer pink uniform [0.22, 0.42] linear. Beta clip 영구 금지 (v5.2).
5. bank_boundary 3-way zone mix 시 bg = pink_baseline (chip 경계 seam 영구 제거).
6. 양 repo byte-identical mirror.

## 관련 doc / agent / skill

- skill: `.claude/skills/wafer-synth/SKILL.md`
- agent: `.claude/agents/wafer-synth.md`
- chip source: [CHIP_SYNTH.md](CHIP_SYNTH.md)
- multi-label master: [CHIP_MULTILABEL_SYNTH.md](CHIP_MULTILABEL_SYNTH.md)
- v5 history: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
- version mgmt: `docs/chip-multilabel/VERSION_HISTORY.md`
- v5.2 manifest: `docs/chip-multilabel/V5_2_REGEN_MANIFEST.md`
