---
name: wafer-synth
description: 6400×6400 wafer map palette PNG 합성 — 32 obj-active class + 9 wafer-canvas class. v5.2 wafer pink uniform [0.22, 0.42] linear + bank seam fix + RingDots fixed positions. CPU multiproc + GPU options.
---

# wafer-synth skill

WM-811K 분포 + chip-internal 결함 합성 → 6400×6400 wafer palette PNG + positions JSON.

## 핵심 사용

### CPU multiproc (canonical, 권장)

```bash
# obj-active 32 class × 200 = 6,400 wafer
python dist_apply/_sample_gen.py --n 200 --workers 6

# wafer-canvas 10 class × 200 = 2,000 wafer (canonical CPU)
python dist_apply/_sample_canvas_gen.py --n 200
```

총 ~8,000 wafer + 2,000 = 10,000. CPU 6 worker 약 25-30 min.

### GPU single-process (alternative)

```bash
python dist_apply/_sample_gen_gpu.py --n 200 --save-workers 8
# → 32 obj-active + 8 NEW_WAFER_CANVAS_CLASSES (mismatch with active 27 — legacy)
```

GPU 는 RingDots, CenterDonut, Row 미지원. **CPU canvas gen 사용 권장**.

## 산출

- `D:/project/data/wm-811k/unknown/<class>/*.png` — 6400×6400 palette PNG
- `D:/project/data/positions/unknown/<class>/*.json` — coord, FTN/QTN, chips list

## v5.2 합성 spec (무엇을 어떻게)

### Wafer pink baseline (chip-aware spatial noise)

```python
wafer_pink = _pink_noise_field_2d(rng, SIZE=6400, exponent=1.5)
# v5.2: Beta clip → uniform [0,1] 매핑 (모든 wafer floor 에 cluster X)
P_FLOOR, P_CAP = 0.22, 0.42
t_wafer = rng.uniform(0, 1)                      # uniform per-wafer
p_bg_field = floor + (cap-floor) * clip(t_wafer + 0.3*(pink-0.5), 0, 1)
# per-pixel:
is_noise = rng.random < p_bg_field
noise_grade = where(rng.random < 0.92, 1, 2)
canvas = where(is_noise, noise_grade, 0)
```

→ wafer 밝기 [floor=0.22 (가장 밝음, 78% white), cap=0.42 (가장 어두움, 58% white)] linear spread.

### Defect chip (obj-active)

#### fork / scratch / scratch_rot — 2-stage smoothstep
```python
chip_p_bg = wafer_pink_field[gy*200:(gy+1)*200, gx*200:(gx+1)*200]
pink_baseline = sample(chip_p_bg)                        # chip seam 없는 wafer 연속 background

is_defect = rng.random < alpha
t2 = clip((alpha - lo)/(hi - lo))  # fork: 0.53/0.90, scratch/scratch_rot: 0.60/0.91
p_2 = smoothstep(t2)
defect_grade = where(p_2, 2, where(0.95, 1, where(0.99, 3, 4)))
grades = where(is_defect, defect_grade, pink_baseline)
```

#### bank_boundary 등 — independent sample + per-pixel choice (v5.2)
```python
edge_g = sample(CUM_EDGE)
center_g = sample(cum_obj_bank)
choice = rng.random
grades = where(choice < w_bg, pink_baseline,         # ★ wafer 와 매끄러움
        where(choice < w_bg+w_edge, edge_g, center_g))
```

→ chip 본체 = wafer pink slice → chip 경계 seam 없음.

### Wafer-canvas (10 class, alpha 함수 직접 그림)

| Class | alpha 패턴 |
|---|---|
| DiagonalSmear | 대각선 한 줄 |
| CrossScratch | + 모양 |
| CrescentArc | 초승달 곡선 |
| ParallelScratches | 평행 줄 3-5 |
| BrokenRing | 끊긴 링 |
| **RingDots** | **FIXED**: 18 dots @ R×0.55 radius, no rotation |
| CenterDonut | 가운데 도넛 (속 빈 원) |
| Row | 가로 한 줄 (PIL Draw line) |
| Starburst | 별빛 폭발 |
| CenterCircle | 가운데 solid disk |

### DEFECT_BUDGET (chip 갯수 per wafer per class)

| class | budget | 비고 |
|---|---:|---|
| Center | 18 | |
| Donut | 30 | |
| Edge-Ring | 70 | |
| **Edge-Bottom** | **20** | v5.2 6→20 |
| **Edge-Top** | **20** | v5.2 6→20 |
| Full | 250 | |
| Thick-Edge | 400 | randomized |
| Normal | 0 | special |

## CPU vs GPU 차이

| 항목 | CPU (`_sample_gen.py`) | GPU (`_sample_gen_gpu.py`) |
|---|---|---|
| 병렬화 | `ProcessPoolExecutor` 4-6 worker | single process + ThreadPool save |
| 시간 (200/class × 32 obj) | ~25 min | ~30 min (single proc, GPU FFT 빠름) |
| wafer-canvas 지원 | 없음 (`_sample_canvas_gen.py` 별도) | NEW_WAFER_CANVAS_CLASSES 8 (legacy mismatch) |
| chip-object crop 저장 | `save_chip_crops` (intensity_tier strong/mid only) | 동일 |
| **권장 사용** | **본 작업 (양 obj + 직접 canvas 호출)** | smoke 또는 GPU only 환경 |

## 옵션 (`_sample_gen.py`)

| Flag | default | 설명 |
|---|---|---|
| `--n N` | 200 | 각 class 합성 갯수 |
| `--workers N` | 4 | 병렬 worker 수 |
| `--only-class CLS` | None | 특정 class 만 |
| `--seed-offset N` | 0 | seed 충돌 회피 |
| `--thick-fork-n N` | 50 | Thick-Edge_fork 만 N (다른 class 1/4) |

## 옵션 (`_sample_canvas_gen.py`)

| Flag | default | 설명 |
|---|---|---|
| `--n N` | 2 | 각 class 합성 갯수 |
| `--classes ...` | CANVAS_CLASSES 10 | 특정 class 만 |

## 환경 변수 (output 경로 override)

```bash
export WAFER_PNG_OUT_DIR=D:/project/data/wm-811k/unknown_v6
export WAFER_JSON_OUT_DIR=D:/project/data/positions/unknown_v6
python dist_apply/_sample_gen.py --n 200 --workers 6
```

## 절대 영구 원칙

1. palette PNG only (mode='P', grade 0-7).
2. 6400×6400 fixed (32×32 grid × 200 chip).
3. RingDots 위치 영구 fixed (R×0.55, 18 dots, th_off=0).
4. wafer pink uniform [0.22, 0.42] linear (Beta clip 폐기 v5.2).
5. bank_boundary 의 chip seam 영구 제거 (independent sample + pink_baseline).

## 관련 파일

- 합성기:
  - `D:/project/known-cnn/dist_apply/_sample_gen.py` (CPU multiproc)
  - `D:/project/known-cnn/dist_apply/_sample_gen_gpu.py` (GPU)
  - `D:/project/known-cnn/dist_apply/_sample_canvas_gen.py` (CPU canvas)
  - `D:/project/known-cnn/dist_apply/_synth_chips_only.py` (chip-only, used by chip-synth skill)
  - `D:/project/known-cnn/dist_apply/_fq_metadata.py` (FTN/QTN JSON)
- 양 repo mirror: `D:/project/unknown-contrastive/_*.py` (root, byte-identical)
- spec doc: `docs/synthesis/WAFER_SYNTH.md`
- agent: `.claude/agents/wafer-synth.md`
- v5 history: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
- version mgmt: `docs/chip-multilabel/VERSION_HISTORY.md`
