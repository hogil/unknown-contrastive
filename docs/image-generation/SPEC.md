# Wafer Synthetic — Technical Spec

이 문서는 모든 수치 사양을 명시한다. 변경 시 `_sample_gen.py`도 같이 수정.

## 1. Canvas / Grid / Chip

```
Canvas size      : 6400 × 6400 pixels
Chip grid        : 32 × 32 cells
Chip pixel size  : 200 × 200 pixels (200 × 32 = 6400 정확히)
Wafer 원 안 chip : ~803 (Center_p_wafer_32.npy ≥ 0.10 기준)
Bank cuts (chip 내부, 모두 200×200 좌표계):
  vertical    : x ∈ {50, 100, 150}
  horizontal  : y ∈ {100}
  → 8 banks per chip (50×100 px each)
```

PNG 용량 추정 (deflate compressed):
- Normal-heavy wafer: 4-6 MB
- Defect 많은 wafer: 5-7 MB
- Near-full: 5-6 MB (defect로 채워져도 압축 가능)

## 2. Palette (32색, fail-map 호환)

`fail-map/utils.py::build_palette` + `fail-map/test_data/color-legends.json` 그대로
사용. 인덱스 순서 고정:

| idx | key | hex | 용도 |
|---:|---|---|---|
| 0 | chip0 | `#FFFFFF` | Grade 0 (pass / 정상) |
| 1 | chip1 | `#9B9B9B` | Grade 1 |
| 2 | chip2 | `#009619` | Grade 2 |
| 3 | chip3 | `#0000FF` | Grade 3 |
| 4 | chip4 | `#D91DFF` | Grade 4 |
| 5 | chip5 | `#FFFF00` | Grade 5 |
| 6 | chip6 | `#FF0000` | Grade 6 |
| 7 | chip7 | `#000000` | Grade 7 |
| 8 | bg | `#DCEEFF` | 배경 (chip 없는 영역, 옅은 light blue) |
| 9 | text | `#000001` | bin 숫자 텍스트 색 |
| 10 | border | `#BEBEBE` | 정상 chip 1px 테두리 |
| 11 | border_inv | `#FF9900` | invalid chip 2px 테두리 (orange) |
| 12 | border_b285 | `#0099FF` | BIN 285 (00P sys defect) |
| 13 | border_b286 | `#FF714F` | BIN 286 |
| 14 | border_b287 | `#66FFCC` | BIN 287 |
| 15 | border_b288 | `#DA26CD` | BIN 288 |
| 16 | border_b290 | `#FFD700` | BIN 290 |
| 17 | border_b291 | `#32CD32` | BIN 291 |
| 18 | border_b300 | `#AAAAAA` | BIN 300 (00C sys defect) |
| 19 | border_b385 | `#00C8FF` | BIN 385 |
| 20 | border_b386 | `#FF00C8` | BIN 386 |
| 21 | border_b388 | `#00FF66` | BIN 388 |
| 22 | border_b389 | `#FF6666` | BIN 389 |
| 23 | border_b390 | `#6666FF` | BIN 390 |
| 24 | border_etc | `#999999` | 기타 BIN ≥ 200 |
| 25-30 | (예약) | `#000000` | (사용 안 함) |
| 31 | invalid_fill | `#FFFFFF` | invalid chip 내부 (white). transparency=31 사용 안 함 |

**중요 변경 vs fail-map 원본**:
- `bg` (idx 8): 원본 `#FEFEFE` → 우리는 `#DCEEFF` (visibility 향상)
- 그 외 모두 동일

Palette 96바이트 빌드:
```python
PALETTE = []
for k in PALETTE_INDEX_TO_KEY: PALETTE.extend(hex_to_rgb(PALETTE_HEX_MAP[k]))
while len(PALETTE) < 96: PALETTE.append(0)
PALETTE[31*3:31*3+3] = [255, 255, 255]                 # idx 31 fill = white
```

**저장 시 transparency 사용 금지**: contrastive 모델 입력으로 활용해야 하므로
모든 픽셀이 명시적 색을 가져야 한다. `img.save(path, optimize=True)` (transparency 인자 없음).

## 3. Baseline Grade Distribution (정상 chip 픽셀)

```python
BASELINE = [0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001]   # grade 0..7
```

특성:
- P(0) = 83%
- P(1) = 15%
- **P(0)+P(1) = 98%** (정상 chip은 거의 0 또는 1만 나옴)
- P(2-7) = 2% (희귀 노이즈, grade 7도 chip 200×200당 평균 4 픽셀 정도 나옴)

이 분포는 **모든 정상 chip**의 픽셀 + **defect chip 내부의 object 영역 바깥 픽셀**에
적용. 곧 wafer 전체에서 P(0)+P(1)이 압도적으로 우세.

## 4. Object Grade Distributions (defect chip 라인/blob 중앙)

각 chip object의 "라인 중앙" 또는 "blob 중심" 픽셀(α=1)에서 사용되는 분포.
**P(0) ≤ 3%** — 라인 중앙은 사실상 grade 1+ 불량 픽셀 밀집:

```python
OBJECT_DISTS = {
    'bank_boundary':  [0.03, 0.30, 0.30, 0.20, 0.10, 0.05,  0.015, 0.005],   # grade 1-3 heavy
    'particle_blast': [0.02, 0.20, 0.30, 0.20, 0.15, 0.08,  0.04,  0.01 ],   # grade 2-5 heavy (severe)
    'scratch':        [0.02, 0.40, 0.30, 0.15, 0.07, 0.03,  0.015, 0.005],   # grade 1-3 heavy
    'scratch_21deg':  [0.02, 0.40, 0.30, 0.15, 0.07, 0.03,  0.015, 0.005],   # scratch와 동일 분포, 라인 각도만 다름
}
```

`invalid_main` object는 grade 분포 적용 안 함 — chip 전체가 palette index 31 (white)로 채워짐.

## 5. Mixing Formula

각 픽셀 grade는 baseline과 object_dist의 **선형 혼합 분포**에서 sampling:

```
P(grade=g | chip=defect, position=(x,y))
  = (1 - α(x,y)) · BASELINE[g]  +  α(x,y) · OBJECT_DIST[g]
```

여기서:
- `α(x,y) ∈ [0, 1]`: chip 내 object 강도 공간장 (object별 함수가 정의됨, 다음 섹션)
- `α=0` (object 영역 밖): pure baseline → 정상 분포
- `α=1` (object 중심): pure object_dist → 압도적으로 grade 1+

Sampling 구현 (cumulative threshold):
```python
cum_mixed = (1-α[..., None]) * CUM_BASE + α[..., None] * cum_obj  # (CHIP, CHIP, 8)
u = rng.random((CHIP, CHIP))
grades = (u[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
```

## 6. Alpha Field (α(x,y) per object)

### 6.1 bank_boundary

3 vertical lines (x=50/100/150) + 1 horizontal (y=100). 각 라인:
- 수직(perpendicular) 방향: σ=10 Gaussian falloff (라인 중앙 α=1, ±20px에서 ~0.135)
- 길이 방향: random Y center + random Y sigma envelope (라인 균일 X, 부분적 강함)
- Peak alpha = **1.0**

```python
def alpha_bank_boundary(rng):
    a = np.zeros((200, 200), dtype=np.float32)
    sigma_perp = 10.0
    for cx in [50, 100, 150]:
        y_center = rng.uniform(40, 160); y_sigma = rng.uniform(50, 90)
        y_env = exp(-((Y - y_center)**2) / (2*y_sigma**2))
        line  = exp(-((X - cx)**2)      / (2*sigma_perp**2)) * y_env
        a = max(a, line)
    for cy in [100]:
        x_center = rng.uniform(40, 160); x_sigma = rng.uniform(50, 90)
        x_env = exp(-((X - x_center)**2) / (2*x_sigma**2))
        line  = exp(-((Y - cy)**2)       / (2*sigma_perp**2)) * x_env
        a = max(a, line)
    return a * 1.0
```

### 6.2 particle_blast

단일 Gaussian blob, 중심·sigma random. 자연스러운 2D gradient.
- center: random in [50, 150]² (chip 중앙 영역)
- sigma: random uniform [22, 35] (넓은 blob)
- Peak alpha = **1.0**

```python
def alpha_particle_blast(rng):
    cx, cy = rng.uniform(50, 150, 2)
    sigma = rng.uniform(22, 35)
    a = exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))
    return a * 1.0
```

### 6.3 scratch (vertical)

2-5개의 vertical 라인 (random count). 각 라인:
- x: random in [15, 185]
- σ: random uniform [3.0, 5.0] (날카롭지만 gradient 가시화)
- y_start ~ U(0, 60), y_end ~ U(140, 200): 부분적 길이
- 길이 방향 Y envelope (random peak): 라인이 균일하지 않음
- Peak alpha = **1.0**

```python
def alpha_scratch(rng):
    a = np.zeros((200, 200), dtype=np.float32)
    n_lines = rng.integers(2, 6)
    for _ in range(n_lines):
        cx = rng.uniform(15, 185); sigma = rng.uniform(3.0, 5.0)
        y_start = rng.uniform(0, 60); y_end = rng.uniform(140, 200)
        in_range = (y_start <= Y) & (Y <= y_end)
        y_peak = rng.uniform(y_start, y_end); env_sigma = rng.uniform(40, 80)
        y_env = exp(-((Y - y_peak)**2) / (2*env_sigma**2))
        line = exp(-((X-cx)**2)/(2*sigma**2)) * in_range * (0.5 + 0.5 * y_env)
        a = max(a, line)
    return a * 1.0
```

### 6.4 scratch_21deg

`scratch`와 동일하지만 **시계방향(오른쪽으로) 21도 회전한 라인**.

수학:
- θ = 21° (radians = 21 · π / 180 ≈ 0.3665)
- 회전된 라인 방향벡터: `(sin θ, cos θ)`
- 라인 수직 거리: `d_perp = (X-cx)·cos θ - (Y-cy)·sin θ`
- 라인 따라 거리: `d_along = (X-cx)·sin θ + (Y-cy)·cos θ`

```python
def alpha_scratch_21deg(rng):
    a = np.zeros((200, 200), dtype=np.float32)
    n_lines = rng.integers(2, 6)
    theta = np.deg2rad(21.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for _ in range(n_lines):
        cx = rng.uniform(20, 180); cy = rng.uniform(60, 140)
        sigma = rng.uniform(3.0, 5.0)
        d_perp  = (X-cx)*cos_t - (Y-cy)*sin_t
        d_along = (X-cx)*sin_t + (Y-cy)*cos_t
        length = rng.uniform(80, 160)
        in_range = abs(d_along) < length/2
        y_peak = rng.uniform(-length/3, length/3); env_sigma = rng.uniform(40, 80)
        y_env = exp(-((d_along - y_peak)**2) / (2*env_sigma**2))
        line = exp(-d_perp**2/(2*sigma**2)) * in_range * (0.5 + 0.5*y_env)
        a = max(a, line)
    return a * 1.0
```

각도 파라미터화: 21° 외 다른 angle 추가하려면 `theta` 값만 바꾸면 됨 (예: 45°, 90°).

### 6.5 invalid_main

α 적용 없음. 해당 chip 전체 픽셀 = palette index 31 (white).

## 7. Defect Chip Budget per Wafer Distribution

```python
DEFECT_BUDGET = {
    'Center':    25,
    'Donut':     40,
    'Edge-Ring': 70,
    'Edge-Loc':   6,    # localized small spot
    'Loc':        6,    # localized anywhere
    'Random':    50,
    'Near-full': 500,   # most of wafer
}
```

`invalid_main` object일 때는 위 budget이 invalid chip 개수가 됨 (defect는 0).

## 8. Random Invalid Chip Count

각 wafer마다 추가로 산재되는 inside-wafer invalid chip:
- `non-invalid_main` object의 경우: **15** chips random scattered (defect와 겹치지 않음)
- `invalid_main` object의 경우: **10** extra chips random scattered

## 9. Object Mixing in Defect Cells

같은 클래스의 wafer라도 defect chip 모두가 동일한 object일 필요 없음:
- `bank_boundary` 클래스: 75% bank_boundary, 25% 다른 object 무작위 (mixed defects)

```python
def pick_mixed_object(primary, rng, mix_ratio=0.25):
    if rng.random() < mix_ratio:
        others = [o for o in CHIP_OBJECTS if o != primary]
        return random.choice(others)
    return primary
```

`invalid_main` 클래스는 mixing 없음 (모든 결함 chip이 invalid).

## 10. Border Rules

| Chip 종류 | 두께 | 색 (palette idx) |
|---|---|---|
| Normal (inside-wafer, no defect) | 1 px | `border` (10) `#BEBEBE` |
| Defect | 2 px | bin별 (idx 12~24) |
| Invalid (inside-wafer) | 2 px | `border_inv` (11) `#FF9900` |
| Outside-wafer 영역 | **없음** | (chip 자체가 없음, bg 색만 채움) |

## 11. Bin Number Text

**Invalid chip만** chip 중앙에 bin 번호 텍스트 그림.

- font: arial.ttf 64pt (없으면 calibri/DejaVu fallback, 최후 default)
- color: `text` (idx 9) `#000001` (거의 검정)
- 위치: chip 중앙 (200×200 chip의 (100,100))

Defect chip은 텍스트 없음 (테두리 색만으로 BIN 식별).

## 12. Wafer Inside Mask

WM-811K cca 이미지에서 학습된 `Center_p_wafer_32.npy`의 셀 중 값 ≥ 0.10인
셀이 wafer 안. 결과: ~803 inside, ~221 outside (32×32=1024 중).

다른 클래스의 `_p_wafer_32.npy`도 거의 동일 (모두 같은 wafer 원형).
일관성 위해 Center 기준 사용.
