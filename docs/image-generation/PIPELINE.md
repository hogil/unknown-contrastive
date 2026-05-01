# Render Pipeline — Step by Step

이 문서는 `_sample_gen.py`의 `render(class_name, object_name, seed)` 함수가
하는 일을 단계별로 설명한다. 새 세션에서 처음부터 작성할 때 그대로 따라하면 됨.

## Pre-requisite: WM-811K 분포 heatmap

`_dist_heatmaps/<Class>_p_defect_32.npy` 8개 (Center/Donut/Edge-Loc/Edge-Ring/
Loc/Near-full/Random + meta) 는 gitignored 로컬 산출물. 학습 절차는 다음 단계로
1회 수행됐고 (재학습 시 git history `441c532` 이전의 `_dist_learn.py` 참고):

각 클래스(Center/Donut/Edge-Loc/Edge-Ring/Loc/Near-full/Random)별:
1. PNG 로드 (224×224 RGBA, 값 = {0=outside, 128=normal, 255=defect})
2. defect mask + wafer mask 추출
3. Wafer 중심·반경 fitting → unit disk 좌표계로 reproject
4. 32×32와 256×256 두 해상도로 누적
5. 정규화: `P(defect|cell) = acc_defect / acc_wafer` (cell이 wafer 안에 있을 확률 ≥ 0.1만 유효)
6. `_dist_heatmaps/<Class>_p_defect_{32,256}.npy/.png` 저장

**주의**: Edge-Loc / Loc는 sample마다 위치가 random이므로 평균 heatmap이
거의 uniform이 됨. 이 두 클래스는 sample-level spot generation을 사용 (다음 섹션).

## Render 단계 (1 sample 생성)

### 1. 초기화

```python
rng = np.random.default_rng(seed)
inside = np.load("_dist_heatmaps/Center_p_wafer_32.npy") >= 0.10  # bool (32, 32)
```

### 2. Kind 결정

- `object_name == 'invalid_main'`: kind = `'00C'` (강제)
- 그 외: 50% 확률로 `'00P'` 또는 `'00C'`

### 3. Defect / Invalid mask 결정

```python
if object_name == 'invalid_main':
    invalid_dist   = select_distribution_chips(class_name, rng, inside)  # 분포 따라 클러스터
    invalid_random = select_random_invalid(rng, invalid_dist, inside, n=10)
    invalid_inside_mask = invalid_dist | invalid_random
    defect_mask = zeros((32, 32), bool)                                  # 없음
else:
    defect_mask = select_distribution_chips(class_name, rng, inside)
    invalid_inside_mask = select_random_invalid(rng, defect_mask, inside, n=15)
invalid_mask = invalid_inside_mask & ~defect_mask
```

`select_distribution_chips`는 wafer 분포 클래스에 따라 다른 알고리즘:

#### 3a. Center / Donut / Edge-Ring / Random / Near-full
heatmap 기반 weighted sampling (without replacement):
```python
hm = np.load(f"_dist_heatmaps/{class_name}_p_defect_32.npy")
flat = (hm * inside.astype(float)).flatten()
flat /= flat.sum()
chosen = rng.choice(1024, size=DEFECT_BUDGET[class_name], replace=False, p=flat)
mask = zeros((32, 32), bool); mask.flat[chosen] = True
```

#### 3b. Edge-Loc
Edge-Ring heatmap에서 random anchor 1개 → 주변에 cluster:
```python
er = np.load("_dist_heatmaps/Edge-Ring_p_defect_32.npy")
i = rng.choice(1024, p=(er*inside).flatten()/sum)
cy0, cx0 = i // 32, i % 32
mask = _cluster_around(cy0, cx0, n=6, spread=1.3, inside, rng)
```

#### 3c. Loc
wafer 내부(중심 거리 < radius·0.65) random anchor → cluster:
```python
ys, xs = np.where(inside)
d = sqrt((ys-16)**2 + (xs-16)**2)
interior = d < 16 * 0.65
k = rng.choice(np.where(interior)[0])
cy0, cx0 = ys[k], xs[k]
mask = _cluster_around(cy0, cx0, n=6, spread=1.3, inside, rng)
```

`_cluster_around`: anchor 주위에 Gaussian random walk으로 N chip 배치:
```python
def _cluster_around(cy0, cx0, n, spread, inside, rng):
    mask = np.zeros((32, 32), bool); placed = 0
    for _ in range(200):
        dy, dx = round(rng.normal(0, spread)), round(rng.normal(0, spread))
        cy, cx = cy0+dy, cx0+dx
        if 0 <= cy < 32 and 0 <= cx < 32 and inside[cy, cx] and not mask[cy, cx]:
            mask[cy, cx] = True; placed += 1
            if placed >= n: break
    return mask
```

### 4. Per-chip bin / object 할당

각 defect / invalid chip에 대해 메타정보 생성:

```python
chip_meta = {}     # (gy, gx) -> {'kind', 'obj', 'bin'}

# Defect chips: object mixing (75% primary, 25% other)
for (gy, gx) in defect_mask:
    obj_actual = pick_mixed_object(object_name, rng, mix_ratio=0.25)
    bin_val = rng.choice(OBJECT_BIN_PREF[obj_actual][kind])
    chip_meta[(gy,gx)] = {'kind':'defect', 'obj': obj_actual, 'bin': bin_val}

# Invalid chips
for (gy, gx) in invalid_mask:
    if kind == '00P':
        bin_val = rng.integers(200, 280)   # 200-279
    else:
        bin_val = rng.integers(200, 300)   # 200-299
    chip_meta[(gy,gx)] = {'kind':'invalid', 'obj': None, 'bin': bin_val}
```

`OBJECT_BIN_PREF` 매핑:
```python
{
    'bank_boundary':  {'00P': [285],      '00C': [300]},
    'particle_blast': {'00P': [287, 288], '00C': [386, 388]},
    'scratch':        {'00P': [290],      '00C': [389]},
    'scratch_21deg':  {'00P': [291],      '00C': [390]},
}
```

### 5. Canvas: baseline noise + outside fill

```python
# 5a. Sample baseline grades on entire 6400x6400 canvas
u = rng.random((6400, 6400))
canvas = np.searchsorted(CUM_BASE, u).astype(np.uint8)
del u

# 5b. Outside-wafer 200x200 cells = bg color (chip 없음)
inside_pix = np.repeat(np.repeat(inside, 200, axis=0), 200, axis=1)   # 6400x6400 bool
canvas[~inside_pix] = IDX_BG                                          # idx 8 = #DCEEFF
```

이로써:
- 정상 chip 영역: baseline 분포 픽셀 (P(0)+P(1) = 98%)
- Wafer 밖 영역: bg 색으로만 fill (border 없음)

### 6. Defect chip alpha modulation

```python
for (gy, gx), meta in chip_meta.items():
    if meta['kind'] != 'defect': continue
    obj = meta['obj']                         # mixed object name
    alpha   = ALPHA_FNS[obj](rng)             # (200, 200) float32, peak=1.0
    cum_obj = np.cumsum(OBJECT_DISTS[obj])    # (8,)
    cum_mixed = (1-alpha[..., None]) * CUM_BASE + alpha[..., None] * cum_obj  # (200, 200, 8)
    u = rng.random((200, 200))
    grades = (u[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
    y0, x0 = gy*200, gx*200
    canvas[y0:y0+200, x0:x0+200] = grades
```

### 7. Invalid chip fill

```python
for (gy, gx), meta in chip_meta.items():
    if meta['kind'] != 'invalid': continue
    y0, x0 = gy*200, gx*200
    canvas[y0:y0+200, x0:x0+200] = 31      # palette idx 31 = white
```

### 8. Borders (inside-wafer chips only)

```python
for gy in range(32):
    for gx in range(32):
        if not inside[gy, gx]: continue              # outside chip 없음 → border 없음
        y0, x0 = gy*200, gx*200; y1, x1 = y0+200, x0+200
        meta = chip_meta.get((gy, gx))
        if meta is None:                              # normal chip
            b, c = 1, IDX_BORDER                     # 1px gray
        elif meta['kind'] == 'defect':
            b, c = 2, BIN_TO_BORDER_IDX[meta['bin']] # 2px BIN-color
        else:                                         # invalid
            b, c = 2, IDX_BORDER_INV                 # 2px orange
        canvas[y0:y0+b, x0:x1] = c
        canvas[y1-b:y1, x0:x1] = c
        canvas[y0:y1, x0:x0+b] = c
        canvas[y0:y1, x1-b:x1] = c
```

### 9. PIL palette image + bin text

```python
img = Image.frombytes('P', (6400, 6400), canvas.tobytes())
img.putpalette(PALETTE)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", 64)

for (gy, gx), meta in chip_meta.items():
    if meta['kind'] != 'invalid': continue           # text는 invalid에만
    text = str(meta['bin'])
    cx_px, cy_px = gx*200 + 100, gy*200 + 100
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text((cx_px - tw/2, cy_px - th/2 - bbox[1]), text,
              fill=IDX_TEXT, font=font)
```

### 10. Yield / Sys / LT / TM 계산

```python
sys_bins = {285,286,287,288,290,291} if kind == '00P' else {300,385,386,388,389,390}
netd = inside.sum()                                  # ~803
gd = sum(1 for m in chip_meta.values() if m['bin'] is not None and m['bin'] < 200)
gd += netd - sum(1 for m in chip_meta.values())     # plus normal chips (no entry)
sys_count = sum(1 for m in chip_meta.values() if m['bin'] in sys_bins)
yield_pct = 100 * gd / netd                          # .1f
sys_pct   = 100 * sys_count / netd                   # .0f for filename, .2f for JSON
TD = rng.choice(['PE', 'EE', 'PT'])                  # filename slot 7 = LT value
LT = rng.choice(['NORMAL', 'PWQ', 'ENGINEER'])       # filename slot 8 = TM value
```

(filename의 `TD/LT` 토큰 위치는 의미상 fail-map 문서의 `LT/TM`. 변수명 vs 의미는 OUTPUT.md 참조.)

### 11. PNG 저장

```python
prefix = rand_prefix(rng)                            # 3 letters + 3 digits, e.g. "abc123"
w_idx  = rng.integers(1, 25)                         # wafer ID 01-24
fname = f"{prefix}_{kind}_{w_idx:02d}_20260501_010000_{yld:.1f}_{syp:.0f}_{TD}_{LT}.png"
img.save(f"D:/project/data/wm-811k/unknown/{class_name}_{object_name}/{fname}",
         optimize=True)                              # NO transparency arg
```

### 12. Positions JSON 저장

(상세 OUTPUT.md 참조)

반드시 `_fq_metadata.add_synthetic_fq_to_json()`를 호출해 다음을 채운 뒤 저장한다.

- `partid`: `PART_<CLASS>_<ROOT>` 형식 synthetic PART ID
- `pgm`: `PGM_SYN_FQ<item_count>_<STEP>` 형식 synthetic program
- `ftn_keys` / `qtn_keys`: 기본 128개씩
- chip별 `f` / `q`: dense int array. 클래스별 hot item 몇 개가 `b >= 200`
  defect/invalid chip과 주변 chip에서 크게 나오도록 boost되어, FTN/QTN heatmap도
  fail-bit 분포와 비슷한 공간 패턴을 갖는다.

새로 generation을 돌리면 partid/part_id/pgm/ftn_keys/qtn_keys/chip f·q가
자동 포함된다. 기존(다른 데서 들여온) JSON을 보정해야 하는 일회성
스크립트는 git history `441c532..fed8c24` 의 `_backfill_fq_positions.py`
참고 (현재 repo에서는 제거됨).

## Performance Notes

- Full canvas baseline sample (6400×6400) = 40M `rng.random()` + `searchsorted`. ~0.5s.
- Per defect chip alpha gen + sample: ~5ms × ~50 chips = 0.25s.
- PNG save with optimize=True: ~5-10s (deflate compression dominates).
- 35 samples 총 시간: ~3-5분.

## 참조 코드

`_sample_gen.py` (repo root)에 위 모든 단계가 구현되어 있음. 다음 세션에서
재현하려면 이 PIPELINE.md를 보고 그대로 작성하거나 `_sample_gen.py`를 그대로 실행.
