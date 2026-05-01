# Output Format — PNG filename + Positions JSON

## 1. 출력 위치

```
D:/project/data/wm-811k/unknown/
└── <Distribution>_<Object>/
    └── <prefix6>_<kind>_<wafer:02d>_<ymd>_<hms>_<yld.1f>_<sys.0f>_<TD>_<LT>.png

D:/project/data/positions/unknown/
└── <Distribution>_<Object>/
    └── <same basename>.json

D:/project/data/wm-811k/classification_chips/             ← chip-object crop (per-chip true label)
└── <obj>/                                                  ← bank_boundary | particle_blast | scratch | scratch_21deg | invalid_main
    └── <wafer_basename_without_yield_sys>_x<x>_y<y>_b<bin>.png
```

PNG와 JSON는 **basename이 동일**. 페어로 생성됨.

## 2. PNG Filename — 9 underscore-separated tokens

`fail-map/docs/png-filename.md` 컨벤션 준수. 참조 데이터:
`D:/project/data/wm-811k/fq_missing_test/ABC234_00C_02_20260315_090000_93.4_26_PE_NORMAL.png`

| idx | 토큰 | 예시 | 형식 | 의미 |
|---:|---|---|---|---|
| 0 | root | `abc123` | 3 lowercase letters + 3 digits | random LOT prefix (sample마다 다름) |
| 1 | step | `00P` | `00P` 또는 `00C` | kind (test type) |
| 2 | wafer | `07` | 2-digit `01`-`24` | wafer ID in lot |
| 3 | ymd | `20260501` | YYYYMMDD | 검사 일자 (고정 사용 가능) |
| 4 | hms | `010000` | HHMMSS | 검사 시간 (고정 사용 가능) |
| 5 | yield | `92.4` | float `.1f` | (BIN<200) / NETD × 100 |
| 6 | sys | `18` | int `.0f` | sys defect bin / NETD × 100 |
| 7 | TD | `PE` | one of {PE, EE, PT} | filename slot — 의미상 fail-map의 LT (Lot Type) |
| 8 | LT | `NORMAL` | one of {NORMAL, PWQ, ENGINEER} | filename slot — 의미상 fail-map의 TM (Test Mode) |

**변수명 vs 의미 주의**:
- 우리 코드의 변수 `TD` (filename 7번 위치) = fail-map 문서의 `LT` 의미값 (PE/EE/PT)
- 우리 코드의 변수 `LT` (filename 8번 위치) = fail-map 문서의 `TM` 의미값 (NORMAL/...)

이건 사용자가 처음 요청할 때 사용한 변수명이 fail-map docs와 다르게 되어
있어서 그대로 유지. JSON 안에서는 fail-map 의미대로 `lt` / `tm` 키 사용.

전체 예: `abc123_00P_07_20260501_010000_92.4_18_PE_NORMAL.png`

## 3. Yield 계산

```
yield = (BIN < 200인 chip 수) / NETD × 100
NETD  = inside-wafer chip 수 (= ~803)
GD    = 정상 chip 수 (chip_meta에 없는 inside-wafer chip) + bin<200 chip in chip_meta
```

Defect chip의 bin은 285+/300+ → < 200 안 만족. 즉 GD = normal_inside_count.

Invalid chip의 bin은 200-279/200-299 → < 200 안 만족.

## 4. Sys 계산

```
sys = sys_count / NETD × 100
sys_count = chip_meta에서 bin이 sys_set에 속하는 chip 수
sys_set:
  00P: {285, 286, 287, 288, 290, 291}
  00C: {300, 385, 386, 388, 389, 390}
```

ETC bin (200-299 중 위에 안 속하는 것)은 sys에 안 들어감. Invalid도 sys 아님.

`invalid_main` 클래스의 경우 sys = 0 (모두 invalid bin).

## 5. Positions JSON Schema

`fail-map/docs/positions-json.md` 스키마 준수. compact 직렬화.

```json
{
  "bucket_b_key": "",
  "root": "abc123",
  "step": "00P",
  "wafer": "W07",                                    // ← W prefix 붙음
  "stime": "20260501_010000",
  "partid": "PART_CENTER_BANK_BOUNDARY_ABC123",
  "part_id": "PART_CENTER_BANK_BOUNDARY_ABC123",     // partid mirror (snake_case 호환)
  "tester": "PE",                                    // = LT 값 (참조 데이터 컨벤션)
  "device": "NORMAL",                                // = TM 값
  "pgm": "PGM_SYN_FQ128_00P",
  "ftn_keys": [1000, 1001, ..., 1127],
  "qtn_keys": [5000, 5001, ..., 5127],
  "netd": 803,
  "gd": 762,
  "yield": "92.40",                                  // .2f
  "sys": "2.61",                                     // .2f
  "tm": "NORMAL",                                    // 정확한 fail-map 의미
  "lt": "PE",                                        // 정확한 fail-map 의미
  "coord": {
    "rot_code": 5,
    "x_min_abs": 0,  "y_min_abs": 0,
    "x_max_abs": 31, "y_max_abs": 31,
    "tiles_w_rot": 32, "tiles_h_rot": 32,
    "grid_edges": {
      "xs": [0, 200, 400, ..., 6400],                // 33 elements, 0..32 × 200
      "ys": [0, 200, 400, ..., 6400]
    },
    "canvas": {"width": 6400, "height": 6400},
    "scale": {"sx": 1.0, "sy": 1.0},
    "border": 1,
    "defect_border": 2,
    "center_rule": {"even_x_zero": "left", "even_y_zero": "down"}
  },
  "chips": [
    {
      "x_abs": 5,                                    // grid 좌표 0..31
      "y_abs": 12,
      "b": "1",                                      // bin (string, 앞쪽 0 제거)
      "f": [120, 84, ..., 233],
      "q": [95, 143, ..., 188],
      "x_cal": -10,                                  // centerized: gx - (32//2 - 1) = gx - 15
      "y_cal": -4,                                   // centerized: gy - 32//2 = gy - 16
      "rect": {
        "x0": 1000, "y0": 2400,
        "x1": 1200, "y1": 2600,
        "quad": [[1000,2400],[1200,2400],[1200,2600],[1000,2600]]
      }
    },
    ...
  ]
}
```

## 6. JSON 생성 규칙

### Chips 리스트
- Inside-wafer cells만 (~803). Outside-wafer cells는 chip이 아니므로 제외.
- 각 chip이 `chip_meta`에 있으면 그 bin 사용, 없으면 random normal bin (1-199):
  ```python
  norm_rng = np.random.default_rng(seed + 99999)
  bin_val = chip_meta[(gy,gx)]['bin'] if (gy,gx) in chip_meta else norm_rng.integers(1, 200)
  ```
- 각 chip에는 `f`/`q` dense array를 넣는다. top-level `ftn_keys`/`qtn_keys`의
  같은 index와 매칭된다.

### PARTID / PGM
- `partid`는 빈 문자열 금지. `_fq_metadata.synthetic_partid()`로
  `PART_<CLASS>_<ROOT>` 형태를 넣는다.
- `pgm`도 빈 문자열 금지. `_fq_metadata.synthetic_pgm()`로
  `PGM_SYN_FQ<item_count>_<STEP>` 형태를 넣는다.

### FTN/QTN
- `_fq_metadata.py`가 담당한다 (`_sample_gen.py`/`_sample_gen_gpu.py`가 import).
- 기본 item 수는 FTN 128개 + QTN 128개. 참조 데이터는 500개지만, unknown 전체
  11600장 처리 용량을 고려해 기본은 128. 필요시 `_fq_metadata.DEFAULT_FQ_ITEM_COUNT`
  상수 조정 후 재생성.
- key 범위:
  - `ftn_keys`: 1000부터 연속
  - `qtn_keys`: 5000부터 연속
- 클래스별로 deterministic하게 몇 개 hot item을 고른다. 해당 hot FTN/QTN item은
  `b >= 200`인 defect/invalid chip과 그 주변 chip에서 값이 크게 나오도록 boost한다.
  따라서 FTN/QTN heatmap도 fail-bit defect 분포와 같은 공간 패턴을 가진다.
- 정상 chip은 low baseline + long-tail noise만 가진다.
- **의도(분석용)**: FTN/QTN ↔ fail-bit map cross-correlation 분석이 가능하도록 하는 것이 목적.
  hot item의 defect-chip 평균값이 normal-chip 평균값보다 약 4-5배 크게 (실측: F hot ratio
  Center_bank_boundary 4.9x / Edge-Ring_scratch_21deg 3.5x) 나오게 만들고, 클래스마다 hot index
  세트가 다르게 (`stable_seed("hot-items", class_label, item_count)`) 생성한다 → 클래스 식별과
  공간 패턴 매칭을 동시에 검증할 수 있다. boost 강도/spread radius/item count 조정 시 이 의도가
  유지되도록 검증 (`_fq_metadata`의 `_chip_scores` × `_hot_indices`).

### x_abs / y_abs
grid 좌표 직접 사용 (0~31).

### x_cal / y_cal (centerized)
fail-map 컨벤션: 짝수 grid에서 zero가 left/down. 32 짝수이므로:
```
x_cal = x_abs - (32 // 2 - 1) = x_abs - 15
y_cal = y_abs - (32 // 2)     = y_abs - 16
```

### rect (pixel)
```
x0 = x_abs * 200; y0 = y_abs * 200
x1 = x0 + 200;     y1 = y0 + 200
quad = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
```

### grid_edges
```
xs = [k*200 for k in range(33)]   # [0, 200, ..., 6400]
ys = [k*200 for k in range(33)]
```

### Yield/Sys 포맷 in JSON
- PNG filename: `.1f` for yield, `.0f` for sys
- JSON: `.2f` for both (more precision)

### 직렬화
compact (no whitespace):
```python
json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))
```

## 7. 검증 체크리스트

새 세션에서 generator 작성 후 확인:

- [ ] PNG palette는 fail-map과 동일 (32색, idx 8 bg = #DCEEFF, idx 31 fill = white)
- [ ] PNG에 transparency 인자 없음 (`img.save(path, optimize=True)`)
- [ ] 정상 chip border 1px gray
- [ ] Defect chip border 2px BIN 색
- [ ] Invalid chip border 2px orange + bin number text 그려짐
- [ ] Outside-wafer 영역 = bg 색만, border 없음
- [ ] Defect chip 안 alpha modulation 적용 (라인 중앙 진하고 멀어질수록 옅어짐)
- [ ] yield/sys 공식 일치 (`(bin<200)/netd*100`, `(sys_set bin)/netd*100`)
- [ ] 파일명 9 underscore tokens
- [ ] JSON `chips` 길이 ≈ 803 (inside-wafer 수)
- [ ] JSON `wafer` 필드는 "W" 접두사 포함 (예: "W07")
- [ ] PNG filename `wafer` 토큰은 W 없음 (예: "07")
- [ ] JSON `partid` / `pgm`은 빈 문자열이 아님
- [ ] JSON top-level `ftn_keys` / `qtn_keys` 존재
- [ ] 각 chip에 `f` / `q` array 존재, 길이는 key 개수와 동일
- [ ] class별 hot FTN/QTN item이 `b >= 200` defect/invalid chip 위치에서 normal chip보다 크게 나옴
