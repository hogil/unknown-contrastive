# Chip 합성 v5 — canonical spec (260507)

★ **이 문서는 영구 보존**. v5 가 known-cnn / unknown-contrastive 양쪽 repo 의 chip 생성 통일 기준.

## 적용 파일 (양 repo 동일 logic)

| 파일 | 위치 | 역할 |
|---|---|---|
| `_synth_chips_only.py` | `dist_apply/` (known-cnn), root (unknown-contrastive) | standalone 200×200 chip-only generator (canonical) |
| `_sample_gen.py` | 동일 | wafer 합성 + chip crop (multiprocessing). canvas baseline 은 wafer pink noise field. |
| `_sample_gen_gpu.py` | 동일 | wafer GPU 가속 합성. pink noise field GPU FFT. |

## 핵심 logic — 4 단계

### Stage 1: alpha map (라인 모양 mask, [0, 1] float)

`ALPHA_FNS[obj](rng)` — fork ㅡ ㅣ / scratch 평행 / scratch_rot 대각 / bank_boundary 격자.
peak 영역 alpha ~1.0, 외곽 ~0.

### Stage 2: pink-noise baseline (chip 의 비-defect 영역, chip 경계 seam 제거)

```python
chip_p_bg = wafer_pink_field[y0:y0+200, x0:x0+200]   # wafer pink noise 의 자기 위치 slice
u = rng.random((200, 200))
is_bg = u < chip_p_bg
u2 = rng.random((200, 200))
chip_bg_grade = where(u2 < 0.92, grade 1, grade 2)   # 92% grade 1 + 8% grade 2
pink_baseline = where(is_bg, chip_bg_grade, grade 0)
```

`_synth_chips_only.py` (chip-only) 는 wafer_pink_field 없으므로 `CUM_BASELINE_TIERS[tier]` searchsorted 사용 (clean / normal / hazy).

### Stage 3: defect 영역 grade 결정 — obj 별 분기

#### A) fork — 2-stage smoothstep `0.50 / 0.88`

```python
u1 = rng.random((200, 200))
is_defect = u1 < alpha                      # P(defect) = alpha
t2 = clip((alpha - 0.50) / (0.88 - 0.50), 0, 1)
p_2 = t2 * t2 * (3 - 2*t2)                  # smoothstep
u2 = rng.random((200, 200))
is_2 = u2 < p_2
u3 = rng.random((200, 200))
defect_other = where(u3 < 0.95, 1,          # 95% grade 1
                where(u3 < 0.99, 3, 4))     # 4% grade 3 + 1% grade 4
defect_grade = where(is_2, 2, defect_other)
grades = where(is_defect, defect_grade, pink_baseline)
```

α=0.5 → p_2=0
α=0.7 → p_2=0.62
α=0.85 → p_2=0.97

#### B) scratch / scratch_rot — 2-stage smoothstep `0.60 / 0.91`

위 동일 (lo_t2=0.60, hi_t2=0.91).
α=0.7 → p_2=0.32
α=0.85 → p_2=0.86

→ fork 보다 grade 2 비율 약간 ↓ (sustained 라인 vs 개별 leg).

#### C) bank_boundary 등 — 3-way zone mix `split 0.45/0.55`

```python
t_low = clip(alpha / 0.45, 0, 1)
t_high = clip((alpha - 0.45) / 0.55, 0, 1)
s_low = smoothstep(t_low); s_high = smoothstep(t_high)
mask_low = (alpha < 0.45)
w_bg = mask_low * (1 - s_low)
w_edge = mask_low * s_low + (1 - mask_low) * (1 - s_high)
w_center = (1 - mask_low) * s_high
cum_mixed = w_bg * CUM_DEFECT_BG + w_edge * CUM_EDGE + w_center * CUM_OBJ[obj]
grades = sample(cum_mixed)
# low-alpha (alpha < 0.05) 영역은 pink_baseline 으로 override (chip seam 제거)
grades = where(alpha < 0.05, pink_baseline, grades)
```

bank_boundary `OBJECT_DISTS` = `[0.003, 0.10, 0.85, 0.11, 0.004, 0.002, 0.001, 0]` — center grade 2 dominant (85%).

### Stage 4: border 추가 (모든 obj)

- defect chip = 2px BIN-color border (palette idx per assigned bin)
- invalid chip = 2px orange border (palette idx 11) + 가운데 큰 텍스트 (font 64px)
- Normal chip = 1px gray border (palette idx 25)

## Normal/Invalid (chip_multilabel/gen_eval_set.py)

- **Normal**: per-chip `Beta(2, 10)` noise probability (mean ~0.17, range ~0.02-0.50). per-pixel grade 0 (1-p_noise) / grade 1 (95% of noise) / grade 2 (5% of noise). palette PNG mode='P'.
- **Invalid**: 전체 grade 0 (white) + 2px orange border (palette idx 11) + 가운데 'B<bin>' 큰 텍스트 (font 64px, palette idx 9).

## 개정 history

| 버전 | 날짜 | 변경 | 사유 |
|---|---|---|---|
| v0 (d70daaf) | 260506 | smoothstep 0.65/0.92, bank 3-way 0.5/0.5 | 기준선 |
| v1 (chip pink) | 260507 | pink noise inline cap 0.50 | 사용자 "정상 영역 noise 다양화" |
| v2-v3 | 260507 | floor 0.10/cap 0.50, cap 0.35 | 사용자 visual 조정 |
| v4 (wafer pink) | 260507 | wafer-level pink field + chip slice | chip 경계 seam 제거 |
| v5 | 260507 | per-obj smoothstep (fork 0.50/0.88, sc/sr 0.60/0.91) + bank 0.45/0.55 | 사용자 visual 통과 |
| **v5.1** (현재) | 260507 | fork 만 미세 dial down: 0.50/0.88 → 0.53/0.90 | 사용자 "fork 진짜 미세하게 pixel 2 줄여보자" |

## 절대 영구 원칙

1. **palette PNG only** (mode='P', grade 0-7). RGB 자유 색 영구 금지.
2. **rotation/flip aug 영구 금지** (학습) — scratch ↔ scratch_rot 구분 깨짐.
3. **TTA 영구 금지** (chip multi-label 추론) — same reason.
4. v5 변경 시 본 doc 의 history 표 + 적용 3 파일 (양 repo) 모두 반영. partial 변경 절대 금지.

## 참고

- 학습 데이터 위치: `D:/project/data/wm-811k/classification_chips/<obj>/*.png`
- chip_multilabel master: `D:/project/data/wm-811k/chip_multilabel/<class>/*.png`
- 시각 sample: `D:/project/known-cnn/docs/chip-multilabel/manager_report/figs/`
