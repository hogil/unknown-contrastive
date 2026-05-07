# v5.2 Chip + Wafer 재합성 manifest

★ 영구 기록. v5.2 spec 적용 (260507 evening).

## Spec 출처

- **Code commit**: known-cnn `(pending)`, unknown-contrastive `(pending)` — 양 repo byte-identical
- **Spec doc**: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md` v5.2 history row
- **VERSION_HISTORY**: `docs/chip-multilabel/VERSION_HISTORY.md` v5.2 row
- **Mirror policy**: 양 repo 5 wafer-gen 파일 byte-identical

## v5.2 spec 차이점 (vs v5.1)

| 항목 | v5.1 | v5.2 | 효과 |
|---|---|---|---|
| **bank_boundary 배경** | 3-way zone mix `w_bg * CUM_DEFECT_BG` (fixed 25.5% sprinkle) | **independent sample + per-pixel choice, bg=pink_baseline** (wafer slice) | bank chip 본체 가 wafer 와 매끄럽게 연속 — chip 경계 seam 제거 |
| **wafer pink baseline 분포** | `clip(Beta(2,8) × (0.5+pink), 0.13, 0.35)` — 대부분 floor 에 cluster | `floor + (cap-floor) × clip(Uniform[0,1] + 0.3×(pink-0.5), 0,1)` with floor=0.22, cap=0.42 | wafer 밝기 spread 가 [floor, cap] 사이 균등 분포 (이전: 90%가 floor에 걸림) |
| **RingDots 위치** | random (r ∈ [0.40, 0.65]·R, n ∈ [14, 23], rotation random) | **FIXED**: r=R×0.55, n=18, th_off=0, sigma=CHIP×0.30, peak∈[0.40,0.60] | 같은 위치/반지름의 18개 dot — wafer 별 brightness 만 random |
| **Edge-Top / Edge-Bottom defect chip 수** | 6 | **20** | edge cluster 명확히 보임 (3.3× ↑) |
| **fork / scratch / scratch_rot smoothstep** | 0.53/0.90 / 0.60/0.91 (변경 없음) | 동일 | — |

## Critical files (양 repo 동일)

| 파일 | known-cnn | unknown-contrastive | 변경 |
|---|---|---|---|
| `_synth_chips_only.py` | `dist_apply/` | root | 변경 없음 |
| `_sample_gen.py` | `dist_apply/` | root | bank else branch + wafer pink uniform + Edge budget |
| `_sample_gen_gpu.py` | `dist_apply/` | root | render_gpu else + render_wafer_canvas + wafer pink uniform |
| `_sample_canvas_gen.py` | `dist_apply/` | root | RingDots fixed positions |
| `_fq_metadata.py` | `dist_apply/` | root | 변경 없음 |

## v5.2 재합성 산출

### 1) `classification_chips/` — chip-object 5 class

**변경 없음** — chip-only 합성기 (`_synth_chips_only.py`) 의 fork/scratch smoothstep 은 v5.1 그대로. v5 manifest 의 1,000 chip 유효.

### 2) `chip_multilabel/` master — multi-label eval set

**변경 없음** — chip 합성 logic 동일. v5 manifest 의 2,250 chip 유효.

### 3) `unknown/` + `positions/unknown/` — wafer maps

**v5.2 재합성** — wafer pink uniform + bank seam fix + RingDots fixed + Edge budget 20.

#### 부분 검증 (10/class)
- 합성 도구: 
  ```bash
  python dist_apply/_sample_gen.py --n 10 --workers 6
  python dist_apply/_sample_canvas_gen.py --n 10
  ```
- 산출: 32 obj-active × 10 + 10 wafer-canvas × 10 = ~420 wafer
- 시각 확인: bank seam 사라짐, RingDots 위치 동일, wafer 밝기 linear spread, Edge cluster ~20 chip

#### 본 합성 (200/class) — pending
- ETA ~25-30 min (CPU multi-proc)
- output: `D:/project/data/wm-811k/unknown/<class>/*.png` + `positions/unknown/<class>/*.json`

## 시각 검증

| 위치 | 내용 |
|---|---|
| `D:/project/known-cnn/_uniform_linear_sample/` | 10 wafer (bright_00 ~ bright_09 sorted, floor 0.22 cap 0.42) |
| `D:/project/known-cnn/_v5_2_smoke/` | bank_boundary 4 (chip seam fix) + RingDots 4 (fixed positions) |
| `D:/project/known-cnn/_floor_cap_4opts/` | 4 (floor, cap) 옵션 × 10 wafer (사용자 선택 결정용) |
| `D:/project/known-cnn/_edge_check_preview/` | Edge-Top/Bottom defect chip 6 → 20 visual |

## 데이터 backup (v5 와 공유)

| 백업 폴더 | 원본 | 크기 |
|---|---|---:|
| `D:/project/data/wm-811k/classification_chips_pre_v5_260507/` | classification_chips/ | 18 MB |
| `D:/project/data/wm-811k/chip_multilabel_pre_v5_260507/` | chip_multilabel/ | 75 MB |
| `D:/project/data/wm-811k/unknown_pre_v5_260507/` | unknown/ | 21 GB |
| `D:/project/data/positions/unknown_pre_v5_260507/` | positions/unknown/ | 6.3 GB |

★ v5.2 별도 backup 만들지 않음 — chip 데이터 변경 없고, wafer 만 새로 생성.

## 재현성 (Reproducibility)

```bash
# Repo state (양 repo)
git checkout <v5.2 commit>

# 1) chip — 변경 없음 (v5 그대로 사용 가능)
python dist_apply/_synth_chips_only.py --per-class 200 --out D:/project/data/wm-811k/classification_chips

# 2) chip_multilabel master — 변경 없음
python -m chip_multilabel.gen_eval_set --out-root D:/project/data/wm-811k/chip_multilabel \
    --per-defect 200 --per-normal 200 --per-invalid 50 \
    --source-strength-pct 50 --seed 42

# 3) wafer obj-active 32 class
python dist_apply/_sample_gen.py --n 200 --workers 6
# 4) wafer-canvas 10 class
python dist_apply/_sample_canvas_gen.py --n 200
```

## 영구 원칙

1. palette PNG only (mode='P', grade 0-7)
2. RGB 자유 색 영구 금지
3. rotation/flip aug 영구 금지 (학습)
4. TTA 영구 금지 (chip multi-label 추론)
5. v5.2 변경 시 본 manifest + `CHIP_SYNTH_V5_SPEC.md` + `VERSION_HISTORY.md` 동시 업데이트
6. 양 repo (known-cnn + unknown-contrastive) byte-identical mirror 유지
