# v5 Chip + Wafer 전체 재합성 manifest

★ 본 문서는 영구 기록. v5 spec 적용 후 데이터 재생성 1회 (260507).

## Spec 출처

- **Code**: `dist_apply/_synth_chips_only.py` + `_sample_gen.py` + `_sample_gen_gpu.py`, commit `582826d` (known-cnn) / `293b6ba` (unknown-contrastive)
- **Spec doc**: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
- **Mirror repos**: known-cnn + unknown-contrastive (3 chip gen 파일 byte-identical)
- **Image-path 절대 규칙**: known-cnn `CLAUDE.md` line 231

## 데이터 backup (보존)

| 백업 폴더 | 원본 위치 | 크기 |
|---|---|---:|
| `D:/project/data/wm-811k/classification_chips_pre_v5_260507/` | `classification_chips/` | 18 MB |
| `D:/project/data/wm-811k/chip_multilabel_pre_v5_260507/` | `chip_multilabel/` | 75 MB |
| `D:/project/data/wm-811k/unknown_pre_v5_260507/` | `unknown/` | 21 GB |
| `D:/project/data/positions/unknown_pre_v5_260507/` | `positions/unknown/` | 6.3 GB |

★ 백업 무단 삭제 금지. 사용자 명시 요청 전 영구 보존.

## 재합성 산출

### 1) `classification_chips/` — chip-object 5 class

- 합성 도구: `python dist_apply/_synth_chips_only.py --per-class 200`
- 시간: 6.3s (159.5 img/s)
- 산출: 5 obj × 200 = 1,000 chip
  - bank_boundary 200, fork 200, scratch 200, scratch_rot 200, invalid_main 200
- 파일 형식: 200×200 palette PNG (mode='P', 8-color palette grade 0-7)

### 2) `chip_multilabel/` master — multi-label eval set

- 합성 도구: `python -m chip_multilabel.gen_eval_set --out-root D:/project/data/wm-811k/chip_multilabel --per-defect 200 --per-normal 200 --per-invalid 50 --source-strength-pct 50 --seed 42`
- 산출: 12 class, 총 2,250 chip + 20 rejected (Normal whiteness sanity 미통과)
  - 4 single (defect): bank_boundary, fork, scratch, scratch_rot — 각 200
  - 6 2-combo: bank_boundary+fork/scratch/scratch_rot, fork+scratch/scratch_rot, scratch+scratch_rot — 각 200
  - Normal: 200 (Beta(2,10) palette)
  - Invalid: 50 (orange border + 큰 텍스트)
- chip source: 위 v5 `classification_chips/` (top 50% by defect strength)
- min-blend RGB 합성 (`np.minimum`)

### 3) `unknown/` + `positions/unknown/` — wafer maps + JSON

- 합성 도구: `python dist_apply/_sample_gen_gpu.py --n 200 --save-workers 8`
- background 실행 (시작 시각 약 260507 진행 중)
- 산출: 32 obj-active class + 8 wafer-canvas class = 40 × 200 = 8,000 wafer 6400×6400 palette PNG + JSON
  - 32 obj-active: Center/Donut/Edge-Ring/Edge-Bottom/Edge-Top/Full × 5 obj + Thick-Edge_invalid_main + Thick-Edge_fork + Normal + Starburst + CommaCluster
  - 8 wafer-canvas: DiagonalSmear, CrossScratch, CrescentArc, SpiralTrail, ParallelScratches, EdgeSmudge, BlobChain, BrokenRing

## 시각 검증

| 위치 | 내용 |
|---|---|
| `D:/project/known-cnn/_wafer_v5_preview/` | 6 wafer 6400×6400 풀해상도 (Donut/fork, Edge-Bottom/scratch_rot, Edge-Ring/scratch, Center/bank_boundary, Edge-Top/fork, Donut/scratch) |
| `D:/project/known-cnn/_chip_revert_preview/synth_v20_fork_scratch_etc/` | 18 chip preview (fork 4, scratch 4, scratch_rot 4, bank_boundary 4, invalid 2) |
| `D:/project/known-cnn/_chip_revert_preview/d8ab78d_palette_beta_Normal_Invalid/` | Normal 8 + Invalid 4 (palette PNG Beta(2,10)) |
| `docs/chip-multilabel/manager_report/figs/` (양 repo) | 30 chip preview (root single + 2-combo + 3-combo OOD overlay + classification_chips_demo + normal_demo) |

## 재현성 (Reproducibility)

본 v5 데이터 정확히 재현하려면:

```bash
# Repo state
git checkout 582826d                            # known-cnn  
git checkout 293b6ba                            # unknown-contrastive (같은 spec)

# 1) chip
python dist_apply/_synth_chips_only.py --per-class 200 --out D:/project/data/wm-811k/classification_chips

# 2) chip_multilabel master
python -m chip_multilabel.gen_eval_set --out-root D:/project/data/wm-811k/chip_multilabel \
    --per-defect 200 --per-normal 200 --per-invalid 50 \
    --source-strength-pct 50 --seed 42

# 3) wafer + wafer-canvas
python dist_apply/_sample_gen_gpu.py --n 200 --save-workers 8
```

seed 정책:
- `_synth_chips_only.py`: `--seed 20260506` (default)
- `gen_eval_set`: `--seed 42`
- `_sample_gen_gpu.py`: deterministic per-class seed offset (`ci*100000 + oi*10000 + s`)

## 변경 history (v5 적용 전후)

| 데이터 | v4 (이전) | v5 (현재) |
|---|---|---|
| chip fork peak grade 2 | 0.65/0.92 (약) | 0.50/0.88 (mid) |
| chip scratch/scratch_rot grade 2 | 0.65/0.92 | 0.60/0.91 (좁은 peak) |
| chip bank_boundary | 3-way 0.5/0.5 | 3-way 0.45/0.55 |
| Normal chip | pink noise FFT (chip 별 spatial) | Beta(2, 10) palette (chip 별 prob, no spatial) |
| wafer baseline | tier searchsorted (clean/normal/hazy) | wafer-level pink noise field 1/f^1.5 (chip seam 제거) |
| GPU fork bug | `unsqueeze(0/1)` shape (200,200,200) | `.unsqueeze` 제거 → (200,200) |

## 절대 영구 원칙 (다시 확인)

1. **palette PNG only** (mode='P', grade 0-7). RGB 자유 색 영구 금지.
2. **rotation/flip aug 영구 금지** (학습) — scratch ↔ scratch_rot 회전 구분 손상.
3. **TTA 영구 금지** (chip multi-label 추론) — 동일 이유.
4. v5 spec 변경 시 본 manifest + `CHIP_SYNTH_V5_SPEC.md` 동시 업데이트 필수.
5. 백업 폴더 (`*_pre_v5_260507/`) 무단 삭제 금지.
