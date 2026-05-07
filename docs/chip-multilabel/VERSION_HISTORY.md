# Chip + Wafer 합성 version history

★ 영구 single source of truth. 모든 chip / wafer 합성 version 의 tag / commit / spec / 백업 / rollback 명령.

매 version 변경 시 **반드시 이 표 + `CHIP_SYNTH_V5_SPEC.md` history 두 곳 동시 업데이트**.

## 통합 history table

| Tag | Date | known-cnn commit | unknown-contrastive commit | fork peak | sc/sr peak | bank | Normal | wafer baseline | 데이터 backup | preview |
|---|---|---|---|---|---|---|---|---|---|---|
| **v0** | 260506 | `d70daaf` | `09ef390` | 0.65/0.92 | 0.65/0.92 | 3-way 0.5/0.5 | RGB sprinkle (white + grey + green dots) | tier searchsorted | `unknown_pre_v5_260507/` 의 일부 (없음, 직접 d70daaf 재생성 필요) | `_chip_revert_preview/d70daaf_RGB_Normal_Invalid/` |
| **v1** | 260506 | `d8ab78d` | (n/a) | 0.65/0.92 | 0.65/0.92 | 3-way 0.5/0.5 | palette PNG Beta(2,10) noise prob | tier searchsorted | (없음) | `_chip_revert_preview/d8ab78d_palette_beta_Normal_Invalid/` |
| **v2-3** | 260507 | `f12c135` | (n/a) | 0.65/0.92 | 0.65/0.92 | 3-way 0.5/0.5 | pink noise FFT 1/f^1.5 | tier searchsorted | (없음, 짧은 iter) | (없음) |
| **v4** | 260507 | (uncommitted) | (n/a) | 0.20/0.50 (강) | 0.20/0.50 | 2-stage (강한 grade 2 dominant) | pink FFT | wafer-level pink noise field | (없음) | `_pink_preview/` (삭제됨) |
| **v5** | 260507 | `582826d` | `293b6ba` | 0.50/0.88 | 0.60/0.91 | 3-way 0.45/0.55 | Beta(2,10) | wafer pink + chip slice | `_pre_v5_260507/` (모두) | `_wafer_v5_preview/` + `docs/chip-multilabel/manager_report/figs/` |
| v5.1 | 260507 | `329ef90` | `b7a53ba` | 0.53/0.90 | 0.60/0.91 | 3-way 0.45/0.55 | Beta(2,10) | wafer pink + chip slice (Beta clip 0.13/0.35) | (v5 backup 공유) | `_fork_v5_1_sample/` |
| **★ v5.2** | 260507 | `2eb12a0` | `292697b` | 0.53/0.90 | 0.60/0.91 | **independent sample + per-pixel choice (bg=pink_baseline → chip seam 제거)** | Beta(2,10) | **wafer pink: uniform [0,1] → linear [0.22, 0.42]** + **RingDots fixed positions (R×0.55, 18 dots)** + **Edge-Top/Bottom DEFECT_BUDGET 6→20** | (v5 backup 공유, 별도 backup 안 만듦) | `_uniform_linear_sample/` (10 wafer linear spread) |

## 각 column 의미

- **fork peak**: 2-stage smoothstep `(alpha - lo) / (hi - lo)` 의 lo/hi 값. 낮을수록 grade 2 (green) 비율 ↑.
- **sc/sr peak**: scratch / scratch_rot 의 동일 smoothstep.
- **bank**: bank_boundary 합성 logic — `2-stage` 또는 `3-way` (zone mix) split 값.
- **Normal**: chip_multilabel 의 Normal chip 합성 방식.
- **wafer baseline**: wafer 의 비-defect chip 영역 grade 분포. v5 부터 wafer pink noise 1/f^1.5 field + 각 chip 자기 위치 slice → chip 경계 seam 제거.

## 데이터 / preview 위치 절대 경로

### Backup data (보존)

```
D:/project/data/wm-811k/classification_chips_pre_v5_260507/
D:/project/data/wm-811k/chip_multilabel_pre_v5_260507/
D:/project/data/wm-811k/unknown_pre_v5_260507/
D:/project/data/positions/unknown_pre_v5_260507/
```

### Active data (현재 version)

```
D:/project/data/wm-811k/classification_chips/
D:/project/data/wm-811k/chip_multilabel/
D:/project/data/wm-811k/unknown/
D:/project/data/positions/unknown/
```

### Preview (시각 sample)

```
D:/project/known-cnn/_chip_revert_preview/synth_v20_fork_scratch_etc/    # 5 chip class × 4
D:/project/known-cnn/_chip_revert_preview/d70daaf_RGB_Normal_Invalid/    # v0 RGB Normal/Invalid
D:/project/known-cnn/_chip_revert_preview/d8ab78d_palette_beta_Normal_Invalid/  # v1 palette
D:/project/known-cnn/_wafer_v5_preview/                                  # v5 wafer 6400×6400
D:/project/known-cnn/_fork_v5_1_sample/                                  # v5.1 fork sample
D:/project/known-cnn/docs/chip-multilabel/manager_report/figs/           # canonical figs (양 repo 동일)
```

## 새 version 만들기 (rollout 절차)

### Step 1 — code 변경

```bash
# 3 파일 (양 repo 모두) 동시 변경:
# - dist_apply/_synth_chips_only.py        (chip-only canonical)
# - dist_apply/_sample_gen.py              (CPU wafer + chip crop)
# - dist_apply/_sample_gen_gpu.py          (GPU wafer + canvas)

# 변경 후 mirror:
cp -f D:/project/known-cnn/dist_apply/_{synth_chips_only,sample_gen,sample_gen_gpu}.py D:/project/unknown-contrastive/
diff -q D:/project/known-cnn/dist_apply/_*.py D:/project/unknown-contrastive/   # 검증
```

### Step 2 — sample 생성 + 사용자 visual approve

```bash
# 6-12 chip / 6 wafer sample → _XXX_sample/ 출력
# 사용자 approve 받기
```

### Step 3 — backup (선택, major version 만)

```bash
# minor (v5.0 → v5.1) 은 backup 생략 (이전 backup 유지)
# major (v4 → v5) 는 새 backup 폴더:
mv D:/project/data/wm-811k/classification_chips D:/project/data/wm-811k/classification_chips_pre_v6_<YYMMDD>
# (chip_multilabel, unknown, positions 동일)
```

### Step 4 — 본 합성

```bash
# chip:
python dist_apply/_synth_chips_only.py --per-class 200 --out D:/project/data/wm-811k/classification_chips
# chip_multilabel master:
python -m chip_multilabel.gen_eval_set --out-root D:/project/data/wm-811k/chip_multilabel \
    --per-defect 200 --per-normal 200 --per-invalid 50 \
    --source-strength-pct 50 --seed 42
# wafer:
python dist_apply/_sample_gen_gpu.py --n 200 --save-workers 8       # GPU (obj-active 32 + canvas 8)
# 또는
python dist_apply/_sample_gen.py --n 200 --workers 6                # CPU multiproc (obj-active 32 만)
```

### Step 5 — version 기록

본 표 + `CHIP_SYNTH_V5_SPEC.md` history 두 곳 동시 업데이트:
- new tag, date, commit hashes (양 repo)
- spec parameters
- 데이터 backup 경로 (있으면)
- preview 경로

### Step 6 — commit + push (양 repo)

```bash
cd D:/project/known-cnn && git add -A && git commit -m "v<tag>: <변경 요약>" && git push
cd D:/project/unknown-contrastive && git add -A && git commit -m "v<tag> mirror" && git push
```

## Rollback 절차 (특정 version 으로 복원)

### 코드만 복원

```bash
# 해당 version commit 으로 3 파일 + spec doc revert
git checkout <commit> -- dist_apply/_synth_chips_only.py dist_apply/_sample_gen.py dist_apply/_sample_gen_gpu.py docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md
# 양 repo 동시
```

### 데이터까지 복원

```bash
# 1) 현재 active 보존 (혹시 모를 사용)
mv D:/project/data/wm-811k/classification_chips D:/project/data/wm-811k/classification_chips_pre_rollback_<YYMMDD>
mv D:/project/data/wm-811k/chip_multilabel D:/project/data/wm-811k/chip_multilabel_pre_rollback_<YYMMDD>
mv D:/project/data/wm-811k/unknown D:/project/data/wm-811k/unknown_pre_rollback_<YYMMDD>
mv D:/project/data/positions/unknown D:/project/data/positions/unknown_pre_rollback_<YYMMDD>

# 2) 백업 → active
mv D:/project/data/wm-811k/classification_chips_pre_v5_260507 D:/project/data/wm-811k/classification_chips
mv D:/project/data/wm-811k/chip_multilabel_pre_v5_260507 D:/project/data/wm-811k/chip_multilabel
mv D:/project/data/wm-811k/unknown_pre_v5_260507 D:/project/data/wm-811k/unknown
mv D:/project/data/positions/unknown_pre_v5_260507 D:/project/data/positions/unknown
```

### 또는 새 합성으로 복원 (백업 없을 때)

위 Step 1 의 `git checkout <commit>` 후 Step 4 본 합성 명령 실행.

## 절대 영구 원칙

1. **모든 version 은 본 표에 기록.** version 누락 = 재현 불가.
2. **데이터 backup 폴더 무단 삭제 금지** (글로벌 룰 + 본 정책). 사용자 명시 요청 전 영구 보존.
3. **양 repo (known-cnn + unknown-contrastive) 동시 변경 + 동시 commit** — 중간 상태 push 금지.
4. **version commit 메시지 prefix**: `v<tag>:` (예: `v5.1:`, `v6:`).
