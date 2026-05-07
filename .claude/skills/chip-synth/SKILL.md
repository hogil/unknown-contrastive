---
name: chip-synth
description: chip-only 200×200 palette PNG 합성 — 5 class (bank_boundary, fork, scratch, scratch_rot, invalid_main). 단일 chip generator, wafer 합성 X. v5.2 fork 0.53/0.90 / sc/sr 0.60/0.91 smoothstep + bank 3-way zone mix.
---

# chip-synth skill

200×200 chip-internal 결함 1개를 palette PNG (mode='P', grade 0-7) 로 합성.

## 핵심 사용

```bash
python dist_apply/_synth_chips_only.py \
    --per-class 200 \
    --out D:/project/data/wm-811k/classification_chips
```

5 obj × 200 = 1,000 chip 합성, ~6-10s.

## 산출

- `D:/project/data/wm-811k/classification_chips/<obj>/<filename>.png` — palette PNG
- 5 obj: `bank_boundary`, `fork`, `scratch`, `scratch_rot`, `invalid_main`
- 파일명: `{prefix}_{kind}_{w_idx:02d}_20260501_010000_{TD}_{LT}_X{x}_Y{y}_B{bin}.png`

## v5.2 합성 spec

| obj | smoothstep | grade 분포 |
|---|---|---|
| **fork** | 2-stage `(α-0.53)/(0.90-0.53)` | peak grade 2 + halo grade 1, defect_other = 95%/4%/1% (1/3/4) |
| **scratch / scratch_rot** | 2-stage `(α-0.60)/(0.91-0.60)` | peak window 좁음, grade 1 dominant 강화 |
| **bank_boundary** | 3-way zone mix split 0.45/0.55 | center grade 2 dominant (cum_obj 85%), edge halo, BG=CUM_BASELINE_TIERS |
| **invalid_main** | 전체 grade 31 (white) + orange border + 큰 텍스트 | bin 번호 가운데 큰 글씨 |

## 옵션

| Flag | default | 설명 |
|---|---|---|
| `--per-class N` | 200 | 각 obj 합성 갯수 |
| `--out PATH` | `D:/project/data/wm-811k/classification_chips` | 출력 root |
| `--classes ...` | 5 obj 전체 | 특정 obj 만 합성 |
| `--seed N` | 20260506 | random seed |
| `--clean-first` | False | 기존 *.png 먼저 삭제 |

## 절대 영구 원칙

1. palette PNG only (mode='P', grade 0-7). RGB 자유 색 영구 금지.
2. fork 의 ㅡ ㅣ 모양 = leg height 70-130 px, leg count 7-9, top horizontal bar.
3. scratch_rot = 21° 대각선 (왼쪽 아래 → 오른쪽 위).
4. bank_boundary = 3 vertical (cx 50/100/150) + 1 horizontal (cy 100) lines.

## 관련 파일

- 합성기: `D:/project/known-cnn/dist_apply/_synth_chips_only.py` (양 repo mirror: unknown-contrastive root)
- spec doc: `docs/synthesis/CHIP_SYNTH.md`
- agent: `.claude/agents/chip-synth.md`
- v5 spec history: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
