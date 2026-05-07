# Synthesis docs — chip + chip-multilabel + wafer 합성 통합 인덱스

3 합성 pipeline 의 spec / agent / skill 진입점.

## 3 합성 pipeline

| Pipeline | 단위 | 출력 | 파일 | doc |
|---|---|---|---|---|
| **chip-synth** | chip 200×200 단일 결함 | `classification_chips/<obj>/*.png` | `dist_apply/_synth_chips_only.py` | [CHIP_SYNTH.md](CHIP_SYNTH.md) |
| **chip-multilabel-synth** | chip 200×200 multi-label master | `chip_multilabel/<class>/*.png` | `chip_multilabel/gen_eval_set.py` | [CHIP_MULTILABEL_SYNTH.md](CHIP_MULTILABEL_SYNTH.md) |
| **wafer-synth** | wafer 6400×6400 (32×32 chip grid) | `unknown/<class>/*.png` + `positions/unknown/<class>/*.json` | `dist_apply/_sample_gen.py` + `_sample_gen_gpu.py` + `_sample_canvas_gen.py` | [WAFER_SYNTH.md](WAFER_SYNTH.md) |

## 의존 관계

```
chip-synth          → classification_chips/ (chip source)
                            ↓
                  chip-multilabel-synth → chip_multilabel/ master (eval set)
                            ↓ (사용)
                          training / inference
                            
wafer-synth         → unknown/ (training data)
                            ↓
                  chip CNN training → wafer CNN training
```

## skill / agent 매핑

| Pipeline | skill | agent |
|---|---|---|
| chip-synth | `.claude/skills/chip-synth/SKILL.md` | `.claude/agents/chip-synth.md` |
| chip-multilabel-synth | `.claude/skills/chip-multilabel-synth/SKILL.md` | `.claude/agents/chip-multilabel-synth.md` |
| wafer-synth | `.claude/skills/wafer-synth/SKILL.md` | `.claude/agents/wafer-synth.md` |
| version mgmt | `.claude/skills/chip-wafer-version-mgmt/SKILL.md` | `.claude/agents/chip-wafer-regen.md` |

## v5 / v5.1 / v5.2 spec history

- 통합 표: `docs/chip-multilabel/VERSION_HISTORY.md`
- canonical spec: `docs/chip-multilabel/CHIP_SYNTH_V5_SPEC.md`
- v5.2 manifest: `docs/chip-multilabel/V5_2_REGEN_MANIFEST.md`

## 양 repo mirror 정책

5 wafer-gen 파일 (양 repo byte-identical):
- `_sample_gen.py`
- `_sample_gen_gpu.py`
- `_sample_canvas_gen.py`
- `_synth_chips_only.py`
- `_fq_metadata.py`

위치:
- `D:/project/known-cnn/dist_apply/`
- `D:/project/unknown-contrastive/` (root)

## 절대 영구 원칙

1. palette PNG only (mode='P', grade 0-7). RGB 자유 색 영구 금지.
2. rotation/flip aug 영구 금지 (chip 학습) — scratch ↔ scratch_rot 깨짐.
3. TTA 영구 금지 (chip multi-label 추론) — 동일 이유.
4. 양 repo mirror 동시 commit + push — partial push 금지.
5. spec 변경 시 VERSION_HISTORY + CHIP_SYNTH_V5_SPEC 동시 업데이트.

## Quick reference

| 목적 | command |
|---|---|
| chip 1,000 합성 | `python dist_apply/_synth_chips_only.py --per-class 200` |
| chip_multilabel master 2,250 합성 | `python -m chip_multilabel.gen_eval_set --out-root ... --per-defect 200 --per-normal 200 --per-invalid 50 --source-strength-pct 50` |
| wafer obj-active 6,400 합성 | `python dist_apply/_sample_gen.py --n 200 --workers 6` |
| wafer-canvas 2,000 합성 | `python dist_apply/_sample_canvas_gen.py --n 200` |
