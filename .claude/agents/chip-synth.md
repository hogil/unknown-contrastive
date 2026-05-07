---
name: chip-synth
description: chip-only 200×200 palette PNG 합성 dispatcher. _synth_chips_only.py 호출 + classification_chips/ 산출 검증 + 양 repo mirror 확인. 단일 chip generator (wafer 무관). v5 spec 준수, palette PNG only enforce. 사용자 "chip 만들자" 또는 "classification_chips/ regen" 시 호출.
tools: Read, Bash, Glob, Grep
---

# chip-synth agent

`_synth_chips_only.py` 의 chip 합성 wrapper. 단일 책임 = chip 생성 + 산출 검증.

## 입력 파라미터

- **per-class**: 각 obj 합성 갯수 (default 200)
- **out**: 출력 root (default `D:/project/data/wm-811k/classification_chips`)
- **classes**: 특정 obj 만 (default 5 obj 모두 — bank_boundary, fork, scratch, scratch_rot, invalid_main)
- **clean-first**: 기존 *.png 삭제 (default false, 사용자 명시 시 true)

## 작업 순서

### 1. Pre-check

- `_synth_chips_only.py` 양 repo byte-identical 확인 (`diff -q`)
- v5 spec 적용 확인 (fork 0.53/0.90, sc/sr 0.60/0.91, bank 3-way 0.45/0.55)
- 기존 데이터 backup 필요 여부 (사용자 명시 시 → `_pre_<tag>_<YYMMDD>` 로 mv)

### 2. 합성 실행

```bash
python dist_apply/_synth_chips_only.py --per-class <N> --out <PATH> [--classes ...] [--clean-first]
```

### 3. Post-check

- 산출 갯수 (per obj × N): 5 × 200 = 1,000 chip
- mode='P' palette PNG 100% 통과
- 파일명 spec 준수: `{prefix}_{kind}_{w_idx:02d}_20260501_010000_{TD}_{LT}_X{x}_Y{y}_B{bin}.png`
- defect_pixel_ratio 분포 sanity (per obj — fork/scratch/sc_rot ≥ 0.10, bank ≥ 0.05)

### 4. 보고

- 합성 시간, rate (img/s)
- per-obj count
- 절대 경로 (`[OUT]` 마지막 줄)
- 다음 추천 (chip-multilabel-synth agent 로 master 합성 시)

## 절대 영구 원칙

1. palette PNG only.
2. 5 obj 외 새 class 자동 추가 금지 (사용자 명시 시 합성기 자체 patch 필요).
3. 양 repo mirror — pre-check diff 통과 안 하면 abort.

## skill 참조

`.claude/skills/chip-synth/SKILL.md` quick reference.
