---
name: wafer-synth
description: wafer 6400×6400 palette PNG 합성 dispatcher. _sample_gen.py (CPU multiproc obj-active 32 class) + _sample_canvas_gen.py (CPU canonical 10 wafer-canvas) 호출. v5.2 spec — wafer pink uniform [0.22, 0.42] linear + bank seam fix + RingDots fixed positions + Edge-Top/Bottom budget 20. unknown/ + positions/unknown/ 산출. GPU 옵션 (legacy mismatch 주의).
tools: Read, Bash, Glob, Grep
---

# wafer-synth agent

wafer 합성 dispatcher. obj-active + wafer-canvas 양쪽 처리.

## 입력 파라미터

- **n**: 각 class 합성 갯수 (default 200)
- **workers**: CPU multiproc worker (default 6)
- **only-class**: 특정 class 만 (filter, optional)
- **scope**: `obj-active` / `wafer-canvas` / `all` (default `all`)
- **gpu**: GPU pipeline 사용 여부 (default false, CPU canonical 권장)

## 작업 순서

### 1. Pre-check

- 5 wafer-gen 파일 (`_sample_gen.py`, `_sample_gen_gpu.py`, `_sample_canvas_gen.py`, `_synth_chips_only.py`, `_fq_metadata.py`) 양 repo byte-identical 확인
- v5.2 spec 적용 확인:
  - `P_FLOOR, P_CAP = 0.22, 0.42` (양쪽 _sample_gen.py + _sample_gen_gpu.py)
  - bank else branch independent sample + per-pixel choice
  - RingDots `r_center = R*0.55, n_dots = 18, th_off = 0`
  - DEFECT_BUDGET[Edge-Top/Bottom] = 20
- 기존 unknown/ 데이터 backup 필요 여부 — 사용자 명시 시 `unknown_pre_<tag>_<YYMMDD>` 로 mv

### 2. 합성 실행

#### CPU canonical (권장)

```bash
# obj-active 32 class via CPU multiproc
python dist_apply/_sample_gen.py --n <N> --workers <W>

# wafer-canvas 10 class via canonical CPU
python dist_apply/_sample_canvas_gen.py --n <N>
```

병렬 실행 가능 (서로 다른 class 디렉토리 → 충돌 X).

#### GPU (alternative)

```bash
python dist_apply/_sample_gen_gpu.py --n <N> --save-workers 8
# 32 obj-active + 8 NEW_WAFER_CANVAS_CLASSES (legacy — RingDots/CenterDonut/Row 미지원)
```

GPU 만 사용하면 active 27 spec 의 RingDots/CenterDonut/Row 누락 — CPU canvas gen 추가 필요.

### 3. Post-check

#### obj-active count check

| class | expected count |
|---|---:|
| Center / Donut / Edge-Ring / Edge-Bottom / Edge-Top / Full × 5 obj | N each (= 30 × N) |
| Thick-Edge_invalid_main | N |
| Thick-Edge_fork | min(N, 50) (특수 정책 default) |
| Normal_bank_boundary | N |
| Starburst / CommaCluster | **fail** (legacy CLASSES, render() 미지원 — skip) |

#### wafer-canvas count check

| class | expected | notes |
|---|---:|---|
| DiagonalSmear, CrossScratch, CrescentArc, ParallelScratches, BrokenRing, RingDots, CenterDonut, Row, Starburst, CenterCircle | N each | 10 class × N |

#### 시각 sanity

- 1 wafer per class 1024 down preview (sample)
- chip 경계 seam 없음 (bank chip 본체 = wafer pink slice 연속 확인)
- RingDots 위치 동일 (다른 seed 비교)
- wafer 밝기 [floor=0.22, cap=0.42] linear spread (사용자 visual)

### 4. 보고

- 합성 시간, rate (wafer/s), per-class count
- 절대 경로 (`[OUT]` 마지막 줄): unknown/ + positions/unknown/
- 부분 / 본 합성 구분
- VERSION_HISTORY / SPEC v5.x row 업데이트 트리거 (별도 chip-wafer-version-mgmt skill)

## 절대 영구 원칙

1. palette PNG only (mode='P', grade 0-7) — 6400×6400.
2. RingDots 위치 fixed (R×0.55, 18 dots, th_off=0).
3. wafer pink uniform [0.22, 0.42] linear (Beta clip 폐기 v5.2).
4. bank_boundary chip 경계 seam 영구 제거 (independent sample + pink_baseline).
5. 양 repo mirror — pre-check diff 통과 안 하면 abort.
6. CPU multiproc 우선 — GPU pipeline 의 NEW_WAFER_CANVAS_CLASSES legacy mismatch 주의.

## skill 참조

`.claude/skills/wafer-synth/SKILL.md` quick reference.

## version mgmt 협조

대규모 spec 변경 (smoothstep/floor/cap/budget 등) 시 `chip-wafer-version-mgmt` skill +
`chip-wafer-regen` agent 와 협조 — 별도 version tag 부여 + VERSION_HISTORY.md update.
