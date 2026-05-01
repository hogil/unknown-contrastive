---
name: image-verification
description: 합성 wafer PNG/JSON 36-class 데이터셋이 spec과 일치하는지 검증. 파일명, 팔레트, 크기, JSON 스키마, chip count, 클래스 폴더 구조 체크.
---

# image-verification skill

## 가장 먼저 읽기

| 문서 | 용도 |
|---|---|
| `docs/image-generation/SPEC.md` | palette/canvas/grid/chip 수치 — 검증 기준 |
| `docs/image-generation/OUTPUT.md` | 9-token 파일명 + JSON schema — 검증 핵심 |
| `docs/image-generation/CLASSES.md` | 36 클래스 enumeration — 폴더 매칭 |

## 검증 항목 5가지

### 1. 클래스 폴더 구조
- `D:/project/data/wm-811k/unknown/`에 36 폴더 (7 dist × 5 obj 중 Thick-Edge는 invalid_main만)
- `D:/project/data/positions/unknown/`에 동일 구조

### 2. 파일명 9 tokens (OUTPUT.md §2)
`{prefix6}_{kind}_{wafer:02}_{ymd}_{hms}_{yld:.1f}_{sys:.0f}_{TD}_{LT}.png`
- prefix: 3 lowercase + 3 digits (`^[a-z]{3}\d{3}$`)
- kind: `00P` 또는 `00C`
- wafer: 01-24
- TD: PE/EE/PT, LT: NORMAL/PWQ/ENGINEER

### 3. PNG (SPEC.md §1, §2)
- mode = `P` (palette indexed)
- size = (6400, 6400)
- 32-color palette (96 bytes)
- 파일 크기 2-15 MB

### 4. JSON 페어 (OUTPUT.md §5)
- 동일 basename JSON 존재
- 필수 키: `bucket_b_key, root, step, wafer, stime, netd, gd, yield, sys, tm, lt, coord, chips`
- `chips` 길이 ≈ 803 (inside-wafer count, 허용 700-850)
- `step` (JSON) == `kind` (filename)
- `wafer` (JSON) == "W" + filename wafer 토큰

### 5. 일관성 cross-check
- filename `yld` ≈ JSON `yield` (반올림 허용)
- filename `sys` ≈ JSON `sys`
- filename `TD` == JSON `tester` and `lt`
- filename `LT` == JSON `device` and `tm`

## 실행

```bash
cd D:/project/unknown-contrastive

# 전체 검사
python _verify.py

# 특정 클래스만
python _verify.py --class Center_bank_boundary

# 클래스당 N개 sampling만 (빠른 체크)
python _verify.py --sample 10

# 첫 N개 실패 사례 표시 (default 3)
python _verify.py --show-errors 5
```

## 산출물

- stdout: 클래스별 (count / ok / fail) 표
- 실패 시 첫 N개 사례 (`fname: error1, error2, ...`)
- exit code 0 (전부 OK) / 1 (실패 있음)

## 금지

- 데이터 파일 삭제·이동·재생성 금지 (검증 전용 read-only)
- `_sample_gen.py` 변경 금지 (image-generation skill 영역)
