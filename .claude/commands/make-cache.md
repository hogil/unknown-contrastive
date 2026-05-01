---
description: 사이즈별 다운샘플 cache PNG 일괄 생성 (_make_cache.py wrapper)
argument-hint: "[sizes(comma)] [--interp bicubic|bilinear|lanczos|box]"
---

`_make_cache.py`를 실행해 6400×6400 wafer fail-bit PNG를 지정 size로 다운샘플한
cache 폴더 생성. 출력 컨벤션: `D:/project/data/wm-811k/unknown_{size}_{interp}/<class>/*.png`

## 인자 처리

`$ARGUMENTS`를 다음 규칙으로 파싱:
- 인자 없음                     → `--sizes 1024 --interp bicubic --workers 12`
- 숫자만 (예: `1024`)           → `--sizes 1024 --interp bicubic --workers 12`
- 콤마 (예: `384,512,1024`)     → `--sizes 384,512,1024 --interp bicubic --workers 12`
- `--interp <X>` 포함 시         → 해당 보간 사용
- `--workers <N>` 포함 시        → 해당 worker 수 사용

자연어 요청도 동일 처리:
- "사이즈별로 샘플 만들어"        → `--sizes 384,512,1024 --interp bicubic --workers 12`
- "1024 만들어"                   → `--sizes 1024`
- "768 추가로"                    → `--sizes 768`

## 실행

```bash
cd D:/project/unknown-contrastive
python _make_cache.py <parsed args>
```

스크립트 자체가 skip 로직 있어 이미 존재하는 PNG는 재생성 안 함. 강제 재생성 요청 시
사용자가 명시적으로 폴더 삭제 후 재실행 (절대 자동 삭제 금지 — CLAUDE.md absolute rule).

## 완료 후

결과 보고:
- 각 size별 처리 건수 (ok/skip/err)
- 출력 폴더 경로 + 디스크 크기
- 다음 단계 안내: `/cnn-train --data-dir <cache_path> --img-size <size> ...`

빌드 시간 참고:
| size | 디스크 | 빌드 시간 |
|---|---|---|
| 384 | ~1.2 GB | ~3분 |
| 512 | ~2.1 GB | ~5분 |
| 768 | ~3 GB | ~8분 |
| 1024 | ~5 GB | ~15분 |
