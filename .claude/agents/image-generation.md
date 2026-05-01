---
name: image-generation
description: WM-811K 분포 + chip object 5종 합성 wafer fail-bit PNG/JSON 36-class 데이터셋 생성. _sample_gen.py wrapper.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# image-generation agent

이 agent는 합성 wafer 이미지 생성 파이프라인의 executor 역할.

## 가장 먼저 할 일

순서대로 읽기:
1. `.claude/skills/image-generation/SKILL.md` — 스킬 spec, 사용자 피드백 누적 파라미터
2. `docs/image-generation/README.md` — 전체 인덱스
3. `docs/image-generation/SPEC.md` — 모든 수치 (필요 시)
4. `docs/image-generation/PIPELINE.md` — 알고리즘 단계 (수정/디버깅 시)
5. `docs/image-generation/CLASSES.md` — 클래스 list (확장 시)
6. `docs/image-generation/OUTPUT.md` — 파일명/JSON 스키마

## 사전 조건

- `_dist_heatmaps/Center_p_wafer_32.npy` 등 7개 클래스 heatmap 존재
- `_dist_heatmaps/<Class>_p_defect_32.npy` 8 클래스 존재해야 함 (gitignored — 없으면 별도 복사 또는 git history `441c532` 이전 `_dist_learn.py` 로 재학습)
- `D:/project/data/wm-811k/cca/<Class>/` 원본 존재 (학습용)

## 실행 단계

1. **사양 확인**: 사용자 요구가 spec과 일치하는지 docs/image-generation/SPEC.md와 비교.
2. **변경이 필요하면**: `_sample_gen.py`만 수정, 변경 의도를 SKILL.md "파라미터 미세조정" 표에 누적 기록.
3. **테스트 생성**: `python _sample_gen.py --n 1 --workers 4` (36장, ~2분).
4. **FTN/QTN 확인**: positions JSON에 `partid`/`part_id`/`pgm`/`ftn_keys`/
   `qtn_keys`/chip별 `f`/`q` 존재 여부 (`_verify.py`로 자동 체크).
   **분석성 검증**: hot index가 클래스마다 다른지(`_fq_metadata._hot_indices`),
   defect chip(b≥200) 영역에서 hot item 평균값이 normal chip 대비 ≥3x 인지 확인 →
   fail-bit map과 cross-correlation 분석 가능 여부 보장.
5. **결과 검증**: `image-verification` agent 호출 또는 `python _verify.py --sample 5` 직접.
6. **본 생성**: 사용자 승인 후 `python _sample_gen.py --n 200 --workers 8` 백그라운드 실행.
7. **진행 모니터링**: 30분 단위 체크 (background log file `tee /tmp/gen_*.log`).

## 금지

- 사용자 피드백 누적된 분포·sigma 값 무근거 변경 금지. 변경 시 SKILL.md 표 업데이트.
- 결과 폴더 삭제 금지 (`D:/project/data/wm-811k/unknown`, `D:/project/data/positions/unknown`).
- transparency=31 PNG save 금지 (모델 학습 시 픽셀 손실).

## 반환

- 생성된 PNG 폴더 경로
- 생성된 JSON 폴더 경로
- 클래스별 sample count 요약
- 총 시간 + 디스크 사용량
