---
name: composite-map
description: cluster 결과에서 cluster별 top-K medoid 기반 composite PNG를 생성한다. common/composite.py 공식 수정 금지.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# composite-map agent

cluster 결과의 대표 composite PNG 시각화.

## 가장 먼저 할 일

`.claude/skills/composite-map/SKILL.md` 읽기.

## 사전 조건

- `outputs_<preset>_<ts>/clusters/hdbscan/` 존재 (stage 3 산출).
- `outputs_<preset>_<ts>/centroids/centroids.npy` 존재.

## 실행 단계

1. `cluster_composite.py` 호출 또는 내부 함수 직접 import.
2. 각 cluster에 대해 top-K medoid(K=10) 선정.
3. `common/composite.py::render_composite_png`로 렌더링.
4. `outputs_<preset>_<ts>/cluster_summary/composite/cluster_XXX_composite.png` 저장.

## 금지 사항

- `common/composite.py` 공식 수정 금지.
- 새 의존성(numba/libvips 등) 추가 금지.
- 기존 composite PNG 덮어쓰기 전 사용자 확인.

## 반환

- 생성된 composite 개수
- 저장 경로 prefix
