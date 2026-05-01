---
name: composite-map
description: cluster 결과를 읽어 cluster별 top-K medoid 기반 composite PNG를 생성한다. 기존 common/composite.py 공식 그대로, 수정 금지.
---

# composite-map — Cluster 대표 composite PNG 생성

## 목적

stage 3이 만든 cluster 폴더에서 각 cluster의 대표 wafer N장을 골라 fail-map
composite PNG (Σ count·g² / N 기반 base + gradient overlay)를 생성한다.

## 입력

- `outputs_<preset>_<ts>/clusters/hdbscan/cluster_XXX_size_YYY/` (stage 3 산출)
- `outputs_<preset>_<ts>/centroids/centroids.npy` (medoid 선정용)

## 산출물

`outputs_<preset>_<ts>/cluster_summary/composite/cluster_XXX_composite.png`:
- RGB PNG (palette LUT base + 4-stop gradient overlay).
- `common/composite.py::render_composite_png` 포맷 그대로.

## 수행 절차

이 stage는 기존 코드 래퍼.

1. `cluster_composite.py` (또는 `contrastive.py::main()` 내부 composite 로직)를
   호출.
2. 각 cluster 폴더에서 top-K medoid 선정 (K=10 기본, `centroids.npy`와
   cosine 거리 최소).
3. `common.palette_io.load_palette_indices` → `common.composite.compute_grade_counts`
   → `compute_base_indices` + `render_composite_png`.
4. 저장 경로에 클러스터 ID 포함.

## 규칙 (금기)

- **`common/composite.py` 공식 수정 금지**. mapviewer의 `api/composite_map.py`와
  바이트 단위 호환 유지. 공식 바꾸면 composite 호환성 깨짐.
- **numba/libvips/turbojpeg 의존 추가 금지**. pure numpy+PIL 유지.
- **기존 composite 덮어쓰기 전 확인**. `outputs_*/`는 글로벌 삭제 금지 폴더.
- **K > cluster_size**인 경우 가능한 만큼만 사용 (silent에서 경고로 변경).

## 검증 기준

- `cluster_XXX_composite.png` 각 cluster에 대해 존재.
- 이미지 크기 = (원본 wafer 크기와 동일) 또는 고정 (`common/composite.py` 내부 규약).
- PNG mode = 'RGB'.

## 예외 케이스

- `cluster_-1` (noise)은 기본 skip. 필요 시 명시 옵션으로 포함.
- cluster_size == 1인 경우 medoid == sample. composite == 원본 (정상 동작).

## 다음 stage

없음. 파이프라인 종단 stage.
