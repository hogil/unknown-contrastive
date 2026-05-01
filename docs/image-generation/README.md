# Wafer Synthetic Image Generation — Design Overview

이 문서는 contrastive learning용 wafer fail-bit 이미지를 처음부터 합성하는
파이프라인 전체를 다음 세션에서도 그대로 재현할 수 있도록 상세히 기술한다.

## 위치

| 파일 | 역할 |
|---|---|
| `docs/image-generation/README.md` | 이 파일 (개요·인덱스) |
| `docs/image-generation/SPEC.md` | 캔버스/그리드/팔레트/분포 수치 사양 |
| `docs/image-generation/PIPELINE.md` | 렌더 알고리즘 단계별 세부 |
| `docs/image-generation/CLASSES.md` | 35개 클래스 정의표 + 기본 chip object 5종 |
| `docs/image-generation/OUTPUT.md` | PNG 파일명 + positions JSON 스키마 |
| `_sample_gen.py` (repo root) | 합성 generator (실행 가능) |
| `_fq_metadata.py` (repo root) | synthetic `partid`/`pgm` + FTN/QTN 생성 규칙 |
| `_backfill_fq_positions.py` (repo root) | 기존 positions JSON에 FTN/QTN backfill |
| `_dist_learn.py` (repo root) | WM-811K 분포 heatmap 추출기 |

## 핵심 아이디어

**fail-bit 데이터의 본질은 확률적 픽셀 밀도장**이다. 깔끔한 기하학적 그림이
아니라 (Bernoulli grade) categorical distribution이 chip 내부 위치에 따라
공간적으로 modulation되어 발생하는 노이즈 합성이다.

실제 wafer:
- 대부분의 chip은 정상이고 정상 chip의 픽셀 분포는 grade 0(정상) ≈ 83%,
  grade 1 ≈ 15%, 그 외 < 2%
- 일부 chip이 결함을 보이며 chip별 결함 모양은 "object class"로 분류됨
  (bank 경계 결함, particle 폭발, scratch 등)
- 어느 chip이 결함인지 wafer 좌표상의 분포가 또 다른 클래스 라벨을 형성
  (Center / Donut / Edge-Ring / Edge-Loc / Loc / Random / Near-full)

따라서 클래스 = (wafer 분포 패턴) × (chip object 패턴) 조합.

## 핵심 수치

| 항목 | 값 |
|---|---|
| 캔버스 | 6400 × 6400 px (palette PNG) |
| Chip grid | 32 × 32 (= 1024 cells; 원형 wafer 안 ~803 cells) |
| Chip size | 200 × 200 px |
| Chip 내부 bank 분할 | 3 vertical + 1 horizontal = 8 banks (50/100/150 vertical, 100 horizontal) |
| Palette | fail-map 32-color (chip0..7 + bg + text + border + bin borders) |
| Baseline grade dist | P=[0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001] |
| Wafer 분포 7종 | Center, Donut, Edge-Ring, Edge-Loc, Loc, Random, Near-full (WM-811K cca/* heatmap에서 학습) |
| Chip object 5종 | bank_boundary, particle_blast, scratch, scratch_21deg, invalid_main |
| Total class 수 | 7 × 5 = 35 |
| Sample per class | 사용자 지정 (기본 1) |

## 빠른 실행

```bash
cd D:/project/unknown-contrastive

# 1. WM-811K 분포 heatmap 추출 (1회)
python _dist_learn.py
# → _dist_heatmaps/<Class>_p_defect_32.npy + .png

# 2. 35개 sample 생성 (PNG + JSON 동시)
python _sample_gen.py
# → D:/project/data/wm-811k/unknown/<class>/<filename>.png
# → D:/project/data/positions/unknown/<class>/<filename>.json

# 기존 JSON에 FTN/QTN이 빠져 있으면 보정
python _backfill_fq_positions.py
```

## 외부 참조 (필독)

| 경로 | 역할 |
|---|---|
| `D:/project/fail-map/` | palette 정의·테두리 규칙·파일명 컨벤션 원본 |
| `D:/project/fail-map/docs/png-filename.md` | 파일명 9-token 스키마 |
| `D:/project/fail-map/docs/palette-and-borders.md` | 32색 팔레트 인덱스·BIN별 테두리 규칙 |
| `D:/project/fail-map/docs/positions-json.md` | positions JSON 스키마 |
| `D:/project/fail-map/utils.py::build_palette` | palette bytes 생성 함수 |
| `D:/project/data/wm-811k/cca/<class>/*.png` | WM-811K class별 wafer 분포 학습용 입력 |
| `D:/project/data/positions/fq_missing_test/` | positions JSON 참조 샘플 |

수정 금지:
- `fail-map/` repo (읽기 전용)
- `mapviewer/` repo (읽기 전용 — composite 공식 원본)

## 다음 단계

- 분포·object·class 추가 시 `SPEC.md` + `_sample_gen.py`만 수정
- 새 chip object 추가: `alpha_<name>` 함수 + `OBJECT_DISTS[<name>]` + `OBJECT_BIN_PREF[<name>]` + `OBJECTS` 리스트에 등록
- 새 wafer 분포 추가: `_dist_learn.py`로 heatmap 만들고 `CLASSES`에 등록
