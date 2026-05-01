---
name: image-generation
description: WM-811K 분포 + chip object 5종을 합성해 36 클래스 wafer fail-bit palette PNG와 positions JSON을 일괄 생성한다. _sample_gen.py 사용, 모든 수치 사양은 docs/image-generation/에 정의.
---

# image-generation skill

이 스킬은 새 세션에서도 이미지를 동일하게 재생성할 수 있도록 docs/image-generation/의
세부 문서를 참조하면서 동작한다.

## 가장 먼저 읽기 (필수)

| 문서 | 읽을 시점 |
|---|---|
| `docs/image-generation/README.md` | **최초 1회** — 전체 개요·인덱스 |
| `docs/image-generation/SPEC.md` | 모든 수치 (캔버스 6400×6400, 32×32 grid, 200×200 chip, palette, baseline/edge/object 분포, alpha 함수) |
| `docs/image-generation/PIPELINE.md` | render() 알고리즘 단계별 설명 (canvas init, defect modulation, invalid fill, border, text, yield/sys 계산) |
| `docs/image-generation/CLASSES.md` | 36 클래스 enumeration (7 wafer dist × 5 chip obj + Thick-Edge_invalid_main) |
| `docs/image-generation/OUTPUT.md` | 9-token 파일명 + JSON schema |

## 핵심 사양 (한눈)

- 캔버스: 6400×6400 palette PNG, 32×32 chip grid, chip 200×200
- baseline (normal chip): P(0)=0.83, P(1)=0.15 — 압도적으로 0+1
- defect zone: 3-way 분포 (DEFECT_BG → EDGE → OBJECT) + 11단계 익스포넨셜 power 가중
- 36 classes: Center/Donut/Edge-Ring/Edge-Loc/Loc/Random/Near-full/Thick-Edge × bank_boundary/particle_blast/scratch/scratch_21deg/invalid_main (Thick-Edge는 invalid_main만)
- 출력: `D:/project/data/wm-811k/unknown/<class>/*.png` + `D:/project/data/positions/unknown/<class>/*.json`
- positions JSON은 반드시 synthetic `partid`, `pgm`, `ftn_keys`, `qtn_keys`, chip별
  `f`/`q`를 포함한다. `_fq_metadata.py` 사용. 클래스별 hot FTN/QTN item은
  `b >= 200` defect/invalid chip 위치와 주변에서 크게 나오게 해 fail-bit 분포와 맞춘다.
  **목적: FTN/QTN ↔ fail-bit cross-correlation 분석.** 클래스마다 hot index 셋이
  다르고 (class-specific signature), defect chip에서 hot item 값이 normal chip 대비 약 4-5배.
  새 클래스 추가/spec 변경 시 이 분석성(클래스 식별 + 공간 패턴 매칭)을 검증한 뒤 commit.

## 사전 조건

1. WM-811K heatmap: `_dist_heatmaps/<Class>_p_defect_32.npy` 8개 클래스
   (repo에 포함됨, fresh clone에서도 즉시 사용 가능). Thick-Edge는 heatmap
   불필요, 코드에서 직접 계산.
2. WM-811K 원본: `D:/project/data/wm-811k/cca/<Class>/*.png` 존재
3. 출력 폴더 쓰기 권한

## 실행

```bash
cd D:/project/unknown-contrastive

# 작은 테스트 (1 sample per class = 36장, ~2분 with 4 workers)
python _sample_gen.py --n 1 --workers 4

# 본 생성 (200 per class = 7200장, ~3-4시간 with 8 workers)
python _sample_gen.py --n 200 --workers 8
```

CLI 옵션:
- `--n N`: 클래스당 샘플 수 (default 200)
- `--workers W`: 병렬 worker (default 4)

새 generation 결과는 `_fq_metadata.add_synthetic_fq_to_json` 통해 partid/
part_id/pgm/ftn_keys/qtn_keys/chip f·q를 자동 포함한다 (FTN 128 + QTN 128).
기본값 변경 시 `_fq_metadata.DEFAULT_FQ_ITEM_COUNT` 조정 후 재생성.

## 클래스 매트릭스 변경

새 wafer distribution 추가:
1. WM-811K 원본 폴더에서 heatmap 학습 → `_dist_heatmaps/<NewClass>_p_defect_32.npy`
   추가 (학습 코드는 git history `441c532` 이전 `_dist_learn.py` 참고)
2. `CLASSES` 리스트에 등록 + `DEFECT_BUDGET`에 chip 수
3. `select_distribution_chips()`에 분기 추가 (heatmap 기반이면 자동)

새 chip object 추가:
1. `alpha_<name>` 함수 정의 (200×200 → float32)
2. `OBJECT_DISTS[<name>]`: zone 중앙 grade 분포
3. `PRIMARY_GRADE[<name>]`: (main, sub) 메타
4. `OBJECTS` 리스트에 등록
5. mixing 단계의 `center_power` dict에 추가 (좁은 center 원할수록 높은 power)

세부 사양은 `docs/image-generation/CLASSES.md` 끝부분 "추가 angle scratch 클래스" 절 참고.

## 파라미터 미세조정 (사용자 피드백 누적 결과)

| 파라미터 | 값 | 이유 |
|---|---|---|
| BASELINE | [0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001] | normal chip 0+1 압도적 |
| DEFECT_BG_DIST | [0.73, 0.25, 0.012, 0.005, ...] | 양호 영역 P(1) 살짝↑ |
| EDGE_DIST | [0.50, 0.40, 0.07, 0.02, ...] | zone 끝 grade 1 너무 높지 않게 |
| OBJECT_DISTS center | main grade 80% | "거의 대부분" |
| BG↔EDGE transition | alpha 0~0.40 (선형) | smooth, 절벽 X |
| EDGE→CENTER transition | alpha 0.40~1.0 (power exp) | 가운데 갈수록 main grade 급증 |
| center_power | bank=6, particle=4, scratch=5, scratch_21deg=8 | object별 center zone 두께 |
| bank_boundary sigma | (0.7, 3.0, 12.0) | center 좁고 line 폭 부드러움 |
| scratch sigma | (1.0, 2.0, 4.0) | 얇은 라인, 5-15개 random |
| scratch_21deg sigma | (0.7, 1.5, 3.0) | 가장 얇은 라인, 12-18개 균일 |
| Y축 산포 (bank_boundary) | 10 segments × U(0.55, 1.0) | 라인 위아래 산포 |
| invalid 위치당 개수 | 15 random + 외곽 자동 | 시각 산만 줄임 |
| chip border 색 | bin별 (낮은 번호 가중치 6:5:4:3:2:1) | 다양하게 |

## 금지

- `BASELINE`, `DEFECT_BG_DIST`, `EDGE_DIST`, `OBJECT_DISTS` 무근거 변경 금지 (사용자 피드백 누적된 값)
- `_sample_gen.py`에서 transparency=31 추가 금지 (모델 입력 위해 모든 픽셀 가시 색)
- 32-color palette 변경 금지 (fail-map 호환)
- 결과 폴더(`D:/project/data/wm-811k/unknown`, `D:/project/data/positions/unknown`) 삭제 금지

## 검증

생성 직후 `image-verification` skill로 결과 검증.
