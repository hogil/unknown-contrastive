---
name: pixel-design
description: 합성 wafer chip-level grade 분포·alpha 함수·클래스 설계. WM-811K cca/* 분포는 학습된 heatmap 사용. 모든 수치는 docs/image-generation/SPEC.md 기준.
---

# pixel-design skill

이 스킬은 wafer 합성 데이터셋의 **설계 단계** — chip 내부 grade 분포, alpha 공간장,
클래스 정의, 미세 파라미터 조정을 담당.

## 가장 먼저 읽기

| 문서 | 용도 |
|---|---|
| `docs/image-generation/SPEC.md` | 모든 수치 (palette, baseline/edge/object 분포, alpha 함수, sigma 값) |
| `docs/image-generation/CLASSES.md` | 36 클래스 + 5 chip object 설계 근거 |
| `docs/image-generation/PIPELINE.md` §3, §6 | chip selection algorithm + alpha mixing 식 |
| `docs/image-generation/README.md` | 핵심 멘탈 모델 (확률적 픽셀 밀도장) |

## 핵심 멘탈 모델

**Wafer fail-bit 데이터의 본질** = 확률적 픽셀 밀도장. 깔끔한 기하학적 그림이 아니라
Bernoulli grade categorical distribution이 chip 내부 위치에 따라 공간적으로
modulation되는 노이즈 합성.

- 정상 chip: P(0)+P(1)≈98%, P(2-7) ≪ 1%
- 불량 chip: chip 내부에 alpha 공간장 정의 → 라인/blob 중심에 grade 1+ 픽셀 밀집

## 설계 단계

### 1. Wafer-level distribution
WM-811K cca/*에서 학습된 heatmap (`_dist_heatmaps/<Class>_p_defect_32.npy`) 또는
사용자 정의 (예: Thick-Edge — 외곽 60% 영역).

8 classes: Center, Donut, Edge-Ring, Edge-Loc, Loc, Random, Near-full, Thick-Edge.

### 2. Chip-level object (alpha field)
5 종류: bank_boundary, particle_blast, scratch, scratch_21deg, invalid_main.

각 object의 alpha 함수는 200×200 → float32 [0, 1] map. center=1, 멀어질수록 0.
SPEC.md §6에 모든 함수 정의.

### 3. Grade distribution
3 layers (alpha에 따른 mixing):
- DEFECT_BG_DIST (chip 양호 영역, alpha=0): grade 0 dominant
- EDGE_DIST (zone 가장자리, alpha~0.4): grade 1 dominant
- OBJECT_DISTS[obj] (zone 중앙, alpha=1): main grade dominant

11단계 익스포넨셜 mixing: alpha 0~0.4 (BG↔EDGE 선형) + 0.4~1.0 (EDGE→CENTER power exp).

### 4. Per-object center power (mixing의 중앙 zone 두께)
- bank_boundary: 6
- particle_blast: 4
- scratch: 5
- scratch_21deg: 8 (가장 좁은 center)

값이 높을수록 main grade dominant 영역이 좁음.

## 사용자 피드백 누적 (변경 시 SKILL.md "이미지 generation skill" 표 함께 갱신)

이 시퀀스는 사용자 피드백 반복으로 도달한 spec. 변경 시 visual sample 보고 확인 필수:

| 단계 | 핵심 변경 | 이유 |
|---|---|---|
| v1 | NEAREST upscale | 사용자: "NEAREST 같은 것 안 됨" → 직접 합성으로 |
| v2 | 직사각형 chip grid + bg 색 | 사용자: "동그라미 그리지 마" |
| v3 | invalid chip 산재 | 사용자 추가 요구 |
| v4 | bin 숫자 텍스트 (invalid only) | fail-map docs 매칭 |
| v5 | 9-token 파일명 + JSON 페어 | fq_missing_test 참조 |
| v6 | DEFECT_BG_DIST 분리 (object 보단 grade 1만 elevated) | 양호 영역 normal 비슷하게 |
| v7 | 3-way mixing (BG/EDGE/CENTER) | 가운데일수록 main grade 비율↑ |
| v8 | 11단계 익스포넨셜 + 객체별 center_power | 더 많이 세분화 + 가운데 밀도↑↑ |
| v9 | bank_boundary Y축 산포 (10 segments) | "라인이 균일하지 않게" |
| v10 | EDGE_DIST P(1) 75→40, BG↔EDGE 0.10-0.20→0-0.40 | "양호 영역과 절벽 부드럽게" |
| v11 | bank_boundary sigma 변경 (0.7/3.0/12.0) | center 1/4 폭, line halo 부드럽게 |
| v12 | scratch 5-15 lines, scratch_21deg 12-18 균일 간격 | "scratch는 적게 불균일, 21deg는 많이 균일" |
| v13 | Thick-Edge_invalid_main 추가 | "외곽 매우 두껍게 한 클래스" |

## 금지

- baseline / DEFECT_BG / EDGE / OBJECT 분포 무근거 변경 금지
- alpha 함수 sigma 무근거 변경 금지
- 사용자 피드백 누적된 위 spec 임의 회귀 금지
- WM-811K 원본 cca/* 분포 데이터 수정 금지 (학습 데이터)
