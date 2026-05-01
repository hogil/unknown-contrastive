# Classes — 35 enumerations

Total = **7 wafer distributions × 5 chip objects = 35 classes**.

## Wafer distributions (7)

WM-811K cca/* heatmap에서 학습. 각 분포는 "어느 chip에 결함이 몰리는지"의 공간
패턴이다.

| Class | n samples | max P | mean P | 패턴 | 학습 algorithm |
|---|---:|---:|---:|---|---|
| Center | 290 | 0.886 | 0.070 | 정중앙 hot spot | heatmap weighted |
| Donut | 47 | 0.723 | 0.146 | 중간 ring (정중앙 약함) | heatmap weighted |
| Edge-Ring | 400 | 0.927 | 0.100 | 외곽 가장자리 ring | heatmap weighted |
| Edge-Loc | 382 | 0.273 | 0.071 | 외곽에 random 1 spot | sample-level (Edge-Ring anchor + cluster) |
| Loc | 279 | 0.154 | 0.051 | wafer 내부 random 1 spot | sample-level (interior anchor + cluster) |
| Random | 75 | 0.636 | 0.289 | 전체 산재 | heatmap weighted |
| Near-full | 9 | 1.000 | 0.693 | 거의 전체 | heatmap weighted |

heatmap 파일: `D:/project/unknown-contrastive/_dist_heatmaps/<Class>_p_defect_32.npy` (32×32 float32).

## Chip objects (5)

각 object는 chip 내부 200×200 픽셀에서 결함 픽셀이 어떤 모양으로 분포하는지의
공간장(α(x,y))과 grade 분포(object_dist).

| Object | α 모양 | sigma | peak α | grade dist (P(0)..P(7)) | 시각적 특징 |
|---|---|---|---:|---|---|
| `bank_boundary` | 3 vertical(x=50/100/150) + 1 horizontal(y=100) Gaussian ridges, 길이별 random envelope | 10 | 1.0 | [0.03, 0.30, 0.30, 0.20, 0.10, 0.05, 0.015, 0.005] | bank 경계 격자 모양, grade 1-3 위주 |
| `particle_blast` | 단일 2D Gaussian blob, center+sigma random | 22-35 | 1.0 | [0.02, 0.20, 0.30, 0.20, 0.15, 0.08, 0.04, 0.01] | 동그란 폭발 자국, grade 2-5 위주 (severe) |
| `scratch` | 2-5개 vertical 라인, 각 라인 random x/sigma/y_envelope | 3-5 | 1.0 | [0.02, 0.40, 0.30, 0.15, 0.07, 0.03, 0.015, 0.005] | 날카로운 세로 자국, grade 1-3 |
| `scratch_21deg` | scratch와 동일 패턴, 시계방향 21° 회전 | 3-5 | 1.0 | [0.02, 0.40, 0.30, 0.15, 0.07, 0.03, 0.015, 0.005] | 오른쪽으로 기울어진 scratch |
| `invalid_main` | (α 사용 안 함) | - | - | (chip 전체 = palette idx 31 white) | 모든 chip이 invalid 상태로 클러스터됨 |

추가 angle scratch 클래스를 만들고 싶으면:
- `alpha_scratch_<angle>deg` 함수 (`_sample_gen.py`의 `alpha_scratch_21deg` 그대로 복사 → `theta` 값만 변경)
- `OBJECT_DISTS[<name>]` 추가
- `OBJECT_BIN_PREF[<name>]` 추가 (사용 가능한 bin 골라서)
- `OBJECTS` 리스트에 등록

## 35 classes 전체 목록

```
Center_bank_boundary       Donut_bank_boundary       Edge-Ring_bank_boundary
Center_particle_blast      Donut_particle_blast      Edge-Ring_particle_blast
Center_scratch             Donut_scratch             Edge-Ring_scratch
Center_scratch_21deg       Donut_scratch_21deg       Edge-Ring_scratch_21deg
Center_invalid_main        Donut_invalid_main        Edge-Ring_invalid_main

Edge-Loc_bank_boundary     Loc_bank_boundary         Random_bank_boundary
Edge-Loc_particle_blast    Loc_particle_blast        Random_particle_blast
Edge-Loc_scratch           Loc_scratch               Random_scratch
Edge-Loc_scratch_21deg     Loc_scratch_21deg         Random_scratch_21deg
Edge-Loc_invalid_main      Loc_invalid_main          Random_invalid_main

Near-full_bank_boundary
Near-full_particle_blast
Near-full_scratch
Near-full_scratch_21deg
Near-full_invalid_main
```

폴더 이름은 `<Distribution>_<Object>` 패턴. ImageFolder 호환을 위해 한 클래스당
한 폴더, 그 안에 wafer 이미지 PNG들이 들어감.

## Defect chip count per class

| Wafer distribution | Defect chip count | Notes |
|---|---:|---|
| Center | 25 | 정중앙 cluster |
| Donut | 40 | 도넛 ring |
| Edge-Ring | 70 | 외곽 ring 따라 |
| Edge-Loc | 6 | 외곽 1 localized |
| Loc | 6 | 내부 1 localized |
| Random | 50 | 산재 |
| Near-full | 500 | 거의 전체 |

`invalid_main` object일 때는 위 budget이 invalid chip 개수가 됨.

추가로 모든 wafer마다:
- `non-invalid_main`: 15 chips random scattered invalid
- `invalid_main`: 10 extra chips random scattered invalid

## Object mixing in defect cells

같은 클래스의 wafer 안에서도 모든 defect chip이 동일 object일 필요는 없다.
"primary object 75% + 다른 object 25%" 비율로 섞어서 자연스러움 추가.

`invalid_main` 클래스는 mixing 없음 (모든 결함 = invalid).

## Sample 분배 전략

연구 단계: 클래스당 1장씩 = 35장 (디버깅·시각 확인용).

Production 학습 데이터셋 권장:
- 클래스당 100~300장 (총 3500~10500장)
- Edge-Loc/Loc는 sample마다 위치가 다르므로 다양성 확보 위해 더 많이 (예: 500장)
- Near-full은 9개 원본 sample밖에 없어 학습 시 다양성 부족. 합성으로 부풀리는 게
  주 목적
