# WM-811K 분석 및 pixel 규칙 설계

- 원본 pickle: `data_raw\LSWMD.pkl` (2.10 GB)
- 분석 일시: 자동 생성, seed=42
- 목표 이미지 크기: 4000×4000 (NEAREST upscale)

## 1. Class 분포 통계

| Class | 사용 가능 | 유효 2D | H median | W median | Upscale factor | Defect 비율 (mean / median / max) |
|---|---:|---:|---:|---:|---:|---|
| Center | 4294 | 4294 | 25 | 27 | 160.0× | 0.232 / 0.250 / 0.727 |
| Donut | 555 | 555 | 41 | 42 | 97.6× | 0.277 / 0.258 / 0.678 |
| Edge-Loc | 5189 | 5189 | 36 | 34 | 111.1× | 0.183 / 0.156 / 0.746 |
| Edge-Ring | 9680 | 9680 | 53 | 52 | 75.5× | 0.151 / 0.140 / 0.451 |
| Loc | 3593 | 3593 | 35 | 34 | 114.3× | 0.150 / 0.142 / 0.698 |
| Near-full | 149 | 149 | 33 | 29 | 121.2× | 0.877 / 0.878 / 1.000 |
| Random | 866 | 866 | 35 | 38 | 114.3× | 0.481 / 0.477 / 0.771 |
| Scratch | 1193 | 1193 | 41 | 38 | 97.6× | 0.101 / 0.092 / 0.393 |
| none | 147431 | 147431 | 33 | 31 | 121.2× | 0.106 / 0.098 / 0.471 |

## 2. Upscaling 방법: NEAREST

원본 wafer 의 중앙값 크기는 class 별로 다르며(표 참고) 4000×4000 으로 upscale 할 경우 upscale factor 가 일반적으로 **75× ~ 154×** 범위. Palette PNG 는 각 픽셀이 `{0..7, 8..13, 31}` 중 하나의 discrete index 를 가지므로 BILINEAR/BICUBIC 등 interpolation 은 palette 에 없는 중간 값을 생성해 계약을 깨뜨림. 따라서 `Image.resize(..., Image.NEAREST)` 를 **유일 허용** 방식으로 사용한다.

부산물: die 블록이 정사각형으로 단순 확대되므로 defect 시각화 크기가 class 별로 다름. fail-map composite pipeline 과 호환.

## 3. Grade 분포 설계

WM-811K 원본 die 값 `{0=outside, 1=normal, 2=defect}` 를 fail-map palette `{31=transparent, 0=Grade0, 1..7=Grade1..7}` 로 매핑한다.

### 3.1 고정 매핑 (재협상 불가)

- `outside (0) → palette 31`: fail-map transparency 규약 (`common/palette_io.py`).
- `normal (1) → palette 0`: Grade0 이 정상 die 표준.

### 3.2 불량 → grade 1..7 분포 후보 비교

| Grade | Exponential ``1/2^(k-1)`` | Power-law ``1/k^1.5`` | Linear ``8-k`` |
|---:|---:|---:|---:|
| 1 | 50.39% | 53.12% | 25.00% |
| 2 | 25.20% | 18.78% | 21.43% |
| 3 | 12.60% | 10.22% | 17.86% |
| 4 | 6.30% | 6.64% | 14.29% |
| 5 | 3.15% | 4.75% | 10.71% |
| 6 | 1.57% | 3.61% | 7.14% |
| 7 | 0.79% | 2.87% | 3.57% |

**선택: Exponential decay.** 근거 3 가지:
1. **현실성** — 실제 wafer 에서 grade 7 (조밀 결함 cluster 내부 die) 는 전체 결함 의 1% 미만이다. Exponential 은 이 희귀성을 직접 반영한다. Linear 는 grade 7 을 3.6% 로 부풀려 과다 생성.
2. **학습 안정성** — Contrastive learning 에서 rare grade 를 과다 생성하면 해당 texture 가 실제 분포와 괴리되어 역효과. Exponential 의 급속 decay 는 grade 5-7 을 소수 guaranteed minority 로 유지.
3. **재현성** — `1/2^(k-1)` 은 명시적 공식, 소수점 6자리 고정. 서로 다른 실행 환경 에서 동일 weight 재생산 가능.

최종 weights (소수점 6자리): `[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]`
재정규화 확률 (합=1): `[0.503937, 0.251969, 0.125984, 0.062992, 0.031496, 0.015748, 0.007874]`

## 4. Split 구성 및 풀 충분성

- **train**: defect 50 × 8 + normal 1600 = 2000 (defect:normal = 400:1600 ≈ 1:4.0)
- **val  **: defect 20 × 8 + normal 640 = 800 (defect:normal = 160:640 ≈ 1:4.0)
- **test **: defect 20 × 8 + normal 640 = 800 (defect:normal = 160:640 ≈ 1:4.0)

| Class | 사용 가능 | 필요 (train+val+test) | OK? |
|---|---:|---:|:---:|
| Center | 4294 | 90 | OK |
| Donut | 555 | 90 | OK |
| Edge-Loc | 5189 | 90 | OK |
| Edge-Ring | 9680 | 90 | OK |
| Loc | 3593 | 90 | OK |
| Near-full | 149 | 90 | OK |
| Random | 866 | 90 | OK |
| Scratch | 1193 | 90 | OK |
| none | 147431 | 2880 | OK |

> 모든 class 의 pool 이 요청 샘플 수를 충족한다. Stage 2 실행 가능.

## 5. 산출 YAML 요약

```yaml
version: 1
seed: 42
size:
- 4000
- 4000
upscale: nearest
mapping:
  outside: 31
  normal: 0
  defect:
    mode: random_skewed
    grades:
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    - 7
    weights:
    - 1.0
    - 0.5
    - 0.25
    - 0.125
    - 0.0625
    - 0.03125
    - 0.015625
texture:
  mode: synthetic_scatter
  defect_perturb_p: 0.3
  normal_scatter_p: 0.03
  normal_scatter_grades:
  - 1
  global_scatter_n: 0
  global_scatter_radius:
  - 2
  - 6
split:
  train:
    defect_n_per_class: 50
    normal_n: 1600
    total: 2000
  val:
    defect_n_per_class: 20
    normal_n: 640
    total: 800
  test:
    defect_n_per_class: 20
    normal_n: 640
    total: 800
paths:
  pkl: data_raw/LSWMD.pkl
  train_out: data/wm811k_train
  val_out: data/wm811k_val
  test_out: data/wm811k_test
```

## 6. 금기 재강조

- BILINEAR / BICUBIC upscale 금지 (palette 깨짐).
- Weight 를 데이터 분석 없이 임의 조정 금지 — 본 문서 근거에 입각.
- Train / Val source wafer 복원추출 금지.
- 기존 `configs/pixel_rules.yaml` 덮어쓰기 전 diff 확인 권장.
