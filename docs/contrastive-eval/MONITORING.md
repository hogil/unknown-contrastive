# MONITORING — 학습 도중 quality 추적

## 목적

CNN supervised 와 달리 SSL contrastive 는 학습 중 **collapse / plateau / overfit** 신호가
loss 만으로 안 보임 (loss 가 작아도 모든 embedding 이 한 점 → 학습 망가짐 ≠ 학습 잘 됨).
별도 신호 필요.

## 매 epoch 출력 (필수, label 무관)

### Alignment + Uniformity (Wang & Isola 2020, ICML)

논문 핵심 결과: **contrastive loss = alignment + uniformity** (asymptotic). 두 metric 의
trajectory 만 봐도 학습 quality 추적 가능, GT 불필요.

#### Alignment (positive pair 가까운 정도)

```
L_align = mean over positive pairs (x, x') of:  ||f(x) - f(x')||²
```

- `f(x)` = L2-normalized embedding (unit hypersphere 위 점)
- positive pair = 같은 wafer 의 augmented 두 view
- **낮을수록 좋음**: 같은 wafer 두 view 가 같은 점에 모임

#### Uniformity (embedding sphere 분포)

```
L_unif = log[ mean over random pairs (xi, xj) of exp(-2 · ||f(xi) - f(xj)||²) ]
```

- `xi, xj` = batch 안 임의의 두 wafer (positive 인지 negative 인지 무관)
- `||f(xi) - f(xj)||²` = unit vector squared distance ∈ [0, 4]
- `exp(-2 · 거리²)` ∈ [exp(-8), 1] = [0.0003, 1]
- log 후 → 작은 음수 ~ 큰 음수. **negative loss, 작을수록 좋음**

값 직관:
- 모든 embedding 이 한 점 (collapse): mean ≈ 1, log ≈ **0** (나쁨)
- embedding 이 sphere 골고루: mean ≈ 0.001, log ≈ **-7** (좋음)
- 정상 학습 시 보통 -3.0 ~ -3.5 범위

#### Trajectory 패턴 → 진단

| trajectory | 진단 | 처방 |
|---|---|---|
| align ↓, unif ↓ | ★ 정상 학습 진행 | 그대로 |
| align ↓, unif **→** (평탄) | collapse 시작 — embedding 한 점 모임 위험 | batch ↑ / queue ↑ / temperature ↓ |
| align **→**, unif ↓ | augment 너무 강함 — positive 가 너무 멀어짐 | augment 약화 |
| align ↑ | 학습 깨짐 — augment / LR 문제 | 즉시 stop, hyperparameter 검토 |

## 매 epoch 출력 (옵션, label 있는 작은 subset)

### k-NN top-1

```
val embedding 으로 각 sample 의 nearest neighbor 1 개 분류 → top-1 accuracy
```

- 학습 안 함, 5초 정도 빠름
- val 의 small labeled subset 만 (e.g., 20 samples / class) 가능
- 학습 도중 supervised quality 빠른 check

| trajectory | 처방 |
|---|---|
| ↑ 진행 좋음 | 그대로 |
| 평탄 — plateau | LR ↓ 또는 epoch stop |
| ↓ 하락 | 즉시 stop, 이전 ckpt rollback |

label 없으면 skip — alignment + uniformity 만 사용.

## Periodic (5 epoch 마다, label 있을 때)

학습 중간 HDBSCAN + Tier 1 metric 산출:
- Completeness, AMI, noise_pct (defect only), class_capture_rate
- class_fragmentation_summary 의 frac_single_cluster

확인:
- Completeness 감소 시 → embedding quality 악화 (즉시 검토)
- noise_pct 증가 시 → encoder 격리 능력 저하

5 epoch 단위 충분 — 매 epoch 하면 학습 시간 늘어남 (HDBSCAN cost).

## 출력 형식 (run.log 매 epoch 끝)

```
Epoch 5/20  G=0.85  Q=0.18  L=0.00  align=0.32  unif=-3.10  knn=86.5%
Epoch 10/20 G=0.78  Q=0.18  L=0.00  align=0.28  unif=-3.25  knn=89.2%
[Periodic eval] Completeness=0.92 AMI=0.91 noise_def=1.2% capture=38/38
```

## Production (label 완전 없음)

production 운영 단계에서 새 학습 / 재학습 시:
- alignment + uniformity 만 monitoring (label 무관)
- k-NN, periodic HDBSCAN metric skip
- 또는 cluster distribution 변화 (drift detection) 만 추적

## 참고

- 공식 PyTorch impl: https://github.com/ssnl/align_uniform
- 우리 적용: `contrastive.py::main()` 의 epoch loop 안에 5 줄 추가하면 됨
