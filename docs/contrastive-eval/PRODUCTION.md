# PRODUCTION — 실제 운영 시나리오 + class imbalance 처리

## 실제 production 시나리오 (사용자 정정)

```
검사장비 → wafer 10000장 입력
  ├─ 정상 (defect 0)              ≈ 8000장 (80%)
  └─ Defect                       ≈ 2000장 (20%)
      ├─ defect_a (1차)             ≈ 1500장
      ├─ defect_b (드물게)           ≈ 300장
      ├─ defect_c (희귀)             ≈ 200장
      └─ unknown defect (새로운 패턴)  ≈ 가끔 등장

Label 가능: 1% 미만 (≈ 100장 정도, defect 안에서도 일부만)
```

→ **Class imbalance + heavy unlabeled + unknown defect 출현 가능**.

## 합성 (test) 시나리오 vs 실제 production 차이

| | 합성 (현재 overall) | 실제 production |
|---|---|---|
| Normal 비율 | 1000/8357 ≈ 12% | **80%** |
| defect class 수 | 38 (균등) | 5~10 (극히 imbalanced) |
| 가장 큰 defect class | 200 | 1500 |
| 가장 작은 defect class | 50 | 50~100 |
| Label 가능 비율 | 100% | **<1%** |
| Unknown defect | 없음 | **있을 수 있음** |

**시사점**: 합성 환경의 metric 수치가 production 에 직접 transfer 안 됨. 비율은 비슷하게 흉내 낼 수 있음.

## 학습 sampling 정책 (다음 학습부터)

### 사용자 명시: per-class sample 수 random (50 ~ 200+)

균등 200/class 가 합성 환경 비현실적. 다음 학습 dispatch 시:
```
class A: 200
class B: 80
class C: 50
class D: 150
class E: 30
... random sample 분포
```

### Production 비율 흉내 (더 현실적)

```
Normal: 학습 set 의 70~80%
Defect total: 학습 set 의 20~30%
defect 안에서도 imbalanced (e.g., 1 class 가 50%, 나머지 분산)
```

권고:
- Normal undersample (production 비율 그대로 학습 시 8000 normal × 4 augment = 너무 많음). 1000 ~ 2000 정도면 충분.
- Defect oversample 또는 weight ↑ (작은 class 가 학습에서 사라지지 않게)

## Class imbalance 대응 (학습 측)

### Sampling
- **Class-balanced sampler** (PyTorch `WeightedRandomSampler`): 각 class 가 균등 빈도로 batch 등장
- **Defect-only oversample**: defect class 들 에 대해서만 균등 sampling, normal 은 random

### Loss weight
- InfoNCE 자체에는 class weight 직접 못 넣음 (class 무관 contrastive)
- HDBSCAN 후 Hungarian matching 또는 cluster purity 분석 시 class weight 사용

### Hard negative mining (별도 docs)
- 작은 class 가 batch 에 자주 안 나오면 negative 로도 적게 등장 → encoding 학습 약함
- Hard mining 으로 가까운 negative 강조 → 작은 class 도 잘 학습

## Unknown defect 대응

### SSL 유지 정책 (사용자 결정)

**SupCon (Supervised Contrastive) 주력 X**:
- SupCon 은 label 직접 사용 → 학습 본 class manifold 만 sharp
- 새 unknown defect 들어오면 known class 로 끌려감 → cluster 안 만듦
- production 에 unknown defect 가능성 있으니 SSL 유지

**대안 — 2-stage hybrid (옵션)**:
1. Stage 1: SSL (현재) — general embedding
2. Stage 2 (옵션): known class 만으로 가벼운 SupCon fine-tune (LR 매우 작게, 1-2 epoch)
   - Full_*** sub-style split 같은 약점 해결
   - LR 작아 unknown defect generalization 손상 최소화

## Inference 시 unknown defect 검출

학습된 encoder + HDBSCAN 후:
- Unknown defect 는 작은 cluster (기존 medoid 와 거리 멀음) 또는 noise (-1) 로 분류 됨
- Operator 가 medoid 검토 시:
  - 작은 cluster (size < 10) → 새 defect 후보
  - noise sample 의 새 패턴 → 새 defect 후보
- Action: 새 cluster 생기면 라벨링 후 spec 추가

## Label 부족 처리 — Partial Label SSL

production 에서는 label 1% 미만:
- Tier 1 metric 평가 시 labeled subset 만 사용
- 같은 cluster 안 unlabeled wafer 는 cluster medoid 와 라벨 추론 (semi-supervised)
- 정기적 active learning: 모호한 cluster 먼저 라벨링 우선

## 평가 시 normal 처리

운영 관점:
- **정상 (Normal) 이 defect cluster 에 들어가면 false alarm**
- **defect 가 normal cluster 에 들어가면 missed detection (가장 위험)**

평가 metric 는 이 두 가지 모두 자연스럽게 penalize:
- Completeness 낮으면 한 class 가 여러 cluster 에 흩어짐 (recall 약함)
- Homogeneity 낮으면 cluster 안 다른 class 섞임 (precision 약함)
- noise_pct (defect only) 높으면 defect 격리 실패

## 절대 룰

- production data 는 절대 학습에 직접 노출 X (privacy / 안정성). 합성으로 대리 검증.
- unknown defect 출현 시 모델 retrain 전까지 cluster -1 (noise) 로 격리 정상.
- false alarm rate 보다 missed detection 이 더 위험 — recall (P1) > precision (P2) 우선.

## 참고

- 합성 데이터 spec 검토 (Full_*** sub-style split 발견) → `DECISIONS.md`
- Hard negative mining 도입 → `HARD_NEGATIVE.md`
- 학습 monitoring → `MONITORING.md`
