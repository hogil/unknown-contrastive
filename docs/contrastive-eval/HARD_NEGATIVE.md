# HARD_NEGATIVE — InfoNCE + Hard Negative Mining

## 동기

기본 InfoNCE 의 한계:
- batch 16 wafer + queue 4096 wafer = 4111 negative
- 99% 가 anchor 와 무관한 멀리 떨어진 wafer (e.g., Edge-Top vs Center)
- "다른 거 = 다르다" 학습 task 가 너무 쉬움
- Encoder 가 fine-grained 차이 학습 안 함
- **결과**: Edge-Top × {bank_boundary, fork, scratch, scratch_rot} 4 sub-class merge (cluster 12)

해결: **어려운 (cosine 큰) negative 만 강조해 학습 task 어렵게**.

논문: Robinson et al. 2021, ICLR. "Contrastive Learning with Hard Negative Samples".

## 변수 정의

batch 1개 안 한 anchor 에 대해:

| 변수 | 정의 |
|---|---|
| `a_i` ∈ R¹²⁸ | anchor 의 augmented view 1 의 embedding (L2 normalized) |
| `z_i` ∈ R¹²⁸ | 같은 wafer 의 augmented view 2 의 embedding (positive) |
| `m_1, ..., m_M` | 다른 wafer 들의 embedding (M = N-1 + queue ≈ 4111) |
| τ | temperature, 0.07 (현재) |
| β | hardness 강도, 새 hyperparameter (시작값 1.0) |

## 원본 InfoNCE 한 줄씩

```python
# Step 1: positive cosine similarity
sim_pos = a_i · z_i             # 스칼라, [-1, 1]

# Step 2: 모든 negative cosine similarity
sim_neg_j = a_i · m_j   for j = 1..M

# Step 3: temperature 적용 logit
score_pos = exp(sim_pos / τ)    # τ 작을수록 차이 강조
score_neg_j = exp(sim_neg_j / τ)

# Step 4: softmax 분모 (positive + 모든 negative 평등)
denom = score_pos + Σ_{j} score_neg_j

# Step 5: anchor 의 contrastive loss
L_i = -log(score_pos / denom)
```

해석: positive 가 negative 들보다 가까우면 loss 작음. 모든 negative 평등 취급.

## Hard Mining 변형 (Robinson 2021)

```python
# Step 1: 각 negative 에 weight 부여
β = 1.0    # 새 hyperparameter
w_j = exp(β · sim_neg_j)
       # cosine 큰 (가까운, 어려운) negative 일수록 큰 w
       # cosine 작은 (먼, 쉬운) negative 일수록 작은 w

# Step 2: weight 정규화 (학습 안정 — 평균 1 되게)
w_j = w_j / mean(w)

# Step 3: weighted softmax 분모
denom_hard = score_pos + Σ_{j} w_j · score_neg_j

# Step 4: 새 loss
L_i = -log(score_pos / denom_hard)
```

## β 효과 직관

| β | w 범위 (sim ∈ [-1, 1]) | 의미 |
|---|---|---|
| **0** | 모두 1 | 원본 InfoNCE (변화 없음) |
| **1** | 0.37 ~ 2.72 (e⁻¹ ~ e¹) | 가까운 negative 가 7배 영향 |
| **2** | 0.13 ~ 7.4 | 56배 — 강한 mining |
| **5** | 0.007 ~ 148 | 너무 sharp → 학습 unstable |

권고 시작값 β = 1.0. 학습 안정 시 1.5, 2.0 시도.

## 실전 효과

**원본 InfoNCE**: 4000+ negative 중 무관한 멀리 wafer 가 99% → 학습 신호 약함 → fine-grained 차이 학습 부족.

**Hard mining 후**: anchor=Edge-Top_bank_boundary 일 때 가까운 (Edge-Top_fork, Edge-Top_scratch) 가 weight ↑ → "이 sub-style 끼리 구분" 압박. Encoder 가 sharp boundary 만들도록 강제.

→ **Edge-Top × bank_boundary cluster 12 merge 같은 confusing cluster 분리 ↑**.

## Production (label 없음) 적용

✅ **label 무관 — production 그대로 작동**:
- positive = augmentation (label 불필요)
- negative weight = cosine similarity (label 불필요)
- β 는 학습 시 선택, inference 단계 영향 X

→ 합성 데이터로 β 검증 → production 학습에 그대로 적용.

## 적용 위치

`D:/project/unknown-contrastive/contrastive.py` 의 InfoNCE 함수:
- `info_nce_global` (line 255 근처)
- `info_nce_with_queue` (line 265 근처)
- `info_nce_local_multi` (line 286 근처) — 옵션

각 함수에 `--hard-mining-beta` 환경변수 / CFG 추가. β=0 (default) 시 원본 동작.

## Verification

학습 후 Tier 1 metric 비교:
- Completeness ↑ (current 0.95 → 0.96+ 기대)
- frac_single_cluster ↑ (current 0.895 → 0.92+ 기대)
- noise_pct 비슷하거나 약간 ↓
- AMI 약간 ↑

성능 큰 변화 없으면 β=2.0 재시도. β=2 에서도 변화 없으면 hard mining 효과 작은 것 (encoder 능력 한계).

## 참고

- Robinson et al. 2021, ICLR — https://arxiv.org/abs/2010.04592
- 공식 구현 PyTorch: GitHub joshr17/HCL (Hard Contrastive Learning)
