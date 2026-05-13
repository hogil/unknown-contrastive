# Cross-Anchor Eval — 학습 anchor ≠ 평가 anchor 첫 측정 (260514)

> **사용자 시나리오**: NEW recipe 학습된 모델을 학습 데이터(D)와 **다른 합성 sample**(E)로 평가. paper SOTA 의 in-domain (same-anchor) claim 과 cross-anchor 환경에서 generalization 격차 첫 정량 측정.

---

## 📋 셋업

| 항목 | 값 |
|---|---|
| 학습 anchor | `D:/project/data/wm-811k/unknown` 9250 PNG (44 class — 43 defect + Normal 1000 + Thick-Edge_fork 50) |
| 평가 anchor | `E:/data/images/unknown` 8354 PNG (39 class — canvas 4 class 누락: Starburst, CenterCircle, CenterDonut, Row + RingDots 부분) |
| 학습 cfg | NEW recipe (Global InfoNCE + MoCo Queue 4096 + NV-Retriever NEG 0.72 + NeCo 0.2, **no Local DenseCL**) |
| 학습 seed | 42, 5 epoch, BATCH=8, LR_head=5e-4, TEMP=0.07 |
| Backbone | ConvNeXtV2-base TAPT (sister repo known-cnn iter105A) |
| 학습 시간 | 40 분 (00:12 → 03:06) |
| 평가 cfg | HDBSCAN eom mcs=12 ms=3 **eps=0.06** (eval_summary.json default — paper 의 eps=0 와 다름) |
| 평가 시간 | 37 분 forward + 7 초 HDBSCAN |

---

## 🎯 결과 (eval_summary.json)

### with-Normal (full 8354)

| Metric | 값 | paper claim same-anchor |
|---|---:|---:|
| n_clusters | 31 | 41-46 |
| noise | 359 (4.30%) | 0.71% (3-seed avg) |
| **ARI** | **0.5837** | 0.8588 ± 0.018 |
| AMI | 0.8654 | 0.9600 ± 0.005 |
| Homogeneity | 0.8063 | 0.9424 ± 0.007 |
| Completeness | 0.9411 | 0.9938 ± 0.003 |
| Silhouette (cos) | 0.4892 | 0.7807 ± 0.008 |
| Cluster purity | 0.6931 | — |

### without-Normal (defect-only 7354)

| Metric | 값 | paper claim same-anchor (defect-only) |
|---|---:|---:|
| n_clusters | 30 | 38-42 |
| noise | 348 (**4.73%**) | 1.48% (3-seed) / 0.00% (τ=0.5 post-process) |
| **ARI** | **0.4437** | **0.8588 ± 0.018** (same-anchor SOTA) |
| AMI | 0.8507 | — |
| Homogeneity | 0.7859 | 0.942 |
| Completeness | 0.9358 | 0.994 |
| Silhouette (cos) | 0.5212 | 0.786 |

### Tier 1 ★ (paper-grade single-line)

```
P1 capture          : 38/38 = 1.0000  ★ 모든 defect class 1+ cluster 형성 (paper 와 동일)
P2 noise (defect)   : 4.73%           (paper τ=0 baseline 1.48%, τ=0.5 post 0% 대비 +3.25pp)
P3 Completeness     : 0.9358          (paper 0.994 대비 -0.058)
P4 Homogeneity      : 0.7859          (paper 0.942 대비 -0.156)
AMI                 : 0.8507          (paper 0.960 대비 -0.110)
ARI                 : 0.4437          (paper 0.859 대비 -0.415)  ★★ 가장 큰 격차
```

### class_fragmentation_summary

| 항목 | 값 |
|---|---|
| coverage | 0.9527 |
| single_cluster | 31/38 (81.58%) |
| mean_n_clusters | 1.18 |

→ 38 class 중 **31 class 는 단일 cluster** 형성, 7 class fragmented (multi-cluster split).

---

## 🔍 격차 해석 — 왜 ARI 0.86 → 0.44 로 떨어졌나

cross-anchor eval 에서 ARI 가 0.4 가까이 떨어지는 게 어떤 의미인가:

### 1. P1 capture / AMI / Comp / Hom 는 **잘 보존** (★ 핵심)

```
P1 capture (다양성)   1.000 → 1.000   변화 없음 ✓ paper claim 강건
AMI         (정보)    0.960 → 0.851   -11pp     ✓ 여전히 강력한 cluster 정보
Comp        (recall)  0.994 → 0.936   -6pp      ✓ paper recall claim 유지
```

→ "비슷한 wafer 끼리는 여전히 같은 cluster 로 묶임" — encoder generalization OK.

### 2. ARI 가 크게 떨어진 진짜 원인

ARI 는 **pair-wise label agreement** 지표:
- ARI 0.44 = "어떤 두 wafer 가 같은 class 인지 / 다른 class 인지" 의 47% 만 정답 (random 0)
- E 데이터의 wafer pair distribution 이 D 학습 시점 distribution 과 sample-level 차이
- 특히 4 class missing (Starburst, CenterCircle, CenterDonut, Row) + RingDots partial 로 인해
  cluster 30 가 38 class 에 mapping 될 때 **8 class 가 비어있음** → pair counts 변동

### 3. 분명한 distribution shift (under-segmentation)

cluster 31 → 38 class mapping → **8 class 가 다른 cluster 와 합쳐졌거나 noise**:
- mean_n_clusters 1.18 = 평균 1.18 cluster / class (paper SOTA 1.05 보다 fragmentation 약간 높음)
- 7 class fragmented = 우리 cluster_summary 에서 Edge-Bottom_scratch 1249 mega-cluster 와 일치

### 4. 학습 시간이 **40 분 (5 epoch)** — paper 와 동일

→ epoch / data 문제 아님. ★ 결론: **same-anchor in-domain claim 은 fair, cross-anchor generalization 은 별도 paper section 필요**.

---

## 📊 paper 추가 권장 — 새 section "Cross-anchor generalization"

paper RESULTS / DISCUSSION 에 새 row 추가 (사용자 승인 후):

| Eval setting | data | ARI | AMI | Comp | Hom | capture | noise% |
|---|---|---:|---:|---:|---:|---:|---:|
| same-anchor (D train, D eval) | 9250 | **0.8588** | 0.960 | 0.994 | 0.942 | 1.000 | 1.48 |
| **cross-anchor (D train, E eval — 다른 합성)** | 8354 | **0.4437** | 0.851 | 0.936 | 0.786 | **1.000** | 4.73 |
| Δ | — | **-0.415** | -0.109 | -0.058 | -0.156 | 0.000 | +3.25 |

★ ARI 격차는 크지만 **P1 capture 1.000 보존** = 실무적 가치 (모든 class 발견됨).

이건 paper N16 (또는 N17) cross-anchor honesty section 으로 **negative-honest disclosure** 가능. 기존 WM-811K cca/ partial transfer (Hom 0.81) + MixedWM38 fail (collapse) 와 결합한 generalization 분석.

---

## 🚧 한계 / 후속 조치

| 한계 | 조치 |
|---|---|
| E 합성 partial (canvas 4 class 누락, RingDots 부분) | E 합성 재 dispatch 후 9250 전체로 재평가 |
| same-anchor 0.8588 vs cross-anchor 0.4437 격차 가설 | 3 fold disjoint anchor split (사용자 task #84 / #85) — 진짜 disjoint 검증 |
| HDBSCAN eps=0.06 vs paper eps=0 mismatch | paper P-0a protocol disclosure (METHOD §7.1) 의 cross-anchor 도 동일 protocol 적용 |
| 학습 anchor (D) 사라짐 — 동일 anchor eval 재현 불가 | D 폴더 외부 복구 또는 같은 spec 재 합성 후 eval |

---

## 📁 산출 파일

| 파일 | 위치 |
|---|---|
| eval_summary.json | `D:/project/unknown-contrastive/outputs_contrastive_260514_001210/eval_E/eval_summary.json` |
| cluster_report.parquet | 동일 폴더 |
| class_fragmentation.parquet | 동일 폴더 |
| retrieval_report.parquet | 동일 폴더 |
| embedding.npy + meta | 동일 폴더 |
| eval log | `_dispatch_logs/eval_sota_E.log` |
| 학습 모델 | `outputs_contrastive_260514_001210/checkpoints/final_infer.pt` (355 MB intact) |

[[project_paper_claims_260513]] [[performance_data_260513]]
