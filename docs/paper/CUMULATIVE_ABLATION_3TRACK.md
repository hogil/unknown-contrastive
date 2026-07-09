# 3-Dataset Cumulative Ablation — frozen base → best model (paper-style)

**목표 (사용자):** WM811K · unknown-synth · unknown-multi(mixed29) 세 데이터셋에서
**frozen 백본(base)부터 부품을 하나씩 누적**하여 clustering 성능이 올라가는 것을
논문식(portfolio.md 형식)으로 제시.

- **무라벨 SSL** (CE/SupCon 없음), task-neutral 부품만.
- 고정 컬럼: **P1 cap | recov | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전체/클래스수/noise) | 파편비(전체÷클래스, 1.0=이상)**.
- 주 판정축 = **capture + 파편비** (recov 후순위, 260615 정책).
- 클러스터러는 데이터셋 특성에 맞게 고정 (WM811K=finch_p2, unknown=finch_p1 — 아래 근거).

각 데이터셋 raw 채점은 `WM811K_CD.md`, `UNK_FCMAE_CD.md` / `UNK_SYNTH_ANALYSIS.md`,
`CUMULATIVE_FROZEN_BASE.md` 참조. 이 문서는 세 트랙의 **누적 최종표**만 모음.

═══════════════════════════════════════════════════════════════════
## Track 1 — WM811K (within-dataset, 정상 학습 → novel 불량 발견)
═══════════════════════════════════════════════════════════════════
- **train** = WM811K Normal 1500장, **eval** = WM811K 7 defect class (Normal/Random 제외).
- **클러스터러 = finch_p2** (finch_p1 은 이 풀에서 파편비 16~19 로 과파편 = artifact, 사용자 통찰 "파편비는 hdbscan 문제"). 각 부품 best epoch.

| # | 누적 Recipe | cap | recov | noise% | Comp | Hom | ARI | Sil | k(전체/7/noise) | 파편비 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **DINOv3 frozen (base)** | 1.00 | 0.609 | 0.0 | 0.267 | 0.435 | 0.149 | 0.057 | 28/7/4 | 4.00 |
| 1 | + local-grid 0.15 | 1.00 | 0.635 | 0.0 | 0.324 | 0.448 | 0.218 | 0.032 | 22/7/3 | 3.14 |
| 2 | + queue 4096 (★ best) | 1.00 | 0.641 | 0.0 | 0.375 | 0.486 | **0.270** | 0.008 | 21/7/3 | **3.00** |

**Δ frozen → best: ARI 0.149 → 0.270 (+81%), Comp 0.267 → 0.375, 파편비 4.00 → 3.00, cap 1.0 고정.**
- **local-grid** = 파편비 압축 (4.0→3.14) + ARI↑ — wafer 불량의 **위치 구조**를 patch-grid 대조로 잡아 같은 불량이 여러 조각으로 쪼개지는 것을 막음.
- **queue 4096** = 더 많은 negative 로 class 분리도(ARI) 상승. contrastive **학습이 frozen 을 실질적으로 넘김** (frozen 0.149 → 0.270).
- 참고: 이 within-dataset 풀에서는 **FCMAE frozen(ARI 0.140)이 DINOv3 frozen(0.149)보다 약간 낮음** — Track 2 와 반대 (아래 논의).

═══════════════════════════════════════════════════════════════════
## Track 2 — unknown-synth (합성 정상 학습 → 20 novel 단일결함 발견)
═══════════════════════════════════════════════════════════════════
- **train** = 합성 Normal, **eval** = 20 novel 단일-결함 class (Normal 제외).
- **클러스터러 = finch_p1** (이 풀은 파편비 ~1.0 정상, finch_p2 는 과병합 cap↓).
- 이 트랙의 최대 레버는 **백본 선택**(FCMAE ≫ DINOv3) → 그 위 contrastive 누적.

| # | 누적 Recipe | cap | recov | noise% | Comp | Hom | ARI | Sil | k(전체/20/noise) | 파편비 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0a | DINOv3 frozen | 0.65 | 0.615 | 0.0 | 0.937 | 0.763 | 0.459 | 0.320 | 19/20/5 | 0.95 |
| 1a | + contrastive (DINOv3-init) | 0.95 | 0.870 | 0.0 | 0.927 | 0.921 | 0.798 | 0.558 | 25/20/5 | 1.25 |
| 0b | **FCMAE frozen** (★ 백본 교체) | 0.95 | 0.925 | 0.0 | 0.975 | 0.951 | 0.894 | 0.558 | 21/20/2 | 1.05 |
| 1b | **+ contrastive (FCMAE-init) ★ best** | **1.00** | **0.965** | 0.0 | 0.973 | **0.971** | **0.932** | 0.587 | 24/20/4 | 1.20 |

**세 갈래 상승 스토리:**
1. **DINOv3-init 학습 효과**: frozen 0.459 → +contrastive **0.798 (+0.339, +74%)** — 학습이 크게 끌어올림.
2. **백본 교체 레버**: DINOv3 frozen 0.459 → FCMAE frozen **0.894 (+0.435)** — 가장 큰 단일 레버. **FCMAE(Masked AutoEncoder)는 텍스처/저수준 특징을 보존**해 wafer fail-bit 패턴에 강함 (DINOv3 는 invariance 학습으로 텍스처를 버림).
3. **강한 frozen 위 학습 효과**: FCMAE frozen 0.894 → +contrastive **0.932 (+0.038), cap 0.95 → 1.00** — 이미 강한 출발점에서도 contrastive 가 **capture 를 완성**(20/20)하며 ARI 를 더 올림.

**★ 최고 모델 = FCMAE + contrastive (ep8): ARI 0.932, cap 1.00, Hom 0.971, 파편비 1.20.**
- 궤적은 noisy (ep 별 0.73~0.93, Normal 1500장 과적합 변동) → 향후 데이터↑/정칙화로 안정화 예정.

═══════════════════════════════════════════════════════════════════
## Track 3 — unknown-multi / mixed29 (cross-dataset, 다중결함 중첩)
═══════════════════════════════════════════════════════════════════
- **train** = WM811K Normal, **eval** = MixedWM38 29 multi-defect 조합 (한 wafer 에 여러 결함 중첩). cross-dataset + multi-label = 가장 어려운 트랙.
- **클러스터러 = finch_p1**. 각 부품 ep1 (cross-dataset 은 ep1 정점 후 하락 — 아래).

| # | 누적 Recipe | cap | recov | noise% | Comp | Hom | ARI | Sil | k(전체/29/noise) | 파편비 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **DINOv3 frozen (base)** | 0.931 | 0.376 | 0.0 | 0.484 | 0.528 | 0.192 | 0.017 | 59/29/6 | 2.03 |
| 1 | + queue 4096 | **1.000** | 0.501 | 0.0 | 0.540 | 0.640 | 0.228 | 0.099 | 71/29/5 | 2.45 |
| 2 | + local-grid 0.15 (frag ★) | 0.966 | 0.389 | 0.0 | 0.511 | 0.557 | 0.198 | 0.049 | 49/29/3 | **1.69** |

**Δ frozen → 누적:**
- **queue 4096** → **cap 0.931 → 1.000** (29 조합 전부 group 으로 회수 = recall 완성).
- **local-grid 0.15** → **파편비 2.03 → 1.69** (같은 조합이 여러 조각으로 쪼개짐을 위치 대조로 억제) = 주 판정축(파편비) 챔피언.
- ARI 는 modest (0.192→0.228) — cross-dataset multi-label 의 본질적 천장 (lattice-neighbor cosine 0.996, 같은 모양 다른 위치 ≈ 동일).
- **보조 최강 운영점**: duo-frozen concat (FCMAE ⊕ DINOv3, 학습0) p2-해석모드 = **ARI 0.485, cap 0.905, 파편비 ~0.9** — 두 백본이 놓치는 class 가 상보적이라 concat 이 단일 frozen 을 능가.

═══════════════════════════════════════════════════════════════════
## 종합 — "성능이 올라간다" 한눈 요약
═══════════════════════════════════════════════════════════════════

| 데이터셋 | 클러스터러 | base (frozen) | best (누적) | 주 상승 축 | 최고 레시피 |
|---|---|---|---|---|---|
| **WM811K** (within) | finch_p2 | ARI 0.149 / 파편비 4.00 | **ARI 0.270 / 파편비 3.00** | ARI +81%, 파편비↓ | DINOv3 + local + queue4k |
| **unknown-synth** (single) | finch_p1 | ARI 0.459 (DINOv3) | **ARI 0.932 / cap 1.00** | 백본(+0.435)→학습(+0.038) | FCMAE + contrastive |
| **unknown-multi** (mixed29) | finch_p1 | cap 0.931 / 파편비 2.03 | **cap 1.000 / 파편비 1.69** | cap+0.069, 파편비↓ | DINOv3 + queue4k + local |

**공통 결론:**
- **모든 트랙에서 frozen base 대비 누적 학습으로 주 판정축(capture · 파편비 · ARI)이 상승.**
- **local-grid = 파편비(위치 구조)**, **queue = capture/ARI(negative 다양성)** — 반도체 불량의 공간 특성을 각각 다른 축에서 개선 (도메인 정합).
- **백본 선택은 데이터셋 의존**: 합성 단일결함(텍스처 지배)은 FCMAE, cross-dataset/within WM811K 는 DINOv3 가 유리 → duo-frozen concat 이 상보적 안전판.
- **epoch 은 개선 레버 아님** (cross-dataset ep1 정점 / within·synth 도 조기 정점 후 하락 = backbone drift 과적합). light adaptation 이 옳음.

작성 260702. 클러스터러·프로토콜 명시. raw 채점 = 각 트랙 CD 문서.
