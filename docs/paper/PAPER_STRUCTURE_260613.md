# Unknown Defect Discovery — Paper Structure (260613)

라벨 없는 wafer 풀에서 "처음 보는 불량"을 그룹으로 발견. 학습-평가 클래스/성분 무겹침,
무라벨(InfoNCE), 백본 frozen-init. 평가 = MixedWM38, 채점 = FINCH p1 (다이얼 0), capture(P1)>recov>noise>Comp>Hom.

## 트랙 (난이도 = multi-label 중첩, 클래스 수 아님 — 260613 발견)
| 트랙 | 클래스 | 성질 | frozen 천장 | 미해결? |
|---|---|---|---|---|
| single7 | 7 | single-label | 0.957 (duo) | 거의 풀림 |
| hard43 | 42 | single-label compositional | 0.978 (duo) | 거의 풀림 |
| **mixed29** | 29 | **multi-label 중첩** | — | ★ 유일 미해결 (SOTA 0.586) |

→ 논문 주 트랙 = **mixed29**. single7/hard43 은 "백본이 푸는 쉬운 칸" 대조군.

---

## TABLE 1 — Baseline 경쟁 (mixed29, FINCH p1, 동일 protocol)

### 1a. 백본 (frozen, 학습 0)
| 백본 | cap | recov | Comp | Hom | 선택 |
|---|---|---|---|---|---|
| DINOv3 frozen | 0.931 | 0.376 | 0.484 | 0.528 | |
| FCMAE frozen | 0.897 | 0.441 | 0.528 | 0.568 | |
| DINOv3+FCMAE duo | 0.793 | 0.450 | 0.552 | 0.586 | |
- 주: frozen 전부 약함(0.38-0.45, cap<1.0). **학습(SimCLR)이 cap→1.0 + recov 0.50→0.60** — mixed29 는 학습 필수 (single-label 과 반대).
- FCMAE 는 single-label(hard43 0.957) 압도하나 mixed29 multi-label 엔 약함. **DINOv3 = mixed29 백본 선택.**

### 1b. SSL method (DINOv3 백본, WM811k 단일 학습 → mixed29, 2ep, FINCH p1) — ★ 측정 완료
| method | cap | recov | 선택 |
|---|---|---|---|
| **SimCLR (q4k)** | 1.000 | **0.5014** | ★ 승자 |
| Barlow Twins | 1.000 | 0.4959 | 근소 2위 |
| MoCo (momentum 0.99) | 0.966 | 0.477 | declining |
| SimSiam | 0.931 | 0.4276 | |
| VICReg | 1.000 | 0.4048 | |
| BYOL | 1.000 | 0.3703 | negative 부재 약함 |
| DINO | 0.862 | 0.3517 | collapse 경향 |
- 옛 novel-track(k-means ARI) 순위와 일치: SimCLR > Barlow≈VICReg > MoCo > SimSiam ≫ DINO,BYOL.
- **선택 모델 = SimCLR + queue4096** (양 트랙·양 채점 일관 승자).

---

## TABLE 2 — Ablation Waterfall (선택=SimCLR q4k 위에 기법 누적, mixed29 FINCH p1)
| # | 누적 기법 | cap | recov | Hom | Δrecov | 출처 |
|---|---|---|---|---|---|---|
| S0 | SimCLR q4k (base) | 1.000 | 0.5014 | 0.640 | — | queue4096 |
| S1 | + LS 0.02 (앙상블 ep1+ep2) | 1.000 | 0.5572 | 0.688 | +0.056 | 옛 검증 LS |
| S2 | + SoftNCE topk20 (분배 0.02) | 1.000 | 0.5862 | 0.705 | +0.029 | arXiv:2212.07158 |
| S3a | + SoftNCE 분배량 0.2 (paper 권장) | 0.966 | **0.5966** | 0.709 | +0.010 | ★ 값-스윕 260613 |
| S3b | (cap-1 유지판: mass02 swap) | 1.000 | 0.5841 | — | — | cap1 운영점 |
| S4 | + kNN 2% noise 후처리 | =유지 | -0.0006 | — | nz→noise 27% | W4 보완 |
- 단조 상승: 0.5014 → 0.5572 → 0.5862 → 0.5966. **각 additive 기법이 누적 상승** (사용자 지적).
- 운영점 2개: cap1.0/0.5862 (전 클래스 발견) vs cap0.966/0.5966 (회수 우선).

---

## 음성 결과 (정직성 — 안 통한 기법, paper-worthy)
NeCo-KL(-0.02) / NV-filter(-0.006) / ig72(즉사) / 4-tool 묶음(상호작용 독 0.353) / τ 0.03-0.1(전부↓, 0.05 정점) /
queue 5120-16384(오목 강등) / local-grid(무익) / MoCo(declining) / NNCLR nn-pos(lattice 병합) / SCE λ0.5(과smoothing) /
head linear·ad(collapse) / FCMAE on mixed29(독) / mass0.1(골짜기) / ls0.025(0.02 정점 아래) / 5-way 희석 /
agglo·spectral 클러스터러(FINCH 미달) / 데이터 Normal 0%·82%(29% 정점) / adapter mixed29 transductive(0.44 정체).

## 미완 (진행/대기)
- TABLE 1 빈칸 채우기: DINOv3/duo frozen + VICReg/BYOL/SimSiam/Barlow/DINO on mixed29 (baseline 경쟁 완성)
- 값-스윕 잔여: topk15/25/30, sce09 (additive 부품 탐색)
