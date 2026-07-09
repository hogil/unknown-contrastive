# UNK_SYNTH 트랙 분석 — unknown 합성 (Normal 학습 → 21 novel defect grouping)

평가 풀: `data/images/synth_clean_contrastive_eval_n50_normal500`
(20 defect class × 10 + Normal × 100 = 300장. EXCL=Normal/Random/R → 채점 20 defect class)
채점기: `python _score_umapfree.py <emb.npy> --skip-umap --pool <위 풀>`
판정 클러스터러: **finch_p1** (이 트랙 정답 클러스터러. finch_p2 는 과병합 cap≈0.30).
주축: capture(P1) + ARI + 파편비. recov 보조.

---

## 260622 15:13 — un_base / un_lg015 8/6-epoch 채점 (finch_p1)

### 채점표 (finch_p1, best epoch = best ARI, ARI 내림차순)

| tag | best ep | P1 cap | recov | P3 Comp | P4 Hom | ARI | Sil | 파편비 | k(전체/클래스/noise) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| **FCMAE_frozen** | — | **0.95** | 0.925 | 0.9748 | 0.9512 | **0.8942** | 0.5584 | 1.05 | 21/20/2 |
| un_base | ep5 | 0.95 | 0.870 | 0.9266 | 0.9213 | 0.7984 | 0.5578 | 1.25 | 25/20/5 |
| un_lg015 | ep1 | 0.90 | 0.875 | 0.9579 | 0.9100 | 0.7914 | 0.5546 | 1.10 | 22/20/4 |
| DINOv3_frozen | — | 0.65 | 0.615 | 0.9371 | 0.7633 | 0.4592 | 0.3198 | 0.95 | 19/20/5 |

전체 epoch finch_p1 ARI 추이 (best 굵게):
- un_base: ep1 .726 / ep2 .680 / ep3 .672 / ep4 .731 / **ep5 .798** / ep6 .756 / ep7 .721 / ep8 .708 (ep5 정점 후 하락)
- un_lg015: **ep1 .791** / ep2 .666 / ep3 .637 / ep4 .681 / ep5 .692 / ep6 .729 (ep1 정점, 이후 회복 못함)

raw(옛다이얼 hdbscan) 행은 전 tag 에서 noise 6~100%, ARI<0.41 — synth_clean 풀에서 raw HDBSCAN 운영 불가 (밀도 부적합). finch_p0(k60~80)은 cap 1.0 이지만 파편비 3~4 (과분할), p2(k4~7)는 cap≤0.35 (과병합). p1 이 유일한 운영 구간.

### 판정: LOSE (현 시점 학습이 최강 frozen 못 넘음)

- **현 winner = FCMAE_frozen finch_p1** (ARI 0.894, cap 0.95, Comp 0.975, Hom 0.951, 파편비 1.05). 학습 산출 중 이를 넘는 부품 없음.
- ★ **컨텍스트 "frozen ARI 0.459" 는 DINOv3_frozen 기준**이었음. 이 풀의 진짜 frozen 천장은 **FCMAE_frozen ARI 0.894** — DINOv3 보다 +0.435 강함. un_base ep5(0.798)/un_lg015 ep1(0.791)는 DINOv3 대비 +74%/+72% 이지만 **FCMAE_frozen 대비 -0.096/-0.103 (열세)**.
- base contrastive(un_base ep5)가 local-grid(un_lg015 ep1)를 ARI/recov 에서 근소 우위 → 컨텍스트 "local-grid 가 base 못 넘음" 재확인. 단 둘 다 FCMAE_frozen 아래.
- 학습은 DINOv3-init 에서 출발했고 DINOv3_frozen(0.459)→un_base ep5(0.798) 로 +0.339 끌어올림. 그러나 도달점이 **FCMAE_frozen 출발점보다 낮음** = init 선택(DINOv3) 이 이 단일라벨 clean 합성 풀에서 불리. (held-out 진짜 novel 에서도 frozen 이 강하다는 260614 결과와 일관 — clean single-label 합성은 백본이 이미 푼다.)

### capture 실패 분석 (다음 후보 근거)

| tag | k_p1 | 놓친 class | 메커니즘 |
|---|--:|---|---|
| FCMAE_frozen | 21 | CenterDonut (1종) | CenterDonut↔Donut 계열 병합 (동심원 토폴로지 근접) |
| un_base ep5 | 25 | Edge-Ring_scratch_rot (1종) | scratch_rot 미세각 + Edge-Ring 형상 혼입, 25클러스터 과분할(파편비 1.25) |
| un_lg015 ep1 | 22 | Edge-Bottom_bank_boundary, Edge-Bottom_scratch_rot (2종) | Edge-Bottom 계열 2종 동시 소실 (local-grid 가 Edge-Bottom 위치단서 흐림) |
| DINOv3_frozen | 19 | 7종 (CenterDonut, Donut_scratch_rot, Edge-Bottom_bb, Edge-Top_bb, Edge-Top_fork, Full_scratch_rot, Row) | bank_boundary/fork/scratch_rot 미세 obj 단서 미해상 |

### 다음 후보 (coordinate-descent, ARI 올릴 가능성 높은 3개)

1. **FCMAE-init 으로 contrastive 재학습 (un_fcmae)** — 최우선.
   근거: 이 풀에서 FCMAE_frozen(0.894) ≫ DINOv3_frozen(0.459). 학습이 DINOv3 를 +0.339 끌어올렸으니, 같은 학습을 **더 높은 출발점(FCMAE)** 에서 시작하면 천장(0.894)+α 가 기대됨. init 만 바꾸는 순수 coordinate-descent (이긴 부품=FCMAE backbone, 이웃값=그 위 contrastive). un_base 와 동일 recipe, lr_bb 만 보수적(FCMAE 이미 강하므로 1e-6 권장 — 과학습 ep5+ 하락 방지).

2. **un_base early-stop + lr_bb 절반 (un_base_lr half, ep≤5 고정)** — base 의 ep5 정점 후 하락(0.798→0.708) = backbone drift 과학습.
   근거: un_base 는 ep5 에서 정점 후 단조 하락, un_lg015 는 ep1 정점. 둘 다 **빠른 정점 + 붕괴** = lr_bb 가 큼. lr_bb 를 절반(현 2e-6→1e-6)으로 낮추고 정점을 ep5 이후로 밀면 더 높은 plateau 가능. recov(현 0.87)도 동반 상승 기대.

3. **duo concat (FCMAE_frozen ⊕ DINOv3_frozen, L2)** — 학습 0, 즉시 검증 가능.
   근거: 260614 held-out 에서 "duo frozen" 이 단일 frozen 각각을 ARI 에서 능가(0.601→0.628, p2 해석 0.846). 두 frozen 은 놓친 class 가 다름(FCMAE=CenterDonut 1종, DINOv3=7종이지만 일부는 FCMAE 가 잡음) → 상보적. concat+L2 후 finch_p1 채점이 FCMAE_frozen 단독(0.894) 초과 가능. 가장 싼 실험이므로 1번과 병행.

우선순위: 3(즉시·무학습) → 1(FCMAE-init 학습) → 2(base lr 튜닝). 1번이 본질적 천장 돌파 후보.

산출 CSV: `result_grouping/_field_unksynth/score_part{1,2,3}.csv` (전 16 임베딩 × 6 클러스터러 raw 행 포함).
