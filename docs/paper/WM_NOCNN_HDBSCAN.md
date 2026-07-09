### wm_fcmae (--method simclr, FCMAE SSL, 실제 WM-811K) 260703

## WM-811K no-CNN (SSL only) — frozen vs +contrastive 누적표 (260703)

- 평가 pool: `data/images/wm811k_eval500_512/eval` = 실제 WM-811K 8 폴더, **Random 제외 → 7 defect class** (Center/Donut/Edge-Loc/Edge-Ring/Loc/Near-full/Scratch), N=3649. capture 분모 = 7.
- 고정 잣대: 임베딩 L2 → UMAP(nc10/nn10/min_dist0/cosine/seed42) → HDBSCAN(mcs10/ms3/leaf/eps0.15). CPU 채점(`_score_umapfree.py`, GPU 학습 무간섭).
- 지표(사용자 260703 지시): **P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil** 만 (k·파편비 제외). 우선순위 cap > noise% > Comp > Hom > ARI > Sil.
- 전부 no-CNN (SSL only). contrastive = `--method simclr`. best ep = 고정 잣대에서 위 우선순위 사전식 선택.

### Table A — 고정 잣대 (UMAP+HDBSCAN) 누적

| track | model (best ep) | P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil |
|---|---|--:|--:|--:|--:|--:|--:|
| frozen | **FCMAE_frozen** | 1.00 | 15.31 | 0.310 | 0.636 | 0.173 | 0.153 |
| frozen | DINOv3_frozen | 1.00 | 16.29 | 0.291 | 0.622 | 0.126 | 0.110 |
| +contrast (DINOv3-init) | wn_base ep5 | 1.00 | 19.09 | 0.291 | 0.617 | 0.131 | 0.074 |
| +contrast (DINOv3-init) | wn_lg015 ep6 | 1.00 | 17.91 | 0.308 | 0.629 | 0.170 | 0.065 |
| +contrast (DINOv3-init) | wn2_q4k ep4 | 1.00 | 20.48 | 0.324 | 0.685 | 0.209 | 0.045 |
| +contrast (FCMAE-init) | wm_fcmae ep1 * | 1.00 | 9.97 | 0.262 | 0.557 | 0.111 | 0.115 |
| +contrast (FCMAE-init) | wm_fcmae ep2 * | 1.00 | **7.78** | 0.258 | 0.551 | 0.121 | **0.178** |

frozen 대비 contrastive delta (같은 백본 frozen 기준, ↓/↑ 값 방향, [+]=개선 [-]=악화):

| model | Δnoise% | ΔComp | ΔHom | ΔARI | ΔSil | 요약 |
|---|--:|--:|--:|--:|--:|---|
| wn_base ep5   vs DINOv3_frozen | +2.80 ↑[-] | −0.000 = | −0.006 ↓[-] | +0.005 ↑[+] | −0.037 ↓[-] | 거의 무변 (noise 소폭 악화) |
| wn_lg015 ep6  vs DINOv3_frozen | +1.62 ↑[-] | +0.017 ↑[+] | +0.007 ↑[+] | +0.045 ↑[+] | −0.046 ↓[-] | 순도↑ / noise↑ |
| wn2_q4k ep4   vs DINOv3_frozen | +4.19 ↑[-] | +0.033 ↑[+] | +0.062 ↑[+] | +0.083 ↑[+] | −0.065 ↓[-] | **순도 최대↑** / noise 최대↑ |
| wm_fcmae ep1  vs FCMAE_frozen  | −5.34 ↓[+] | −0.048 ↓[-] | −0.079 ↓[-] | −0.062 ↓[-] | −0.038 ↓[-] | noise 대폭↓(recall↑) / 순도↓ |
| wm_fcmae ep2  vs FCMAE_frozen  | −7.53 ↓[+] | −0.052 ↓[-] | −0.084 ↓[-] | −0.052 ↓[-] | +0.026 ↑[+] | **noise 전행 최저** / Sil>frozen / ARI ep1대비 회복 / Comp·Hom 여전↓ |

`*` wm_fcmae 는 **ep1~ep2 (학습 진행중)** — 잠정. ep3+ 완주 시 동일 고정 잣대 재채점 필수. ep1→ep2 추세: noise 계속↓(9.97→7.78, 전행 최저 갱신), Sil 는 frozen 초과(0.115→0.178>0.153), ARI 소폭 회복(0.111→0.121), 단 Comp/Hom 은 여전히 frozen 하회. 순도 회복 지속 시 WIN 승격 후보.

### Table B — 보조 finch_p1 (UMAP-free, noise=0 고정)

| track | model (best ep) | P1 cap | noise% | Comp | Hom | ARI | Sil |
|---|---|--:|--:|--:|--:|--:|--:|
| frozen | FCMAE_frozen | 1.00 | 0.0 | 0.252 | 0.593 | 0.063 | 0.011 |
| frozen | DINOv3_frozen | 1.00 | 0.0 | 0.241 | 0.566 | 0.062 | 0.004 |
| +contrast (DINOv3) | wn_base ep5 | 1.00 | 0.0 | 0.241 | 0.542 | 0.072 | −0.039 |
| +contrast (DINOv3) | wn_lg015 ep6 | 1.00 | 0.0 | 0.254 | 0.559 | 0.080 | −0.020 |
| +contrast (DINOv3) | wn2_q4k ep4 | 1.00 | 0.0 | 0.271 | 0.603 | 0.087 | −0.017 |
| +contrast (FCMAE) | wm_fcmae ep1 * | 1.00 | 0.0 | 0.229 | 0.569 | 0.052 | 0.044 |
| +contrast (FCMAE) | wm_fcmae ep2 * | 1.00 | 0.0 | 0.227 | 0.567 | 0.046 | 0.046 |

finch_p1 은 noise 를 안 만들므로(전부 회수) P2 무차별 → Comp/Hom/ARI 로만 서열. 고정 잣대와 동일 결론: wn2_q4k 가 DINOv3-contrastive 중 순도 최고, wm_fcmae 는 frozen 대비 순도 소폭↓.

### 판정 (WM-811K no-CNN)

1. **백본 서열 (frozen)**: **FCMAE_frozen > DINOv3_frozen** — 고정 잣대 P2~P4·ARI·Sil 전 지표에서 FCMAE 우위 (noise 15.31<16.29, Comp 0.310>0.291, Hom 0.636>0.622, ARI 0.173>0.126, Sil 0.153>0.110). 보조 kNN retrieval top1 도 FCMAE 0.745 > DINOv3 0.702 로 일치. **실제 WM-811K 에서 FCMAE 백본이 DINOv3 보다 낫다.**

2. **contrastive 가 P1~P4 를 올리나 → MIXED (전 지표 동시 개선 아님, trade-off 재배치)**:
   - **cap(P1)** 은 모든 행 1.00 (7/7 전 class 메인 등장) — 이미 천장. contrastive 로 못 올림(올릴 여지 없음).
   - **DINOv3-init contrastive**: noise(P2) 를 **악화**(16.3→17.9~20.5) 시키나 Comp/Hom/ARI(순도) 는 **개선**. wn2_q4k ep4 가 순도 정점(Hom 0.685, ARI 0.209 — 전 행 최고). 즉 "덜 회수하되 더 순수" 방향.
   - **FCMAE-init contrastive (wm_fcmae ep1)**: 정반대 — noise(P2) 를 **대폭 개선**(15.3→9.97, 전 행 최저) 하나 Comp/Hom/ARI(순도) 는 **하락**. 즉 "더 많이 회수하되 덜 순수" 방향.
   - 사용자 우선순위(cap>noise)에 순수 사전식 대입 시 **wm_fcmae ep2 가 리더**(cap 1.00 동률 → noise 7.78 최저, ep1 9.97 다음). 단 순도(P3/P4)는 최하위권 → **KEEP(준후보, trade-off)** 표기. ep2 에서 Sil 은 frozen 초과·ARI 회복 시작 → 순도 반등 조짐이나 Comp/Hom 미회복이라 확정 보류.

3. **결론**: 실제 WM-811K 에서 contrastive 는 frozen 을 **전 지표 동시 압도하지 못한다**. cap 은 이미 1.00 포화, 나머지는 noise↔순도 trade-off 를 백본 방향에 따라 반대로 밀 뿐. frozen FCMAE 가 균형점 기준 여전히 강함. **재채점 권고: wm_fcmae ep2+ 완주 후 동일 고정 잣대로 재채점 — ep1 의 noise 급감이 순도 회복과 함께 유지되면 WIN 승격 후보.**

## ★ 실제 WM-811K no-CNN(SSL only) 누적 ablation — finch_p2 (260703 취합)
train=WM-811K Normal 1500 (SSL, label 미사용), eval=WM-811K 7 defect (label 채점만). 백본 SSL-pretrained(DINOv3/FCMAE), CNN 사전학습 없음. best epoch by ARI.

| 누적 recipe | P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil |
|---|---|---|---|---|---|---|
| DINOv3 frozen (base) | 1.0 | 0.0 | 0.267 | 0.435 | 0.149 | 0.057 |
| + local-grid 0.15 (ep6) | 1.0 | 0.0 | 0.324 | 0.448 | 0.218 | 0.032 |
| + queue 4096 (ep4) ★ | 1.0 | 0.0 | 0.365 | 0.510 | 0.280 | 0.049 |
| (ref) FCMAE frozen | 1.0 | 0.0 | 0.279 | 0.450 | 0.140 | 0.061 |
| (ref) FCMAE+contrastive ep1 | 1.0 | 0.0 | 0.251 | 0.435 | 0.111 | 0.091 |

→ no-CNN SSL contrastive: ARI 0.149→0.280 (+88%), Comp/Hom 단조 상승. cap 1.0/noise 0% (FINCH 구조적).
FCMAE frozen ≈ DINOv3 frozen (합성과 달리 실제 저해상도 맵에선 백본 우위 사라짐).
umap_hdbscan(진짜 noise): frozen noise 16.3%→학습 20-27% (real map noise 0% 불가). ARI frozen 0.126→wn_lg015 ep4 0.224.
wm_fcmae ep1 finch_p1(k159) | 1.0 | 0.7 | 0.0 | 0.2291 | 0.569 | 0.052 | 0.0439 | 159/7/22 | 22.71
wm_fcmae ep1 umap_hdbscan(고정잣대) | 1.0 | 0.6287 | 9.97 | 0.2624 | 0.5566 | 0.1106 | 0.1148 | 91/7/14 | 13.0
wm_fcmae ep2 finch_p1(k158) | 1.0 | 0.6891 | 0.0 | 0.2266 | 0.5665 | 0.0461 | 0.0461 | 158/7/22 | 22.57
wm_fcmae ep2 umap_hdbscan(고정잣대) | 1.0 | 0.6495 | 7.78 | 0.2583 | 0.5513 | 0.1211 | 0.1783 | 88/7/9 | 12.57
wm_fcmae ep3 finch_p1(k150) | 1.0 | 0.6844 | 0.0 | 0.2269 | 0.5568 | 0.0481 | 0.0307 | 150/7/19 | 21.43
wm_fcmae ep3 umap_hdbscan(고정잣대) | 1.0 | 0.6192 | 12.35 | 0.2594 | 0.5649 | 0.1019 | 0.1673 | 90/7/10 | 12.86
wm_fcmae ep4 finch_p1(k135) | 1.0 | 0.6736 | 0.0 | 0.2303 | 0.5493 | 0.0516 | 0.0182 | 135/7/19 | 19.29
wm_fcmae ep4 umap_hdbscan(고정잣대) | 1.0 | 0.6226 | 12.42 | 0.2648 | 0.5716 | 0.1047 | 0.1734 | 89/7/12 | 12.71
wm_fcmae ep5 finch_p1(k149) | 1.0 | 0.6742 | 0.0 | 0.2242 | 0.5471 | 0.0495 | 0.0475 | 149/7/22 | 21.29
wm_fcmae ep5 umap_hdbscan(고정잣대) | 1.0 | 0.6124 | 11.31 | 0.2594 | 0.5505 | 0.1124 | 0.1302 | 87/7/11 | 12.43
wm_fcmae ep6 finch_p1(k147) | 1.0 | 0.6739 | 0.0 | 0.2293 | 0.5574 | 0.0511 | 0.0489 | 147/7/22 | 21.0
wm_fcmae ep6 umap_hdbscan(고정잣대) | 1.0 | 0.6096 | 13.53 | 0.2758 | 0.5708 | 0.1259 | 0.1104 | 80/7/10 | 11.43
wm_fcmae ep7 finch_p1(k147) | 1.0 | 0.6867 | 0.0 | 0.2256 | 0.5558 | 0.0429 | 0.0229 | 147/7/19 | 21.0
wm_fcmae ep7 umap_hdbscan(고정잣대) | 1.0 | 0.6082 | 12.73 | 0.2662 | 0.5656 | 0.0999 | 0.1074 | 78/7/7 | 11.14
wm_fcmae ep8 finch_p1(k144) | 1.0 | 0.6809 | 0.0 | 0.2292 | 0.555 | 0.0518 | 0.0101 | 144/7/23 | 20.57
wm_fcmae ep8 umap_hdbscan(고정잣대) | 1.0 | 0.6182 | 12.48 | 0.2595 | 0.5712 | 0.094 | 0.1176 | 88/7/10 | 12.57

### wm_fcmae_full (--method simclr --use-queue --queue-size 4096 --nv-filter 0.95 --neco 0.2, FCMAE SSL, 실제 WM-811K) 260703

## ★ WM no-CNN 최종 결론 (260703) — FCMAE 트랙 기각
FCMAE-init contrastive (wm_fcmae ep5-8, wm_fcmae_full ep1-2) finch_p2 ARI 0.10~0.16 = FCMAE frozen(0.14) 수준, DINOv3+queue 승자(0.280)에 한참 미달.
→ **WM(실제 저해상도 map) no-CNN best = DINOv3 frozen 0.149 → +local 0.218 → +queue4096 0.280.** FCMAE 는 WM 에서 dead-end (합성/DTD 와 반대 — 백본 우위는 도메인 의존). wm_fcmae_full 조기중단(dead-end 확정).
wn_star2_b4 ep2 finch_p2(k25) | 1.0 | 0.6255 | 0.0 | 0.3153 | 0.4563 | 0.1895 | 0.0645 | 25/7/6 | 3.57
wn_star2_b4 ep2 umap_hdbscan(고정잣대) | 1.0 | 0.6083 | 22.86 | 0.3129 | 0.6492 | 0.1784 | 0.0767 | 76/7/6 | 10.86
