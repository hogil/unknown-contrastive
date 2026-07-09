### un_nv (--method simclr --nv-filter 0.95, DINOv3 no-CNN) 260703

---

## no-CNN 누적 ablation (HDBSCAN 기준 + FINCH 병기) — 260703

- 백본: **generic frozen ImageNet only** (DINOv3 / FCMAE / duo=concat). CNN/TAPT/라벨 사전학습 **미사용**.
- 채점기: `_score_umapfree.py --skip-umap` (CPU, `CUDA_VISIBLE_DEVICES=""`).
- 풀: `data/images/synth_clean_contrastive_eval_n50_normal500` = 20 defect class ×10 + Normal 100 = 300. 채점은 20 defect, Normal 은 제외.
- 컬럼: **P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전체/20/noise) | 파편비**. 주 판정 = P1·P2 (목표 cap 1.000 / noise 0%).
- 두 클러스터러: `hdbscan_raw` = 옛 tight 다이얼 (mcs12/ms15/leaf/eps0.06) raw 1024-D 직접. `finch_p1` = parameter-free 계층(2단계).
- best epoch = FINCH p1 capture 최대(동률 ARI). 같은 best 모델의 hdbscan_raw + finch_p1 둘 다 병기.

### [A] FINCH p1 (parameter-free — noise=0% 구조적, 목표 도달 가능 경로)

| stage | recipe (best ep) | P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전/20/nz) | 파편비 |
|---|---|---|---|---|---|---|---|---|---|
| frozen | DINOv3_frozen | 0.65 | 0.0 | 0.937 | 0.763 | 0.459 | 0.320 | 19/20/5 | 0.95 |
| frozen | FCMAE_frozen | 0.95 | 0.0 | 0.975 | 0.951 | 0.894 | 0.558 | 21/20/2 | 1.05 |
| frozen | duo_frozen (concat) | 0.80 | 0.0 | 0.994 | 0.902 | 0.799 | 0.584 | 19/20/3 | 0.95 |
| +contrastive | DINOv3-init un_base (ep5) | 0.95 | 0.0 | 0.927 | 0.921 | 0.798 | 0.558 | 25/20/5 | 1.25 |
| +contrastive | FCMAE-init un_fcmae (ep8) | **1.00** | **0.0** | 0.973 | 0.971 | **0.932** | 0.587 | 24/20/4 | 1.20 |
| +component local0.15 | un_fcmae_lg (ep4) | **1.00** | **0.0** | 0.956 | 0.951 | 0.884 | 0.584 | 24/20/4 | 1.20 |
| +component local0.05 | un_fcmae_lg005 (ep5) | 0.95 | 0.0 | 0.990 | 0.965 | 0.917 | 0.662 | 25/20/6 | 1.25 |

화살표 (cap→100% / noise→0%):
- DINOv3: frozen 0.65 → +contrastive **0.95** (cap ↑↑, +0.30). noise 0%→0% (유지).
- FCMAE: frozen 0.95 → +contrastive **1.00** (cap ↑ 목표 도달), ARI 0.894→**0.932** ↑. noise 0% 내내.
- +local component: cap 유지(1.00 / 0.95) 이나 ARI 는 plain ep8(0.932) 대비 **하락**(lg 0.884, lg005 0.917) — local 항은 이 풀에서 순이득 없음(무~미미한 음).

### [B] hdbscan_raw (옛 tight 다이얼, raw 1024-D 직접 — 목표 미도달)

| stage | recipe (동일 best ep) | P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전/20/nz) | 파편비 |
|---|---|---|---|---|---|---|---|---|---|
| frozen | DINOv3_frozen | 0.10 | 29.0 | 1.000 | 0.149 | 0.040 | 0.652 | 2/20/0 | 0.10 |
| frozen | FCMAE_frozen | 0.20 | 64.0 | 1.000 | 0.544 | 0.380 | 0.657 | 5/20/1 | 0.25 |
| frozen | duo_frozen (concat) | 0.20 | 58.0 | 1.000 | 0.560 | 0.428 | 0.638 | 4/20/1 | 0.20 |
| +contrastive | DINOv3-init un_base (ep5) | 0.10 | 32.0 | 1.000 | 0.131 | 0.033 | 0.450 | 2/20/0 | 0.10 |
| +contrastive | FCMAE-init un_fcmae (ep8) | 0.20 | 66.0 | 0.968 | 0.605 | 0.515 | 0.708 | 5/20/1 | 0.25 |
| +component local0.15 | un_fcmae_lg (ep4) | 0.20 | 65.0 | 0.967 | 0.493 | 0.345 | 0.544 | 4/20/1 | 0.20 |
| +component local0.05 | un_fcmae_lg005 (ep5) | 0.20 | 52.5 | 0.974 | 0.500 | 0.329 | 0.630 | 5/20/1 | 0.25 |

- hdbscan_raw 는 **어느 no-CNN recipe 도 목표 미도달**: cap ≤ 0.30 (전체 epoch 통틀어 최고 = fcmae_lg005 ep2/ep8 의 0.30), noise 는 학습해도 ~50% 바닥 (best noise = lg005 ep2/ep8 의 51.5–52.5%). 옛 tight eps0.06 다이얼이 raw 1024-D cosine 밀도에서 2–6 클러스터로 붕괴 + 대량 noise. → **학습만으로 noise 0% 불가**.

### 판정: target 근접

- **FINCH 경로: 목표 도달.** best no-CNN recipe = **FCMAE-init + plain contrastive `un_fcmae` ep8** → **P1 cap = 1.000, P2 noise = 0%**, ARI 0.932 / Hom 0.971 / Comp 0.973. 차선 `un_fcmae_lg` ep4 도 cap 1.000 / noise 0% (ARI 0.884 로 열세). ※ FINCH 는 noise 라벨이 없어 P2=0% 는 구조적으로 자동 충족 — 실질 변별은 cap·ARI·Hom.
- **HDBSCAN 경로: 목표 미도달.** raw 위 tight 다이얼에서 noise 는 학습으로 0% 로 수렴 안 됨. **후처리로만 0% 달성 가능** (학습 아님):
  - τ-reassignment: noise 점을 가장 가까운 클러스터 centroid(cosine)로 τ 임계 재배정 → noise% 강제 0. (옛 트랙 후처리와 동일 성격.)
  - 또는 parent 고정 잣대인 UMAP(10d, cosine) 선차원축소 후 HDBSCAN — 역사적으로 noise≈0. (이번 run 은 `--skip-umap` 강제라 미산출.)
  - 둘 다 임베딩 후단계지 학습 부품 아님 — 명시.

### 다음 후보 (P1·noise 목표 강화 — no-CNN 부품/값)

1. **plain FCMAE contrastive 안정화 (양자화 아님, 스케줄 부품).** ep8 만 cap 1.00 이고 이웃 ep7/ep9 는 0.80 로 요동 (단일 lucky epoch). EMA teacher / momentum-encoder 항을 더해 cap=1.00 을 여러 epoch 에 걸쳐 lock → 목표의 robust 재현. local 항은 순이득 없으니 **plain 유지**.
2. **local weight coordinate-descent 중간값 0.08–0.10.** lg005(0.05) 는 ARI/Hom 최고(0.917/0.965)지만 cap 0.95 상한, lg(0.15) 는 cap 1.00 도달하나 ARI 0.884 로 열세. 두 끝점 사이 0.08–0.10 이 cap 1.00 + 高 ARI 동시 확보 후보 (coordinate-descent 이웃값).
3. **(HDBSCAN noise→0 전용, 후처리)** τ-reassignment sweep (τ ∈ {0.15,0.20,0.25}) 를 un_fcmae ep8 hdbscan_raw 위에 적용해 noise 0% 하 cap 회복폭 측정 — 학습 아닌 후단계임을 표에 별도 열로 표기.

> 요약: no-CNN 만으로 **FINCH 기준 목표(cap 1.000 / noise 0%) 도달** (un_fcmae ep8). HDBSCAN raw 기준은 학습만으로 미도달 → τ-reassignment/UMAP 후처리 필요. 상승은 frozen→+contrastive 스텝에서 확실(특히 DINOv3 0.65→0.95, FCMAE ARI 0.894→0.932); local component 는 이 풀에서 무이득.

---

## [C] UMAP+HDBSCAN (고정잣대) 실측 — `--skip-umap` 제거 후 재채점 260703

- 위 [A]/[B] 는 `--skip-umap` 강제라 UMAP+HDBSCAN 미산출이었음(line 50). 이번엔 **`--skip-umap` 제거** → 공식 고정잣대 `run_umap_hdb` (UMAP n_components=10/n_neighbors=10/min_dist=0/cosine/seed42 → HDBSCAN mcs10/ms3/leaf/eps0.15) 로 재채점. CPU (`CUDA_VISIBLE_DEVICES=""`), umap 0.5.12 설치 확인.
- 채점 대상 = parent 지정 7개 (전부 no-CNN, generic frozen backbone). CSV: `result_grouping/_field_unksynth/nocnn_umaphdb_scores.csv`.

| stage | recipe | P1 cap | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전/20/nz) | 파편비 |
|---|---|---|---|---|---|---|---|---|---|
| frozen | DINOv3_frozen | 0.85 | 9.0 | 0.946 | 0.904 | 0.814 | 0.492 | 20/20/4 | 1.00 |
| frozen | FCMAE_frozen | 0.90 | 0.5 | 0.980 | 0.937 | 0.870 | 0.589 | 23/20/5 | 1.15 |
| frozen | duo_frozen (concat) | 0.90 | 0.5 | 0.989 | 0.945 | 0.884 | 0.599 | 22/20/4 | 1.10 |
| +contrastive | DINOv3-init un_base (ep5) | 0.85 | 8.0 | 0.984 | 0.938 | 0.882 | 0.623 | 21/20/4 | 1.05 |
| +contrastive | FCMAE-init un_fcmae (ep5) | 0.90 | 4.5 | **1.000** | **0.970** | **0.936** | **0.682** | 22/20/4 | 1.10 |
| +contrastive | FCMAE-init un_fcmae (ep8, [A] winner) | 0.80 | **0.0** | **1.000** | 0.899 | 0.770 | 0.590 | 21/20/5 | 1.05 |
| +component local0.05 | un_fcmae_lg005 (ep5) | 0.85 | 3.5 | **1.000** | 0.940 | 0.875 | 0.655 | 21/20/4 | 1.05 |

### 판정: **고정잣대에서 joint target (cap 1.000 & noise 0%) — 도달 recipe 없음**

- cap ceiling = **0.90** (defect 20 중 2종은 어떤 no-CNN recipe 로도 main-cluster 자격 미달). frozen FCMAE·duo, contrastive ep5 가 동률 상한.
- noise 0% 는 **un_fcmae ep8 단독** 도달 — 단 대가로 cap 0.90→0.80, ARI 0.936→0.770 하락 (defect 2종이 noise 흡수되며 소멸). "옛 43-class 의 noise 0%" 는 **재현되나(ep8), cap 1.0 과 동시 성립은 불가**.
- 사전식 P1>P2>P3>P4 승자 = **duo_frozen** (cap 0.90 / noise 0.5 / Comp 0.989) — 즉 **고정잣대에서는 plain frozen concat 이 P1·P2 최우수, contrastive 는 P1/P2 를 못 올림**. contrastive 이득은 ARI/Hom(순도)로만 발현.
- **KEEP(준후보, 사용자 260612): un_fcmae ep5** — cap 0.90(상한 동률)이나 ARI 0.936 / Hom 0.970 / Comp 1.000 / Sil 0.682 / 파편비 1.10 로 전 순도지표 최고. cap<1.0 단독 탈락 금지 규칙 적용 → best no-CNN 순도 recipe 로 유지.

화살표 (고정잣대):
- DINOv3: frozen 0.85/9.0 → +contrastive(base ep5) 0.85/8.0 (cap 유지, ARI 0.814→0.882↑). FINCH([A]) 의 0.65→0.95 급등과 달리 고정잣대에선 cap 정체.
- FCMAE: frozen 0.90/0.5 → +contrastive ep5 0.90/4.5 (cap 유지, ARI 0.870→**0.936**↑, Hom 0.937→0.970↑, Comp→1.000; noise 0.5→4.5↑) → ep8 0.80/0.0 (noise 제거되나 cap·ARI 붕괴). ep5→ep8 은 **noise↔cap 트레이드**, over-train collapse 패턴(cross-DS epoch↑ 붕괴 메모와 일치).
- local0.05: cap 0.85 (frozen 0.90 대비 ↓), ARI 0.875 < plain ep5 0.936 — 고정잣대에서도 local 항 무이득~음.

### 3-잣대 상충/일치 (한 줄)

- **FINCH p1**: cap→1.00 도달·noise 0% (라벨 부재로 구조적 자동)·파편비 1.05–1.25 — joint target "도달"이나 P2=0 은 자동. **raw-HDBSCAN(옛다이얼)**: cap ≤0.20·noise 29–68% 로 이 풀에서 완전 붕괴(사망). **UMAP+HDBSCAN(고정잣대)**: cap 0.80–0.90·noise 0–9%·ARI 0.77–0.94·파편비 1.0–1.15 로 유일한 충실 심판 → joint target **미도달, cap 상한 0.90** 확정. → 세 잣대 모두 FCMAE-family ≫ DINOv3-raw 는 일치, contrastive→ARI/순도↑ 도 FINCH·고정잣대 일치. **상충 = "target 도달" 여부**: FINCH 는 noise 무배출·미세 파편으로 "도달" 선언, 고정잣대(정직 심판)는 "cap 0.90 천장·noise0=cap대가" 로 미도달.

### 다음 후보 (고정잣대 cap 0.90 천장 돌파용)

1. **cap 0.90 = 미포착 2 defect class 식별 우선.** 20 중 어느 2종이 main-cluster 미달인지 per-class 로 뽑아(고정잣대 pred), 그 2종의 (a) 합성 강도 부족인지 (b) 인접 class 와 임베딩 융합인지 진단 → 융합이면 그 pair 만 hard-negative 강화. (라벨은 진단에만, 학습 선택엔 미사용.)
2. **ep5↔ep8 사이 early-stop lock (ep6–ep7).** ep5 는 cap 0.90·순도최고, ep8 은 noise0·cap0.80 — 사이 구간에서 cap 0.90 유지하며 noise 0.5→0 으로 내리는 지점 탐색 (coordinate-descent, 스케줄 부품). EMA teacher 로 요동 억제.
3. **(고정잣대 noise→0 유지하며 cap 회복)** un_fcmae ep5 UMAP 임베딩 위 τ-reassignment (noise→최근접 centroid, τ∈{0.15,0.20,0.25}) — 학습 아닌 후단계 열로 별기. ep5 의 noise 4.5% 를 0 으로 흡수 시 cap 손실폭 측정.

---

## cap 0.90 천장 진단: 미포착 2 defect class 식별 (260703, no-CNN 트랙)

풀 `synth_clean_contrastive_eval_n50_normal500` (20 defect ×10 + Normal 100). 고정잣대 UMAP(dim10/cosine/seed42)+HDBSCAN(mcs10/ms3/leaf/eps0.15), nn=10. 라벨=폴더명, **진단 채점 전용**(학습 레버 미사용). capture=메인클래스로 등장한 class 수/20. 각 class 10장이 어느 클러스터로 갔는지 + 그 클러스터의 majority(=흡수 상대)로 분해.

### 미포착 2종 (un_fcmae_ep5 = best 순도 recipe, cap 0.90 = 18/20)

| 미포착 class | 10장 행선 | 흡수 상대(클러스터 majority) | 최근접 class centroid (cosine) | 실패 메커니즘 |
|---|---|---|---|---|
| **Center_bank_boundary** | 9장 → **noise(-1)**, 1장 → Donut_bb 클러스터 | Donut_bank_boundary | Donut_bb **0.900**, Edge-Ring_bb 0.760 | **밀도 붕괴(dissolution)**: mcs=10 을 채울 응집 부족 → 9/10 noise 배출. 남은 1장이 최근접 Donut_bb 로 흡수 |
| **Edge-Top_bank_boundary** | 10장 전부 → 클러스터 14 (Edge-Bottom_bb 10장과 **10+10 동거**) | Edge-Bottom_bank_boundary | Edge-Bottom_bb **0.952** | **인접 class 융합(fusion)**: Edge-Bottom_bb 와 완전 병합, majority tie(10:10) 를 Edge-Top 이 패배 → capture 실패 |

- 두 실패는 **메커니즘이 정반대**: Center_bb = 신호 흩어짐(noise), Edge-Top_bb = 신호 겹침(fusion). "합성 강도 부족" 은 Center_bb 쪽(9/10 noise), "임베딩 융합" 은 Edge-Top_bb 쪽(cos 0.952).
- Edge-Top_bb 는 순수 tie-break 패배 — 클러스터 14 는 20장(10+10)이라, majority 를 Edge-Bottom 이 가져가면 Edge-Top 은 자동 탈락. **분리(split)해야 회복**, tie 를 이겨선 안 됨.

### 임베딩 교차 추적 (frozen → contrastive)

| class | FCMAE_frozen | un_fcmae_ep5 | un_fcmae_ep8 | 해석 |
|---|---|---|---|---|
| Center_bank_boundary | MISS (9→Donut_bb) | MISS (9→noise) | **CAPTURED** | contrastive epoch↑ 가 Center_bb 를 Donut 에서 밀어냄 → ep5 는 중간(흩어져 noise), ep8 에 자기 클러스터 형성 |
| Edge-Top_bank_boundary | **CAPTURED** (9/10 own) | MISS (→Edge-Bottom) | MISS (→Edge-Bottom) | frozen 은 top/bottom 분리, **contrastive 가 오히려 융합**(cos 0.952) — 방향 반대 |
| (참고) Center_scratch_rot | MISS (→Donut_sr) | CAPTURED | MISS | ep5 가 Center-rot 계열을 Donut 에서 분리 |

- **핵심 상충**: contrastive 학습은 **Center 계열을 Donut 에서 떼어내는 이득**과 **Edge top/bottom 위치 구분을 무너뜨리는 손해**를 동시에 낸다. frozen(cap 0.90; 놓침=Center_bb+Center_sr)과 ep5(cap 0.90; 놓침=Center_bb+Edge-Top_bb)는 **놓치는 2종이 서로 다름** → 어떤 단일 고정잣대 config 도 20/20 미달.

### 회복 레버 정량 테스트 (다이얼 미변경, 임베딩·nn 부품만)

| config | cap | noise% | 미포착 |
|---|---|---|---|
| ep5 nn10 (baseline) | 0.90 | 6.3 | Center_bb, Edge-Top_bb |
| ep5 nn15 (고정잣대 허용값) | 0.90 | 15.3 | Center_bb, Edge-Top_bb (noise↑, 무이득) |
| concat(frozen⊕ep5) L2 nn10 | 0.90 | 9.7 | Center_bb, Edge-Top_bb |
| concat(frozen⊕ep5) L2 **nn15** | 0.90 | 10.3 | **Donut_bb**, Edge-Top_bb (Center_bb **회복**, Donut_bb 붕괴로 상쇄) |
| co-assoc consensus(frozen,ep5) | 0.90 | **2.3** | Center_bb, Edge-Top_bb (noise 최저이나 융합 미해결) |
| [w·frozen⊕ep5] w∈{0.5,1,2,3,5} × nn{10,15} | ≤0.90 | — | **Edge-Top_bb 12/12 config 전부 실패** |

- **Edge-Top_bb 는 모든 admixture(frozen 가중 포함)에서 실패** — ep5 성분이 조금이라도 섞이면 L2-cosine 상에서 top/bottom 이 다시 붙는다. 오직 **순수 frozen** 만 분리. → contrastive 임베딩이 top/bottom 공간정보를 능동적으로 삭제.
- Center_bb 는 concat@nn15 또는 ep8 에서 회복되나 **각각 Donut_bb·Edge-Top 등 다른 class 를 대신 깨뜨려** cap 순증 없음.

### 원인 추정 (통합)

두 미포착 class 는 **흡수 상대와 오직 공간 위치로만 구분**된다: Edge-Top_bb↔Edge-Bottom_bb = 수직 위치(위 vs 아래 edge band), Center_bb↔Donut_bb = 반경 위치(중심 vs 링). 백본의 **global-average pooling 이 이 공간 layout 을 뭉개고**, contrastive 목적이 그 뭉갬을 심화(Edge-Top). = "합성 강도 부족" 보다 **"pool 이 위치신호를 버림 + 인접 class 근접"** 이 지배 원인. (Center_bb 는 밀도까지 약해 이중고.)

### no-CNN 회복 제안 (값-스윕/부품, 1~2)

1. **[부품] 공간 2영역(top/bottom · center/ring) region-pooled 임베딩** (학습 0, 다이얼 0): frozen 백본 feature map 을 global-avg 대신 **상/하 반분(또는 center/annulus) 분할 pool 후 concat**. Edge-Top↔Edge-Bottom 을 구조적으로 선형분리 → tie 자체 제거. frozen 이 이미 global-pool 로 cos 0.952·9/10 분리하므로 region-pool 은 이를 결정적으로 벌림. **동일 부품이 Center_bb↔Donut_bb(center/ring)도 동시 겨냥** → 단일 부품으로 2종 모두 타겟. 검증: 새 pooling 으로 재임베딩만(백본·학습·다이얼 불변) 후 고정잣대 재채점.
2. **[값-스윕] Edge-Top 전용 frozen-only 채널 + Center 전용 ep-schedule 의 열-분리 보고**: cap 은 단일 임베딩으론 0.90 이 상한이므로, frozen(Edge-Top 포착)·ep5(Center-rot 포착)·ep8(Center_bb 포착) 을 **별 행으로 병기**하고 "class 별 최적 임베딩" 을 명시(라벨 미사용, 사후 보고용). 실제 승격은 제안1 의 region-pool 이 3종을 한 임베딩에서 동시 포착하는지로 판정.

**결론**: cap 0.90 천장의 병목은 (i) Edge-Top_bank_boundary — contrastive 가 Edge-Bottom 과 융합(cos 0.952, 全 config 실패, 최난), (ii) Center_bank_boundary — 밀도 붕괴로 9/10 noise + 잔여 1장 Donut_bb 흡수. 둘 다 **공간위치=구분축**이며 global-avg-pool 이 근본 원인 → **region-pooled frozen 임베딩(제안1)** 이 단일 no-CNN 부품으로 두 실패를 동시 겨냥하는 1순위 후보. 나머지 P1~P4 지표 영향은 재채점으로 확정 필요.
