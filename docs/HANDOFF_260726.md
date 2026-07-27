# Handoff — unknown-contrastive, 260726

작성: Claude Code 세션 종료 시점. 인수자: codex.
목적: 사내 실데이터에서 **unknown grouping 으로 신규불량 발생을 감지**하는 것.

> **Binding update (2026-07-26):** the canonical current objective and rules
> are [`D:\project\unknown-contrastive\docs\ABSOLUTE_RULES.md`](ABSOLUTE_RULES.md).
> This handoff remains historical evidence and does not authorize CCA/my-lot or
> any non-approved root for future work.

## Codex live continuation — 2026-07-26 19:20 KST onward

- User-direct, promotion-locked base run is live: seed 42, LR 0.004, 20
  epochs, B4, sampling 0.25, `REPRO_WORKERS=8`, and hard PyTorch VRAM
  fraction 0.40. It may produce evidence, but cannot become champion before
  split closure and the required A/B/C panel decision.
- Training output:
  `D:\project\unknown-contrastive\runs\may_repro\abl_provisional_strict_base_s42_w8_user_direct_B4_260726_192011`
  (parent PID at launch: 34284). Launcher logs:
  `D:\project\unknown-contrastive\runs\campaign_state\provisional_base20_w8_260726_192005`.
- Live throughput changed from about 7.6 s/batch with zero workers to about
  1.2 s/batch after worker startup. Epoch 1 completed in 311.3 s. The
  Windows `__main__` spawn guard remains mandatory and
  `persistent_workers=false`.
- The first full split audit found zero byte-identical/path duplicates, but
  18 filename-token source-block overlaps. The 18 validation rows were removed
  from both validation manifests, leaving 4,178 aligned rows; training was not
  changed. Current manifests:
  `D:\project\unknown-contrastive\data\pools\v2\unknown\strict_novel_val.json`
  (SHA-256 `aa2da7f8ff4ef63d5fe7312c80828f443efec52cf7bf42a8ef6b6008bf8446f6`)
  and
  `D:\project\unknown-contrastive\data\pools\v2\unknown\strict_novel_val_unlabeled.json`
  (SHA-256 `9f3870e2a5c5a0af5d56bc013463ce68a1308dd984e1fc0a4b4b67b60838e397`).
  Pre-repair manifests are retained under
  `D:\project\unknown-contrastive\runs\campaign_state\audits\manifest_backups`.
- The pre-repair audit is retained at
  `D:\project\unknown-contrastive\runs\campaign_state\audits\strict_novel_train_vs_val.pre_blockrepair.1cac19395cc38ba0.json`.
  A one-read-per-image replacement audit is running to
  `D:\project\unknown-contrastive\runs\campaign_state\audits\strict_novel_train_vs_val.json`.
- Latest focused contract suite after provenance, panel, Rule-C, GPU, and
  audit changes: **168 passed**.

---

## 0. 한 줄 요약

오늘 세션의 산출은 **새 성능 수치가 아니라 측정 체계의 교정**이다. 기존 성능 주장 5건 중
4건이 잘못된 측정 지점 또는 잘못된 대조군에서 나온 것이었고, 그중 하나(severstal)는
**과소평가**였다(정정 후 전 지표 압승). 진짜 신규 성과는 **시간축 배포 운영점 확정** 하나다.

---

## 1. 배포 사다리 — 현재 검증 상태

사용자 정의 우선순위: ①모델 직접 predict > ②레시피 학습 > ③recipe sweep 학습 > ④TAPT+sweep

| 순위 | 방법 | 상태 |
|---|---|---|
| ① zero-shot | 우리 모델을 사내 데이터에 바로 적용 | ❌ **실패 확정**. cca 14 source 중 frozen 을 이기는 게 0개. champion 이 최악(P1 6/7→3~4/7, noise 59.7→95.8) |
| ② 레시피 학습 | B4 recipe 로 사내 데이터 학습 | ✅ **작동**. severstal(산업 결함, 누수 0)에서 다이얼 정정 후 전 지표 압승 |
| ③ sweep 학습 | 레시피를 사내에서 스윕 | ⏳ 10셀 학습 완료, **채점 미완** ← 다음 작업 |
| ④ TAPT+sweep | CNN TAPT 후 스윕 | 미착수. new-domain 에는 불리하다는 기존 기록(260724) |

### ★ 사다리 ②③④ 전부에 걸린 숨은 전제: **라벨 없이 고를 수 있는가**

사내엔 라벨이 없다. 그런데 우리가 "최고"를 고른 방식은 대부분 ARI/Comp = 라벨 기반이었다.

| 무엇을 고르나 | 라벨 없이 | 상태 |
|---|---|---|
| **epoch** (run 안에서) | Rule C | ✅ severstal 외부검증 통과 (오라클과 0.25pp) |
| **레시피** (run 사이) | ? | ⏳ **미해결 — 다음 작업의 핵심** |
| **다이얼** (HDBSCAN mcs/ms) | **없음** | ❌ **실패 확정** (아래 3-C) |

**Rule C** = `gate 통과 → k ≥ 자기 run 의 75th 퍼센타일 유지 → argmin noise`

---

## 2. 오늘 확정된 진짜 성과

### 2-A. 시간축 신규불량 감지 운영점 (배포 가능)

**P10 / m_min = REF 크기 50th-pct(=11) / K=2** → **FAR 0**, 검출 비용 lag +1~3배치 / 누적 20~90장.

- 기존 운영점(P10/25th/K1)은 오경보 8건/4배치였다
- **P1/P2 퍼센타일은 쓰지 마라** — 표본 n=55 에서 P1 은 통계량이 아니라 관측 최솟값이고,
  캘리브레이션 창을 t{1,2,3}→t{2,3,4} 로 옮기면 임계가 0.0258 이동한다(전체 범위 0.05 의 절반).
  P10/P20 은 0.002~0.006 으로 5~15배 안정
- **Normal 오경보는 (b) 경계샘플 jitter** — 모든 Normal 알람이 streak=1(한 배치 뒤 소멸),
  K2/K3 생존자는 전부 CenterCircle/CenterDonut/Row. **K≥2 만으로 완전 억제.** size 필터 불필요
- **size-filter 는 나쁜 지렛대**: floor 를 100th-pct 로 올리면 FAR 0 이지만 size05 가 6배치 내내 미검출

### 2-B. champion 의 시간축 우위 = **운영점 안정성** (lag 배수가 아니라)

| novel class | frozen 최적점 | 총 lag | champion 최적점 | 총 lag |
|---|---|--:|---|--:|
| CrossScratch | P10, m_min=10, **K3** | 13 | **P10, m_min=16, K1** | 5 |
| DiagonalSmear | P10, m_min=17, **K2** | 10 | **P10, m_min=16, K1** | 6 |

lag 비율은 2.6배→1.67배로 흔들리지만, **champion 최적점은 novel 클래스가 바뀌어도 고정**이다.
배포에서는 신규 불량을 보기 전에 운영점을 정해야 하므로 **이게 실질 우위**다.
(BrokenRing 3번째 확인 미완)

### 2-C. severstal — 사다리 ② 가 산업 도메인에서 작동한다

**다이얼 정정 후** (mcs20/ms5/leaf, k=5=실제 클래스 수):

| arm | P1 | seed_noise | Comp | Hom | ARI | k |
|---|---|--:|--:|--:|--:|--:|
| frozen | 2/4 | 70.15 | 0.505 | 0.573 | 0.522 | 4 |
| **adapted ep17** | **3/4** | **59.50** | **0.859** | **0.758** | **0.840** | **5** |

**frozen 은 k=4~6 영역 30셀 전부에서 P1 ≤ 2/4** — 3번째 결함 클래스를 자기 클러스터로 분리 못 한다
(4/4 를 사려면 k=17·noise 76.7% 지불). adapted 는 같은 영역에서 3/4 를 3개 다이얼에서 낸다.
**숫자 차이가 아니라 능력 차이.**

⚠ 단 **z0(랜덤 헤드) 대비는 mcs20 에서 미측정** — 감사 문서에 placeholder 로 열려 있다.

---

## 3. 오늘 발견한 측정 오류 4건 (전부 같은 구조)

**공통 구조: 우리에게 유리한(또는 그냥 물려받은) 측정 지점 하나에서 재고 결론을 냈다.**

### 3-A. 다이얼이 pool 기하에 안 맞았다 → severstal 결론이 뒤집힘

**규칙: `mcs ≈ n/k 의 10%`** (pool 6개로 검증)

| pool | n | k | n/k | mcs6 = ?% | 맞는 mcs |
|---|--:|--:|--:|--:|--:|
| mwm38_clean546 | 546 | 9 | 60.7 | 9.9% | 6 ✅ |
| anchor_avg30_repro | 2260 | 43 | 52.6 | 11.4% | 5 ✅ |
| unknown_eval100 | 4149 | 42 | 98.8 | 6.1% | 10 |
| severstal_pilot | 995 | 5 | 199.0 | **3.0%** ❌ | **20** (스윕 승자와 일치) |
| **v2 strict_novel_train** | 12647 | 22 | 574.9 | **1.0%** ❌❌ | **57** |
| **v2 strict_novel_val** | 4196 | 10 | 419.6 | **1.4%** ❌❌ | **42** |

**★ v2 는 7배 어긋나 있다. v2 학습 결과를 mcs6 으로 평가하면 결론이 무효다.**
⚠ **260727 정정: k 를 입력으로 요구하는 것은 폐기됐다.** k 는 모르는 게 전제다.
다이얼은 `min_cluster_size` 원래 의미("몇 장 이상 뭉쳐야 그룹인가") 또는
**bootstrap 안정성 최대**로 라벨·k 없이 정한다. 위 표는 사후 설명일 뿐 입력 규칙이 아니다.
⚠ pool 전체 클러스터링(grouping)에만 적용. **temporal 은 배치 단위**라 mcs6 이 타당하다.

### 3-B. 대조군이 frozen 이었다 → 올바른 대조군은 z0(랜덤 head)

anchor 에서 **학습 안 한 랜덤 projection head 가 frozen 을 ARI +0.048 / Hom +0.018 / P1 +2 로 이긴다.**
부호는 pool 마다 뒤집힌다(severstal 은 frozen > z0). **무시 가능한 null 이 아니다.**

**용량을 맞춰야 한다** — champion 이 2-head concat+L2 앙상블이면 z0 도 랜덤 2-head 여야 한다.
1-head 랜덤과 비교했을 때 "Hom 짐 / ARI 동률"이던 게, 용량 맞추자 **"둘 다 강건 승"으로 뒤집혔다.**

### 3-C. 다이얼을 라벨 없이 고를 방법이 없다 (배포 제약, 실측)

severstal 168셀에서 무라벨 대리지표 vs ARI Spearman:
- `Sil`: adapted ρ 0.80~0.93 인데 **frozen 은 −0.18** — 어느 arm 인지 미리 모르므로 사용 불가
- `over_merge==0` 게이트: **frozen 의 k=2 병합 치팅을 못 잡는다**(20%-of-n 임계는 "거대 blob"만 잡음)
- frozen-leaf 는 **k 와 ARI 가 ρ −0.96** → noise 최소화 선택이 곧장 병합 치팅으로 간다
- `stability`/`coherence` 도 arm·method 따라 부호 뒤집힘

**⚠ ARI 로 다이얼을 고르지 마라** — frozen best-ARI 가 k=2, adapted best-ARI 가 k=3 (둘 다 병합).
정직한 비교 영역은 **k ≈ 실제 클래스 수**.

### 3-D. 퍼센타일 임계가 불안정했다 (2-A 참조)

---

## 4. z0 감사 결과 — 헤드라인 주장 5건

문서: `docs/audit/z0_control_audit_260726.md` (414줄)

| claim | 판정 | 내용 |
|---|---|---|
| #1 flagship (clean546 champion) | **MIXED, 생존** | 용량매칭 z0-ensemble(n=10) 대비 noise 37.2 vs 48.8±4.2 **전 범위 밖 승**, Hom 0.923 vs 0.789±0.054 **전 범위 밖 승**, ARI 0.858 **90th pct**, Comp 0.838 30th pct 소수패 |
| #2 severstal | **MIXED, 미확정** | noise 는 z0 를 10.4pp 이김, Hom 은 z0 에 짐. **단 mcs6 에서 측정 — mcs20 재검증 필요** |
| #4 cycle-1 운영점 | **MIXED, 헤드라인 무효** | reassign 후 noise 가 랜덤 헤드와 거의 같음. **reassign 전 값(37.2)만 인용하라** |
| #5 temporal FAR | **STANDS, 강화** | champion 운영점에서 champion 0, frozen 1.25, **z0 3.75~4.0** — 랜덤이 frozen 보다 나쁨 |
| #3 무라벨 5.7/7 | **SUPERSEDED** | 최적화 루프 이전 상태. 최종 상태는 #1 이 감사함 |

**철회 0건.** 단 두 가지는 반드시 지켜라:
- **"P1 7/7 달성" 을 헤드라인으로 쓰지 마라** — 랜덤 헤드도 10% 확률로 도달한다.
  쓸 수 있는 건 **"champion 은 7/7 을 매번 재현한다"**(z0 는 1/10)
- **reassign 후 noise 를 인용하지 마라** — 어떤 임베딩에든 나오는 바닥값이다

### ⚠ 미해결 숫자 불일치 (인수자가 확인할 것)
champion ensemble 의 reassign 후 noise 가 두 보고에서 다르다: **5.1 vs 3.85** (reassign 전은 37.2 로 일치).
같은 실행에서 champion 과 z0 를 같은 함수로 다시 재야 #4 판정이 확정된다.
3.85 가 맞다면 #4 는 "동률"이 아니라 "축소되지만 여전히 우위"로 바뀐다.

### 배포 기본값 — **앙상블이 맞다** (검증 완료)
reassign 후(=실제 배포 지점) 비교: single noise 2.93 / ensemble 3.85 → 격차 0.92pp 는 **잡음폭 안 = P2 동률**
→ P3/P4 로 넘어가면 앙상블 승(Comp +0.039, Hom +0.040, ARI +0.074). P1 은 둘 다 7/7, **잡는 클래스도 동일**
`{C,D,EL,ER,L,NF,S}`. `grouping_deploy.py` 기본값(앙상블) 유지.

---

## 5. 진행 중이던 작업 (전부 중단됨, 재개 가능)

### 5-A. ★ severstal recipe sweep — **학습 10셀 완료, 채점 미완** ← 최우선

`runs/severstal/pilot/` 에 20 epoch × 9셀 + 8 epoch 대조군 1셀. **배선 전 셀 검증 완료 — 정확히 1축만 변함**:

```
cell       TEMP   NEG    QUEUE   LR      LOCAL
base       0.2    0.72   16384   0.004   true    (runs/severstal/may_repro/abl_B4_260726_100437 재사용)
t010       0.1    ·      ·       ·       ·
t030       0.3    ·      ·       ·       ·
neg060     ·      0.60   ·       ·       ·
neg085     ·      0.85   ·       ·       ·
q4096      ·      ·       4096   ·       ·
q32768     ·      ·      32768   ·       ·
lr002      ·      ·      ·       0.002   ·
lr008      ·      ·      ·       0.008   ·
nolocal    ·      ·      ·       ·       false
```
전부 EPOCHS 20 / SEED 42 / batch 64 / sampling 0.25 / NUM_WORKERS 0.

#### ★ 채점 결과 (세션 종료 직전 확보 — mcs6 10/10, mcs20 8/10)

`runs/clean546/severstal_pilot_<cell>.json` (mcs6) / `severstal_pilot_<cell>_mcs20.json` (mcs20)

**mcs6/ms3** — k 22~28 (실제 5클래스 대비 과분할, P1 전 셀 4/4 포화)
```
cell      ep  seed_nz   k    P1   noise   Comp     Hom     ARI    frag
base       3   69.45   24   4/4   70.94  0.3700  0.8213  0.2392   6.00
t010      13   67.74   25   4/4   69.43  0.3785  0.8273  0.2352   6.25
t030       4   67.34   28   4/4   69.81  0.3401  0.7907  0.2033   7.00
neg060    14   67.04   28   4/4   69.94  0.3497  0.8056  0.1800   7.00
neg085    16   67.24   25   4/4   68.55  0.3933  0.8272  0.2551   6.25
q4096     15   69.35   22   4/4   72.08  0.4142  0.8489  0.3010   5.50
q32768    20   62.91   25   4/4   63.52  0.4109  0.8428  0.2757   6.25
lr002     15   66.23   23   4/4   69.81  0.4197  0.8822  0.2770   5.75
★ lr008    9   59.10   25   4/4   60.00  0.4073  0.8172  0.4319   6.25
nolocal   17   66.83   24   4/4   69.43  0.3641  0.7708  0.2259   6.00
```

**mcs20/ms5** — k 5~7 (기하 타당 영역, P1 이 3/4 로 변별됨). nolocal 미완
```
cell      ep  seed_nz   k    P1   noise   Comp     Hom     ARI    frag
base       1   70.05    5   3/4   70.44  0.5372  0.5243  0.3895   1.25
t010      17   74.87    6   3/4   74.09  0.5752  0.7120  0.5032   1.50
t030       1   68.84    5   3/4   68.55  0.5598  0.5399  0.4198   1.25
neg060    20   66.23    5   3/4   63.90  0.5266  0.4459  0.2768   1.25
neg085    20   67.74    5   3/4   65.03  0.4185  0.3742  0.1521   1.25
q4096     15   71.76    7   3/4   69.18  0.5578  0.7480  0.4427   1.75
q32768†   20   66.53    6   3/4   66.16  0.6134  0.7526  0.6108   1.50
lr002     14   74.17    6   3/4   73.46  0.5803  0.7319  0.4734   1.50
★ lr008    6   57.89    6   3/4   56.98  0.5872  0.6883  0.6296   1.50
```

**승자 = `lr008` (LR 0.008, base 0.004 의 2배). 두 다이얼 모두에서 1위.**
- mcs20 기준 base 대비: seed_noise 70.05→**57.89**, ARI 0.3895→**0.6296**(+62% 상대), Comp +0.050, Hom +0.164
- mcs6 기준 base 대비: seed_noise 69.45→**59.10**, ARI 0.2392→**0.4319**(+81% 상대)
- 사전등록 마일스톤 `noise ≤ 57.74` 와 대비: lr008@mcs20 noise **56.98** — 다만 마일스톤은
  mcs6 frozen(77.74) 기준으로 정의된 값이라 **직접 비교는 무효**. 다이얼을 맞춘 재정의 필요

† **2026-07-26 19:45 교정**: q32768 mcs20 결과는 이미
`D:\project\unknown-contrastive\runs\clean546\severstal_pilot_q32768_mcs20.json`
(SHA-256 `27d566ca853ac80183393b2d3778343a4cb74121cd59ba575ba819147487e4a4`)에
완료돼 있었다. `lr008`은 q32768보다 seed_noise가 8.64pp 낮고 ARI가 0.0188
높아 승자를 유지한다. 다만 이 파일은 현재 evaluator 수정 전 생성돼 selection
snapshot/manifest/backbone SHA가 없으므로 **역사적 보조 증거**로만 사용하며
덮어쓰지 않는다. nolocal은 현재 승인된 후속 범위 밖이라 재채점하지 않는다.

#### ★★ task #20 — 무라벨 셀 선택: **순위는 못 매기지만 승자는 맞춘다**

```
                          rho(-seed_noise, ARI)   rho(k, ARI)   argmin(seed_noise) vs argmax(ARI)
mcs6/ms3   (n=10)               +0.297              -0.573              lr008 == lr008  MATCH
mcs20/ms5  (n=9)                +0.000              +0.730              lr008 == lr008  MATCH
```

- **전체 순위 상관은 쓸모없다** — seed_noise ρ는 +0.30→0.00으로 사라지고,
  k ρ는 −0.57→+0.73으로 다이얼에 따라 부호가 뒤집힌다. 무라벨 지표로 셀 순위를
  강건하게 매기는 건 불가능
- **그러나 `argmin(seed_noise)` 가 두 다이얼 모두에서 정답 셀(lr008)을 집어냈다.**
  배포에서 필요한 건 순위가 아니라 **승자 하나**이므로 이게 실무적으로 중요한 결과
- ⚠ **증거는 약하다** — 승자 일치 2건(다이얼 2개, pool 1개)뿐이다. 현재 승인된
  다른 데이터셋에서 재현해 argmin(seed_noise)가 다시 승자를 맞춰야 배포 규칙 후보로 승격
- ⚠ 이 분석은 seed 1개(42) 기준이다. 상위 셀 3-seed 확장 후 재확인 필요

**미리보기** (트레이너 내장 다이얼 — 최종 채점 아님):
`q32768 66.4% < lr002 70.3% < q4096 72.8% < t030 73.2% < lr008 74.1% < neg060 74.6% <
nolocal 75.2% < neg085 76.4% < t010 79.0%` (k 8~10, 실제 클래스 5)
→ **queue 를 2배로 키운 게 가장 좋고, temp 를 낮춘 게 가장 나쁘다.**

**해야 할 일**:
1. **두 다이얼로 채점**: `mcs6/ms3/leaf/eps0.06`(기록 연속성) + `mcs20/ms5/leaf/eps0.06`(기하 타당)
2. **★ task #20 — 무라벨 셀 선택 규칙** (사다리 ③ 의 직접 증거):
   - (A) 라벨 순위 = ARI 내림차순
   - (B) 무라벨 후보 = `seed_noise` / frag·k / `coherence` / `stability` / `Sil`
   - (A)(B) 의 **Spearman ρ**. ρ≥0.7 인 후보가 있으면 그게 배포용 셀 선택 규칙
   - **전부 낮으면 그것도 1급 결과** = "사내 배포엔 소량 라벨링 예산 필수"
   - n=10 명시, **잡음폭 밖으로 갈리는 셀 쌍에 한정한 순위 일치율**도 병기
   - **(B) 계산 시 라벨을 곁눈질하면 검증 무효**
   - **두 다이얼 각각에서 계산** — 같은 셀이 이기면 강건, 다르면 사내 난이도 2배
   - 스크립트 이미 작성됨: `_severstal_label_free_cell_selection.py`
3. 상위 4~5셀만 seed 1/2 추가 (3-seed 게이트)

**채점 시 필수**: `seed_noise`(reassign 전) 병기, P1 은 클래스 목록까지, 판정은 P1>P2>P3>P4 사전식
(잡음폭 안은 동률로 처리하고 다음 축으로).

### 5-B. v2 clean 학습 — **학습 속도 문제로 막혀 있음**

- pool: `data/pools/v2/unknown/strict_novel_train.json` (12,647 / Normal+known 21),
  val `strict_novel_val.json` (4,196 / novel 10), **test `strict_novel_test.json` 는 SEALED — 열지 마라**
- 입력 검증 완료: 파일 누락 0, row-order·label mismatch 0/0, train∩val 클래스 겹침 0
- 계획: plain FCMAE, seed 42→1→2, LR 4e-3, batch 16, 20ep, tag `_v2clean_s{seed}`
- **원본 경로로는 11시간/seed.** 두 가속 시도 상태는 5-C 참조
- **★ 평가는 반드시 mcs6 과 mcs≈42 두 다이얼로** (3-A 참조)

### 5-C. 학습 가속 — 두 갈래

**(1) 384 사전 리사이즈 캐시 = FAIL** (`runs/cache_equiv/VERDICT.md`)
- 원인: `_cache_images_384.py` 가 "파이프라인의 `Resize(384)` 는 항등"을 전제했는데
  **eval transform 에서만 참**. 학습은 `RandomResizedCrop((384,384), scale=(0.94,1.0))` 이라
  384 source 에서는 crop 372 → **업샘플**(원본은 6205 → 다운샘플)
- 결과: P1 30/37→23/37, ARI 0.8008→0.6196, Rule C 가 다른 epoch 선택(ep4 vs ep2),
  cache 는 select_rule 끼리도 불일치(noise→ep2, rich→ep5)
- **384 캐시는 eval 전용으로 유효** — manifest `cache_note` 에 `eval_only: true` 태그 박아둠
- **사이징 규칙**: crop 최소 선형배율 √0.94=0.9695 → `cache_size ≥ 396`. 448 채택
- **448 캐시는 생성 완료** (`E:/data/images/unknown_448/anchor_avg30_repro`, 2260/2260),
  **페어드 검증 미실행**
- ⚠ **448 도 실패하면 더 큰 캐시를 시도하지 마라** — 남는 원인은 2단 리샘플링이고 크기로 안 풀린다

**(2) DataLoader 워커 수정 = 코드 완료, GPU 검증 미실행** ← 이쪽이 우세
- 진짜 병목은 해상도가 아니라 **단일 스레드 decode**: 배치당 8.25s 중 GPU 0.58s = **93% 대기**
- 막고 있던 원인 2개 (둘 다 수정 완료):
  1. **`_may_ablation.py` 에 `__main__` 가드 없음** — 70줄 전부 top-level 실행 →
     Windows spawn 워커가 드라이버를 통째로 재실행(과거 "PID 하나에 run_dir 2개" 사고의 원인).
     `_may_repro_src.py` 에는 가드가 **있다**(1067행) — 드라이버가 범인이었다
  2. `_may_repro_src.py:195` `T.Lambda(lambda ...)` pickle 불가 → 모듈 레벨 `AddGaussianNoise` 클래스로 교체
- 검증됨(GPU 불필요): 가드 없으면 카운터 5줄 + `_check_not_importing_main` RuntimeError,
  있으면 1줄. 실제 transform pickle 왕복 + `num_workers=2` DataLoader 완주 확인
- 계획: N=8 (16 논리코어, 다른 에이전트가 ~5개 사용 중이라 13 은 과함), `persistent_workers=False` 유지
- **대조군 arm A 는 이미 있다** — `runs/may_repro/abl_paired_orig_B4_260726_114836/` +
  채점 `runs/clean546/eval_cache_equiv_orig_{noise,rich}.json`. **arm B(워커)만 돌리면 된다**
- **픽셀이 안 바뀌므로 캐시보다 안전**. RNG 스트림만 달라진다(seed 수준 변동, 데이터 분포 변화 아님)

### 5-D. 미완 항목
- v1 BrokenRing 공정성 재검증 (champion 최적점 3번째 확인점)
- temporal z0 arm 자기 최적점 격자 (`f_z0_seed{1,2,42}.npy` 존재, 실행 중 중단됨)
  ⚠ `gamma_reused_from_champion` — z0 가 champion 의 gamma 를 물려받았다. arm 별 값이면 재캘리브레이션 필요
- severstal z0 를 mcs20/ms5 에서 재측정 (#2 확정용)
- cca(7 class)에서 무라벨 셀 선택 재현 (severstal 은 P1 이 포화될 수 있어 2차 확인 필요)
- 캠페인 러너 Tier2 배선 (`scripts/run_unknown_campaign.py`)

---

## 6. 절대 규칙 / 함정

### 데이터
- **데이터셋은 `E:/data/images/<dataset>/` 에만.** 하위셋용 폴더 신규 생성 금지 — manifest 로 선택
- **하드링크·심볼릭 링크·junction 절대 금지.** 해결책은 원본 코드의 이미지 선택 경로 변경
- manifest: `{"root": "...", "files":[{"path":"<class>/<file>","label":"<class>"}...]}`,
  생성기 `scripts/make_pool_manifest.py`, 소비 `scripts/_common.py::resolve_pool()`
- **정렬은 반드시 `resolve_pool()`** — 임의 정렬 시 임베딩 행↔라벨 매칭이 조용히 깨진다
- **학습/실험 결과 폴더 삭제 절대 금지**

### 측정
- **잡음폭**: noise ±2.28pp / ARI ±0.019 / Hom ±0.005 / Comp ±0.033 / **P1 spread 0**
- **민감도 순위**(실측): P1(∞) > ARI(3.87배) > Hom(3.22배) > noise(1.72배) > **Comp(0.87배, 가장 관대)**
  → **Comp 단독 통과 금지**
- **loss 궤적 일치는 등가성의 증거가 아니다** — 소수점 3자리 일치인데 ARI 는 0.18 갈렸다
- 파이프라인 변경 등가성 1차 증거 = **Rule C 가 양쪽에서 같은 epoch 을 고르는가**
- **frozen 행이 두 arm 에서 비트 단위로 같은지** 매번 확인 (채점 하네스 건전성 대조)
- **reassign 전/후를 반드시 분리 보고**

### 실행
- **GPU 1 프로세스만.** 병렬 학습 금지
- **`REPRO_WORKERS` 는 위 5-C(2) 수정 전까지 금지** (수정됨, 단 미검증)
- `REPRO_NEG` 아니라 **`REPRO_IGNORE_NEG_SIM`** (`_may_ablation.py:44`) — 오타 시 조용히 무시됨
- `git commit <path>` 로 경로 명시 — `git add -A` 는 다른 에이전트 작업을 삼킨다
- 새 `.sh`/실행 `.py` 는 `git update-index --chmod=+x` 로 100755
- cp949 `UnicodeEncodeError`: `logger` 경로는 exit code 0(무해), **`print()` 경로는 exit 1**.
  `PYTHONIOENCODING=utf-8` 로 회피

---

## 7. 핵심 파일

| 경로 | 내용 |
|---|---|
| `_may_repro_src.py` | 트레이너 본체(~47KB). env: `REPRO_BATCH/TEMP/LR/QUEUE/IGNORE_NEG_SIM/EPOCHS/SAMPLING/SEED/BACKBONE/TAG/USE_LOCAL`. 학습 transform 191~197행 |
| `_may_ablation.py` | 셀 드라이버(70줄). `CELLS` dict 가 env 를 덮어씀(`M.CFG.update(cell_cfg)` 가 마지막) |
| `_grouping_eval.py` | 채점. `--mcs --ms --eps --method --select-rule {noise,rich_noise} --k-percentile --feat-cache --proj-dir --proj`. **frozen/z0/selected 를 자동으로 함께 출력** |
| `grouping_deploy.py` | 배포 엔트리(100755). champion 기본값 = 2-head 앙상블 + `--reassign nearest_q90` |
| `scripts/_common.py::resolve_pool()` | 디렉토리/manifest 양쪽 처리 |
| `runs/severstal/dial_sweep/REPORT.md` | 168셀 다이얼 진단 전문 |
| `runs/cache_equiv/VERDICT.md` | 384 캐시 FAIL 판정 |
| `docs/audit/z0_control_audit_260726.md` | z0 감사 414줄 |
| `runs/severstal/pilot/` | sweep 10셀 (채점 대기) |
| `~/.claude/projects/D--project-unknown-contrastive/memory/` | 메모리 (MEMORY.md 가 인덱스) |

**champion 체크포인트**:
```
runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints/proj_ep20.pt
runs/sweep/abl_best_s1_B4_260724_111053/checkpoints/proj_ep18.pt
```
**백본**: `weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth`

**B4 recipe**: temp 0.20 / queue 16384 / ignore_neg 0.72 / local 0.5 / batch 64 / 20ep / sampling 0.25 / LR_HEAD 4e-3

---

## 8. 사내 배포 시 현재까지의 요구사항

1. **frozen 을 기준선으로 삼고 사다리 ②(레시피 학습)부터 시작한다.** zero-shot 은 쓰지 마라
2. **k 는 받지 않는다** — 모르는 게 전제. 다이얼은 min_cluster_size 운영값 또는 안정성 자동선택.
   그 다음 `n/(15k)` ~ `n/(8k)` 범위만 좁게 sweep
3. **다이얼은 pool 마다 다시 정한다.** 다른 pool 값을 이식하지 마라
4. 시간축 운영점 **P10 / m_min=REF 50th-pct / K=2**, 단 **그 pool 에서 FAR 을 다시 재라**
   (배경 클래스 종수가 많을수록 frozen FAR 이 나빠진다: 배경 10종 1.25 → 21종 2.00/batch)
5. **검출 하한 ~20~90장 누적** (주입률 무관, 절대 장수로 보임)
6. 개선을 주장하려면 **z0(랜덤 head) 대조군을 용량 맞춰 함께 측정**해야 한다
