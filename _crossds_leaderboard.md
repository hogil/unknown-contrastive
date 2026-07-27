# cross-dataset / pool 채점 leaderboard

(260726 이전 이력은 `docs/archive/root_results_260726/_crossds_leaderboard.md` 로 이관됨 — 이
파일은 그 이후 새 항목만 append. 잣대/판정 규칙은 `.claude/agents/loop-analyzer.md` 참조.)

## 260726 — ★★ 신규 트랙: 시간축 novelty 탐지 (detection lag / FAR) — 최초 측정

팀리드 지시: 기존 P1~P4/ARI 는 전부 정적 지표("한 풀을 얼마나 잘 묶었나")뿐, 최종목표
("신규불량 발생 시 감지")를 재는 시간축 지표(감지 지연/오경보율)가 없었다. 이 항목이 그
정의·측정 최초 결과.

**시뮬**: `data/pools/temporal/unknown_novelty_sim/` (manifest만, 폴더 복제/링크 없음).
배경(known) = Normal + 10 known-defect class(= `unknown_train_defectaware_260710.json` 학습
클래스와 동일). novel = `CrossScratch`(마스터에만 존재, 어떤 학습 pool 에도 없음 — 진짜
미학습 클래스, 누수 자체검증 PASS). t=1-4 calibration(REF 확정) / t=5-8 FAR-test(held-out
순수배경) / t=9-14 novel-window(배치당 5/10/20/30장 sweep, 4 변형이 배경 스트림 공유).
다이얼: **HDBSCAN mcs6/ms3/leaf/eps0.06 고정**(euclidean, raw L2 backbone feature 위,
UMAP 없음 — clean546/unknown_eval100 기존 프로토콜과 동일 다이얼).

**arm**: frozen FCMAE(무학습) vs champion(=`fcmae_ad1_t010_s1_ep4` residual adapter,
`unknown_train_defectaware_260710` 로 이미 학습 완료된 기존 champion — 이번 실험은
inference-only, 재학습 0). 두 arm 이 frozen backbone forward 를 공유해 GPU 비용 절반.

> ★★ **범위 한계 (팀리드 260726 지시로 명시)**: 여기서 champion 은 시뮬의 배경(known) 도메인
> (`unknown_train_defectaware_260710`) 위에서 **이미 학습된** 적응 모델이다. 즉 이 WIN 은
> **in-domain 적응 이득**이며 `project_final_goal_ladder_260726` 의 **1순위(zero-shot, 모델
> 그대로 predict) 검증이 아니다** — champion 은 2순위(레시피학습)에 해당한다. novel 클래스
> (CrossScratch)는 champion 학습에 쓰이지 않았지만, 배경 10 known-defect + Normal 은 champion
> 이 이미 본 데이터다. "우리 모델 그대로 갖다 쓰면 FAR 0" 로 일반화하지 말 것 — 다른 배경
> 도메인(학습 안 한 known 클래스 구성)에서는 재검증 필요.

**"새 그룹 탄생" 판정** (라벨 無, 그 run 자신의 분포로 임계): REF=t4 클러스터 centroid 고정
→ calibration(t1-3 vs REF) 분포의 P-th percentile = 미매칭 임계(절대값 아님, ig72 교훈 준수)
→ 크기 ≥ m(REF 클러스터 크기 25th pct) candidate → 연속 K 배치 지속되면 그 t 에서 ALARM.
라벨은 사후채점(정적 P1-P4 + majority-label TP 판정)에만 사용.

### FAR (t=5-8, 4개 held-out 순수배경 배치 — 변형 무관, 배경 공유라 값 동일)
| 임계 P(pct) | 지속 K | frozen FAR(건/4배치) | champion FAR(건/4배치) |
|---|---|--:|--:|
| 5  | 1 | 3 (0.75/배치) | **0** |
| 10 | 1 | 5 (1.25/배치) | **0** |
| 20 | 1 | 8 (2.00/배치) | 1 (0.25/배치) |
| 5  | 2 | 3 (0.75/배치) | **0** |
| 10 | 2 | 3 (0.75/배치) | **0** |
| 20 | 2 | 5 (1.25/배치) | **0** |
| 5  | 3 | 0 | **0** |
| 10 | 3 | 0 | **0** |
| 20 | 3 | 0 | **0** |

- frozen FAR alarm 의 주인 클래스: RingDots 32건/ParallelScratches 24건/Center_scratch 24건/
  Edge-Ring_scratch 8건/Normal 16건/Edge-Top_fork 4건 — **여러 known 클래스가 동시에
  불안정**(구조적 문제). champion FAR alarm: RingDots 4건뿐(P20/K1 한 지점) — 단일 잔여
  약점, 체계적 불안정 아님.

### 운영점 P10/K1 — detection lag + 최소검출크기 (novel-window, majority=CrossScratch 인 TP만)
| 배치당 novel 장수 | frozen: lag(배치)/누적novel장/TP건수 | champion: lag(배치)/누적novel장/TP건수 |
|---|---|---|
| 5  | 1배치 / 10장 / 2건 | 3배치 / 20장 / 2건 |
| 10 | 0배치 / 10장 / 3건 | 1배치 / 20장 / 2건 |
| 20 | 0배치 / 20장 / 8건 | 0배치 / 20장 / 4건 |
| 30 | 0배치 / 30장 / 8건 | 1배치 / 60장 / 2건 |

**판정: WIN — champion 이 시간축(FAR)에서 결정적 우위, lag 은 근소 열세(트레이드오프 아님, 구조적 안정성의 대가).**
- P10/K1(가장 민감한 채택 가능 설정)에서 champion FAR=**0/4배치**, frozen FAR=**5/4배치**(1.25/배치) — **5배+ 격차**.
- 반면 frozen 의 "TP" 는 lag 은 짧아 보이지만 그 lag=0~1 배치인 alarm 들 중 다수가 사실 t=9 이전(FAR-test 구간)부터
  이미 지속되던 만성 오탐(RingDots 등)의 연장선 — 진짜 novel 신호가 아니라 "원래도 시끄럽던 클러스터가 우연히
  novel-window 로 넘어온" 착시. champion 은 같은 P/K 에서 FAR=0 이므로 TP 이벤트가 진짜 신호일 신뢰도가 높다.
- ★ **정적 vs 시간축 괴리 확인**: 배치별 정적 P1/noise%(t5-8) 는 두 arm 이 비슷했다(frozen 노이즈 13-26%,
  champion 14-24%, 둘 다 P1=1.0)인데 반해, 그 뒤에서 파생되는 FAR 는 5배 이상 차이 — **정적 스냅샷만으론
  이 차이가 전혀 안 보인다.** 시간축 지표가 정적 지표로 잡히지 않는 새 정보를 준다는 팀리드 가설이 실증됐다.
- 최소검출크기: champion P10/K1 기준 5장/배치는 검출되지만 lag 3배치(누적 20장) 필요, 10장/배치 이상은
  1배치 이내(누적 20장 내외)로 안정적 검출 — **"배치당 장수"보다 "누적 20장 내외"가 검출의 실질 문턱**으로 보임(4개
  size 모두 cum_novel_at_detect≈20 에 수렴, size30 만 60 — 표본잡음 가능성, 재현 필요).
- 운영점 추천: **champion arm, P10, K1** (FAR=0, size≥10 은 0-1배치 지연, size=5 도 3배치 내 검출).
  K 를 3까지 올려도 champion 은 이미 P10/K1 에서 FAR=0 이라 추가 이득 없이 lag 만 늘어남 — K=1 이 pareto-optimal.

### 다음 실험 1개 (정량 근거)
**champion 의 잔여 FAR 근원(RingDots, P20/K1 1건) 격리 + size05 min-detectable 재현성 검증(다른 novel 클래스로 반복)**.
- 근거: (1) RingDots 는 이미 champion 학습 pool 의 일원인데도 유일 불안정 클래스 — 클래스 자체 성질(예:
  intra-class variance ↑)인지 pool 내 비중(84/batch 누적) 문제인지 미분리. (2) size05(=cum 20장) 의 lag=3배치
  결과가 CrossScratch 고유 특성인지 일반 문턱인지는 novel 클래스 1종만으론 확정 불가 — 다른 held-out 클래스
  (예: DiagonalSmear, BrokenRing)로 동일 시뮬 반복해 "누적~20장 문턱" 가설을 교차검증해야 함.
- ★ 팀리드 승인(260726): DiagonalSmear **와** BrokenRing 둘 다 실행 — 결과는 아래 섹션에 append.

산출: `result_grouping/temporal_novelty_260726/{f0_frozen.npy,f_champion.npy,paths_index.json,
temporal_novelty_report.json,summary_tables.csv}`. 재현: `python scripts/make_temporal_novelty_pools.py`
→ `python scripts/run_temporal_novelty_embeddings.py` → `python scripts/run_temporal_novelty_analysis.py`.

## 260726 — Severstal blind rehearsal: frozen(A) vs zero-shot champion ensemble(B) vs in-domain adaptation(C, Rule C vs OLD rule)

사전등록: `runs/severstal/pre_registered_gates.json` + `runs/severstal/pre_registered_gates_amendment_01.json`
(라벨 확인 전 gate 고정 → A/random_z0 확인 후 P1 4-class 포화 발견 → 판정 우선순위를 P1(필요조건만) → P2(noise%) > P3(Comp) > ARI 로 amend, 이 amend는 라벨 안 본 상태에서 구조적 사실만으로 정직하게 결정).

- Pool: `data/pools/severstal_pilot260726.json` (995장, classid.json 기반 single-label만, class당 cap 200 — Class1=200/769, Class2=195/195, Class3=200/4759, Class4=200/516, Normal=200/5902)
- Dial: HDBSCAN mcs6/ms3/leaf/eps0.06 고정, raw L2 backbone feature, no UMAP (grouping_deploy.py 기본값과 동일).
- (C) 학습 레시피: TEMP0.20/QUEUE16384/NEG0.72/BATCH64/EPOCHS20/LR_HEAD4e-3/SEED42, severstal 무라벨 20 epoch 적응.

### 채점표

| arm | tool | k | lf noise% | off P2(noise) | P3(Comp) | P4(Hom) | ARI | AMI | frag | P1 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| (A) frozen | `_grouping_eval.py` | 17 | 76.7 | 77.7 | 0.416 | 0.876 | 0.331 | — | — | 4/4 |
| (A) random_z0 (미학습 대조군) | `_grouping_eval.py` | 17 | 79.6 | 79.8 | 0.382 | 0.822 | 0.228 | — | — | 4/4 |
| (B) zero-shot champion ensemble (reassign=nearest_q90) | `grouping_deploy.py --offline-eval` | 15 | 7.84(reassign전 noise, 참고용) | 6.16* | 0.2034 | 0.3483 | 0.179 | 0.246 | 3.75 | 4/4 |
| (C) adapt, select-rule=noise(OLD) → ep17 | `_grouping_eval.py` | 24 | 66.8 | 69.4 | 0.364 | 0.771 | 0.226 | — | — | 4/4 |
| (C) adapt, select-rule=rich_noise(Rule C, k_pct75) → ep17 | `_grouping_eval.py` | 24 | 66.8 | 69.4 | 0.364 | 0.771 | 0.226 | — | — | 4/4 |

\* (B)의 noise=6.16은 A/C와 다른 스크립트(`grouping_deploy.py`)의 `reassign=nearest_q90` 후처리 적용값 — A/C(`_grouping_eval.py`, 후처리 없음)와 파이프라인이 달라 **직접 비교 불가**(아래 캐비앗 참고). noise_pct_pp band=2.28, Comp band=0.033, ARI band=0.019 (밴드 미만 delta는 동률).

### 핵심 발견
1. **OLD rule과 Rule C가 동일 epoch(ep17) 선택** — 20 epoch 전부 gate 통과(over_merge=0, stab≥0.75, coh≥0.80), Rule C의 k≥p75(=24.00) 제한(9/20 후보) 적용 후에도 non_noise_pct 최대가 여전히 ep17이라 OLD rule과 완전 일치. 이 pool에서 Rule C는 다른 후보를 고르지 않음(무해하지만, 이 1개 pool만으론 "이점"도 입증 안 됨 — 향후 pool에서 gate-통과 epoch 수가 많고 초반 under-clustered epoch이 non_noise_pct로 우연히 이길 때 Rule C 차별점이 드러날 것으로 예상).
2. **P1 gate amendment 재확인**: frozen도 random_z0(미학습 랜덤 헤드)도 P1=4/4 — 4-class pool에서 P1은 변별력 0(랜덤도 천장 도달). amend된 P2>P3>ARI 우선순위로 판정 필요성이 실측으로 재확인됨.
3. **within-tool(같은 `_grouping_eval.py`) 비교, C vs A**: C가 P2(1순위, noise)에서 A를 이김(69.4 vs 77.7, Δ8.3pp > band 2.28pp) 하지만 P3(Comp, 0.364 vs 0.416, Δ0.052>band)와 ARI(0.226 vs 0.331, Δ0.105>band)에서는 A에 짐 — 하위축 열세폭이 band의 1.6~5.5배로 작지 않음.
4. **C vs B**: B의 noise=6.16은 압도적으로 낮지만 다른 파이프라인(reassign 후처리) 값이라 직접비교 불가. reassign 없는 지표(Comp, ARI)로는 C가 B를 이긴다(Comp 0.364>0.2034 Δ0.161, ARI 0.226>0.179 Δ0.047 — 둘 다 band 초과).

### 판정: MIXED → KEEP(준후보)
- vs (A) frozen: **MIXED** — 우선순위 1위 축(P2)은 C 승, 2·3위 축(P3/ARI)은 A 승. amend된 우선순위를 엄격 lexicographic으로 적용하면 C가 전체 승이라고도 볼 수 있으나 하위축 열세폭이 작지 않아 단순 WIN 태깅은 과장 — trade-off 표 그대로 보존, MIXED로 기록(`feedback_cap_not_sole_gate.md` 원칙 — 지표 일부 열세만으로 전면 폐기 금지, 반대로 일부 우세만으로 전면 승리 선언도 금지).
- vs (B) zero-shot ensemble: 파이프라인 불일치(reassign 유무)로 noise% 직접비교 불가 → 판정 보류. same-tool 지표(Comp/ARI)에서는 C가 B를 이김.
- 종합: **KEEP(준후보)**. 전면 폐기 사유 없음(band 초과 승패가 혼재하고 B와의 비교는 캐비앗으로 미해소) — leaderboard에 유지.

### 최적조건 도출 — 다음 실험 1개 (정량 근거)
**C(ep17 체크포인트)에 B와 동일한 `reassign=nearest_q90` 후처리를 적용해 apples-to-apples 재채점.**
근거: A/C는 raw HDBSCAN noise가 66.8~79.6%로 압도적으로 높은데 B만 유일하게 6.16%(reassign 후)다 — 이 gap이 "학습(적응) 효과"가 아니라 "후처리 효과"일 가능성이 커서, 이 실험 없이는 "adaptation이 zero-shot보다 못하다"는 결론이 성급하다. 반대로 reassign 후에도 C가 B에 크게 못 미치면, 그건 진짜 in-domain 적응이 이 champion recipe로는 zero-shot 앙상블을 못 이긴다는 확정 증거가 된다.
명령: `python grouping_deploy.py --proj runs/severstal/may_repro/abl_B4_260726_100437/checkpoints/proj_ep17.pt --pool data/pools/severstal_pilot260726.json --offline-eval` (reassign 기본 적용, mcs6/ms3/leaf/eps0.06 script default 유지).

산출: `runs/clean546/severstal_adapt_ruleC.json`, `runs/clean546/severstal_adapt_noise.json`, `runs/severstal/zeroshot_ens/{offline_summary.json,summary.json}`, `runs/severstal/adapt_train.log`, `runs/severstal/may_repro/abl_B4_260726_100437/checkpoints/proj_ep*.pt`.
재현:
```
python _grouping_eval.py --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth --pool data/pools/severstal_pilot260726.json --proj-dir runs/severstal/may_repro/abl_B4_260726_100437/checkpoints --tag sev_adapt_ruleC --mcs 6 --ms 3 --select-rule rich_noise --out-name severstal_adapt_ruleC
python _grouping_eval.py --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth --pool data/pools/severstal_pilot260726.json --proj-dir runs/severstal/may_repro/abl_B4_260726_100437/checkpoints --tag sev_adapt_noise --mcs 6 --ms 3 --select-rule noise --out-name severstal_adapt_noise
```


### 260726 addendum -- reassign apples-to-apples 해소 (위 최적조건 실험 완료, 원문 무수정)

위 "최적조건 도출"에서 제안한 "C(ep17)에 B와 동일 reassign 적용" 실험을 완료했고, 동시에 B도
reassign=none(raw)으로 재채점해 A/B/C를 완전히 같은 조건(후처리 없음)으로도 비교 가능하게
만들었다 -- "핵심 발견" 4번의 "다른 파이프라인이라 직접비교 불가" 캐비앗을 해소한다.

| arm | pipeline | reassign | k | P2(noise) | Comp | Hom | ARI | frag |
|---|---|---|--:|--:|--:|--:|--:|--:|
| (A) frozen | `_grouping_eval.py` | (없음, raw만 지원) | 17 | 77.74 | 0.416 | 0.876 | 0.331 | 4.25 |
| (B) zero-shot ens | `grouping_deploy.py --offline-eval` | none | 15 | 77.74 | 0.3211 | 0.6759 | 0.2336 | 3.75 |
| (C) adapt ep17 | `_grouping_eval.py` | (없음, raw만 지원) | 24 | 69.4 | 0.364 | 0.771 | 0.226 | 6.00 |
| (B) zero-shot ens | `grouping_deploy.py --offline-eval` | nearest_q90 | 15 | 6.16 | 0.2034 | 0.3483 | 0.179 | 3.75 |
| (C) adapt ep17 | `grouping_deploy.py --offline-eval` | nearest_q90 | 24 | 7.04 | 0.2538 | 0.5302 | 0.1596 | 6.00 |

- **raw(reassign 없음) apples-to-apples**: 이제 B도 raw로 있으므로 A/B/C가 전부 같은 조건이다.
  C가 P2(1순위, noise)에서 A/B를 모두 이긴다(69.4 vs 77.74/77.74, 둘 다 Δ>8pp>band) -- "핵심발견"3번의
  "C가 A를 이긴다"가 B에도 그대로 성립함을 확인. 단 frag(6.00 vs 4.25/3.75)와 Comp/Hom/ARI는
  C가 A에 못 미친다 -- 판정 MIXED는 그대로, "다른 파이프라인이라 불가"였던 B 비교 캐비앗만 해소.
- **reassign 후 apples-to-apples**: C(7.04)와 B(6.16)는 Δ0.88pp < band(2.28pp) → **동률**(더는
  "B가 압도적으로 낮다"고 볼 근거 없음 -- 그 격차는 인코더 우위가 아니라 reassign 후처리 유무
  차이였다). Comp는 C가 B를 band 초과로 이긴다(Δ0.0504>0.033), Hom도 C가 크게 이긴다(Δ0.182).
  ARI는 Δ0.019로 band 경계(사실상 동률).
- **결론**: adaptation(C)이 zero-shot 앙상블(B)보다 열등하다는 근거는 raw/reassign 어느 기준으로도
  없다 -- 오히려 두 기준 모두에서 C가 B와 동률이거나 앞선다. 다만 팀리드 milestone(원문 그대로,
  raw noise 기준 frozen 77.74 대비 -20pp 이상)은 **A(frozen) 대비** 정의된 것이라 이 발견과
  무관하게 그대로 미달(69.4 > 57.74)이다 -- "recipe sweep으로 더 개선 필요"라는 팀리드 결론은
  안 바뀐다. 이 addendum이 바꾸는 것은 "B(zero-shot)가 C(adaptation)보다 우월하다"는 **잘못된
  인상만**이다.
- 판정(MIXED→KEEP)은 원문 그대로 유지 -- A 대비 열세(frag/Comp/Hom/ARI)가 이 addendum과 무관하게
  남아있으므로 결론을 바꿀 근거가 아니다.

산출: `runs/severstal/zeroshot_ens_raw/{offline_summary.json,summary.json,groups.csv,offline_eval.csv}`,
`runs/severstal/adapt_ep17_reassign/{offline_summary.json,summary.json,groups.csv,offline_eval.csv}`.
재현:
```
python grouping_deploy.py --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth   --proj runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints/proj_ep20.pt runs/sweep/abl_best_s1_B4_260724_111053/checkpoints/proj_ep18.pt   --pool data/pools/severstal_pilot260726.json --out runs/severstal/zeroshot_ens_raw --offline-eval --device cpu
python grouping_deploy.py --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth   --proj runs/severstal/may_repro/abl_B4_260726_100437/checkpoints/proj_ep17.pt   --pool data/pools/severstal_pilot260726.json --out runs/severstal/adapt_ep17_reassign --offline-eval --reassign nearest_q90 --device cpu
```

## 260726 — 시간축 novelty 트랙: 2차 novel 클래스 교차검증 (DiagonalSmear + BrokenRing)

팀리드 승인 후속: 위 260726 시간축 트랙의 "누적~20장 문턱"이 CrossScratch 고유 특성인지
일반 성질인지 확인. 배경 스트림·다이얼·판정규칙 **완전 동일**, novel 클래스만 1축 교체
(`data/pools/temporal/unknown_novelty_sim_diagonalsmear/`, `..._brokenring/` — 배경
batch_01~08.json md5 동일 확인 완료, 누수 자체검증 3개 sim 전부 PASS). 임베딩은 기존
`f0_frozen.npy`/`f_champion.npy` 캐시를 재사용하고 novel 클래스 780장만 추가 추출
(`run_temporal_novelty_embeddings.py` 캐시 확장 방식으로 재작성).

### 결과 1 — FAR 은 novel 클래스와 무관하게 항상 동일 (설계상 당연 — 교차검증 아님, 명시)
P10/K1 에서 champion FAR=0, frozen FAR=5 (건/4배치) 가 CrossScratch/DiagonalSmear/BrokenRing
**세 sim 전부 완전히 동일한 수치**로 나왔다. 이는 독립 재현이 아니라 **설계가 의도한 대로
작동했다는 확인**이다 — FAR 는 t=5-8 순수배경 구간에서만 측정되고 배경은 세 sim 이 바이트
단위로 동일(md5 확인)하므로 novel 클래스가 FAR 계산에 전혀 관여하지 않는다. champion 의
유일 FAR 근원(RingDots, P20/K1)도 동일한 이유로 3개 sim 모두에서 그대로 나타난다 — "일반
성질인지 클래스 특성인지"는 이 축으로는 검증되지 않는다(애초에 검증 대상이 아니었음, 정정).

### 결과 2 — ★ 검증 대상이었던 "누적~20장 문턱"은 3개 novel 클래스에서 강하게 재현됨
champion arm, P10/K1, TP(majority=novel 클래스) 첫 검출 시점의 누적 novel 장수:

| 배치당 novel | CrossScratch | DiagonalSmear | BrokenRing |
|---|--:|--:|--:|
| 5  | 20장 (lag 3배치) | **20장 (lag 3배치)** | **20장 (lag 3배치)** |
| 10 | 20장 (lag 1배치) | 30장 (lag 2배치) | 30장 (lag 2배치) |
| 20 | 20장 (lag 0배치) | **20장 (lag 0배치)** | **20장 (lag 0배치)** |
| 30 | 60장 (lag 1배치) | 60장 (lag 1배치) | 30장 (lag 0배치) |

size05·size20 은 **3개 클래스 전부 완전 일치**(각각 lag 3배치/누적20, lag 0배치/누적20).
size10·size30 은 20~30, 30~60 범위로 근접(같은 자릿수, 완전일치는 아님 — 표본 노이즈 추정).
→ **"누적 20~30장 문턱" 가설이 CrossScratch 특이값이 아니라 이 임베딩+다이얼의 일반 성질임을
3-클래스 교차검증으로 확정**. 사내 배포 스펙 근거값으로 사용 가능: "신규불량 종류·유입속도와
거의 무관하게, 대략 20~30장이 누적되면(가장 민감한 champion/P10/K1 설정 기준) 감지된다."

### 결과 3 — frozen 의 "빠른 감지"도 3개 클래스에서 거의 완전 동일 재현 → 기존 해석(착시) 보강
frozen 은 size10/20/30 전부 **3개 클래스 모두 t=9(lag 0배치)** 로 동일 — 이는 frozen 이 novel
신호를 잘 잡아서가 아니라, frozen 이 원래도 과분할(k=23~46) 상태라 **어떤 새 동질 이미지
묶음이 들어와도 novel 클래스 종류와 무관하게 즉시 자기 클러스터를 형성**하기 때문으로 재해석.
클래스를 바꿔도 결과가 안 바뀐다는 사실 자체가 "frozen 이 CrossScratch 를 알아본 게 아니라
구조적으로 아무 새 묶음이나 빨리 분리한다"는 기존 진단(260726 상단 섹션)을 강화한다.

### 판정: 원 WIN 유지, 근거 강화 (CONFIRMED — 클래스 일반성 확보)
- FAR 5배+ 격차는 재확인 대상이 아니었음(설계상 배경-only라 novel 무관, 위에서 명시).
- ★ 새로 확정된 것: **최소검출크기(누적 ~20~30장) 가 일반 성질**이라는 점 — 이번 교차검증의
  실질 성과. 배포 스펙 문서화 시 이 값을 CrossScratch 단일측정이 아니라 3-클래스 공통값으로
  인용 가능.
- 여전히 범위 한계 적용: champion 은 배경 도메인(known 10 class+Normal) 에 in-domain 적응된
  모델 — novel 클래스 3종(CrossScratch/DiagonalSmear/BrokenRing) 모두 배경 학습 pool 에는
  없었지만, 이는 "배경 도메인 위에서 다양한 종류의 새 불량을 감지하는 능력"의 일반성 검증이지
  "전혀 다른 배경 도메인으로 zero-shot 전이"의 검증은 아니다.

산출: `result_grouping/temporal_novelty_260726/temporal_novelty_report_diagonalsmear.json`,
`temporal_novelty_report_brokenring.json` (+ 원본 `temporal_novelty_report.json` 재실행 결과
동일함을 회귀검증 완료). manifest: `data/pools/temporal/unknown_novelty_sim_{diagonalsmear,brokenring}/`.
재현: `python scripts/make_temporal_novelty_pools.py --novel-class DiagonalSmear` (또는
`BrokenRing`) → `python scripts/run_temporal_novelty_embeddings.py` (캐시 확장, 배경 재추출 없음)
→ `python scripts/run_temporal_novelty_analysis.py --sim-name unknown_novelty_sim_diagonalsmear`.


## 260726 — 시간축 v2(leakage-free) frozen arm: FAR 격자 확장 + 근본원인 규명

배경: v1 champion(`fcmae_ad1_t010_s1_ep4`) 의 학습 pool 이 SHA 감사로 eval/holdout/anchor 와
겹침 확정 -> v2(`strict_novel_train.json`, 21 known class) 로 재실행 중(champion 학습은 perf-anchor
담당, 448px 재캐시 검증 대기중). 이 항목은 그 사이 **frozen arm 만으로 먼저 끝낸** 결과.

### v1 대비 v2 frozen 자체 비교 (P10/m_min=25th-pct 고정, 배경 클래스 10->21종)
| | v1 frozen | v2 frozen |
|---|--:|--:|
| FAR K1 | 5/4배치(1.25/배치) | 8/4배치(2.00/배치) |
| FAR K2 | 3/4배치(0.75/배치) | 3/4배치(0.75/배치) |
| FAR K3 | **0** | 2/4배치(0.50/배치) |
| k(t=8) | 25 | 37 |

v1 의 "K3 면 FAR 0" 은 그 pool(10-class 배경) 특유의 성질이었다 — 배경 종수가 늘면(21종) persistence
만으로는 FAR 0 에 도달 못 한다. 격자 확장(P<5, size-filter 축) 으로 원인 분해.

### 격자 확장 결과: FAR=0 은 size-filter 보다 threshold(P) 축이 훨씬 싸다
| 운영점 | m_min | FAR(K1) | size05 lag/누적 | size10 | size20 | size30 |
|---|--:|--:|---|---|---|---|
| 기존(P10, 25pct) | 7 | 8/4배치 | 1배치/10장 | 0/10 | 0/20 | 0/30 |
| ★ P1, 25pct (싼 해법) | 7 | **0** | 1배치/10장 | 0/10 | 0/20 | 0/30 |
| P1, 100pct (비싼 해법) | 30 | 0 | **MISS(6배치 내 미검출)** | 2배치/30 | 5배치/120 | 4배치/150 |

**P1(가장 엄격한 novelty-match 임계) + 기존과 같은 얕은 size 하한(m_min=7)** 만으로 FAR=0 이
K1(가장 관대한 지속조건)에서부터 달성되고, 검출 lag 은 원래 느슨한 설정과 동일하거나 오히려 소폭
개선 — 대가가 없는 개선. DiagonalSmear/RingDots 두 novel 클래스 모두에서 재현.
size-filter 를 극단(100th pct, m_min=30)까지 올려도 FAR=0 은 되지만 size05 는 6배치 내 영구 미검출,
size20 은 lag 5배치/누적120장 — **size-filter 단독은 비싼 레버, threshold 축이 싼 레버**.

### Normal 오경보 근본원인 = (b) 경계-요동, 자연 소멸 (K≥2 만으로 이미 제거됨)
alarm 이벤트의 confirm_t/birth_t/streak 를 직접 덤프: **Normal 이 주인인 오경보는 전부 streak=1**
(그 배치 한 번 뜨고 다음 배치엔 재흡수 — 지속 안 됨). K2/K3 까지 살아남는 오경보는 전부
CenterCircle/CenterDonut/Row(=Normal 아님) — 이쪽이 (a) 진짜 구조적 소형 파편.
**결론: Normal 의 불안정성은 이미 K≥2 만으로 완전 제거된다** — size-filter 도 threshold 강화도
필요 없음. 사내 배포에서 Normal 이 다수라는 우려는 이 메커니즘상 완화된다(단, (a) 그룹은 여전히
size-filter/threshold 축이 필요).

### 판정: 코드 구조 수정 + 새 발견 2건
- FAR 를 arm-level 로 1회만 계산하도록 스키마 변경(`far_grid`) — size05/10/20/30 4행 중복 저장
  버그성 구조 제거. 두 novel-class sim 의 FAR 값이 완전히 동일함(dict 비교로 재확인)은 **버그가
  아니라 배경 공유 설계상 당연** — 보고 시 "FAR 는 1회 측정, 검출 lag/최소크기만 2-클래스 교차검증"
  명시 필수.
- ★ 신규 운영점 후보(frozen arm 기준): **P1, m_min=25th-pct(7), K1** — FAR 0, lag 무손실.
  champion arm 나오면 이 운영점 그대로 적용해 비교 예정.

산출: `result_grouping/temporal_novelty_v2_260726/temporal_novelty_report_v2_{diagonalsmear,ringdots}.json`
(신규 스키마: `arms.<arm>.far_grid[P][Mpct]` + `arms.<arm>.variants.<size>.novel_grid[P][Mpct]`).
재현: `python scripts/run_temporal_novelty_analysis.py --sim-name unknown_novelty_sim_v2_diagonalsmear
--emb-dir result_grouping/temporal_novelty_v2_260726 --arms frozen` (RingDots 는 --sim-name 만 교체).


## 260726 — v1 champion-vs-frozen 공정성 재검증 (task #27) — 원 결론 생존

배경: v2 grid 확장에서 frozen 도 P1 로 조이면 FAR 0 이 나온다는 걸 발견 → v1 의 "champion 이
P10/K1 한 지점에서 frozen 을 이겼다"는 결론이 그 측정점의 아티팩트일 수 있다는 의심(같은 날 다른
트랙에서 다이얼 아티팩트 2건이 이미 확인됨 — `project_equivalence_protocol_260726`).

**검증**: v1 report 를 확장 격자(P∈{1,2,5,10,20}×Mpct∈{0,25,50,75,100}×K∈{1,2,3})로 재계산 후,
P1/P2 는 표본 불안정(같은 날 실측 확정)이라 제외하고 **안정적인 P(10/20)만으로 각 arm 자신의
lag-최소 FAR=0 지점**을 독립 탐색.

| arm | 자기 최적점 (P,m_min,K) | FAR | lag[05,10,20,30] | 합계 | 미검출 |
|---|---|---|---|--:|--:|
| frozen | P10, m_min=10(25pct), **K3** | 0 | [3,4,4,2] | 13 | 0 |
| champion | P10, m_min=16(25pct), **K1** | 0 | [3,1,0,1] | **5** | 0 |

- **Q1(누가 더 빠른가)**: champion 압승, 합계 lag 5 vs 13(2.6배). size20 은 champion 즉시(lag0),
  frozen 은 4배치.
- **Q2(최적 P 가 다른가)**: 아니오 — **양쪽 다 P10 이 자기 최적**(원래 측정점과 우연히 일치,
  cherry-pick 아님으로 확인). 대신 **지속조건(K) 이 다르다** — frozen 은 K3 필요, champion 은
  K1 로 충분. "더 느슨한 조건에서 안전"이라는 실질적 우위가 P축 대신 K축에서 나타남.
- **Q3(역전 구간)**: 없음. 안정 P 전 구간에서 champion 상위 6개 셀이 frozen 상위 6개 셀을 전부 지배.

**판정: 원 결론 생존 — 이번엔 진짜 모델 효과, 측정점 우연 아님.** P1 이 재현됐던 [[project_equivalence_protocol_260726]]
사례(#3 "champion 이 FAR 우수 — P10 한 지점" → 검증중) 를 이 결과로 **확정(CONFIRMED)** 처리.

산출: `result_grouping/temporal_novelty_260726/temporal_novelty_report.json` (양쪽 arm 확장격자로 재계산,
기존 임베딩 재사용, 재추출 없음). `scripts/run_temporal_novelty_p1_stability_check.py` 일반화(--emb-dir/--sim-name/--arm).
