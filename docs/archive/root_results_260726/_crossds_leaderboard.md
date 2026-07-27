| 260611 | q4k_ig72 | q4k + ignore0.72 (포트폴리오값 이식) | ep1: cap 0.966/recov 0.359/noise 15.5/Hom 0.584, loss 0.087 즉사 | LOSE — 임계값은 분포 기준 이식 필요 (절대값 0.72 = 우리 분포에선 negative 몰살) |
| 260611 | (dispatch) wm811k_v3 | pool_v3 20,149장 (Normal 82% 현업분포) q4k 2ep | 진행중 | — |

## 260612 — 4-tool GPU run (ep1) + adaptor v1 (E 트랙) 중간 채점
| run | 잣대 | capture | recov | noise% | k | 판정 |
|---|---|---|---|---|---|---|
| q4k ep1 (SOTA) | FINCH p1 | **1.000** | **0.501** | 0 | 66 | 기준 |
| q4k ep1 (SOTA) | Louvain res6 | 1.000 | 0.494 | 2.1 | 57 | 기준 |
| 4tool ep1 (q4k+NeCoKL+NV0.95+LS0.02, b4) | FINCH p1 | 0.931 | 0.353 | 0 | 67 | **LOSE** |
| 4tool ep1 | Louvain res6 | 0.862 | 0.332 | 0 | 40 | LOSE |
| 4tool ep1 | UMAP+HDB 고정잣대 | 0.931 | 0.332 | 20.2 | 59 | LOSE |
| adaptor_v1 ep1 (E 트랙, γ≈0 ≈frozen) | FINCH p1 | 0.966 | 0.357 | 0 | 63 | LOSE |
| adaptor_v1 ep5/10/20 | FINCH p1 | 0.97/1.0/0.93 | 0.247/0.241/0.236 | 0 | 99/108/90 | LOSE (ep↑ 악화) |

- 4-tool 묶음(4변수 동시) ep1 패배 — 용의자 1순위 NV-filter 0.95 (hard negative 제거 = q4k 연료 차단. ig72 교훈의 재림 가능성).
- adaptor v1: ep1 정점(≈frozen 수준) 후 학습할수록 악화 — cached-aug InfoNCE 가 fmap 질서를 깎음. LR/뷰 재설계 필요.
- 다음: 4tool ep2-4 완주 후 단일변수 분해 (q4k+NeCo only / q4k+NV only / q4k+LS only, 각 2ep GPU).

## 260612 — 4-tool 묶음 최종 (ep1-4 + ens) → LOSE 확정, 분해 착수
| run | 잣대 | capture | recov | k | 판정 |
|---|---|---|---|---|---|
| q4k ep1 (SOTA) | FINCH p1 | 1.000 | 0.501 | 66 | 기준 |
| 4tool ep2 (단일 ep 정점) | FINCH p1 | 0.966 | 0.406 | 67 | LOSE |
| 4tool ep4 | FINCH p1 | 0.897 | 0.360 | 72 | LOSE (ep2 후 하강) |
| 4tool ens(ep1-4 concat) | FINCH p1 | **1.000** | 0.401 | 64 | LOSE (cap 동률→recov 패) |

- ep 곡선: 0.353 → **0.406(ep2)** → 0.378 → 0.360. ep1-정점 아님 (NeCo/LS 가 곡선 모양 바꿈).
- ens 이 cap 1.0 복구 — 묶음에서도 ep-앙상블 보정 효과 재확인.
- → 단일변수 분해 dispatch: q4k+NeCo / q4k+NV / q4k+LS (각 2ep, unknown_loop_decomp).

## 260612 — 분해 1/3: q4k+NeCo (2ep) → 거의 동률 LOSE (독 아님)
| run | 잣대 | capture | recov | Comp | Hom | k |
|---|---|---|---|---|---|---|
| q4k ep1 (SOTA, 재채점 full row) | FINCH p1 | 1.000 | 0.5014 | 0.5402 | 0.6396 | 66 |
| q4k+NeCo ep1 | FINCH p1 | 1.000 | 0.4814 | 0.5213 | 0.6325 | 69 |
| q4k+NeCo ep2 | FINCH p1 | 0.931 | 0.4214 | 0.4669 | 0.5756 | 75 |

- NeCo 단독 = 전 지표 -0.01~-0.02 미세 악화. 4-tool 붕괴(recov 0.35)의 주범 아님 → 용의자 NV/LS 로 압축.
- ep1-정점 패턴 복귀 (NeCo 만으론 곡선 모양 안 바뀜 — 4-tool 의 ep2 정점은 NV/LS 영향 추정).
- ⚠ Louvain res6 주의: _score_umapfree.py (networkx k15) 가 이전 세션 구현과 달라 SOTA 재채점 cap 0.828 (기록 1.000 과 불일치). FINCH p1 은 정확 재현 (0.5014 ✓) — Louvain 행은 구현 통일 전까지 참고용.

## 260612 — 분해 2/3: q4k+NV (2ep) → 거의 동률 LOSE (독 아님)
| run | 잣대 | capture | recov | Comp | Hom | k |
|---|---|---|---|---|---|---|
| q4k+NV ep1 | FINCH p1 | 1.000 | 0.4952 | 0.5371 | 0.6426 | 67 |
| q4k+NV ep2 | FINCH p1 | 0.931 | 0.4579 | 0.5358 | 0.6283 | 64 |

- NV 단독 ep1 = SOTA 와 사실상 동률 (recov -0.006, Hom +0.003). ep1-정점 유지.
- NeCo 무죄 + NV 무죄 → 남은 가설: LS 단독 독 vs 3-부품 상호작용 독.
- NV-only 는 patch forward 없어 GPU 8.5GB / 635s/ep (4tool 12.9GB / 1206s 대비 경량).

## 260612 — 분해 3/3: q4k+LS ep1 → MIXED (recov 신기록, capture 2클래스 상실)
| run | 잣대 | capture | recov | Comp | Hom | k |
|---|---|---|---|---|---|---|
| q4k+LS ep1 | FINCH p1 | 0.931 | **0.5386** | **0.5521** | **0.6720** | 72 |

- recov/Comp/Hom 전부 SOTA 초과 — LS 0.02 는 단독 이득 부품 (4-tool 독은 상호작용으로 확정 수순).
- capture 상실 2종 = C+EL+S, D+EL+L+S (고차 콤보 — lattice 이웃 흡수).
- 다음: ep2 + ens(ep1+ep2) — cap 1.000 복구되면 신규 SOTA.

## 260612 — ★★ 신규 SOTA: ens_q4k_ls12 (q4k ep1 + LS ep1 + LS ep2 concat+L2)
| run | 잣대 | capture | recov | Comp | Hom | k | 판정 |
|---|---|---|---|---|---|---|---|
| (구) q4k ep1 | FINCH p1 | 1.000 | 0.5014 | 0.5402 | 0.6396 | 66 | 강등 |
| q4k+LS ep2 | FINCH p1 | 0.966 | 0.5159 | 0.5411 | 0.6527 | 71 | — |
| q4k+LS ens(ep1+2) | FINCH p1 | 0.966 | 0.5607 | 0.5546 | 0.6711 | 74 | cap 1클래스 미달 |
| ★ ens_q4k_ls12 (이종 3-way) | FINCH p1 | **1.000** | **0.5572** | **0.5602** | **0.6883** | 75 | **WIN — 신규 SOTA** |
| ens +nv1 (4-way) | FINCH p1 | 0.966 | 0.5634 | 0.5609 | 0.6943 | 76 | cap 깨짐 |
| ens +nv1+nc1 (5-way) | FINCH p1 | 0.966 | 0.5710 | 0.5602 | 0.6953 | 77 | cap 깨짐 |

- 원리: q4k(전 클래스 커버) × LS(recov/순도) 이종 보완 — diversity > quantity (자매 repo H-ens 교훈 재현).
- recov +0.056, Comp +0.020, Hom +0.049 — noise 0%, FINCH p1 다이얼 0개 유지.
- 4-tool 묶음 독 = 상호작용 확정 (단독: NeCo -0.02 / NV -0.006 / LS +0.04).
- 다음: LS 이웃값 0.01 / 0.05 coordinate-descent (2ep GPU each).

### 판정 규칙 갱신 (사용자 260612) + 4/5-way 재판정
- cap < 1.0 단독 사유 탈락 금지 — 나머지 지표 압도적이면 KEEP(준후보). k 표기 = 클러스터수/전체불량수 (mixed29 N=29).
- 재판정: 4-way ens(+nv1) cap 0.966/recov 0.5634/Hom 0.6943 → **KEEP**, 5-way ens(+nv1+nc1) cap 0.966/recov 0.5710/Hom 0.6953 → **KEEP** (29종 중 28종 메인 + recov/Hom 최고치 — cap 1 SOTA 와 공동 운영 후보).

## 260612 — LS 이웃값 sweep (0.01/0.05) + 앙상블 재조합 (FINCH p1, k=클러스터/29)
| run | capture | recov | Comp | Hom | k |
|---|---|---|---|---|---|
| q4k+LS0.01 ep1 | **1.000** | 0.5172 | 0.5209 | 0.6557 | 83/29 |
| q4k+LS0.05 ep1 | 0.966 | 0.4772 | 0.5544 | 0.6274 | 58/29 |
| ens_q_l01ab (q+ls01 ep1,2) | 1.000 | 0.5531 | 0.5485 | 0.6719 | 72/29 |
| ★ ens_q4k_ls12 (챔피언 유지) | 1.000 | 0.5572 | 0.5602 | 0.6883 | 75/29 |

- LS 곡선: 0.01 → cap1.0/recov 0.517 (단일 cap-1 신기록) / 0.02 → cap0.931/recov 0.539 / 0.05 → 0.477. 스위트스폿 0.01-0.02.
- LS0.05 는 k 58 로 응집 (Comp 0.554 최고) 하지만 recov 손실 — 과한 smoothing 은 콤보 구분 뭉갬.
- q4k 부품 = cap 1.0 필수 기둥 (q 없는 조합 전부 cap<1.0).
- 다음: pool_v3 (20,149장, Normal 82% 현업분포) 에 SOTA 부품 2종 (q4k / q4k+LS0.02) GPU 재실행.

## 260612 — ens 재조합 잔여 4종 채점 (GPU 0분, 기존 부품 concat) + W2 파편화 측정 (FINCH p1, k=클러스터/29)
| run | capture | recov | Comp | Hom | k | 판정 |
|---|---|---|---|---|---|---|
| ens_q_l01a (q+ls01 ep1, 2-way) | 1.000 | 0.5497 | 0.5536 | 0.6686 | 73/29 | LOSE (챔피언 -0.008) |
| ens_q_l02ab_l01a (챔피언+ls01a, 4-way) | 1.000 | 0.5503 | 0.5499 | 0.6834 | 75/29 | LOSE (부품 추가 = 희석) |
| ens_q_l01ab_l02ab (5-way) | 0.966 | 0.5393 | 0.5722 | 0.6747 | 67/29 | LOSE |
| ens_l01a_l02ab (q 없음, 3-way) | 0.931 | 0.5276 | 0.5638 | 0.6643 | 67/29 | LOSE (q 기둥 재확인) |

- 후보 (b) "ens 에 ls01 추가/교체" 전 변형 종결 — ls01 부품은 어느 조합에서도 챔피언(0.5572) 미달. 부품 수 늘리기 = 희석 (diversity > quantity 재확인).
- ★ W2 파편화 측정 (챔피언 ens_q4k_ls12, FINCH p1 75/29): 29/29 클래스 메인 등장, 24 클래스가 2+ 클러스터로 파편 (최다 D+EL 6개, D+ER 5개). 불순 멤버 44.3%, 그중 50.8% 가 main 의 lattice 인접 (토큰 1개 차). 클러스터 centroid 최근접 = 동족 27% / lattice 인접 41% / 기타 32%.
- → 파편은 동족 이웃이 아니라 lattice 이웃 속에 박혀 있음. FINCH p2 자연병합(80→27)이 cap 1.0→0.724 파괴로 실증 — **후처리 병합으로 못 고침, 임베딩에서 동족 콤보를 당겨야 함** (LS 가 정확히 이 레버).
- 참고: p0 (k306/29) recov 0.7124 / Hom 0.8226 — 미세 파편 허용 시 상한. p1↔p0 갭 = 임베딩 개선 여지.
- v3 (pool_v3 20,149장 Normal 82%): 12:07 기준 gstep 600/10074 학습 중 (ETA ~2h) — 임베딩 생성 후 채점 예정.

## 260612 — EAC co-association 앙상블 (Fred&Jain 2005) → LOSE (음성 기록)
- base: SOTA 부품 3 임베딩 × FINCH p1/p2 (6 partition) co-assoc → average-link + lifetime cut.
- lifetime cut 양극화 (k599 미세 ↔ k21 과병합) — 중간 plateau 부재. p0 포함 시 k1240 (더 악화).
- oracle-k 진단 (k40~120): 최고 k120 = cap 0.966/recov 0.535 — **oracle 줘도 챔피언 (1.000/0.5572) 미달**.
- 결론: partition-level 합의 < concat 거리융합 (우리 임베딩들은 파편 위치까지 서로 닮아 합의가 새 정보를 못 만듦).
  paper-agent 1순위였으나 실측 기각. 후속 후보 (DBCV 선택병합 3-A, kNN-percentile noise 4-A) 는 별도 트랙 유지.

## 260612 — v3 (20,149장, Normal 82% 현업분포) q4k → 조기 종결 LOSE
| run | 잣대 | capture | recov | Comp | Hom | k |
|---|---|---|---|---|---|---|
| v1 q4k ep1 (5,149장) | FINCH p1 | 1.000 | 0.5014 | 0.5402 | 0.6396 | 66/29 |
| v3 q4k ep1 (20,149장) | FINCH p1 | 0.931 | **0.2745** | 0.3744 | 0.4528 | 71/29 |

- v3 ep1 은 v1 ep1 의 4배 step 을 쓰고도 recov 반토막 — **데이터 양 < 데이터 구성**.
  Normal 82% 풀에선 pair 대부분이 Normal끼리 → loss(0.643 < v1 0.769) 가 Normal 분산 정리에 소진, defect 구조 희석.
- master 조기 종결: ep2/ls02 run 정보가치 낮음 — task 삭제, GPU 를 LS0.015 (analyzer 1순위) 로 전환.
  교훈: 현업분포 풀을 쓰려면 Normal 다운샘플 or Normal-aware 페어링 (과제중립 범위 내) 필요.
- ckpt/ep1 npy 보존 (삭제 안 함). 30분 자가치유 schtasks 패턴 (unknown_loop_ls015) 계승.

## 260612 — LS 0.015 (스위트스폿 보간점) → LOSE, LS 0.02 앙상블 파트너 확정
| run | 잣대 | capture | recov | Comp | Hom | k |
|---|---|---|---|---|---|---|
| ls015 ep1 단독 | FINCH p1 | 0.966 | 0.5221 | 0.5527 | 0.6532 | 66/29 |
| ens [q4k, ls015 ep1,2] | FINCH p1 | 1.000 | 0.5469 | 0.5607 | 0.6717 | 69/29 |
| ★ ens_q4k_ls12 (챔피언 유지) | FINCH p1 | 1.000 | 0.5572 | 0.5602 | 0.6883 | 75/29 |

- 0.015 는 정직한 보간점 (cap/recov 모두 0.01~0.02 사이) — 앙상블에선 0.02 부품이 우월 (recov 차 +0.010).
- LS 다이얼 종결: 0.01/0.015/0.02/0.05 4점 — 단독 cap-1 은 0.01, 앙상블 파트너는 0.02.
- 다음: SoftNCE (top-K hardest negative 에만 smoothing — paper 2순위) + kNN-percentile noise 후처리 (W4, CPU).

## 260612 — ★ kNN-percentile noise 후처리 (W4 첫 돌파) — 챔피언에 채택
| run (FINCH p1) | capture | recov | noise% | nz→noise% | k |
|---|---|---|---|---|---|
| ens_q4k_ls12 (후처리 전) | 1.000 | 0.5572 | 0 | 0.0 | 75/29 |
| ★ + kNN 15-th NN cosine 거리 상위 2% → -1 | **1.000** | 0.5566 | 0.21 | **28.0** | 75/29 |
| + 5% | 1.000 | 0.5455 | 3.31 | 30.0 | 75/29 |
| + 10% | 0.966 | 0.5255 | 8.14 | 37.0 | 75/29 |

- FINCH 의 noise 부재 (전원 배정) 보완 — R/Normal 28% 를 noise 로 정답 배치하면서 cap/recov 무손실 (-0.0006).
- percentile (상대) 기준 — ig72 절대 임계 교훈 준수. 2% 가 스위트스폿 (5% 부터 defect 희생 시작).
- 신규 운영점: **ens_q4k_ls12 + FINCH p1 + kNN2% 후처리**. paper-agent 4-A 채택 (1순위 EAC 는 기각, 2/4순위가 적중).

## 260612 — ★★★ 신규 SOTA: ens_q4k_ls12_soft1 (4-way) + kNN2% 운영점
| run (FINCH p1) | capture | recov | Comp | Hom | nz→noise% | k |
|---|---|---|---|---|---|---|
| softnce ep1 단독 (LS0.02→top-20 분배) | 1.000 | 0.5234 | 0.5452 | 0.6607 | — | 70/29 |
| softnce ep2 단독 | 1.000 | 0.4759 | 0.5002 | 0.6342 | — | 85/29 |
| (구 SOTA) ens_q4k_ls12 | 1.000 | 0.5572 | 0.5602 | 0.6883 | — | 75/29 |
| ★ ens_q4k_ls12_soft1 (4-way: q+ls02e1,2+soft1) | **1.000** | **0.5869** | **0.5693** | **0.7043** | — | 77/29 |
| ★★ 운영점 = 위 + kNN2% 후처리 | 1.000 | 0.5862 | 0.5694 | 0.7045 | **27.0** | 77/29 |

- SoftNCE ep1 = 단일 cap-1 신기록 (0.5234) — top-K 분배가 균등 LS 의 고차콤보 capture 손실을 막음 (가설 적중).
- 이전 4-way (nv/neco/ls01 추가) 는 전부 cap 붕괴였는데 softnce 는 **자기가 cap-1 이라 희석 없이 diversity 만 추가**.
- 하루 누적: recov 0.5014 → 0.5572 → **0.5869** (+0.0855, W1), nz→noise 0 → 27% (W4).
- 다음 후보: softnce topk 사다리 (10/40), softnce+ls02 혼합 비율, DBCV 선택병합 (W2 잔여).

## 260612 — topk 사다리 + DBCV 선택병합 판정
| run (FINCH p1) | capture | recov | Comp | Hom | k | 판정 |
|---|---|---|---|---|---|---|
| soft k10 ep1/ep2 | 0.931 / 0.931 | 0.491 / 0.461 | — | — | 66, 62/29 | LOSE (질량 과집중) |
| soft k20 ep1 (SOTA 부품) | 1.000 | 0.5234 | 0.5452 | 0.6607 | 70/29 | ★ 정점 |
| soft k40 ep1 | 0.966 | 0.4986 | 0.5435 | 0.6507 | 70/29 | LOSE |
| SOTA + DBCV-gated p1→p2 병합 (6건) | 0.966 | 0.5697 | 0.5883 | 0.6869 | 69/29 | LOSE (cap 희생 > Comp 이득) |

- **topk=20 오목 정점 확정** (10/20/40 사다리). SoftNCE 다이얼 종결.
- DBCV 병합: 무라벨 게이트로도 "유일 메인 흡수" 못 막음 — W2 는 여전히 임베딩 쪽 수단만 유효 (analyzer 측정과 합치).
- 다음: head 사다리 (ad/linear, q4k+soft k20 레시피 위, repo-agent 경고대로 직접 dispatch — 옛 supervisor queue=0 함정 회피).

## 260612 — head 사다리 종결: mlp(BN 2층) 압승
| run (FINCH p1) | capture | recov | k | 판정 |
|---|---|---|---|---|
| mlp head (SOTA 부품, softnce k20 ep1) | 1.000 | 0.5234 | 70/29 | ★ 유지 |
| ad head (Dropout→512→ReLU) ep1/2 | 0.966 / 0.966 | 0.491 / 0.420 | 63, 52/29 | LOSE |
| linear head ep1 | 0.655 | 0.203 | 27/29 | 대패 (loss 0.29 collapse — 비선형 projector 필수 재현) |

- "작은 head = 덜 오염" 가설 기각 — InfoNCE 불변성 흡수에 BN+비선형 필수 (SimCLR 원논문 발견 재현).
- AD repo 패턴 (작은 head 승) 은 supervised CE 용 — contrastive projector 엔 외삽 불가가 교훈.

## 260612 — SCE λ0.5 기각 + 앙상블 폭 포화 확정
| run (FINCH p1) | capture | recov | k | 판정 |
|---|---|---|---|---|
| sce ep1 (λ0.5, τt0.07) | 0.897 | 0.4669 | 68/29 | LOSE (관계질량 50% = 과 smoothing) |
| sce ep2 | 0.931 | 0.4345 | 59/29 | LOSE |
| 5-way ens (s2 / l05b / l01a 추가 3종) | 1.000 | 0.5655~0.5793 | 75~78/29 | LOSE — 4-way 가 폭 최적 |

- LS family 전 탐사 종결: 균등 0.01-0.05 / topk 10-40 / SCE λ0.5 → **승자 = 균등 0.02 + topk 20 (SoftNCE)**.
- 앙상블 폭 4가 포화점 (3-way 0.5572 < 4-way 0.5869 > 5-way 0.579 — 오목).
- backlog: SCE λ0.9 (가벼운 관계질량) — 후순위.
- 진행: defect-only 풀 (Normal 0%) q4k 2ep — "학습풀 Normal 필요성" 격리 (v3 기울기 추적).

## 260612 — 데이터 구성 곡선 완성: Normal 비율 오목, v1 (29%) 정점권
| 풀 (q4k 레시피, FINCH p1 ep1) | Normal% | capture | recov | 판정 |
|---|---|---|---|---|
| defect-only 3,649장 | 0% | 0.931 | 0.4317 | LOSE — Normal = 유익한 대비 배경 |
| ★ v1 5,149장 | 29% | 1.000 | 0.5014 | 정점권 (현행 유지) |
| v3 20,149장 | 82% | 0.931 | 0.2745 | LOSE — Normal 잠식 |

- Normal 비율 0/29/82% 3점 오목 곡선 — 구성 트랙 종결, v1 유지. (backlog: 50% 점 — 후순위)

## 260612 — 옛 트랙 (7-class, q4096 temp0055 시대) 대조 분석 (사용자 제공 표)
- **queue 사다리 오목 재확인**: 옛 q5120 < q4096 (image_cap 0.437<0.467) — 우리 q8k/q16k 후보 영구 강등.
- **local grid 무익 재확인**: 옛 "local grid=8 개선 없음" — 큐의 q4k+local_grid 항목 제거.
- **ignore0.72 패턴 = NV-filter 판정과 동형**: negative 제거류는 한 지표 소폭 ↑ + 나머지 ↓ — 계열 전체 닫음.
- **★ 파편비 ~2.5× 트랙 불변**: 옛 7→13~19 (1.9~2.7×), 현 29→66~83 (2.3~2.9×) — W2 는 데이터 아니라
  InfoNCE+밀도클러스터링의 구조 성질. "임베딩 당기기만 유효" 진단 강화.
- **과분할 상한 동형**: 옛 7/183 image_cap 0.865 ↔ 현 FINCH p0 recov 0.71 — 미세 partition 회수 상한 존재.
- **retrieval (top1/top5) 보조지표 이식** (`_score_umapfree.py`): SOTA ens 0.686/0.598, q4k 0.614/0.518,
  softnce 0.622/0.551 — recov 와 동조. 옛 0.905/0.883 은 in-domain 7-class 라 직접 비교 불가 (과제 난이도 차).

## 260612 — hybrid smoothing 기각, smoothing 트랙 완전 종결
| run (FINCH p1) | capture | recov | k(불량/전체/29) | 판정 |
|---|---|---|---|---|
| hybrid (uni0.01+topk20×0.01) ep1/ep2 | 0.966/0.931 | 0.523/0.506 | 69/75, 72/75 | LOSE |
| ens 부품교체 (soft1→hyb1) | 1.000 | 0.5766 | 84/89 | LOSE (챔피언 0.5869) |

- smoothing 설계 공간 소진: 균등(4점)/topk(3점)/SCE(1점)/hybrid(1점) — 승자 = topk20 0.02 순정.
- 다음: dino_fixed_v2 구현 (K128 + kmeans init + head warmup — 비-InfoNCE 계열 마지막 카드).

## 260612 — transductive mixed29 (도메인 적응) 판정: 단일 강세, 앙상블 LOSE
| run (FINCH p1) | capture | recov | k(불량/전체/29) | retrieval t1/t5 | 판정 |
|---|---|---|---|---|---|
| tx ep1 | 0.966 | 0.5297 | 74/79 | 0.599/0.529 | — |
| tx ep2 | 1.000 | 0.5228 | 74/81 | 0.634/0.562 | 단일 cap-1 공동 2위 (★ep2 정점 — tx 는 곡선 다름) |
| 5-way champ+tx2 | 0.966 | 0.5828 | 76/81 | **0.717/0.633 (신기록)** | LOSE (cap 깨짐) |
| 4-way soft1→tx2 교체 | 1.000 | 0.5793 | 77/82 | 0.712/0.630 | LOSE (0.5869 미달) |

- 도메인 적응 단일 효과 실증 (+0.02 vs q4k) — 6분/ep 라 가성비 최고 부품군.
- 단 v1-부품들과 조합 시 cap 충돌 — retrieval 은 신기록인데 FINCH 회수로 미전환 (granularity 불일치 추정).
- 챔피언 유지: ens_q4k_ls12_soft1 + kNN2% (1.000/0.5862/77,82/29).

## 260612 — τ 0.1 기각 (τ 다이얼 종결: 0.05 유지)
| run (FINCH p1) | capture | recov | k(불량/전체/29) | 판정 |
|---|---|---|---|---|
| t01 ep1 | 0.931 | 0.4531 | 52/58 | LOSE — τ↑ = hardness 페널티 둔화 → 국소 분별 손실 |

## 260612 — τ0.1 / NN-positive 기각 (신호-설계 3종 트랙 마감)
| run (FINCH p1) | capture | recov | k(불량/전체/29) | retrieval t1 | 판정 |
|---|---|---|---|---|---|
| t01 ep1/ep2 (τ 0.1) | 0.931/0.966 | 0.453/0.404 | 52/58, 80/86 | 0.560 | LOSE — τ 0.05 종결 |
| nnpos ep1 (NNCLR w0.3) | 0.966 | 0.491 | 70/76 | 0.553 | LOSE — lattice 인접 NN 이 가족 병합 견인 (예측된 부작용 실증) |

## 260612 — dino_fixed_v2 종결 (mixed29): collapse 수리 성공, 성능 LOSE
| run (FINCH p1) | capture | recov | k(불량/전체/29) | retrieval t1 | 판정 |
|---|---|---|---|---|---|
| dino_fixed_v2 ep1 (head warmup) | 0.931 | 0.381 | 54/60 | 0.471 | — |
| dino_fixed_v2 ep2 (bb 학습) | 0.966 | 0.399 | 61/67 | 0.474 | LOSE — v1 collapse (ARI 0.17) 는 고침, InfoNCE 에 0.12 열세 |

- v2 3부품 (K128 + kmeans init + head warmup) 구현 완료 — 수리 자체는 paper ablation 행 가치.
- DINO 계열 종결. 비-InfoNCE 카드 소진 → single7 신트랙으로 자원 전환 (frozen duo 0.9571 기준선).

## 260612 — ★ single7 (far-novel) 신트랙 개설 — frozen duo 가 무학습 SOTA
| run (FINCH p1, k=불량/전체/7) | cap | recov | k | Comp | Hom | ARI | retrieval |
|---|---|---|---|---|---|---|---|
| DINOv3 frozen | 1.000 | 0.8886 | 20/25 | 0.587 | 0.859 | 0.461 | 0.937 |
| FCMAE frozen (★단일 역전) | 1.000 | 0.9000 | 16/20 | 0.677 | 0.895 | 0.560 | 0.960 |
| ★ duo concat (학습 0) | **1.000** | **0.9571** | 15/20 | 0.703 | 0.935 | 0.652 | 0.971 |

- multi-label combo 가 난이도 본체였음이 확정 (단일은 frozen 으로 거의 해결). FCMAE 가 single 에선 DINOv3 역전.
- HDBSCAN raw (옛 다이얼) 작동하나 과병합 (cap 0.57-0.71) — FINCH 본선 유지.
- 학습 정당 행: frozen / NR (Normal+Random만) / tx (무라벨 transductive). WM-811K 단일 학습 모델 = class 재발견이라 무효.

## 260612 — single7: NR 곡선 상승 + 3-way SOTA 갱신
| run (FINCH, k=불량/전체/7) | cap | recov | k | Comp | Hom | ARI | 비고 |
|---|---|---|---|---|---|---|---|
| NR ep2 단독 p1 | 1.000 | 0.9371 | 19/27 | 0.623 | 0.899 | 0.491 | ep1→ep2 상승 — ep4 연장 |
| ★ NR ep2 단독 p2 (해석모드) | 1.000 | 0.9257 | 7/9/7 | 0.865 | 0.873 | **0.8314** | k=정답 수렴 + ARI 신기록 |
| ★ ens [nr2, FCMAE, D3] p1 | 1.000 | 0.9629 | 17/24 | 0.694 | **0.9437** | 0.635 | SOTA (Comp 기준 nr1판 제침) |
| 4-way (nr1+nr2+듀오) | 1.000 | 0.9486 | 17/24 | — | — | — | 희석 (폭 3 포화 재현) |

## 260612 — single7: adapter (txad) 검증 + SOTA 갱신
| run (FINCH, k=불량/전체/7) | cap | recov | k | Comp | Hom | ARI | 비고 |
|---|---|---|---|---|---|---|---|
| txad ep1→4 (γ=0 하한 출발) | 1.0 | 0.886→0.920→0.891→0.937 | — | — | — | — | frozen 바닥 보장 작동, ep8 연장 |
| ★ 회수모드 SOTA [NR2,FCMAE,txad2] p1 | 1.000 | **0.9657** | 18/24 | 0.676 | 0.9438 | 0.612 | 신기록 |
| ★ 해석모드 [txad2,FCMAE,D3] p2 | 1.000 | 0.9314 | **7/8** | 0.871 | 0.896 | 0.8276 | k=정답+cap1 동시 |
| 4-way | 1.000 | 0.9457 | 17/22 | — | — | — | 폭 3 포화 3번째 재현 |
- adapter head (γ=0 residual) = E트랙 v1 실패를 이미지 직접 학습으로 살려냄. 55s/ep 사살불가 체급.

## 260612 — ★ single7 최종 SOTA (금일): ens [NR2, FCMAE, txad4]
| 모드 | 구성 | cap | recov | k | Comp | Hom | ARI |
|---|---|---|---|---|---|---|---|
| ★ 회수모드 (p1) | [NR ep2, FCMAE frozen, txad ep4] concat | 1.000 | 0.9657 | 16/23/7 | 0.7145 | 0.9449 | 0.6687 |
| ★ 해석모드 (p2) | [txad ep4, FCMAE frozen, DINOv3 frozen] concat | 1.000 | 0.9286 | **7/8/7** | 0.875 | 0.893 | 0.8303 |
- txad 곡선: 0.886→0.920→0.891→**0.937(ep4)**→0.931→0.897 — ep4 정점.
- 하루 궤적 (single7): frozen 0.889 → duo 0.957 → +NR 0.963 → **+adapter 0.9657** / 해석모드 ARI 0.74→0.83.

## 260613 — MoCo (momentum encoder) 첫 검증 + 하드 데이터셋 난이도
### hard43 (E:/unknown 42-class 합성 compositional, far-novel, FINCH p1)
| run | cap | recov | Comp | Hom | ARI | retrieval | 판정 |
|---|---|---|---|---|---|---|---|
| DINOv3 frozen | 1.000 | 0.8476 | 0.801 | 0.894 | 0.600 | 0.921 | - |
| FCMAE frozen | 1.000 | 0.9565 | 0.811 | 0.972 | 0.623 | 0.980 | FCMAE 합성 압도 |
| frozen 듀오 (학습 0) | 1.000 | 0.978 | 0.831 | 0.983 | 0.679 | 0.986 | hard43 SOTA |
| MoCo ep1/ep2 (Normal만) | 1.000 | 0.742/0.720 | - | - | 0.43 | 0.871 | LOSE frozen -0.26 |
- 핵심: 42-class 합성이 오히려 쉬움 (frozen 0.978). 난이도 = class 수 아니라 multi-label 중첩. single7 0.957 / hard43 0.978 / mixed29 0.586.
- MoCo = frozen 대비 큰 LOSE (Normal-only 정보부재 + 합성은 백본이 이미 풀어 마진 0).
- FCMAE = 합성 압도(0.957) but mixed29 독(0.441). 백본 선택 = 트랙 의존.
- mixed29 클러스터러 교체 전부 FINCH 미달 (agglo <=0.557, spectral 0.572) = 임베딩 천장.

## 260613 — mixed29 긴-epoch 검증 (사용자: epoch 적은게 문제?) + 값-스윕 배터리 착수
- adapterN3_mx (transductive, γ unbounded) 곡선: ep1 0.382/ep2 0.381/ep3 0.396/ep4 0.402/ep5 0.386/ep6 0.406/ep8 (아래) — 0.38-0.41 박스 정체. cap 0.97→0.86 악화. mixed29 transductive adapter = combo lattice 당김 구조 한계 0.40.
- MoCo 장기 (ckpt 이어 ep4): 0.477→0.461→0.419 단조 하락 — epoch 더 줘도 악화 (declining 확정 제외).
- 교훈: epoch 효과 method 의존 — adapter 미세상승후 정체 / MoCo 하락. 둘 다 SOTA(0.586) 미달.
- paper 처방 적용: adapter scale tanh-bound (AdaptFormer s<1) = ep↑ 악화 처방, 신규 run 부터 적용.
- 값-스윕 8종 착수 (sw_lsmass02/01 분배량, ls025/03, topk15/25/30, sce09) — 누적 전략으로 SOTA 갱신 시도.

## 260613 — ★ 값-스윕: SoftNCE 분배량 (paper 적중)
| run (FINCH p1) | cap | recov | Hom | k(불량/전체/29) | 판정 |
|---|---|---|---|---|---|
| 분배량 0.02 (기존 softnce topk20) ep1 | 1.000 | 0.5234 | 0.661 | 70/74 | 기존 |
| ★ 분배량 0.2 (paper 권장) ep1 | 1.000 | 0.5276 | — | — | 단일 신기록 |
| ★ +mass0.2 5-way | 0.966 | 0.5966 | 0.7087 | 72/77 | KEEP 공동SOTA (recov+0.010) |
| swap cap-1 [q,ls1,ls2,mass02] | 1.000 | 0.5841 | — | — | cap-1 유지, SOTA 살짝 아래 |
- paper 발견 실증: 우리 분배량 0.02 = 논문 권장 0.2 의 1/10. 0.2 로 올리니 단일 +0.004, 앙상블 recov +0.010.
- 두 운영점 공존: cap1.0/0.5862 (전 클래스 발견) vs cap0.966/0.5966 (회수 우선, 1 콤보 손실).

## 260613 — 값-스윕 8종 최종 (사용자: 값 조절 여러 번) → mass0.2 유일 WIN
| 스윕 단일 (FINCH p1) | cap | recov | 판정 |
|---|---|---|---|
| ★ mass 0.2 (분배량, paper값) | 1.000 | 0.5276 | WIN — 단일 신기록, 앙상블 0.5966 |
| mass 0.1 | 1.000 | 0.4676 | 골짜기 (0.02·0.2 쌍봉) |
| ls 0.025 / 0.03 | 1.0/0.931 | 0.50 / 0.47 | LS 정점 0.02 확정 |
| topk 15 / 25 / 30 | 0.966/0.931/1.0 | 0.51/0.54/0.45 | topk 정점 20 확정 (25 cap↓) |
| sce λ0.9 | 0.931/1.0 | 0.43/0.46 | SCE 미달 |
- 8종 중 mass0.2 만 additive. SoftNCE 분배량 = 우리 0.02 → 논문 0.2 가 정답 (10× 미탐이 핵심 발견).
- ★ 신규 운영점 ens_sota5_addm1 [q,ls1,ls2,softnce20,mass0.2] = cap0.966/recov0.5966/Comp0.579/Hom0.709/ARI0.285/Sil0.113 (72/77/29) — cap 외 6지표 중 5개 SOTA 초과 → KEEP 공동SOTA.

## 260613 — ★ Baseline 경쟁 완료 (Table 1b, mixed29 FINCH p1 best-ep recov)
| method | cap | recov | 순위 |
|---|---|---|---|
| SimCLR q4k | 1.000 | 0.5014 | ★ 1 (선택) |
| Barlow Twins | 1.000 | 0.4959 | 2 |
| MoCo m0.99 | 0.966 | 0.477 | 3 |
| SimSiam | 0.931 | 0.4276 | 4 |
| VICReg | 1.000 | 0.4048 | 5 |
| BYOL | 1.000 | 0.3703 | 6 |
| DINO | 0.862 | 0.3517 | 7 |
- 옛 novel-track ARI 순위와 일치. SimCLR 선택 확정. 논문 4-표 구조 완성 (docs/paper/PAPER_STRUCTURE_260613.md).

## 260613 — Barlow 이종 앙상블 = 파편(W2) 레버 (recov 와 직교)
| 운영점 | cap | recov | ARI | Sil | 클러스터(불량/전체/29) |
|---|---|---|---|---|---|
| recov-max [q,ls1,ls2,softnce,mass02] | 0.966 | 0.5966 | 0.285 | 0.113 | 72/77 |
| ARI-max [+barlow 6-way] | 0.966 | 0.5834 | 0.2912 | — | 65/70 |
| cap-1 SOTA [q,ls1,ls2,softnce] | 1.000 | 0.5862 | 0.255 | 0.081 | 77/82 |
- Barlow(공분산 method) diversity = 파편 77→70, ARI +0.036. mass0.2(recov 레버)와 보완. 운영점 3분화.
- 발견: 이종 SSL method 앙상블이 same-method ep 앙상블보다 파편 줄임 — baseline 경쟁의 2위(Barlow) 가 부품 가치.

## 260614 — TEMI (SCAN-loss head, frozen-feature deep clustering) C-sweep: 클러스터수↔capture trade-off 정량화
| 설정 | cap | recov | ARI | 클러스터(불량/전체/29) |
|---|---|---|---|---|
| FINCH (기존 클러스터러) | 1.000 | 0.586 | 0.255 | 77/82/29 |
| TEMI C=45 | 0.897 | 0.469 | 0.269 | 43/45/29 |
| TEMI C=35 | 0.897 | 0.451 | 0.282 | 33/35/29 |
| TEMI C=30 | 0.793 | 0.401 | 0.280 | 28/30/29 (≈이상) |
- ★ 클러스터수 29 압축 가능하나 capture 동반 하락 (1.0→0.79). multi-label 콤보의 근본 trade-off 정량 증명:
  현 임베딩으론 "29 깔끔 클러스터 + 29 유형 전수발견" 동시 불가 (C+EL+S ↔ C+EL lattice 중첩).
- 첫 구현 TEMI-PMI 는 붕괴(k=1) → SCAN 엔트로피 항(λ5)으로 교체 후 정상. head 선택=train-loss 최저.
- TEMI recov < FINCH (같은 임베딩) — 클러스터러로는 FINCH 우위. TEMI 가치 = "29 근처 해석모드" 단독.

## 260614 — ★★★ 신규 cap-1 SOTA: +KoLeo (clean 천장 돌파 0.586→0.605)
| 운영점 (FINCH p1) | cap | recov | Comp | Hom | ARI | k(불량/전체/29) |
|---|---|---|---|---|---|---|
| 기존 cap-1 SOTA [q,ls1,ls2,softnce20] | 1.000 | 0.5862 | 0.569 | 0.704 | 0.255 | 77/82 |
| ★ +KoLeo [+m_koleo ep1] (5-way) | 1.000 | **0.6048** | 0.562 | 0.711 | 0.247 | 82/86 |
| +KoLeo+mass (6-way) | 1.000 | 0.5966 | 0.563 | 0.710 | 0.242 | 82/87 |
- m_koleo = SimCLR + SoftNCE-topk20 + KoLeo uniformity reg(0.1). 단일 cap-1 신기록 0.5621 (mass0.2 0.5276 제침).
- ★ cap 손실 없이 0.586 천장 돌파 (+0.0186). 오늘 처음 cap-1 갱신. KoLeo(DINOv2 uniformity)가 진짜 additive 부품.
- 배터리 결산: 분배량/topk 2D = (0.2,20) 단일봉우리 종결 / dcl·hardneg = 파편레버(recov 약) / koleo = ★WIN.

## 260614 — KoLeo 배터리 결산 (강도 스윕 + 결합)
| run 단일 (FINCH p1) | cap | recov | 판정 |
|---|---|---|---|
| koleo 0.05 | 1.000 | 0.5421 | < 0.1 |
| ★ koleo 0.1 (m_koleo) | 1.000 | 0.5621 | 정점, 신규 SOTA 부품 |
| koleo 0.2 | 1.000 | 0.4531 | < 0.1 |
| koleo 0.5 | 0.897 | 0.4166 | 하락 |
| koleo×mass0.2 결합 | 1.000 | 0.54 | 시너지 없음 (별도 부품이 최선) |
- KoLeo 강도 정점 0.1 단일봉우리. 두 WIN(koleo+mass) 한 모델 합성 시너지 X.
- ★ 신규 SOTA 확정: ens_s1_ko [q,ls1,ls2,softnce20,koleo0.1] cap1.0/recov0.6048/Hom0.711 (82/86/29).
- 누적: 0.5014→0.5572→0.5862→0.6048 (cap1.0 유지).

## 260614 — dual-temperature (arXiv:2203.17248) 음성 결과
| run 단일 (FINCH p1) | cap | recov | 판정 |
|---|---|---|---|
| dt_softnce (dual-τ+softnce) | 1.000 | 0.5028 | < softnce 0.5234 |
| dt_koleo (dual-τ+softnce+koleo) | 0.966 | 0.5048 | < koleo 0.5621 (간섭) |
- dual-τ per-anchor 가중이 이미 튜닝된 SoftNCE/KoLeo 설계와 간섭 → 무익. queue-4096 few-neg 처방이나 우리 트랙엔 X.

## 260614 — local-grid (DenseCL) 현 mixed29 첫 실행 → 무익 (옛 7-class 결과 재확인)
| run (FINCH p1) | cap | recov | k | 판정 |
|---|---|---|---|---|
| m_local ep1 (--local 1.0) | 0.828 | 0.362 | 53/58 | LOSE (42분/ep 느림) — local-grid 양 트랙 무익 확정 |
- NeCo·MoCo·local-grid 전부 현 mixed29에서 무익 재확인. patch-level/momentum 계열 = wafer 도메인 부적합.

## 260614 — ★ unknown 데이터셋 held-out (21/21 disjoint) frozen baseline
| 운영점 | cap | recov | ARI | 클러스터(불량/전체/21) |
|---|---|---|---|---|
| DINOv3 frozen | 1.000 | 0.789 | 0.525 | 42/48/21 |
| FCMAE frozen | 1.000 | 0.962 | 0.601 | 54/60/21 |
| duo frozen (학습0) | 1.000 | 0.970 | 0.628 | 52/58/21 |
| duo frozen p2 (해석) | 0.905 | 0.867 | 0.846 | 19/21/21 (이상 도달) |
- ★ held-out 진짜 novel 인데도 frozen 0.970 — single-label 합성은 백본이 풂. 난이도=라벨중첩, mixed29 만 난제 재확정.

## 260615 — ★ 파편 병합 (centroid 계층병합, 라벨無) — 클러스터↓ + ARI↑
| 병합 후 k | cap | Comp | Hom | ARI | Sil | k(전체/29/noise) |
|---|---|---|---|---|---|---|
| 없음(SOTA) | 1.000 | 0.562 | 0.711 | 0.247 | 0.093 | 86/29/4 |
| → 60 | 0.931 | 0.587 | 0.664 | 0.302 | 0.122 | 60/29/4 |
| → 40 | 0.862 | 0.607 | 0.621 | 0.312 | 0.174 | 40/29/4 |
| → 29 | 0.655 | 0.621 | 0.560 | 0.274 | 0.238 | 29/29/4 |
- 86→60 병합: 클러스터 -26, ARI +0.055/Comp +0.025/Sil +0.029, cap만 -0.069 (2종 손실). 군집 품질 향상.
- 절대임계 centroid 병합은 거리집중으로 붕괴 → 목표-k 계층병합(average-link)이 정답. 60-40 스위트스폿, 29는 cap 0.655 (콤보 겹침 한계).

## 260725 — FCMAE 어댑터 temp 0.10 seed1 고정잣대 교차검증 (pool=unknown_eval100, 32 target ≠ mixed29)
run tag: `fcmae_ad1_t010_s1_ep4` (residual adapter, pure SimCLR, freeze-backbone, temp0.10, ep4, seed1)
설정: 채점 pool `data/images/unknown_eval100` (4149장, 32 strict-novel target, excl 13), 고정잣대 4-클러스터러 병행.

| emb (seed1 ep4) | FINCH-p2 P1/recov | Louvain-res6 P1/recov | ★UMAP-nn10 P1/recov | UMAP-nn15 P1/recov |
|---|---|---|---|---|
| frozen (기준) | 32/32 · 0.9259 | 31/32 · 0.9309 | 30/32 · 0.9281 | 27/32 · 0.8387 |
| temp0.05 | 29/32 · 0.8347 | 31/32 · 0.9366 | 27/32 · 0.8387 | 27/32 · 0.8384 |
| temp0.10 | 32/32 · 0.9244 | 31/32 · 0.9322 | 27/32 · 0.8400 | 27/32 · 0.8394 |

판정: **MIXED (primary 잣대 기준 LOSE)**
- FINCH-p2/Louvain-res6 수치는 screen 표와 **완전 일치** (채점 버그 없음 재현 확인).
- ★ 신규: primary 고정잣대 **UMAP-nn10+HDBSCAN(mcs10/ms3/leaf/eps0.15)** 에서 temp0.10 은 frozen 대비 **dP1 −3 (30→27) / drecov −0.088** — temp0.05(27/32,−0.089) 와 사실상 동일. 온도축이 밀도-기반 잣대를 못 움직임.
- temp0.10 은 어떤 클러스터러에서도 frozen 을 P1+recov 로 **초과 못함** (FINCH recov 0.9244<frozen 0.9259). best case = frozen-tie (비-퇴행), WIN 아님.
- 자동 폐기 아님 (KEEP 조건 미충족 — cap·recov 동반 열세라 상위지표 압도 없음). base 채택 안 함, frozen 유지.
- 어댑터 트랙 이력 재확인: "ep≈frozen 정점, 학습할수록 악화" — 온도 최적점도 frozen 천장을 못 넘음.

## 260725 — real-domain clean546 cycle-1 (inference-only 운영점: soft-reassign + ensemble)
Track: real MixedWM38 clean546 (단일결함 7종 C/D/EL/ER/L/NF/S ×50 + 배경 Normal/R ×98 = 546). **학습 0** (고정 champion 임베딩 위 운영점만). 임베딩 = raw GAP f→proj→L2. Dial 봉인: HDBSCAN **mcs6/ms3/leaf** (champion, EL/ER split) 고정, soft-reassign conf 만 운영점. k 표기 = 클러스터/7불량.
(주: 이 트랙은 mixed29 UMAP-nn10 잣대 대신 **euclidean 128-d 위 mcs6ms3leaf** 가 확립 프로토콜 — May-dial mcs12/ms15/leaf/eps0.06 병기. 우선순위 = P1 > noise(P2) > P3 comp > P4 hom, ARI/AMI/Sil 보조.)

### champion 확정 — temp0.20 champion 3-seed 전부 mcs6ms3leaf 에서 P1 7/7 재현
| emb (sel ep) | k | noise% | P1 | P2n | P3 comp | P4 hom | Sil | ARI | AMI |
|---|---|---|---|---|---|---|---|---|---|
| ★ s42 t20 ep20 (champ) | 18/7 | 29.1 | **7/7** | 28.6 | 0.765 | 0.855 | 0.666 | 0.692 | 0.792 |
| s42 n82 ep20 | 16/7 | 30.6 | 7/7 | 31.1 | 0.808 | 0.846 | 0.701 | 0.723 | 0.814 |
| s1 best_s1 ep18 | 17/7 | 43.4 | 7/7 | 40.9 | 0.771 | 0.860 | 0.649 | 0.703 | 0.796 |
| s2 best_s2 ep17 | 18/7 | 36.3 | 7/7 | 37.1 | 0.776 | 0.905 | 0.616 | 0.768 | 0.820 |
| (참고) s42 May-dial | 9/7 | 25.5 | 6/7 | 20.3 | 0.991 | 0.875 | 0.820 | 0.790 | 0.926 |

- champion 임베딩 원천 = `runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints/proj_ep20.pt` (s42), `abl_best_s1_B4_260724_111053/checkpoints/proj_ep18.pt`, `abl_best_s2_B4_260724_111604/checkpoints/proj_ep17.pt`. proj_ep npy 캐시 없음 → raw-GAP+proj 재계산.
- ★ 발견(재현 버그): 메모리의 "NEG 0.82" 은 **명목값** — `_may_ablation.py` B4 셀이 `IGNORE_NEG_SIM=0.72` 를 하드코드하고 sweep 이 export 한 `REPRO_NEG` 를 무시(오직 `REPRO_IGNORE_NEG_SIM` 만 반영). run_info/run.log 전부 0.72. **실제 champion 은 temp0.20 · NEG 0.72 로 학습됨** → NEG 0.82 는 이 트랙에서 미검증 레버(아래 criterion 4).

### (a) high-confidence soft-reassign (봉인 다이얼 conf≥0.8, champ s42 mcs6ms3leaf)
noise 점을 최근접 cluster L2-centroid(cosine=conf) 로 conf≥thr 일 때만 재배정 (라벨無).
| 운영점 | moved | k | noise% | P1 | P2n | P3 | P4 | Sil | ARI |
|---|---|---|---|---|---|---|---|---|---|
| base | — | 18/7 | 29.1 | 7/7 | 28.6 | 0.765 | 0.855 | 0.666 | 0.692 |
| conf≥0.90 | 126/159 | 18/7 | **6.0** | 7/7 | 4.6 | 0.724 | **0.864**(+) | 0.603 | 0.631 |
| conf≥0.80 | 148/159 | 18/7 | **2.0** | 7/7 | 1.4 | 0.705 | 0.846 | 0.580 | 0.627 |

### (b) ensemble (concat + L2) 재클러스터 + reassign
| 운영점 | k | noise% | P1 | P2n | P3 | P4 | Sil | ARI | AMI |
|---|---|---|---|---|---|---|---|---|---|
| ens s42+s1 base | 16/7 | 37.2 | 7/7 | 38.0 | 0.838 | 0.923 | 0.737 | 0.858 | 0.869 |
| ★ ens s42+s1 +conf≥0.90 | 16/7 | **5.1** | 7/7 | 4.0 | 0.757 | **0.890**(+) | 0.587 | **0.707**(+) | 0.809 |
| ens s42+s1 +conf≥0.80 | 16/7 | **0.9** | 7/7 | 0.3 | 0.732 | 0.882 | 0.558 | 0.693 | 0.789 |
| ens s42+s2 base | 15/7 | 35.0 | 6/7 | 39.4 | 0.850 | 0.942 | 0.735 | 0.862 | 0.886 |
| ens s42+s2 +conf≥0.80 | 15/7 | 1.1 | 7/7 | 0.3 | 0.754 | 0.864 | 0.524 | 0.700 | 0.794 |
| ens 3seed base | 18/7 | 47.4 | 7/7 | 54.9 | 0.788 | **0.966** | 0.643 | 0.810 | 0.850 |
| ens 3seed +conf≥0.80 | 18/7 | 0.7 | 7/7 | 0.3 | 0.673 | 0.863 | 0.397 | 0.602 | 0.740 |

판정: **WIN (신규 inference-only 운영점 — cheap 으로 noise 대폭↓, P1 7/7 & P4 미악화 달성)**
- 성공기준 충족: **P1 7/7 유지 & P4 미악화하며 noise 유의미 감소** = YES.
- ★ best 신규 운영점 = **ens[s42+s1] concat+L2 → mcs6ms3leaf → soft-reassign conf≥0.90**: noise **29.1→5.1%** (5.7×↓), P4 0.855→**0.890(+0.035)**, ARI 0.692→**0.707(+0.015)**, AMI +0.017, P3 0.765→0.757(≈flat), P1 7/7. 유일 손실 = Sil(0.666→0.587). 사전식 잣대(P1 tie→P2)에서 base 를 **지배**.
- max-noise-kill 운영점 = 같은 조합 conf≥0.80: noise **29.1→0.9%** (31×↓), P4 0.882(+0.027), ARI 0.693(=), P3 0.732(−0.033). noise 를 사실상 제거하면서 P4↑·ARI 동률.
- 단일 임베딩(앙상블 없이)만 원하면 champ_s42 +conf≥0.90: noise→6.0%, P4 +0.009, P1 7/7 (단 P3/ARI −0.05 — 앙상블이 이 손실을 흡수).
- ★ 메커니즘: ensemble concat 은 **purity 레버**(P4 0.855→0.923, ARI 0.692→0.858) 지만 경계점을 noise 로 밀어 noise↑ — 단독으론 목표(noise↓) 역행. soft-reassign 이 그 noise 를 conf-gated 회수 → **두 부품 상보**. 앙상블의 높은 base purity 가 reassign 의 오배정 손실을 상쇄해 순수 reassign 대비 ARI/P4 유지.
- 폭 주의: 2-seed(s42+s1) 스위트스폿. **3-seed concat 은 over-concentrate** (base P4 0.966 최고지만 reassign 후 Sil 0.40·ARI 0.60 붕괴) — mixed29 "diversity>quantity, 폭3 포화" 재현.
- 봉인 준수: mcs/ms 불변, reassign conf 는 허용된 운영점(memory dinov3_ncd "재배정 conf0.8만"), ensemble concat+L2 는 고정잣대 보조행. 라벨 누수 0 (centroid=embedding cosine).
- 참고: kNN-percentile noise 는 noise 를 늘리는 방향이라 목표(noise↓)와 반대 — 미평가(과제 지시대로 제외).

### 다음 atomic 실험 1개 (정량 근거)
**NEG override 배선 수정 후 NEG 값 재스윕 {0.78, 0.82, 0.86} @ temp0.20** (encoder-side, 1 atomic).
- 근거: (1) 위 버그로 "NEG 0.82" 가 **한 번도 실제 학습 안 됨** — 메모리 headline 이 미검증 레버. (2) champion 의 유일 실약점 = **과분할**(k=18/7, frag 2.57 — 각 class ~2.5 조각) → 이것이 base noise 29% 근원. (3) NEG(ignore-neg-sim)↑ 는 EL↔ER 경계의 false-negative masking 을 풀어 두 class 를 negative 로 되살림 = 경계 분해능↑ → encoder 단계에서 조각 병합(k↓, noise↓) 유도 가능. mixed29 KoLeo0.1/SoftNCE-topk20 는 절대값 이식 위험(트랙 구조 상이)이라 후순위; NEG-fix 는 **이 트랙에서 이미 nominal 승자로 지목됐으나 미적용된** coordinate-descent 이웃값이라 최우선.
- 구현: `_recipe_sweep.sh` 가 `REPRO_NEG` 대신 `REPRO_IGNORE_NEG_SIM` 를 export 하도록(또는 B4 셀 하드코드 제거) 수정 후 dispatch. 판정 = mcs6ms3leaf P1 7/7 유지하며 base noise(29%)가 encoder 단계에서 내려가는가 → 내려가면 reassign 전 순수 P4↑.

## 260725 — cycle-2: NEG 재스윕 (wiring bug 수정 후) → LOSE, NEG 축 종결
| NEG 실제값 (temp0.20, selEp) | P1 | noise% | P4 | ARI | 판정 |
|---|---|---|---|---|---|
| 0.72 (champion, 구 n78/n82/n86=전부 0.72였음) | 6/7 | 20.57-21.14 | 0.8737-0.8757 | 0.788-0.792 | ★ 유지 |
| 0.78 (n78x, ep18) | 6/7 | 22.29 | 0.8726 | 0.780 | LOSE |
| 0.82 (n82x, ep18) | 6/7 | 22.0 | 0.8731 | 0.784 | LOSE |
| 0.86 (n86x, ep18) | 6/7 | 20.86 | 0.8737 | 0.785 | 동률 (무이득) |

- ★ 버그 정정: `_recipe_sweep.sh` REPRO_NEG → REPRO_IGNORE_NEG_SIM (B4 셀 0.72 하드코딩을 뚫는 유일 env). "NEG 0.82 best" 옛 기록은 명목값 — 실제 전부 0.72 학습이었음.
- NEG {0.78,0.82,0.86} 전부 0.72 이하 — "NEG↑=EL/ER 경계 선명화" 가설 기각. encoder knob NEG 축 진짜 탐색 완료·종결.
- 학습 비용 실측: config당 ~2.5분 (clean546 546장, GPU) — 이 트랙 빠른 반복 가능.

## 260725 — cycle-3: 운영점 pair 견고성 확정 + n86x 부품가치 + 배포 v2 (전부 inference-only)
Track: real-domain clean546 (7 defect + Normal/R). 잣대: euclidean 128/256-d 위 **mcs6ms3leaf 봉인** + soft-reassign conf 운영점. 임베딩 = raw GAP f→proj→L2, ens = concat+L2. 학습 0.

### (1) pair 견고성 — 세 seed-pair 전부 × conf {0.90, 0.80}
| 운영점 | k | noise% | P1 | P2n | P3 | P4 | Sil | frag | ARI | AMI |
|---|---|---|---|---|---|---|---|---|---|---|
| ens s42+s1 base | 16/7 | 37.2 | 7/7 | 38.0 | 0.838 | 0.923 | 0.737 | 2.29 | 0.858 | 0.869 |
| ★ ens s42+s1 +conf≥0.90 | 16/7 | **5.1** | 7/7 | 4.0 | 0.757 | **0.890** | 0.587 | 2.29 | 0.707 | 0.809 |
| ens s42+s1 +conf≥0.80 | 16/7 | 0.9 | 7/7 | 0.3 | 0.732 | 0.882 | 0.558 | 2.29 | 0.693 | 0.789 |
| ens s42+s2 base | 15/7 | 35.0 | 6/7 | 39.4 | 0.850 | 0.942 | 0.735 | 2.14 | 0.862 | 0.886 |
| ens s42+s2 +conf≥0.90 | 15/7 | 7.1 | **7/7** | 6.9 | 0.785 | 0.875 | 0.582 | 2.14 | 0.715 | 0.820 |
| ens s42+s2 +conf≥0.80 | 15/7 | 1.1 | 7/7 | 0.3 | 0.754 | 0.864 | 0.524 | 2.14 | 0.700 | 0.794 |
| ens s1+s2 base | 19/7 | 41.9 | 7/7 | 43.7 | 0.726 | 0.887 | 0.634 | 2.71 | 0.654 | 0.773 |
| ens s1+s2 +conf≥0.90 | 19/7 | 4.0 | 7/7 | 2.9 | 0.664 | 0.869 | 0.531 | 2.71 | 0.575 | 0.736 |
| ens s1+s2 +conf≥0.80 | 19/7 | 0.4 | 7/7 | 0.0 | 0.655 | 0.867 | 0.495 | 2.71 | 0.567 | 0.729 |

판정: **WIN — "운영점 견고" 확정.** 세 pair 모두 conf 운영점(0.90/0.80 전부)에서 **P1 7/7 & noise 한 자리** (0.90: 4.0–7.1%, 0.80: ≤1.1%). pair 선택에 비민감.
- s42+s2 는 base 6/7 이지만 reassign 후 7/7 회복 (noise-induced 결손 — 재배정이 dominant 구성을 복원). 구조적 병합 아님.
- ★ champion 운영점 유지 = **ens[s42+s1] +conf≥0.90**. 사전식 strict(P1→noise) 로는 s1+s2 (4.0%) 가 0.90 에서 1.1pp 앞서지만, noise 축은 conf 로 자유 조절 가능(모든 pair 가 0.80 에서 ≤1.1%) — **matched-noise (conf0.80) 비교에서 s42+s1 이 s1+s2 를 P3/P4/ARI 전부 지배** (0.732/0.882/0.693 vs 0.655/0.867/0.567). s1+s2 의 noise 이점은 구조 열세(frag 2.71, k 19)의 부산물. 260612 규칙대로 trade-off 명시 후 s42+s1 유지.
- 부품 원리 재확인: s42 가 들어간 pair (frag 2.14–2.29) > s42 없는 pair (frag 2.71) — s42(t20 champion)가 구조 품질의 anchor.

### (2) n86x 부품가치 (보너스) — LOSE, 폐기
| 운영점 | k | noise% | P1 | P3 | P4 | ARI |
|---|---|---|---|---|---|---|
| ens s42+n86x base | 15/7 | 28.9 | **6/7** | 0.873 | 0.826 | 0.680 |
| ens s42+n86x +conf≥0.90 | 15/7 | 5.3 | **6/7** | 0.785 | 0.822 | 0.636 |
| ens s42+n86x +conf≥0.80 | 15/7 | 1.1 | **6/7** | 0.761 | 0.817 | 0.632 |

- **모든 운영점에서 P1 6/7 고정** — reassign 으로도 회복 불가 = 구조적 클래스 병합 (s42+s2 의 noise-induced 6/7 과 질적으로 다름). P4 도 전 pair 최저 (0.817–0.826).
- 예측 확인: n86x(NEG0.86, 근사 0.72 복제)는 **seed-diversity 대체재가 못 됨** — 독립 seed (s1/s2, 동일 recipe 다른 seed) 의 view-diversity 가 recipe-이웃 복제보다 명백히 우월. n8xx 계열 앙상블 부품 폐기.

### (3) 배포 산출물 v2 — `result_grouping/deliverable_clean546_v2/`
운영점 = ens[s42+s1] concat+L2 → mcs6ms3leaf → soft-reassign conf≥0.90. 산출 = groups.csv / representatives(12/group) / composites / summary.json / offline_eval.csv / offline_summary.json (_grouping_deliverable.py 동일 포맷, 앙상블+reassign 지원 scratchpad 생성기 cycle3_deliver.py). 기존 deliverable_clean546/ 불변.
- runtime(무라벨): n 546, k 16, noise 5.13%, over-merge 0, mean coherence 0.9694, mean stability 0.8825 (동일 파이프라인 bootstrap co-assignment).
- offline(숨긴 라벨): **P1 7/7, P2n 4.0, P3 0.757, P4 0.890, ARI 0.707** — 채점표 정확 일치.
- 그룹 구성: 7 defect 전부 majority 등장 (C .977 / NF .927 / D 1.0×2 / L .820 / EL .944 / S 1.0×2 / ER .972+.542), 배경 Normal×3·R×3. 약점 = group_012 (ER .542, EL 혼입 — EL/ER 경계) 와 L .820 — encoder-side 한계, 다이얼 문제 아님.

### 다음 실험 1개 (최종목표 = 사내 실데이터 unknown predict 관점)
**(a) 사내-리허설: `data/images/unknown_eval100` (4149장, 32 strict-novel target) 에 end-to-end 방법론 리허설** — frozen 진단 → B4 recipe (temp0.20/NEG0.72/queue) 2-seed adaptation → 무라벨 선택 ladder → ens+reassign 운영점 → deliverable 포맷 산출.
- 근거: (1) 운영점이 pair·conf 양 축에서 견고 확정 → 남은 최대 리스크 = **새 풀 전이** (규모 8× + 클래스 32종, 방법론의 유일 미검증 가정). (2) (b) HDBSCAN 이웃 미세조정은 봉인 다이얼 인접 + 이득 축(noise)이 reassign 으로 이미 해결 — 남은 약점(EL/ER purity, frag 2.3)은 encoder/data-side 라 다이얼 미세조정 기대이득 낮음. (3) 비용: clean546 config 당 ~2.5분 실측 → 4149장 ~8× ≈ 20분/seed × 2 = GPU ~40분, 저렴. (4) unknown_eval100 은 채점 리그 기존재 (260725 FCMAE 어댑터 교차검증 행) — frozen 기준선 (FINCH 32/32·0.926 등) 과 직접 비교 가능.
- 단 새 풀에선 mcs6ms3leaf 가 546-규모 튜닝값 — 다이얼 선택은 사용자 결정 사항 (봉인 정책), 리허설 설계 시 명시 필요.

## 260726 — clean546 recipe-축 16셀 mcs6/ms3/leaf/eps0.06 재채점 (May-dial mcs12/ms15/leaf/eps0.06 대비, capture-ceiling 검증)

Track: real MixedWM38 clean546 (단일결함 7종 + Normal/R 배경 546장). **재학습 없음** — `runs/sweep/abl_sw_*_B4_260724/25_*/checkpoints/proj_ep*.pt` (260724/25 학습된 기존 16셀 체크포인트) 재채점만. 스코어러 `_grouping_eval.py`에 `--mcs/--ms/--eps/--method` 옵션 신규 추가(default = May값 12/15/leaf/0.06, 후방호환 diff 0 검증됨) + `--out-name`(신규 파일명, 기존 `eval_sweep_*.json` 무변경) + `--feat-cache`(backbone frozen feature 1회 추출·재사용). **다이얼: mcs6/ms3/leaf/eps0.06** (May-dial 대비 mcs 12→6, ms 15→3만 하향, method/eps 불변). 무라벨 선택 게이트(over_merge==0 & stability≥0.75 & coherence≥0.80 → non_noise_pct 최대) 로직 불변. GPU는 다른 프로세스(Ollama llama-server, VRAM 14.8/16.4GB) 점유 중이라 전부 CPU 실행.

사전검증: 신규 옵션 미지정(default) 재실행 시 기존 `eval_sweep_t20.json`과 완전 일치(후방호환 확인). mcs6/ms3 재채점한 t20 ep20 수치가 `_crossds_leaderboard.md` 260725 cycle-1 champion 행(k18/7, noise29.1, P1 7/7, P4 0.855, ARI 0.692)과 완전 일치 — 다이얼 재현 정확성 확인.

### 잡음 폭 (동일cfg 4셀: t20/n78/n82/n86 — 전부 TEMP0.20·NEG0.72·QUEUE16384·SEED42 동일. n78/n82/n86은 260724 배선버그로 REPRO_NEG가 무시되어 사실상 t20 재실행 3회였음 — 260725 cycle에서 이미 확인된 사실, 여기선 잡음폭 산출용으로 재활용)
게이트가 셀마다 다른 epoch(ep5/ep5/ep5/ep4)을 골랐음에도 지표는 좁게 수렴 — 이것이 이 계측기(mcs6/ms3/leaf)의 run-to-run 잡음 폭이며, 아래 모든 판정의 임계값이다.

| 지표 | 4값(t20/n78/n82/n86) | min | max | 스프레드 |
|---|---|---|---|---|
| P1 | 6/7, 6/7, 6/7, 6/7 | — | — | **0**(전부 동일 — P1 차이는 폭과 무관하게 즉시 유의미) |
| noise%(P2, defect-only) | 20.29/19.43/21.71/20.86 | 19.43 | 21.71 | **±2.28pp** |
| ARI | 0.788/0.791/0.779/0.772 | 0.772 | 0.791 | **±0.019** |
| Hom(P4) | 0.871/0.873/0.868/0.870 | 0.868 | 0.873 | **±0.005** |
| Comp(P3) | 1.000/1.000/0.978/0.967 | 0.967 | 1.000 | **±0.033** |

matched-capacity(오라클 P1=7/7 지점, 아래 표2) 기준으로도 동일 4셀 noise 스프레드 = 20.57–22.86 (±2.29pp) — floor 크기가 용량-매칭 여부와 무관하게 일정, 진짜 측정잡음임을 재확인.

### 표1 — 16셀 재판정 (게이트-선택 epoch 기준, k=클러스터수/7불량, frag=k/7)
| 셀 | 실제 cfg | selEp | P1 | noise%(P2) | Comp | Hom | ARI | Sil | k | frag | 판정 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t05 | TEMP0.05 | ep07 | 6/7 | 27.71 | 0.925 | 0.846 | 0.733 | 0.698 | 14/7 | 2.000 | LOSE |
| t07 | TEMP0.07 | ep16 | 6/7 | 15.14 | 0.913 | 0.857 | 0.750 | 0.757 | 14/7 | 2.000 | MIXED |
| t10 | TEMP0.10 | ep08 | 6/7 | 15.43 | 0.926 | 0.872 | 0.762 | 0.723 | 13/7 | 1.857 | MIXED |
| t12 | TEMP0.12 | ep17 | 6/7 | 14.29 | 0.913 | 0.859 | 0.752 | 0.767 | 14/7 | 2.000 | MIXED |
| t15 | TEMP0.15 | ep06 | 6/7 | 23.14 | 0.962 | 0.865 | 0.764 | 0.781 | 12/7 | 1.714 | LOSE |
| ★t20 | TEMP0.20(base) | ep05 | 6/7 | 20.29 | 1.000 | 0.871 | 0.788 | 0.803 | 12/7 | 1.714 | 기준 |
| t25 | TEMP0.25 | ep06 | 6/7 | 19.14 | 1.000 | 0.873 | 0.793 | 0.783 | 12/7 | 1.714 | 동률 |
| t30 | TEMP0.30 | ep08 | 6/7 | 18.57 | 0.981 | 0.857 | 0.778 | 0.802 | 12/7 | 1.714 | 동률(경계) |
| q32k | QUEUE32768 | ep04 | 7/7 | 26.29 | 0.960 | 0.877 | 0.789 | 0.684 | 12/7 | 1.714 | MIXED(P1↑,frag유지) |
| q64k | QUEUE65536 | ep10 | 7/7 | 30.29 | 0.764 | 0.889 | 0.756 | 0.626 | 20/7 | 2.857 | MIXED(P1↑,전반열세) |
| n78 | NEG0.72(버그,nom.78=t20dup) | ep05 | 6/7 | 19.43 | 1.000 | 0.873 | 0.791 | 0.801 | 12/7 | 1.714 | 동률 |
| n82 | NEG0.72(버그,nom.82=t20dup) | ep05 | 6/7 | 21.71 | 0.978 | 0.868 | 0.779 | 0.772 | 13/7 | 1.857 | 동률 |
| n86 | NEG0.72(버그,nom.86=t20dup) | ep04 | 6/7 | 20.86 | 0.967 | 0.870 | 0.772 | 0.771 | 13/7 | 1.857 | 동률 |
| n78x | NEG0.78 | ep05 | 6/7 | 21.43 | 0.957 | 0.868 | 0.769 | 0.764 | 12/7 | 1.714 | 동률(경계) |
| n82x | NEG0.82 | ep08 | 6/7 | 20.86 | 0.933 | 0.859 | 0.759 | 0.764 | 14/7 | 2.000 | 동률(근접열세) |
| n86x | NEG0.86 | ep14 | 7/7 | 26.57 | 0.811 | 0.954 | 0.859 | 0.691 | 18/7 | 2.571 | MIXED(frag대가,미채택) |

### 표2 — matched-capacity (오라클: 각 셀 20-epoch 중 P1=7/7 달성 epoch들 중 최저 noise%(P2) 지점 — recipe축 판정의 근거)
| 셀 | cfg | 최저noise ep(오라클, P1=7/7 중) | P1 | noise%(P2) | Comp | Hom | ARI | Sil | k | frag | Δnoise vs t20-matched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| t05 | TEMP0.05 | ep09 | 7/7 | 25.14 | 0.916 | 0.900 | 0.830 | 0.603 | 15/7 | 2.143 | +4.28pp |
| t07 | TEMP0.07 | ep15 | 7/7 | 23.71 | 0.868 | 0.923 | 0.858 | 0.708 | 16/7 | 2.286 | +2.85pp |
| t10 | TEMP0.10 | ep06 | 7/7 | 22.29 | 0.869 | 0.893 | 0.790 | 0.586 | 15/7 | 2.143 | +1.43pp |
| t12 | TEMP0.12 | ep11 | 7/7 | 22.57 | 0.883 | 0.921 | 0.845 | 0.693 | 17/7 | 2.429 | +1.71pp |
| t15 | TEMP0.15 | ep11 | 7/7 | 19.71 | 0.903 | 0.887 | 0.787 | 0.714 | 16/7 | 2.286 | -1.15pp |
| ★t20 | TEMP0.20(base) | ep15 | 7/7 | 20.86 | 0.865 | 0.878 | 0.801 | 0.715 | 14/7 | 2.000 | 기준(matched) |
| t25 | TEMP0.25 | ep17 | 7/7 | 22.57 | 0.860 | 0.872 | 0.809 | 0.748 | 15/7 | 2.143 | +1.71pp |
| t30 | TEMP0.30 | ep16 | 7/7 | 20.57 | 0.864 | 0.872 | 0.813 | 0.735 | 16/7 | 2.286 | -0.29pp |
| q32k | QUEUE32768 | ep04 | 7/7 | 26.29 | 0.960 | 0.877 | 0.789 | 0.684 | 12/7 | 1.714 | +5.43pp |
| q64k | QUEUE65536 | ep10 | 7/7 | 30.29 | 0.764 | 0.889 | 0.756 | 0.626 | 20/7 | 2.857 | +9.43pp |
| n78 | NEG0.72(버그dup) | ep15 | 7/7 | 20.57 | 0.865 | 0.878 | 0.801 | 0.716 | 14/7 | 2.000 | -0.29pp |
| n82 | NEG0.72(버그dup) | ep15 | 7/7 | 22.86 | 0.871 | 0.881 | 0.813 | 0.722 | 14/7 | 2.000 | +2.00pp |
| n86 | NEG0.72(버그dup) | ep16 | 7/7 | 20.57 | 0.878 | 0.887 | 0.816 | 0.708 | 15/7 | 2.143 | -0.29pp |
| n78x | NEG0.78 | ep15 | 7/7 | 20.57 | 0.860 | 0.898 | 0.819 | 0.723 | 16/7 | 2.286 | -0.29pp |
| n82x | NEG0.82 | ep11 | 7/7 | 22.57 | 0.867 | 0.922 | 0.865 | 0.729 | 16/7 | 2.286 | +1.71pp |
| n86x | NEG0.86 | ep16 | 7/7 | 21.14 | 0.871 | 0.889 | 0.823 | 0.746 | 16/7 | 2.286 | +0.28pp |

- 16셀 전부 20-epoch 안에 P1=7/7 도달 epoch 존재(오라클 레벨, 예외 0) — 표에 "없음" 행 없음.
- 이 표의 t20/n78/n82/n86 매칭-용량 noise 스프레드(20.57–22.86, ±2.29pp)가 위 잡음폭과 동일 규모 → **temp 0.10–0.30, NEG 0.72–0.86 는 이 폭 안에서 전부 동률**(t15=19.71 최저, t30=20.57 동률, n78/n86=20.57 동률). **temp0.05(+4.28pp)와 q64k(+9.43pp)만 폭 밖 확정 LOSE.** q32k(+5.43pp)는 noise만 보면 폭 밖 열세이나 frag=1.714로 t20 매칭-용량 지점(frag 2.000)보다도 구조가 좋고 Comp=0.960 최고치 — **noise만 열세, 구조는 우위인 트레이드오프**로 별도 표기(자동 LOSE 아님, MIXED 유지).

### ★ n86x — SOTA 후보 아님, frag 대가 확인 (팀리드 재검토 반영)

| 지점 | selEp? | P1 | noise%(P2) | Comp | Hom | ARI | Sil | frag | k |
|---|---|---|---|---|---|---|---|---|---|
| ep14 (게이트 선택) | ✅ | 7/7 | 26.57 | 0.811 | **0.954** | **0.859** | 0.691 | 2.571 | 18/7 |
| ep16 (frag-matched, 오라클 최저noise-7/7 지점) | — | 7/7 | 21.14 | 0.871 | 0.889 | 0.823 | 0.746 | 2.286 | 16/7 |
| ★t20 기준(ep05) | ✅ | 6/7 | 20.29 | 1.000 | 0.871 | 0.788 | 0.803 | 1.714 | 12/7 |

- Hom/ARI 16셀(게이트-선택 기준) 중 최고치(ep14)는 k=18(frag 2.571)로 쪼갠 대가다 — Comp 1.000→0.811 동반 하락이 그 증거(그룹을 작게 쪼갤수록 정의상 Hom↑, Comp↓). 사용자 정책(260615, 파편비+capture 우선·recov/Hom 후순위)상 **frag 비용을 차감하지 않은 Hom/ARI 단독 최고치를 SOTA로 표기하는 것은 정책 위반이다 → 미채택.**
- **게이트 퇴화의 실증 사례**: n86x 자체 20-epoch 안에도 P1=7/7 epoch이 여럿(6/20)인데, 게이트가 고른 ep14를 **ep16이 지배한다** — noise −5.4pp / Comp +0.06 / frag 개선(2.571→2.286), 잃는 건 Hom(-0.065)·ARI(-0.036)뿐. 즉 **7/7 epoch들 안에서조차 게이트가 최선을 못 고른다.** mcs6/ms3에서 quality-gate(over_merge/stability/coherence)가 사실상 항상 통과(16셀 전부 passed=[1..20], gate_failed 0건)되어 선택이 raw noise_pct 최저값으로 퇴화한 결과 — capture도 frag도 게이트에 반영되지 않는다.
- **미채택 확정 사유**: (1) frag 대가 미차감 (2) 단일 run(seed 반복 미검증) — 2-seed 재현은 재학습이 필요하나 GPU가 Ollama에 점유돼 보류 중(사용자 GPU 해제 문의 중). GPU 확보 후 dispatch 예정.

### 결론 (260725 cycle 대비 재확인/보강, 변경 없음)
1. **capture ceiling = HDBSCAN 해상도 문제였다는 가설 확정.** May-dial(mcs12/ms15)에서는 16셀×20epoch=320칸 전부 P1=6/7 고정(예외 0)이었으나, mcs6/ms3에서는 **16/16셀 전부** P1=7/7 도달 epoch 보유(6/20~17/20). encoder/recipe가 원인이 아니었다.
2. **recipe 축(temp/queue/NEG) 자체의 순위는 매칭-용량 비교로도 거의 안 바뀐다.** temp 0.10–0.30·NEG 0.72–0.86은 잡음폭 안에서 동률, q32k만 구조(frag/Comp) 개선으로 격상 후보, q64k·temp0.05는 확정 LOSE — 260726 이전 결론과 방향 일치, 근거만 더 견고해짐(오라클 매칭-용량 비교로 재확인).
3. **무라벨 선택 게이트가 이 다이얼에서 P1-blind 하다는 사실이 새로 드러남** — over_merge/stability/coherence 임계가 비구속적이라 순수 noise-최저 선택으로 퇴화. 게이트 로직은 지시대로 불변 유지했으나, 이 특성 자체는 다음 방법론 논의에 필요.
4. 다음 축 추천 1순위(유지): **LR_HEAD** — 16셀 전부 LR_HEAD=0.001 고정(run_info.json 확인), `_recipe_sweep.sh` 시그니처에 슬롯 있음에도 미탐색인 유일한 원래-축.
5. n86x 2-seed 재현검증은 GPU 확보 후 별도 dispatch(팀리드 보류 결정, 재학습 필요·CPU 비경제적).

산출: `_grouping_eval.py`(수정, 후방호환) / `runs/clean546/eval_d6_sweep_{t05,t07,t10,t12,t15,t20,t25,t30,q32k,q64k,n78,n82,n86,n78x,n82x,n86x}.json`(신규 16개, 기존 `eval_sweep_*.json` 무변경).

## 260726 addendum — cycle-3 교차참조 정정 + 핵심 2문장 확정 (팀리드 재검토 반영, 위 260726 섹션 원문은 무수정)

### n86x 교차참조 정정
위 "★ n86x — SOTA 후보 아님" 절은 이 파일 상단 **cycle-3 "(2) n86x 부품가치 (보너스) — LOSE, 폐기"** 행(260725)과 모순되지 않는다 — 서로 다른 레벨의 판정이다:
- **cycle-3 판정(앙상블 부품 레벨)**: ens[s42+n86x]는 base/conf0.90/conf0.80 전 운영점에서 **P1 6/7 고정**(soft-reassign으로도 회복 불가 = 구조적 클래스 병합) → "n86x(NEG0.86)는 seed-diversity 대체재가 못 됨 — 독립 seed(s1/s2)의 view-diversity가 recipe-이웃 복제보다 명백히 우월" → n8xx 계열 앙상블 부품 **폐기 확정**.
- **260726 발견(단일 셀, 앙상블 아님, mcs6/ms3 채점)**: n86x ep14/ep16이 P1 **7/7** 도달. 이는 cycle-3의 앙상블 판정을 뒤집지 않는다 — 다이얼(May↔mcs6/ms3)과 앙상블-유무가 서로 다른 별개 측정축이고, ep14는 frag 2.571(파편화 대가)로 위 절에서 이미 미채택 처리됨.
- **정정된 결론**: n86x는 두 레벨 모두에서 채택 후보가 아니다. cycle-3(앙상블 LOSE·구조적 병합)와 260726(frag 대가·단일 run 미검증)은 서로 다른 각도에서 같은 결론(미채택)에 도달했을 뿐 — 모순 없음.

### 우선순위 정정
위 §결론 4-5의 "LR_HEAD 1순위, n86x 2-seed 재현 후순위" 배치의 근거를 명확히 한다: **cycle-3이 이미 NEG-recipe-이웃(n8xx) 방향을 앙상블 부품으로 기각**했으므로, 같은 GPU 예산이면 **LR_HEAD(미탐색 원래 축)가 우선**이다. n86x 2-seed 재현은 "frag 대가가 단일-run 노이즈인지 진짜인지"만 확인하는 저가치 확인 작업으로 재분류한다 — LR_HEAD 스윕 이후 여력 있을 때만 착수 권고.

### 핵심 2문장 (후속 세션이 재발명하지 않아야 할 것)
1. **P1 6/7 천장은 encoder가 아니라 다이얼 해상도였음을 recipe 축 전체(16셀×20epoch)에서 확정: May-dial 0/320 → mcs6/ms3 186/320.**
2. **게이트가 mcs6/ms3에서 non-binding으로 퇴화한다(over_merge/stability/coherence 임계를 사실상 모든 epoch이 통과 → 선택이 raw noise_pct 최저 epoch 하나로 퇴화). 그 결과 게이트는 P1을 전혀 보지 않고 epoch을 고르며, 16셀 중 그 선택이 우연히 P1=7/7과 맞아떨어진 건 q32k·n86x 2건뿐이고 나머지 14셀은 6/7-epoch을 선택했다.**
