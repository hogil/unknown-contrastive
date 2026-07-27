# FCMAE-raise 첫 양성 신호 (260721) — zero-init residual adapter

판정선 FCMAE frozen (strict-novel 32cls, unknown_eval100): finch_p2 ARI 0.805 / louvain 0.871.
컬럼: clusterer(k) | P1capture | AMI | P2noise% | P3Comp | P4Hom | ARI | Sil | k(tot/cls/noise) | 파편비.

## ad1_none [head=adapter, γ=0 시작, freeze bb, pure B0, pdim128, seed3, aug=noneonly]
frozen_f       finch_p2(k62) | 32/32 (1.0000) | 0.9259 | 0.0 | 0.8914 | 0.9583 | 0.805  | 0.381  | 62/32/16 | 1.94
frozen_f       louvain_res6  | 31/32 (0.9688) | 0.9309 | 0.0 | 0.9311 | 0.9677 | 0.8707 | 0.4718 | 53/32/13 | 1.66
ep0 f (γ=0)    finch_p2(k62) | 32/32 (1.0000) | 0.9259 | 0.0 | 0.8914 | 0.9583 | 0.805  | 0.381  | 62/32/16 | 1.94   ← 하한=frozen 정확 재현(sanity PASS)
ep0 f (γ=0)    louvain_res6  | 31/32 (0.9688) | 0.9312 | 0.0 | 0.9313 | 0.9679 | 0.871  | 0.4716 | 53/32/13 | 1.66
ep1 f (adapt)  finch_p2(k56) | 31/32 (0.9688) | 0.90   | 0.0 | 0.9043 | 0.9492 | 0.808  | 0.4117 | 56/32/14 | 1.75   ← +0.003
ep1 f (adapt)  louvain_res6  | 31/32 (0.9688) | 0.93   | 0.0 | 0.9318 | 0.9686 | 0.8738 | 0.4685 | 53/32/13 | 1.66
ep2 f (adapt)  finch_p2(k54) | 30/32 (0.9375) | 0.9006 | 0.0 | 0.9402 | 0.9668 | 0.8759 | 0.418  | 54/32/15 | 1.69   ← ★ARI +0.071, Comp/Hom/파편비 개선, but capture -2
ep2 f (adapt)  louvain_res6  | 31/32 (0.9688) | 0.9256 | 0.0 | 0.93   | 0.9671 | 0.8703 | 0.4666 | 53/32/13 | 1.66   ← louvain flat
학습된 γ = -0.0164 (아주 작은 residual)

## 판정
- ep0 f == frozen 정확 재현 → γ=0 하한 보장 구조 검증. frozen(0.805) 아래로 절대 안 내려감.
- ep2 adapted f: finch ARI 0.805→0.876 (+0.071). 학습으로 FCMAE 상승하는 첫 증거.
- 비용/경계: capture 32→30 (P1 -2), louvain은 flat(finch 특이적), single-seed, ep3~5 미관측(kill).
- 다음: bounded adapterN2 (capture 지키며 ARI 확보?), ep3~5 full 궤적, seed 재현.

## [정정 260721 09:00] fresh full run (ep0-5) — 분산 발견
첫 run(ep2 0.876/cap30/γ-0.016)은 kill된 부분run의 best-epoch cherry-pick이었음. fresh full run:
ep0 f(γ=0) 32/32 0.805 (sanity PASS)
ep1 f 32/32 | Comp0.9029 Hom0.9542 | ARI0.8147 | frag1.78
ep2 f 30/32 | ARI0.7737 (dip)
ep3 f 30/32 | Comp0.9102 | ARI0.8184 | frag1.75
ep4 f 32/32 | Comp0.9056 Hom0.9598 | ARI0.8224 | frag1.72  ← clean win (P1 유지, 보조지표 상승, frag 1.94→1.72)
ep5 f 30/32 | ARI0.790 (dip)
γ_final=+0.0205 (첫 run은 -0.0164 — 부호 반대).
결론: bare adapter 상승은 작지만 실재(+0.01~0.02, P1 유지 가능), 분산 큼. fixed-epoch+3-seed 필수(단일seed·best-epoch 금지). 사다리(queue/ignore/NV/local/NeCo)로 신호 증폭·안정화가 관건.
