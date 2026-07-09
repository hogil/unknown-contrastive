# WM Coordinate-Descent Ablation (260610 V1 recipe 이어받기)

목적: 이전 ablation(260610, `SIMCLR_COMPONENT_ABLATION` / `SIMCLR_VALUE_SWEEP`)에서 찾은 최적 recipe를
**현재 잣대(WM 7-class, finch, no-UMAP, no-CNN)로 재현 + coordinate-descent로 확장**.
방법: 한 옵션 여러 값 sweep → 최적 고정 → 다음 옵션 추가 → sweep → 최적 고정 → ... (누적).

## 이전 결과 요약 (260610, eval=class-disjoint novel Donut/Edge-Loc, 잣대 당시 k-means[금지]+HDB)
| stage | recipe | k-means ARI | HDB ARI |
|---|---|---|---|
| B0 raw FCMAE | — | 0.067 | 0.478 |
| C0 Base SimCLR | — | 0.498 | 0.289 |
| C1 +queue only | queue | 0.222 (하락) | — |
| C5 +local+neco | local+neco | 0.512 | — |
| **V1 (best)** | **queue+ignore0.70+local0.3** | **0.636** | **0.705** |
| V9 | queue+ignore0.70+local0.1+neco0.1 | 0.571 | — |

교훈: local weight 매우 민감(0.1→0.006, 0.3→0.636). ignore0.70 = winner 핵심. queue 단독은 하락(local/ignore와 병용해야).
★ 주의: 이전은 2-class novel + k-means라 현재 7-class finch 숫자와 **직접 비교 금지**. recipe 가이드로만.

## 현재 잣대 기준선 (WM 7-class, finch_p2, seed 3, ep3)
| config | ARI | Comp | Hom | cap | 파편비 |
|---|---|---|---|---|---|
| frozen | 0.149 | 0.267 | 0.435 | 1.0 | 4.0 |
| old champion (wmstar: local0.15, no-ignore, queue) | 0.145 | 0.289 | 0.466 | 1.0 | 5.1 |
| wmv (champion+VICReg) | 0.151 | 0.236 | 0.364 | 1.0 | 4.3 |

## Coordinate-descent 계획
- **Stage 1 (진행 중)**: queue+ignore0.70 고정, **local sweep {0.2, 0.3, 0.4}** → 최적 local L*. (wm_ig70_local03 = V1 재현)
- Stage 2: queue+local L* 고정, **ignore sweep {off, 0.6, 0.7, 0.8, 0.9}** → 최적 ignore I*.
- Stage 3: +neco sweep {0.05, 0.1, 0.2} → 최적.
- Stage 4: +nv-filter / vicreg / ls / koleo / temp / queue-size 순차 추가·sweep.
- 각 stage 승자 → 다음 stage 기준선. 최종 = 누적 최적 recipe.
- seed 누적으로 승자 mean±std.

## Stage 1 결과 (finch_p2, ep3, seed3)
| config | local | ARI | Comp | Hom | cap | 파편비 | 판정 |
|---|---|---|---|---|---|---|---|
| frozen (ref) | — | 0.149 | 0.267 | 0.435 | 1.0 | 4.0 | 기준 |
| old champion (local0.15,no-ignore) | — | 0.145 | 0.289 | 0.466 | 1.0 | 5.1 | 나쁜 recipe |
| wm_ig70_local02 | 0.2 | 0.215 | 0.355 | 0.537 | 1.0 | 3.4 | ↑ |
| **wm_ig70_local03 (=V1)** | **0.3** | **0.237** | 0.363 | 0.519 | 1.0 | **3.0** | ★ 현재 best |
| wm_ig70_local04 | 0.4 | 0.183 | 0.315 | 0.480 | 1.0 | 3.86 | ↓ (과다) |

★ **Stage 1 승자 = local 0.3** (peak: 0.2→0.3 상승, 0.4 하락 = 260610 재확인). **L* = 0.3.**
→ Stage 2: local0.3 고정, ignore sweep {off/0.6/0.65/0.7(=V1,0.237)/0.75/0.8/0.9}.

## ★★ Stage 2 반전 (260707): ignore × epoch 상호작용 — ep10 실측이 신기록 발굴
사용자 directive "epoch 10까지는 다 해봐야지" → 전 config resume ep7-10 + ep3/6/8/10 채점. 결과:
- **ep3 단면의 결론(ig70 최적)은 단면 착시**. 강한 마스킹(0.75)은 늦게 만개: ig75 s3 곡선 0.158(ep3)→0.205(ep6)→**0.293(ep8)**→0.255(ep10) 단조상승 후 ep8 峰.
- **★ 신기록: ig75@ep8 = ARI 0.313 ± 0.020 (2-seed: s3 0.293 / s4 0.333)**, cap 1.0, noise 0, **파편비 2.57 (18/7, 역대최저)**, Comp 0.372/0.448, Hom 0.482/0.520.
- 이정표: frozen 0.149 → old champ 0.146 → ig70@ep3 0.237 → **ig75@ep8 0.313 (+110% vs frozen)**.
- ig70 은 ep3 조기峰 후 하락(ep6 cap 붕괴 0.857 일시 관찰) — ig75 는 전 epoch cap 1.0.
- ep10 2-seed: ig75 0.259±0.006(최안정), ig80 0.249±0.051, ig90 0.235±0.026, ig70 0.210±0.018.
- **새 표준: 모든 sweep 은 epochs 10 + ep3/6/8/10 채점** (조기峰 가정 금지). seed5 로 3-seed 확정 예정.
→ Stage 3 (neco): 새 base = queue4096+ignore0.75+local0.3, ep8 중심 평가.

★ 발견: **WM은 fragile 아님.** old champion(local0.15,no-ignore) ARI 0.145 → V1(local0.3+ignore0.70) **0.237 (+0.092, +63%)**. ignore+높은local이 파편비 5.1→3.0(과분할 해소, capture 1.0 유지)로 ARI 급등. 이전 260610 ablation 미참조가 "fragile" 오판의 원인이었음.
