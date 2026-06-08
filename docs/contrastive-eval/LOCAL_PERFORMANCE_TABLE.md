# Local Contrastive Performance Table

작성일: 2026-06-08

목적: 서버에서 최근 돌린 조건과 별개로, 로컬 `D:\project\unknown-contrastive`에 남아 있는 검증 산출물 기준으로 “CNN backbone 이후 contrastive 옵션을 하나씩 추가했을 때 실제 성능이 좋아졌던 근거”를 고정한다.

## 결론

로컬 검증 기준으로 옵션 추가 효과는 있었다. 특히 production-relevant metric인 `noise_pct`, `class_capture_rate`, `completeness`가 좋아졌다.

- Avg30 anchor: `Global InfoNCE -> +Local -> +Queue -> +NEG` 순서에서 AMI/ARI/noise가 개선.
- n500 full: `Global InfoNCE only -> +Queue -> +NeCo -> post τ=0.5` 순서에서 noise가 `15.78% -> 0.00%`, capture가 `0.9337 -> 0.9619`로 개선.
- 단, Homogeneity/ARI는 크게 오르지 않았다. 즉 TAPT/CNN backbone이 이미 강했고, contrastive 옵션은 “근본 분리력”보다 noise 감소와 capture 안정화에 더 크게 기여했다.

## Avg30 Anchor Ablation

데이터: avg30 anchor, 약 900 wafer  
조건: ConvNeXtV2 TAPT backbone frozen, epoch 5, image 384, HDBSCAN `mcs=12`, `ms=3`, `eom`  
출처: `docs/paper/PERFORMANCE_HISTORY.md`, `outputs_contrastive_260511_181441/tier1_B4.json`

| cell | 추가 옵션 | AMI | ARI | noise_pct | n_cluster | 해석 |
|---|---|---:|---:|---:|---:|---|
| B0 | Global InfoNCE only | 0.929 | 0.823 | 6.20% | 37 | baseline |
| B1 | + Local InfoNCE | 0.939 | 0.851 | 3.93% | 37 | ARI 상승, noise 감소 |
| B3 | + MoCo Queue | 0.950 | 0.846 | 1.31% | 36 | noise 크게 감소 |
| B4 | + NEG filter 0.72 | 0.956 | 0.860 | 0.52% | 37 | 이 조건의 best |
| B5 | + NeCo 0.2 | 0.950 | 0.856 | 0.96% | 37 | B4 대비 약간 regression |

Avg30 기준 B0 -> B4:

| metric | baseline B0 | best B4 | 변화 |
|---|---:|---:|---:|
| AMI | 0.929 | 0.956 | +0.027 |
| ARI | 0.823 | 0.860 | +0.037 |
| noise_pct | 6.20% | 0.52% | -5.68pp |

## n500 Full Ablation

데이터: 30-class x 500 + Normal 2000, 총 약 19,250 wafer  
조건: ConvNeXtV2 TAPT backbone frozen, projection 128, epoch 5, image 384  
출처: `outputs_contrastive_260520_204348`부터 `outputs_contrastive_260521_214532`까지의 `tier1_*.json`

| # | recipe | capture | noise_pct | completeness | homogeneity | AMI | ARI | n_cluster |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Global InfoNCE only | 0.9337 | 15.78% | 0.9468 | 0.8348 | 0.8855 | 0.5489 | 40 |
| 2 | + Local DenseCL | 0.9361 | 13.87% | 0.9502 | 0.8111 | 0.8734 | 0.5314 | 37 |
| 3 | + MoCo Queue 4096 | 0.9356 | 9.45% | 0.9474 | 0.8368 | 0.8870 | 0.5596 | 41 |
| 4 | + NEG 0.72 | 0.9250 | 8.23% | 0.9485 | 0.8291 | 0.8831 | 0.5683 | 40 |
| 5 | + NeCo 0.2 | 0.9559 | 6.66% | 0.9660 | 0.8208 | 0.8861 | 0.5648 | 35 |
| 6 | final 4-tool | 0.9559 | 6.66% | 0.9660 | 0.8208 | 0.8861 | 0.5648 | 35 |
| 7 | final + post τ=0.5 | 0.9619 | 0.00% | 0.9679 | 0.8184 | 0.8856 | 0.5489 | 35 |

n500 기준 baseline -> final:

| metric | baseline #1 | final #7 | 변화 |
|---|---:|---:|---:|
| capture | 0.9337 | 0.9619 | +0.0282 |
| noise_pct | 15.78% | 0.00% | -15.78pp |
| completeness | 0.9468 | 0.9679 | +0.0211 |
| homogeneity | 0.8348 | 0.8184 | -0.0164 |
| AMI | 0.8855 | 0.8856 | +0.0001 |
| ARI | 0.5489 | 0.5489 | 0.0000 |
| n_cluster | 40 | 35 | -5 |

## 최근 260608 계열과 구분

최근 서버/신규 조건 계열은 위 표와 같은 “성능 개선 근거”로 보면 안 된다. 다수 run이 HDBSCAN에서 `clusters=0`, `noise=100%`로 무너졌고, 이는 학습 옵션이 전부 무효라는 뜻보다는 현재 데이터/클러스터링 조건이 맞지 않았다는 신호다.

예:

| run | capture | noise_pct | clusters | 해석 |
|---|---:|---:|---:|---|
| `260608_190520_wm811k_pretrain...` | 0.5833 | 41.67% | 4 | 일부 잡힘, noise 과다 |
| `260608_185245_wm50_direct...` | 0.0000 | 100.00% | 0 | clustering 실패 |
| `260608_183346_wm50_direct...` | 0.0000 | 100.00% | 0 | clustering 실패 |
| `260608_181811_wm50_direct...` | 0.0000 | 100.00% | 0 | clustering 실패 |

따라서 현재 판단은 다음이 맞다.

1. 로컬 과거 검증에서는 옵션 추가 효과가 있었다.
2. 최근 서버/신규 조건에서는 같은 효과가 재현되지 않았다.
3. 현업 over-merge 문제는 기존 metric만으로 부족하므로 `largest_group_pct`, cluster purity, representative/composite 육안 검증을 함께 봐야 한다.

