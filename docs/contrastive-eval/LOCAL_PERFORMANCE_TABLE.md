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

## 2026-06-08 Local Rerun: Current-Best 21

목적: 로컬 `wm811k_50`에서 기존 current-best family 조건을 현재 코드로 다시 실행해, CNN baseline 대비 embedding 이웃 품질이 유지되는지 확인했다.

실행:

- train: `data/images/wm811k_50/train` (320 images)
- eval: `data/images/wm811k_50/eval` (80 images, 8 classes)
- backbone: `runs/260608_073602_cnn_ddp/cnn/best_model.pth`
- condition: `21_fn085_pseudo005_local075_epochscan`
- 핵심 옵션: `queue_size=16384`, `ignore_neg_sim=0.85`, `pseudo_pos_weight=0.05`, `local_weight=0.75`, `local_window=4`, frozen backbone, `img_size=384`, `proj_dim=128`
- local runtime fix: `--num-workers 4`

산출:

- sweep: `runs/260608_195446_local_currentbest21_nw4_260608_195446`
- model: `runs/260608_195449_local_currentbest21_nw4_260608_195446_21_fn085_pseudo005_local075_epochscan/contrastive/best_model.pt`
- t-SNE/kNN: `result_grouping/260608_200054_local_currentbest21_nw4_260608_195446_summary`

Embedding kNN same-class rate:

| model | dim | top1 | k3 | k5 | k7 | k9 | 해석 |
|---|---:|---:|---:|---:|---:|---:|---|
| CNN baseline | 1024 | 0.7375 | 0.7167 | 0.6975 | 0.6607 | 0.6347 | 가장 가까운 1개는 강함 |
| Current-best 21 rerun | 128 | 0.7000 | 0.7042 | 0.7075 | 0.6982 | 0.6444 | k5/k7/k9 이웃 안정성은 개선 |

Delta vs CNN:

| metric | delta |
|---|---:|
| top1 | -0.0375 |
| k5 | +0.0100 |
| k7 | +0.0375 |
| k9 | +0.0097 |

HDBSCAN tier1:

| n_total | n_clusters | noise_pct | capture | completeness | homogeneity | AMI | ARI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0 | 100.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

해석:

- 로컬 current-best 21은 CNN보다 `top1`은 낮지만, `k5/k7/k9`는 오른다. 즉 단일 최근접보다 주변 이웃 묶음이 더 안정적이다.
- HDBSCAN 기본 tier1 조건은 아직 맞지 않는다. 이번 run의 embedding 진단에서 최근접 cosine 거리 `<0.06` 비율이 `96.2%`였으므로, 고정 epsilon/leaf 조건만으로는 noise 또는 merge 실패가 쉽게 난다.
- 다음 실험의 1차 목표는 HDBSCAN 값 조정이 아니라 embedding kNN 지표 개선이다. HDBSCAN sweep은 같은 embedding에서 후처리가 너무 강한지/느슨한지 확인하는 진단 도구로만 본다.

같은 embedding으로 HDBSCAN parameter만 sweep:

- output: `runs/260608_195449_local_currentbest21_nw4_260608_195446_21_fn085_pseudo005_local075_epochscan/contrastive/hdbscan_param_sweep.csv`

| method | eps | min_cluster_size | min_samples | clusters | noise_pct | largest_group_pct | capture | purity | completeness | homogeneity | AMI | ARI | 해석 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| leaf | 0.10 | 2 | 1 | 17 | 12.5% | 15.0% | 0.750 | 0.857 | 0.602 | 0.763 | 0.522 | 0.472 | 세분화 강함, purity 우선 |
| eom | 0.10 | 2 | 1 | 15 | 11.2% | 15.0% | 0.738 | 0.831 | 0.618 | 0.755 | 0.546 | 0.497 | 세분화 + 약간 응집 |
| eom | 0.00 | 5 | 2 | 6 | 7.5% | 17.5% | 0.688 | 0.743 | 0.756 | 0.697 | 0.673 | 0.579 | AMI/ARI 균형 best |
| eom | 0.00 | 10 | 2 | 6 | 7.5% | 17.5% | 0.688 | 0.743 | 0.756 | 0.697 | 0.673 | 0.579 | mcs 5~10 동일 결과 |

판단:

- 기본 `min_samples=15`가 이 작은 eval set에서는 너무 강해서 전부 noise로 보냈다.
- `min_samples=2`로 낮추면 같은 embedding에서 cluster가 살아난다.
- `leaf + mcs=2, ms=1`은 많이 쪼개고 purity를 올리는 조건이다.
- `eom + mcs=5~10, ms=2, eps=0`은 cluster 수는 적지만 AMI/ARI 균형이 가장 낫다.

## 2026-06-08 Embedding Follow-up: Temperature / Local Weight

목적: 21번 조건에서 embedding 분리 압력을 직접 바꾸는 두 축을 확인했다. `NCE_TEMP`를 낮추는 조건과 `local_weight`를 올리는 조건 모두 21번보다 나빴다.

| condition | temp | local_weight | top1 | k5 | k7 | k9 | score(top1,k5,k7) | delta score vs 21 | 판단 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 21 current-best | 0.07 | 0.75 | 0.7000 | 0.7075 | 0.6982 | 0.6444 | 0.7019 | 0.0000 | 유지 |
| 22 local100 | 0.07 | 1.00 | 0.7000 | 0.6875 | 0.6768 | 0.6444 | 0.6881 | -0.0138 | 탈락 |
| 23 lower-temp | 0.05 | 0.75 | 0.6625 | 0.6900 | 0.6714 | 0.6361 | 0.6746 | -0.0273 | 탈락 |

산출:

- 22 sweep: `runs/260608_201937_local_embed22_nw4_260608_201937`
- 22 model: `runs/260608_201941_local_embed22_nw4_260608_201937_22_fn085_pseudo005_local100_epochscan/contrastive/best_model.pt`
- 22 t-SNE/kNN: `result_grouping/260608_202536_local_embed22_nw4_260608_201937_summary`
- 23 sweep: `runs/260608_201206_local_embed23_nw4_260608_201206`
- 23 model: `runs/260608_201209_local_embed23_nw4_260608_201206_23_fn085_pseudo005_local075_temp005_epochscan/contrastive/best_model.pt`
- 23 t-SNE/kNN: `result_grouping/260608_201808_local_embed23_nw4_260608_201206_summary`

결론:

- `temp`를 낮추는 것이 항상 더 엄격한 embedding을 만드는 것은 아니다.
- `local_weight`도 `0.75 -> 1.00`은 과했고, 주변 이웃 안정성(k5/k7)이 떨어졌다.
- 현재 `wm811k_50` 로컬 조건에서는 `NCE_TEMP=0.07`, `local_weight=0.75`가 낫다.
- 다음 embedding 개선 축은 `pseudo_pos_weight`, `IGNORE_NEG_SIM`, `proj_dim`, `train_sampling_ratio` 쪽으로 봐야 한다.
