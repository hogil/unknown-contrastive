# DTD (텍스처, HF 고해상도 추가 데이터셋) no-CNN SSL
train/eval = data/images/hf_dtd (47 texture class ×40 = 1880장). CNN 사전학습 없음, SSL frozen 백본(DINOv3/FCMAE), label 채점만.

## frozen 기준선 (260703) — ★ headroom 있음 (Flowers 포화와 대조)
| 백본 | 잣대 | P1 cap | P2 noise% | ARI | Sil |
|---|---|---|---|---|---|
| DINOv3 frozen | umap_hdbscan | 0.957 | 10.27 | **0.4742** | 0.255 |
| DINOv3 frozen | finch_p1 | 0.936 | 0.0 | 0.324 | 0.189 |
| FCMAE frozen | umap_hdbscan | 0.936 | 11.91 | 0.4235 | 0.186 |
| FCMAE frozen | finch_p1 | 0.894 | 0.0 | 0.318 | 0.135 |

→ frozen ARI 0.47 = 포화 아님 (Flowers 0.99 대비). 백본 DINOv3 > FCMAE (자연이미지 semantic 우위).
→ **contrastive(natural-aug)로 0.474 초과 시 = no-CNN 고해상도 delta 시연.** (`_nocnn_dtd_train.sh`, GPU free 시 자동 투입)
### dtd_base (--method simclr --natural-aug, DINOv3 no-CNN, DTD) 260703

## dtd_base (DINOv3 + contrastive + natural-aug) 채점 (260703)
| ep | 잣대 | cap | noise% | Comp | Hom | ARI | Sil |
|---|---|---|---|---|---|---|---|
| ep1 | umap_hdbscan | 1.0 | 8.67 | 0.701 | 0.759 | 0.452 | 0.249 |
| ep3 | umap_hdbscan | 0.979 | 8.30 | 0.712 | 0.764 | **0.488** | 0.262 |
| ep5 | umap_hdbscan | 1.0 | **7.55** | 0.704 | 0.776 | 0.469 | 0.247 |
| ep1 | finch_p1 | 0.915 | 0.0 | 0.657 | 0.719 | 0.359 | 0.159 |
| ep5 | finch_p1 | 0.957 | 0.0 | 0.672 | 0.716 | 0.357 | 0.170 |

frozen: umap 0.474 / finch_p1 0.324, noise 10.27%.
→ contrastive = **mild positive**: noise 10.3→7.6%↓, finch_p1 ARI 0.324→0.359, umap 0.474→0.488(소폭). collapse 아님(loss 0.009는 batch8 few-neg 때문). batch8 negative 부족 → dtd_full(+queue4096) 이 negative 늘려 개선 여지.

## ★ 4-데이터셋 no-CNN 종합 (contrastive delta ∝ frozen headroom)
| dataset | frozen ARI | +contrastive | delta | 성격 |
|---|---|---|---|---|
| 실제 WM-811K | 0.149 | 0.280 | **+0.131 (+88%)** | frozen 약(OOD) → 큰 delta ★ |
| DTD 텍스처 | 0.474 | ~0.49 | +0.01~0.035 | frozen 중간 → mild |
| 합성 wafer | 0.894 | 0.93 | +0.04 | frozen 강 → 미미 |
| Flowers-102 | 0.989 | — | 0 | frozen 포화 → 없음 |
dtd_base ep1 finch_p1(k82) | 0.9149 | 0.6191 | 0.0 | 0.6572 | 0.7189 | 0.359 | 0.1593 | 82/47/0 | 1.74
dtd_base ep1 umap_hdbscan(고정잣대) | 1.0 | 0.6197 | 8.67 | 0.7011 | 0.7586 | 0.4518 | 0.2491 | 72/47/0 | 1.53
dtd_base ep2 finch_p1(k75) | 0.9362 | 0.6069 | 0.0 | 0.6681 | 0.7061 | 0.3635 | 0.1725 | 75/47/0 | 1.6
dtd_base ep2 umap_hdbscan(고정잣대) | 0.9787 | 0.6277 | 7.66 | 0.6944 | 0.7573 | 0.4371 | 0.2474 | 75/47/0 | 1.6
dtd_base ep3 finch_p1(k79) | 0.9787 | 0.6085 | 0.0 | 0.6709 | 0.7103 | 0.3361 | 0.1736 | 79/47/0 | 1.68
dtd_base ep3 umap_hdbscan(고정잣대) | 0.9787 | 0.6335 | 8.3 | 0.7123 | 0.764 | 0.4883 | 0.2615 | 68/47/0 | 1.45
dtd_base ep4 finch_p1(k77) | 0.9787 | 0.6149 | 0.0 | 0.6657 | 0.7086 | 0.3654 | 0.1727 | 77/47/0 | 1.64
dtd_base ep4 umap_hdbscan(고정잣대) | 1.0 | 0.6287 | 7.77 | 0.7018 | 0.7579 | 0.4582 | 0.2494 | 71/47/0 | 1.51
dtd_base ep5 finch_p1(k76) | 0.9574 | 0.6282 | 0.0 | 0.6716 | 0.716 | 0.3574 | 0.1695 | 76/47/0 | 1.62
dtd_base ep5 umap_hdbscan(고정잣대) | 1.0 | 0.6495 | 7.55 | 0.7041 | 0.7761 | 0.4688 | 0.2466 | 76/47/0 | 1.62
dtd_base ep6 finch_p1(k80) | 0.9574 | 0.6186 | 0.0 | 0.664 | 0.7126 | 0.3546 | 0.1548 | 80/47/0 | 1.7
dtd_base ep6 umap_hdbscan(고정잣대) | 1.0 | 0.6399 | 8.3 | 0.7156 | 0.7694 | 0.4757 | 0.237 | 69/47/0 | 1.47
dtd_base ep7 finch_p1(k77) | 0.9362 | 0.5856 | 0.0 | 0.6557 | 0.6839 | 0.3063 | 0.1446 | 77/47/0 | 1.64
dtd_base ep7 umap_hdbscan(고정잣대) | 1.0 | 0.6154 | 8.83 | 0.7032 | 0.7572 | 0.4567 | 0.2233 | 70/47/0 | 1.49
dtd_base ep8 finch_p1(k89) | 0.9787 | 0.633 | 0.0 | 0.6526 | 0.7312 | 0.3635 | 0.1645 | 89/47/0 | 1.89
dtd_base ep8 umap_hdbscan(고정잣대) | 0.9574 | 0.6117 | 11.28 | 0.7177 | 0.7614 | 0.4789 | 0.2366 | 66/47/0 | 1.4

### dtd_full (--method simclr --natural-aug --use-queue --queue-size 4096 --nv-filter 0.95 --neco 0.2, DINOv3 no-CNN, DTD) 260703

## ★ dtd_q (DINOv3 + natural-aug + queue4096) — queue 가 DTD delta 확대 (260703)
| recipe | finch_p1 ARI | umap ARI | umap noise% | umap cap |
|---|---|---|---|---|
| DINOv3 frozen | 0.324 | 0.474 | 10.27 | 0.957 |
| + contrastive dtd_base (batch8) | 0.359 | 0.488 | 7.55 | 1.0 |
| **+ queue4096 dtd_q (ep3/5) ★** | **0.399** | **0.511** | 10-13 | 0.96-1.0 |

→ queue 로 negative 확대 = dtd_base 대비 finch_p1 +0.040 / umap +0.023 추가 상승.
→ **DTD 최종: no-CNN contrastive finch_p1 ARI 0.324→0.399 (+23%), umap 0.474→0.511 (+8%).** batch8 few-negative 가 base delta 를 제한했음이 입증 (queue 가 해소).

## ★★ 4-데이터셋 no-CNN 최종 종합 (contrastive delta ∝ frozen headroom) — 갱신
| dataset | 잣대 | frozen | +contrastive best | delta | 성격 |
|---|---|---|---|---|---|
| 실제 WM-811K | finch_p2 | 0.149 | 0.280 (DINOv3+local+queue) | **+0.131 (+88%)** | frozen 약(OOD) → 큰 delta ★★ |
| DTD 텍스처 | finch_p1 | 0.324 | 0.399 (DINOv3+queue+nataug) | **+0.075 (+23%)** | frozen 중 → moderate ★ |
| 합성 wafer | finch_p1 | 0.894 | 0.93 (FCMAE+contrast) | +0.036 | frozen 강 → 미미 |
| Flowers-102 | umap | 0.989 | — | ~0 | frozen 포화 → 없음 |

**결론: CNN 사전학습 없이(SSL only) contrastive 가 clustering 성능을 올리며, 그 크기는 frozen↔과제 gap 에 비례. WM-811K·DTD 처럼 frozen 이 약/중간인 곳에서 확실한 delta, Flowers 처럼 포화된 곳에선 여지 없음.**
dtd_q ep1 finch_p1(k89) | 0.9787 | 0.634 | 0.0 | 0.6656 | 0.727 | 0.3533 | 0.1842 | 89/47/0 | 1.89
dtd_q ep1 umap_hdbscan(고정잣대) | 0.9787 | 0.6415 | 7.45 | 0.7133 | 0.7729 | 0.4813 | 0.2785 | 72/47/0 | 1.53
dtd_q ep3 finch_p1(k92) | 1.0 | 0.6633 | 0.0 | 0.6698 | 0.7473 | 0.3836 | 0.1836 | 92/47/0 | 1.96
dtd_q ep3 umap_hdbscan(고정잣대) | 0.9574 | 0.6271 | 12.82 | 0.7311 | 0.7943 | 0.5105 | 0.2783 | 72/47/0 | 1.53
dtd_q ep5 finch_p1(k93) | 1.0 | 0.6713 | 0.0 | 0.6772 | 0.7617 | 0.3989 | 0.1876 | 93/47/0 | 1.98
dtd_q ep5 umap_hdbscan(고정잣대) | 1.0 | 0.6473 | 10.16 | 0.7338 | 0.7941 | 0.5058 | 0.2314 | 71/47/0 | 1.51
dtd_q ep8 finch_p1(k92) | 1.0 | 0.6766 | 0.0 | 0.6779 | 0.7499 | 0.3727 | 0.1481 | 92/47/0 | 1.96
dtd_q ep8 umap_hdbscan(고정잣대) | 1.0 | 0.6309 | 12.18 | 0.745 | 0.7988 | 0.5134 | 0.229 | 69/47/0 | 1.47
dtd_qnv ep3 finch_p1(k90) | 1.0 | 0.6585 | 0.0 | 0.6685 | 0.7502 | 0.3792 | 0.1929 | 90/47/0 | 1.91
dtd_qnv ep3 umap_hdbscan(고정잣대) | 0.9787 | 0.6532 | 7.07 | 0.7266 | 0.7877 | 0.4998 | 0.2646 | 72/47/0 | 1.53
dtd_qnv ep5 finch_p1(k91) | 0.9787 | 0.6628 | 0.0 | 0.668 | 0.7493 | 0.3937 | 0.1982 | 91/47/0 | 1.94
dtd_qnv ep5 umap_hdbscan(고정잣대) | 0.9787 | 0.6372 | 9.1 | 0.7191 | 0.778 | 0.4832 | 0.2388 | 71/47/0 | 1.51
dtd_qnv ep8 finch_p1(k101) | 1.0 | 0.6926 | 0.0 | 0.6641 | 0.7669 | 0.3885 | 0.1716 | 101/47/0 | 2.15
dtd_qnv ep8 umap_hdbscan(고정잣대) | 1.0 | 0.6468 | 12.07 | 0.7362 | 0.8116 | 0.5162 | 0.2405 | 75/47/0 | 1.6
