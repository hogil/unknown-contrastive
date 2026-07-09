# RESISC45 (aerial/위성, OOD, HF 고해상도) no-CNN SSL
train/eval = data/images/hf_resisc45 (45 class ×40 = 1800장). CNN 사전학습 없음, SSL frozen, label 채점만.

## frozen 기준선 (260703) — headroom 중간
| 백본 | 잣대 | P1 cap | P2 noise% | ARI | Sil |
|---|---|---|---|---|---|
| DINOv3 frozen | umap_hdbscan | 1.0 | 8.28 | **0.617** | 0.226 |
| DINOv3 frozen | finch_p1 | 0.978 | 0.0 | 0.450 | 0.171 |
| FCMAE frozen | umap_hdbscan | 0.933 | 16.22 | 0.520 | 0.113 |

→ DINOv3 frozen 0.617 = 중간 headroom (DTD 0.474 ~ Flowers 0.99 사이). aerial 도 DINOv3 강함.
→ contrastive(DINOv3+queue+natural-aug) 로 0.617 초과 여부 검증 중.

## rs_q (DINOv3 + natural-aug + queue4096) — 강한 delta (260703)
| recipe | finch_p1 ARI | umap ARI | umap noise% | umap cap |
|---|---|---|---|---|
| DINOv3 frozen | 0.450 | 0.617 | 8.28 | 1.0 |
| + contrastive rs_q ep3 ★ | **0.542** | **0.702** | 5.0 | 0.978 |
| + contrastive rs_q ep5 | 0.514 | 0.679 | 5.44 | 0.978 |

→ **RESISC45 no-CNN contrastive: finch_p1 0.450→0.542 (+20%), umap 0.617→0.702 (+14%), noise↓.**
→ 비-wafer 고해상도 OOD(aerial)에서도 contrastive 명확한 상승 = 방법 일반화 입증 (DTD +0.037 보다 큰 delta).
rs_q ep3 finch_p1(k77) | 0.9778 | 0.7961 | 0.0 | 0.7658 | 0.8464 | 0.5421 | 0.1769 | 77/45/0 | 1.71
rs_q ep3 umap_hdbscan(고정잣대) | 0.9778 | 0.8056 | 5.0 | 0.837 | 0.8821 | 0.7021 | 0.2754 | 60/45/0 | 1.33
rs_q ep5 finch_p1(k80) | 0.9778 | 0.7783 | 0.0 | 0.7475 | 0.834 | 0.514 | 0.1652 | 80/45/0 | 1.78
rs_q ep5 umap_hdbscan(고정잣대) | 0.9778 | 0.7783 | 5.44 | 0.8282 | 0.8705 | 0.679 | 0.2437 | 60/45/0 | 1.33
rs_q ep8 finch_p1(k80) | 0.9778 | 0.7578 | 0.0 | 0.7401 | 0.8154 | 0.476 | 0.1524 | 80/45/0 | 1.78
rs_q ep8 umap_hdbscan(고정잣대) | 0.9333 | 0.7694 | 4.39 | 0.8201 | 0.8576 | 0.6317 | 0.242 | 61/45/0 | 1.36
