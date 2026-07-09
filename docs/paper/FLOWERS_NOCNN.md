### Flowers-102 frozen (no-CNN SSL backbone) 260703

=== DINOv3_frozen.npy === (retrieval top1=1.000 top5=0.999)
method | P1 capture | recov | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전체/클래스수/noise) | 파편비(전체÷클래스)
finch_p0(k210) | 1.0 | 1.0 | 0.0 | 0.888 | 1.0 | 0.7375 | 0.3926 | 210/102/0 | 2.06
finch_p1(k76) | 0.7451 | 0.7451 | 0.0 | 1.0 | 0.9122 | 0.6984 | 0.6823 | 76/102/0 | 0.75
finch_p2(k16) | 0.1569 | 0.1569 | 0.0 | 1.0 | 0.5437 | 0.1511 | 0.1679 | 16/102/0 | 0.16
finch_p3(k3) | 0.0294 | 0.0294 | 0.0 | 1.0 | 0.1571 | 0.0118 | 0.0408 | 3/102/0 | 0.03
louvain_res6 | 0.6176 | 0.6176 | 0.0 | 1.0 | 0.8776 | 0.6622 | 0.6177 | 63/102/0 | 0.62
hdbscan_raw(옛다이얼) | 0.1275 | 0.1265 | 65.39 | 0.9984 | 0.6544 | 0.3703 | 0.4219 | 13/102/0 | 0.13
umap_hdbscan(고정잣대) | 0.9902 | 0.9902 | 0.0 | 1.0 | 0.9971 | 0.9891 | 0.9115 | 101/102/0 | 0.99

=== FCMAE_frozen.npy === (retrieval top1=0.997 top5=0.980)
method | P1 capture | recov | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전체/클래스수/noise) | 파편비(전체÷클래스)
finch_p0(k237) | 1.0 | 0.998 | 0.0 | 0.8652 | 0.9992 | 0.6632 | 0.3028 | 237/102/0 | 2.32
finch_p1(k83) | 0.8137 | 0.8098 | 0.0 | 0.9972 | 0.9384 | 0.7998 | 0.4826 | 83/102/0 | 0.81
finch_p2(k10) | 0.098 | 0.098 | 0.0 | 0.9967 | 0.392 | 0.0652 | 0.0878 | 10/102/0 | 0.1
louvain_res6 | 0.5686 | 0.5686 | 0.0 | 0.9914 | 0.8525 | 0.613 | 0.3793 | 58/102/0 | 0.57
hdbscan_raw(옛다이얼) | 0.0392 | 0.0373 | 83.63 | 1.0 | 0.415 | 0.1937 | 0.3601 | 4/102/0 | 0.04
umap_hdbscan(고정잣대) | 0.9706 | 0.9706 | 0.1 | 0.9993 | 0.9908 | 0.968 | 0.5789 | 99/102/0 | 0.97
[CSV] D:\project\unknown-contrastive\result_grouping\_field_flowers102\scores.csv

## ★ 해석 (260703) — Flowers-102 는 frozen 포화 (headroom 없음)
올바른 잣대(102 class → umap_hdbscan k=101 또는 finch_p0 k=210; finch_p2 는 k=16 과병합 = 틀림):
- DINOv3 frozen: cap 0.990 / noise 0.0 / Comp 1.0 / Hom 0.997 / **ARI 0.989** / Sil 0.912
- FCMAE frozen:  cap 0.971 / noise 0.1 / **ARI 0.968** / Sil 0.579
- 백본 서열 = **DINOv3 > FCMAE** (자연이미지 semantic SSL 우위 — wafer 텍스처의 FCMAE 우위와 반대).

**contrastive delta ∝ frozen headroom (전 데이터셋 법칙 확정):**
| dataset | frozen ARI | +contrastive | 효과 |
|---|---|---|---|
| 실제 WM-811K (OOD, frozen 약함) | 0.15 | 0.28 | ✅ +88% |
| 합성 wafer (frozen 강함) | 0.89 | 0.93 | ⚠️ 미미 |
| Flowers-102 (ImageNet-인접) | 0.989 | (여지 0) | ❌ 포화 |

→ Flowers 는 frozen 이 이미 풀어 contrastive SSL train 무의미. "학습이 성능을 올린다"는 **frozen↔과제 gap 이 클 때만**. no-CNN contrastive 의 가치가 실증되는 곳 = frozen 이 약한 **실제 wafer(WM-811K)**.
