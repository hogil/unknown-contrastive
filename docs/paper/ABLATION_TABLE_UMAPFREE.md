# no-CNN SSL Contrastive — Ablation 성능향상표 (UMAP-free, finch raw, k-free)

## Protocol 라벨 (260708 — claim 오염 방지)
각 셀은 둘 중 하나로 분류. 같은 표에 섞을 때 protocol 컬럼 명시:
- **[D] open-set discovery**: train 에 불량 0 (Normal-only). eval = 처음 보는 불량. → "미지 불량 발굴" claim. (WM 전 셀, unknown 의 unk_base/unk75)
- **[T] transductive/현업 mining**: 무라벨 pool 전체(불량 포함) 학습+클러스터. → "현업 무라벨 더미 그룹핑" claim. (unkall/wmall, RESISC/DTD self-train)

방법론 (정당성):
- **CNN 사전학습 0** — 백본 = frozen DINOv3 (`convnext_base.dinov3_lvd1689m`, self-supervised, wafer/label 무학습). lr_bb 2e-6.
- **라벨 0 학습** — InfoNCE(SimCLR) contrastive, Normal/train pool 만. class label 은 **채점에만**.
- **k-free, UMAP-free 클러스터링** — FINCH(파라미터 0, cosine, raw 임베딩 직접). k-means 금지, UMAP 금지(`--skip-umap`, 계산 자체 안 함).
- **잣대 규칙 (통일)**: "capture ≥ 0.9 유지하는 가장 성긴 finch 레벨". WM(7 class)=finch_p2, RESISC(45)/DTD(47)=finch_p1. (p 더 올리면 capture 붕괴 → recall 우선 원칙 위반.)
- seed 3(단일, ep3 고정 = 정직 rung). mean±std 는 seed 누적 중.

## ① WM-811K (7 defect class, finch_p2)
| rung | P1 cap | noise% | Comp | Hom | ARI | Sil | 클러스터/정답(파편비) |
|---|---|---|---|---|---|---|---|
| frozen (무학습) | 1.00 | 0.0 | 0.267 | 0.435 | 0.149 | 0.057 | 28/7 (4.0) |
| SSL base | 1.00 | 0.0 | 0.265 | 0.437 | 0.158 | 0.047 | 31/7 (4.4) |
| +local | 1.00 | 0.0 | 0.275 | 0.437 | 0.158 | 0.087 | 29/7 (4.1) |
| +local+queue(MoCo) champion | 1.00 | 0.0 | 0.289 | 0.466 | 0.145 | −0.019 | 36/7 (5.1) |
| **Δ frozen→champion** | 0 | 0 | **+0.022** | **+0.031** | −0.004 | | |

→ fragile: frozen 이 ARI 천장 근처. 부품은 Comp/Hom 만 소폭↑, ARI 는 파편화로 평평.

## ② RESISC45 (45 class, aerial, finch_p1)
| rung | P1 cap | noise% | Comp | Hom | ARI | Sil | 클러스터/정답(파편비) |
|---|---|---|---|---|---|---|---|
| frozen | 0.978 | 0.0 | 0.736 | 0.769 | 0.450 | 0.171 | 67/45 (1.49) | 기여 — |
| SSL base | 1.00 | 0.0 | 0.742 | 0.832 | 0.494 | 0.166 | 81/45 (1.8) | **+0.044** |
| +queue(MoCo) champion | 1.00 | 0.0 | 0.760 | 0.850 | 0.546 | 0.209 | 83/45 (1.84) | **+0.052** |
| **Δ frozen→champion** | **+0.022** | 0 | +0.024 | **+0.081** | **+0.096** ★ | +0.038 | |

→ ★ 강한 향상 + **부품 기여 분리**: SSL 자체 +0.044, MoCo queue +0.052 (거의 균등 가산). capture 0.978→1.0(놓친 class 잡음), Hom +0.081.

## ③ DTD (47 class, texture, finch_p1)
| rung | P1 cap | noise% | Comp | Hom | ARI | Sil | 클러스터/정답(파편비) |
|---|---|---|---|---|---|---|---|
| frozen | 0.936 | 0.0 | 0.641 | 0.679 | 0.324 | 0.189 | 75/47 (1.6) | 기여 — |
| SSL base | 0.979 | 0.0 | 0.660 | 0.722 | 0.353 | 0.190 | 83/47 (1.77) | **+0.029** |
| +queue(MoCo) champion | 1.00 | 0.0 | 0.671 | 0.736 | 0.370 | 0.181 | 87/47 (1.85) | **+0.017** |
| **Δ frozen→champion** | **+0.064** | 0 | +0.030 | **+0.057** | **+0.046** ★ | | |

→ 향상 + **부품 기여 분리**: SSL 자체 +0.029, MoCo queue +0.017. capture 0.936→1.0, Hom +0.057.

## 헤드라인
| dataset | ARI Δ | capture Δ | Hom Δ | 판정 |
|---|---|---|---|---|
| RESISC45 | +0.096 | 0.978→1.00 | +0.081 | ★ 강한 향상 |
| DTD | +0.046 | 0.936→1.00 | +0.057 | 향상 |
| WM-811K | −0.004 | 1.00→1.00 | +0.031 | fragile (Comp/Hom만) |

- **UMAP-free 가 오히려 delta ↑** (RESISC UMAP +0.076 → finch-raw +0.096; DTD +0.020 → +0.046). UMAP 없이도 contrastive(MoCo queue)가 clustering 향상 입증 = 더 정당.
- delta ∝ frozen headroom 법칙 유지: RESISC/DTD(중 headroom) 큰 향상, WM(frozen 천장 근처) fragile.

## 상태 / 다음
- ✅ **3-데이터셋 frozen→base→+component 분해 완성** (seed 3, ep3). RESISC 부품 기여 SSL +0.044 / queue +0.052, DTD +0.029 / +0.017, WM fragile.
- 🔄 **인코더 부품 실험 가동**: wmv(VICReg) / wmneco(NeCo) / wmnv(NV-Retriever) / wmls(LS) / rsv / dtdv — 13-config ladder 에 편입, WM 파편비↓+ARI↑ 노림.
- 🔄 seed 누적 → 3-seed mean±std (현재 seed 4 진행 중).
- 잣대 규칙: capture ≥ 0.9 유지 최성긴 finch 레벨 (WM p2 / RESISC·DTD p1). k-means·UMAP 금지.
