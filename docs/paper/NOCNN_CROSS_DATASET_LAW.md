# no-CNN SSL contrastive — 5-데이터셋 cross-dataset 법칙 (260703)
CNN 사전학습 없음(SSL frozen 백본 DINOv3/FCMAE + InfoNCE contrastive), label 채점만. clustering = FINCH/UMAP+HDBSCAN. 지표 P1~P4/ARI/Sil.

## frozen → +contrastive delta (데이터셋별)
| dataset | 도메인 | 잣대 | frozen ARI | +contrastive best | delta | frozen headroom |
|---|---|---|---|---|---|---|
| **실제 WM-811K** | wafer(실측, OOD) | finch_p2 | 0.149 | 0.280 (DINOv3+local+queue) | **+0.131 (+88%)** | 약(큰 gap) ★★ |
| **RESISC45** | aerial/위성(OOD) | finch_p1 | 0.450 | 0.542 (DINOv3+queue+nataug) | **+0.092 (+20%)** | 중(gap 큼) ★ |
| **DTD** | 텍스처(OOD) | finch_p1 | 0.324 | 0.399 (DINOv3+queue+nataug) | **+0.075 (+23%)** | 중 ★ |
| 합성 wafer | wafer(합성) | finch_p1 | 0.894 | 0.93 (FCMAE+contrast) | +0.036 | 강 |
| Flowers-102 | 자연물(ImageNet-인접) | umap | 0.989 | — | ~0 | 포화 |

## 결론
1. **CNN 사전학습 없이(순수 SSL) contrastive 가 clustering 성능을 올린다** — wafer(실측) +88%, aerial +20%, 텍스처 +23%.
2. **delta 크기 ∝ frozen↔achievable 천장의 gap** — frozen 이 약/중간(WM/RESISC45/DTD)이면 큰 delta, 포화(Flowers)면 여지 없음.
3. **핵심 부품 = queue(negative 확대)**. batch8 few-negative 를 queue4096 이 해소 (DTD/RESISC45 실증). nv-filter/neco 는 자연이미지에서 무이득. local-grid+queue 는 wafer 에서 최강.
4. **백본은 도메인 의존**: 자연이미지 DINOv3(semantic) 우위, wafer 텍스처 FCMAE 우위(합성). 실측 저해상도 WM 은 DINOv3≈FCMAE.
5. **일반화**: 방법이 wafer 전용 아님 — aerial/텍스처 OOD 에서도 동일하게 작동.

## ★ 3-seed robustness 교정 (260703) — 단일-seed → mean±std
| dataset | frozen | 단일-seed(낙관) | **3-seed mean±std (정직)** | delta | 안정성 |
|---|---|---|---|---|---|
| 실제 WM-811K | 0.149 | 0.280 (+88%) | **0.222 ± 0.054** | +0.073 (+49%) | variable |
| RESISC45 (aerial) | 0.617 | 0.702 | **0.702 ± 0.012** | +0.085 | robust ★ |
| DTD (텍스처) | 0.474 | 0.513 | **0.500 ± 0.019** | +0.026 | robust |

→ 단일-seed WM 0.280 은 high outlier. **정직한 headline = 3-seed**. 세 데이터셋 전부 양수 delta(CNN 없이 상승) 유지, RESISC45 가 최강건 positive. (best-epoch 선택이라 약간 낙관적 — fixed-epoch/무라벨선택 재분석은 다음 단계.)
