# unknown 합성 데이터셋 성능표 (E:/data/images/unknown)

portfolio.md 형식. 42-class compositional 합성 wafer (Center_scratch, Edge-Bottom_fork 등).
채점 FINCH p1 (회수모드) / p2 (해석모드, 클러스터수 이상치 근접).
★ 결론: single-label 합성은 held-out 이어도 frozen 백본이 풂 (학습 마진 거의 0). 난이도 본질 = multi-label 중첩(mixed29).

## A. held-out 21/21 disjoint split (학습 21종 / 평가 21종 — 진짜 novel)
| Recipe | M1 (capture) | M2 (noise %) | M3 (Completeness) | M4 (Homogeneity) | ARI | AMI | Sil | k(불량/전체/21) |
|---|---|---|---|---|---|---|---|---|
| DINOv3 frozen | 1.0000 | 0.00% | 0.7388 | 0.8414 | 0.5246 | 0.7472 | 0.229 | 42/48/21 |
| FCMAE frozen | 1.0000 | 0.00% | 0.7679 | 0.9682 | 0.6005 | 0.8220 | 0.245 | 54/60/21 |
| ★ duo frozen (학습0) p1 | 1.0000 | 0.00% | 0.7796 | 0.9768 | 0.6276 | 0.8359 | 0.219 | 52/58/21 |
| ★ duo frozen p2 (해석모드) | 0.9048 | 0.00% | 0.9655 | 0.9232 | 0.8457 | 0.9388 | 0.485 | 19/21/21 |
| 승자레시피 NR학습 (softnce+koleo) | 1.0000 | 0.00% | 0.7336 | 0.8951 | 0.5572 | 0.7648 | 0.254 | 47/52/21 |
- 학습(0.764 AMI) < frozen(0.836) — single-label 은 학습이 frozen 못 넘음. duo frozen p2 가 클러스터 19~21개로 이상(21) 정확 도달 + ARI 0.846.

## B. hard43 (Normal만 학습, 42 eval)
| Recipe | M1 (capture) | M2 (noise %) | M3 (Completeness) | M4 (Homogeneity) | ARI | AMI | Sil | k(불량/전체/42) |
|---|---|---|---|---|---|---|---|---|
| DINOv3 frozen | 1.0000 | 0.00% | 0.8012 | 0.8935 | 0.5996 | 0.7992 | 0.252 | 83/88/42 |
| FCMAE frozen | 1.0000 | 0.00% | 0.8113 | 0.9716 | 0.6231 | 0.8429 | 0.245 | 100/104/42 |
| ★ duo frozen (학습0) p1 | 1.0000 | 0.00% | 0.8310 | 0.9832 | 0.6787 | 0.8665 | 0.255 | 97/102/42 |
| ★ duo frozen p2 (해석모드) | 0.8333 | 0.00% | 0.9891 | 0.9305 | 0.8255 | 0.9525 | 0.461 | 36/38/42 |

## 대조 — mixed29 (multi-label, 진짜 난제)
| 트랙 | 성질 | frozen recov/AMI | 학습 효과 |
|---|---|---|---|
| unknown held-out (21) | single-label | 0.970 / 0.836 | 없음 (학습<frozen) |
| unknown hard43 (42) | single-label | 0.978 / 0.867 | 없음 |
| mixed29 (29) | **multi-label 중첩** | 0.45 / — | ★ 학습이 0.50→0.60 끌어올림 (유일 난제) |
