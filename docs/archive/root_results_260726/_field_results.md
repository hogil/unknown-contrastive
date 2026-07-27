# 현업 시뮬레이션 — real cca 8-class 불균형 풀, 라벨·k 없이 그룹핑

pool: 1550장, 분포 {'C+EL': 50, 'C+EL+L': 50, 'C+EL+L+S': 50, 'C+EL+S': 50, 'C+ER': 50, 'C+ER+L': 50, 'C+ER+L+S': 50, 'C+ER+S': 50, 'C+L': 50, 'C+L+S': 50, 'C+S': 50, 'D+EL': 50, 'D+EL+L': 50, 'D+EL+L+S': 50, 'D+EL+S': 50, 'D+ER': 50, 'D+ER+L': 50, 'D+ER+L+S': 50, 'D+ER+S': 50, 'D+L': 50, 'D+L+S': 50, 'D+S': 50, 'EL+L': 50, 'EL+L+S': 50, 'EL+S': 50, 'ER+L': 50, 'ER+L+S': 50, 'ER+S': 50, 'L+S': 50, 'Normal': 50, 'R': 50}

우선순위: P1 capture(불량 회수) > P2 noise% > P3 Completeness > P4 Homogeneity. 라벨은 평가만.
**Random 클래스는 채점 제외** (비정형 — 특정 모양 아님. 클러스터링/학습엔 포함, GT=Random 만 metric 에서 drop).
hdb_unsup_sel = 라벨 없이 silhouette 로 cfg 선택 / hdb_dbcv_sel = 라벨 없이 DBCV(밀도 지표) 로 선택
(둘 다 현업 가능). oracle = AMI 상한 참조 (라벨로 cfg 선택 — 현업 불가, 진단용).

**capture (P1)** = 메인(다수) 클래스로 등장한 클래스 수 / 전체 클래스 수. (예: 10종 중 메인 2종류 → 2/10)
recov (보조) = 자기가 메인인 클러스터에 회수된 이미지 비율의 클래스 평균 (잡탕 셋방 불인정).
noise% = defect 가 noise 로 버려진 비율 (낮을수록 좋음) / nz→noise% = noise 클래스(Random·Normal)가 올바르게 noise 로 빠진 비율 (높을수록 좋음 — 정답 배치 = noise).

| embedding | rule | **capture** | recov | noise% | nz→noise% | k | Comp | Hom | AMI | ARI |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| DINOv3_frozen | hdb_default | 0.0345 | 0.0345 | 14.97 | 32.0 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| DINOv3_frozen | hdb_unsup_sel | 0.1034 | 0.0393 | 15.79 | 26.0 | 3 | 0.4631 | 0.0074 | 0.0057 | 0.0002 |
| DINOv3_frozen | hdb_dbcv_sel | 0.2069 | 0.0669 | 74.28 | 50.0 | 6 | 0.9038 | 0.3224 | 0.4408 | 0.173 |
| DINOv3_frozen | hdb_oracle | 0.1034 | 0.0772 | 65.17 | 41.0 | 3 | 0.988 | 0.3436 | 0.4965 | 0.181 |
| DINOv3_frozen | kmeans_khat(k̂=2) |  |  | 0.0 |  | 2 |  |  |  | 0.0379 |
| umap10+DINOv3_frozen | hdb_default | 0.1379 | 0.1117 | 0.0 | 0.0 | 4 | 0.8419 | 0.2993 | 0.434 | 0.1174 |
| umap10+DINOv3_frozen | hdb_unsup_sel | 0.1034 | 0.1034 | 0.0 | 0.0 | 3 | 0.9186 | 0.2943 | 0.4409 | 0.1161 |
| umap10+DINOv3_frozen | hdb_dbcv_sel | 0.1034 | 0.1034 | 0.0 | 0.0 | 3 | 0.9186 | 0.2943 | 0.4409 | 0.1161 |
| umap10+DINOv3_frozen | hdb_oracle | 0.4138 | 0.2076 | 14.34 | 2.0 | 13 | 0.6073 | 0.446 | 0.488 | 0.2192 |
| umap10+DINOv3_frozen | kmeans_khat(k̂=5) |  |  | 0.0 |  | 5 |  |  |  | 0.1161 |
| FCMAE_frozen | hdb_default | 0.0345 | 0.0345 | 0.0 | 0.0 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| FCMAE_frozen | hdb_unsup_sel | 0.0345 | 0.0345 | 0.0 | 0.0 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| FCMAE_frozen | hdb_dbcv_sel | 0.1379 | 0.0731 | 72.83 | 3.0 | 4 | 0.9663 | 0.3575 | 0.5024 | 0.2098 |
| FCMAE_frozen | hdb_oracle | 0.3448 | 0.0503 | 93.59 | 23.0 | 11 | 0.8829 | 0.7894 | 0.7552 | 0.6722 |
| FCMAE_frozen | kmeans_khat(k̂=2) |  |  | 0.0 |  | 2 |  |  |  | 0.0387 |
| umap10+FCMAE_frozen | hdb_default | 0.1379 | 0.1145 | 0.0 | 0.0 | 4 | 0.8628 | 0.2935 | 0.4303 | 0.1166 |
| umap10+FCMAE_frozen | hdb_unsup_sel | 0.1034 | 0.1034 | 0.0 | 0.0 | 3 | 0.8848 | 0.2856 | 0.4268 | 0.1151 |
| umap10+FCMAE_frozen | hdb_dbcv_sel | 0.1034 | 0.1034 | 0.0 | 0.0 | 3 | 0.8848 | 0.2856 | 0.4268 | 0.1151 |
| umap10+FCMAE_frozen | hdb_oracle | 0.5862 | 0.3041 | 39.38 | 0.0 | 23 | 0.6795 | 0.6266 | 0.6022 | 0.3951 |
| umap10+FCMAE_frozen | kmeans_khat(k̂=5) |  |  | 0.0 |  | 5 |  |  |  | 0.1151 |

## 클러스터 인구조사 — best: DINOv3_frozen

### ① 도미넌트(defect) 클러스터

| 주인 클래스 | 클러스터 수 | 크기 분포 |
|---|--:|---|
| C+EL | 1 | 7 |
| C+EL+L | 0 |  ⚠️ 형성 실패 |
| C+EL+L+S | 0 |  ⚠️ 형성 실패 |
| C+EL+S | 0 |  ⚠️ 형성 실패 |
| C+ER | 1 | 145 |
| C+ER+L | 0 |  ⚠️ 형성 실패 |
| C+ER+L+S | 0 |  ⚠️ 형성 실패 |
| C+ER+S | 0 |  ⚠️ 형성 실패 |
| C+L | 0 |  ⚠️ 형성 실패 |
| C+L+S | 0 |  ⚠️ 형성 실패 |
| C+S | 0 |  ⚠️ 형성 실패 |
| D+EL | 1 | 5 |
| D+EL+L | 0 |  ⚠️ 형성 실패 |
| D+EL+L+S | 0 |  ⚠️ 형성 실패 |
| D+EL+S | 0 |  ⚠️ 형성 실패 |
| D+ER | 1 | 5 |
| D+ER+L | 0 |  ⚠️ 형성 실패 |
| D+ER+L+S | 0 |  ⚠️ 형성 실패 |
| D+ER+S | 0 |  ⚠️ 형성 실패 |
| D+L | 0 |  ⚠️ 형성 실패 |
| D+L+S | 0 |  ⚠️ 형성 실패 |
| D+S | 1 | 10 |
| EL+L | 0 |  ⚠️ 형성 실패 |
| EL+L+S | 0 |  ⚠️ 형성 실패 |
| EL+S | 0 |  ⚠️ 형성 실패 |
| ER+L | 1 | 202 |
| ER+L+S | 0 |  ⚠️ 형성 실패 |
| ER+S | 0 |  ⚠️ 형성 실패 |
| L+S | 0 |  ⚠️ 형성 실패 |

### ② noise 클러스터 (Random/Normal 주인 — 존재 자체가 감점, 목표는 -1 강등)

| 클러스터 | 크기 | 순도 | 구성 top3 |
|---|--:|--:|---|
| #0 | 21 | 1.00 | R 21 |
| #3 | 28 | 1.00 | Normal 28 |

### ③ noise(-1) 버림통: 1127장 (72.7%)
