# 현업 시뮬레이션 — real cca 8-class 불균형 풀, 라벨·k 없이 그룹핑

pool: 1779장, 분포 {'Center': 290, 'Donut': 47, 'Edge-Loc': 382, 'Edge-Ring': 609, 'Loc': 279, 'Near-full': 9, 'Random': 75, 'Scratch': 88}

우선순위: P1 capture(불량 회수) > P2 noise% > P3 Completeness > P4 Homogeneity. 라벨은 평가만.
hdb_unsup_sel = 라벨 없이 silhouette 로 cfg 선택 / hdb_dbcv_sel = 라벨 없이 DBCV(밀도 지표) 로 선택
(둘 다 현업 가능). oracle = AMI 상한 참조 (라벨로 cfg 선택 — 현업 불가, 진단용).

| embedding | rule | capture | noise% | k | Comp | Hom | AMI | ARI |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| DINOv3_frozen | hdb_default | 0.8206 | 6.8 | 2 | 0.1934 | 0.0111 | 0.0186 | -0.0104 |
| DINOv3_frozen | hdb_unsup_sel | 0.7298 | 11.19 | 2 | 0.1874 | 0.0115 | 0.0192 | -0.0114 |
| DINOv3_frozen | hdb_dbcv_sel | 0.7298 | 11.19 | 2 | 0.1874 | 0.0115 | 0.0192 | -0.0114 |
| DINOv3_frozen | hdb_oracle | 0.178 | 60.26 | 4 | 0.6072 | 0.7373 | 0.6633 | 0.6012 |
| DINOv3_frozen | kmeans_khat(k̂=2) |  | 0.0 | 2 |  |  |  | 0.2458 |
| umap10+DINOv3_frozen | hdb_default | 0.2779 | 13.49 | 28 | 0.3042 | 0.5858 | 0.3842 | 0.157 |
| umap10+DINOv3_frozen | hdb_unsup_sel | 0.1355 | 38.45 | 66 | 0.2708 | 0.664 | 0.3375 | 0.0686 |
| umap10+DINOv3_frozen | hdb_dbcv_sel | 0.3086 | 11.35 | 38 | 0.2951 | 0.5928 | 0.373 | 0.1332 |
| umap10+DINOv3_frozen | hdb_oracle | 0.33 | 16.02 | 15 | 0.3333 | 0.5301 | 0.3997 | 0.2085 |
| umap10+DINOv3_frozen | kmeans_khat(k̂=7) |  | 0.0 | 7 |  |  |  | 0.2377 |
| FCMAE_frozen | hdb_default | 0.8492 | 5.45 | 2 | 0.1842 | 0.0104 | 0.0173 | -0.0107 |
| FCMAE_frozen | hdb_unsup_sel | 0.7781 | 8.77 | 2 | 0.1788 | 0.0106 | 0.0176 | -0.0114 |
| FCMAE_frozen | hdb_dbcv_sel | 0.8492 | 5.45 | 2 | 0.1842 | 0.0104 | 0.0173 | -0.0107 |
| FCMAE_frozen | hdb_oracle | 0.1582 | 63.35 | 4 | 0.7558 | 0.8041 | 0.7771 | 0.8876 |
| FCMAE_frozen | kmeans_khat(k̂=5) |  | 0.0 | 5 |  |  |  | 0.3883 |
| umap10+FCMAE_frozen | hdb_default | 0.3001 | 5.45 | 32 | 0.2913 | 0.5836 | 0.3721 | 0.1388 |
| umap10+FCMAE_frozen | hdb_unsup_sel | 0.1367 | 30.19 | 69 | 0.2863 | 0.7042 | 0.3674 | 0.0732 |
| umap10+FCMAE_frozen | hdb_dbcv_sel | 0.2421 | 11.69 | 50 | 0.2974 | 0.6306 | 0.3787 | 0.1248 |
| umap10+FCMAE_frozen | hdb_oracle | 0.4143 | 8.83 | 15 | 0.347 | 0.5168 | 0.4064 | 0.2393 |
| umap10+FCMAE_frozen | kmeans_khat(k̂=3) |  | 0.0 | 3 |  |  |  | 0.0398 |
