# Unknown Mixed / Defect-Aware Queue 260710

- eval: D:\project\unknown-contrastive\data\images\unknown_eval100
- defect-aware train: D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710
- all-unlabeled train: D:\project\unknown-contrastive\data\images\unknown_train_all
- embeddings: D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings
- log: D:\project\unknown-contrastive\_unknown_mixed_queue_260710.log

Queue order:

1. `unkda_base`: train `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710`, score strict novel by excluding the train defect classes.
2. NV threshold sweep: `0.50`, `0.60`, `0.70`, `0.75`, `0.80`, `0.85`, `0.90`, `0.93`, `0.95`, `0.97`, `0.98`, `0.99`; same split, one threshold per 10-epoch run. Together with `unkda_base` (`NV=0`), this yields thirteen curve points. `0.70/0.75/0.80` form the local three-point sweep around the expected useful range. The low range was added after NV0.90 masked only `0.03%` of in-batch candidates at ep6.
3. `unkda_q4k`: queue4096 only; this is the queue-axis univariate control missing from the first queue.
4. `unkda_adapter_frozen`: residual adapter only with backbone LR fixed at zero.
5. `unkda_fcmae`: same defect-aware split with FCMAE initialization; no NV/queue/local combination.
6. `unkall_base`: train `D:\project\unknown-contrastive\data\images\unknown_train_all`, score field-mixed.

Acceptance rule: do not add queue+NV or another multi-option recipe until its individual options beat `unkda_base` on the same strict-novel split.

## NV sweep performance plot

- plot: `D:\project\unknown-contrastive\docs\paper\figs_unknown_nv_sweep_260710.png`
- machine-readable scores: `D:\project\unknown-contrastive\docs\paper\UNKNOWN_NV_SWEEP_SCORES_260710.csv`
- per-recipe summary: `D:\project\unknown-contrastive\docs\paper\UNKNOWN_NV_SWEEP_SUMMARY_260710.csv`
- update command: `python D:\project\unknown-contrastive\scripts\plot_unknown_nv_sweep.py`

The plot shows FINCH-p2 and Louvain epoch trajectories, frozen baselines, and both the best-observed and latest available epoch. Best-observed values are exploratory only; they are not accepted results until the seed gate is satisfied.

Frozen strict-novel baseline (32 unseen classes, `grade_only`, 2026-07-10 rescore):

| clusterer | capture | recov | noise% | Comp | Hom | ARI | Sil | fragment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FINCH p2 | 0.9688 | 0.8506 | 0.0 | 0.8302 | 0.8918 | 0.7090 | 0.3045 | 1.81 |
| Louvain r6 | 0.9688 | 0.8553 | 0.0 | 0.8684 | 0.9075 | 0.7850 | 0.3775 | 1.62 |

Palette check: retaining the original PNG background reduced FINCH-p2 ARI to 0.6456 and Louvain ARI to 0.7519, so `grade_only` remains the primary preprocessing mode.

Defect-aware train split:

- manifest: `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710\manifest_260710.json`
- eval: `D:\project\unknown-contrastive\data\images\unknown_eval100`
- embeddings: `D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings`
- production review output root: `D:\project\unknown-contrastive\result_grouping\_production_review_260710`

Operational note: field data does not provide a perfect "Normal" oracle. In deployment, Normal should be treated as a high-confidence background set from production metadata and conservative filtering, not as a guaranteed truth label.

### unkda_base (defect-aware strict novel)
#### unkda_base ep1 backbone
unkda_base ep1 F finch_p1(k209) | 1.0 | 0.8241 | 0.0 | 0.6155 | 0.8817 | 0.3259 | 0.1228 | 209/32/45 | 6.53
unkda_base ep1 F finch_p2(k60) | 0.9688 | 0.7478 | 0.0 | 0.7521 | 0.8251 | 0.5927 | 0.2609 | 60/32/12 | 1.88
unkda_base ep1 F louvain_res6 | 0.9375 | 0.7475 | 0.0 | 0.7617 | 0.8277 | 0.614 | 0.3414 | 58/32/14 | 1.81
unkda_base ep1 F hdbscan_raw(옛다이얼) | 0.625 | 0.4163 | 47.95 | 0.9377 | 0.8863 | 0.766 | 0.5893 | 30/32/7 | 0.94
#### unkda_base ep2 backbone
unkda_base ep2 F finch_p1(k198) | 1.0 | 0.8484 | 0.0 | 0.6399 | 0.8946 | 0.3593 | 0.1214 | 198/32/45 | 6.19
unkda_base ep2 F finch_p2(k58) | 0.875 | 0.7847 | 0.0 | 0.8021 | 0.8491 | 0.5959 | 0.2954 | 58/32/12 | 1.81
unkda_base ep2 F louvain_res6 | 0.9375 | 0.8091 | 0.0 | 0.8056 | 0.8696 | 0.6878 | 0.3349 | 57/32/13 | 1.78
unkda_base ep2 F hdbscan_raw(옛다이얼) | 0.6562 | 0.4316 | 45.22 | 0.9375 | 0.8786 | 0.7518 | 0.6045 | 33/32/8 | 1.03
#### unkda_base ep3 backbone
unkda_base ep3 F finch_p1(k190) | 1.0 | 0.8103 | 0.0 | 0.6287 | 0.8767 | 0.3372 | 0.1203 | 190/32/40 | 5.94
unkda_base ep3 F finch_p2(k54) | 1.0 | 0.7241 | 0.0 | 0.7585 | 0.8126 | 0.538 | 0.3041 | 54/32/10 | 1.69
unkda_base ep3 F louvain_res6 | 1.0 | 0.7588 | 0.0 | 0.7735 | 0.8439 | 0.6281 | 0.343 | 57/32/11 | 1.78
unkda_base ep3 F hdbscan_raw(옛다이얼) | 0.625 | 0.4012 | 52.24 | 0.9047 | 0.9143 | 0.7739 | 0.6214 | 33/32/6 | 1.03
#### unkda_base ep3 projection
unkda_base ep3 Z finch_p1(k224) | 1.0 | 0.7462 | 0.0 | 0.5619 | 0.8176 | 0.2298 | 0.1371 | 224/32/59 | 7.0
unkda_base ep3 Z finch_p2(k57) | 0.9062 | 0.6671 | 0.0 | 0.6999 | 0.7499 | 0.4351 | 0.2515 | 57/32/11 | 1.78
unkda_base ep3 Z louvain_res6 | 0.9062 | 0.7072 | 0.0 | 0.6882 | 0.7813 | 0.4958 | 0.3465 | 63/32/14 | 1.97
unkda_base ep3 Z hdbscan_raw(옛다이얼) | 0.5312 | 0.3469 | 59.03 | 0.874 | 0.9051 | 0.7433 | 0.6652 | 29/32/4 | 0.91
#### unkda_base ep4 backbone
unkda_base ep4 F finch_p1(k202) | 1.0 | 0.8128 | 0.0 | 0.6155 | 0.873 | 0.3245 | 0.1206 | 202/32/43 | 6.31
unkda_base ep4 F finch_p2(k49) | 0.9375 | 0.7309 | 0.0 | 0.7798 | 0.8146 | 0.5842 | 0.3243 | 49/32/11 | 1.53
unkda_base ep4 F louvain_res6 | 1.0 | 0.7609 | 0.0 | 0.779 | 0.8368 | 0.6454 | 0.3467 | 53/32/11 | 1.66
unkda_base ep4 F hdbscan_raw(옛다이얼) | 0.6562 | 0.3959 | 52.49 | 0.9622 | 0.9101 | 0.8537 | 0.6485 | 26/32/6 | 0.81
#### unkda_base ep6 backbone
unkda_base ep6 F finch_p1(k198) | 1.0 | 0.7775 | 0.0 | 0.6004 | 0.8469 | 0.3373 | 0.1417 | 198/32/40 | 6.19
unkda_base ep6 F finch_p2(k45) | 0.8125 | 0.6766 | 0.0 | 0.7718 | 0.7782 | 0.5433 | 0.2542 | 45/32/11 | 1.41
unkda_base ep6 F louvain_res6 | 0.9375 | 0.7247 | 0.0 | 0.7397 | 0.8063 | 0.585 | 0.3409 | 55/32/12 | 1.72
unkda_base ep6 F hdbscan_raw(옛다이얼) | 0.5938 | 0.3781 | 53.79 | 0.9191 | 0.8999 | 0.7736 | 0.6141 | 30/32/6 | 0.94
#### unkda_base ep6 projection
unkda_base ep6 Z finch_p1(k212) | 1.0 | 0.7053 | 0.0 | 0.5434 | 0.7837 | 0.2071 | 0.0947 | 212/32/48 | 6.62
unkda_base ep6 Z finch_p2(k53) | 0.9375 | 0.5722 | 0.0 | 0.6517 | 0.6871 | 0.3576 | 0.1358 | 53/32/11 | 1.66
unkda_base ep6 Z louvain_res6 | 0.9062 | 0.6069 | 0.0 | 0.6387 | 0.711 | 0.4027 | 0.2576 | 56/32/9 | 1.75
unkda_base ep6 Z hdbscan_raw(옛다이얼) | 0.5 | 0.2347 | 71.23 | 0.8869 | 0.861 | 0.7121 | 0.7033 | 22/32/4 | 0.69
#### unkda_base ep8 backbone
unkda_base ep8 F finch_p1(k194) | 1.0 | 0.7991 | 0.0 | 0.6125 | 0.8681 | 0.3101 | 0.1071 | 194/32/45 | 6.06
unkda_base ep8 F finch_p2(k45) | 0.875 | 0.6819 | 0.0 | 0.7874 | 0.7959 | 0.5336 | 0.2533 | 45/32/8 | 1.41
unkda_base ep8 F louvain_res6 | 0.9375 | 0.6953 | 0.0 | 0.7748 | 0.8131 | 0.608 | 0.3024 | 45/32/9 | 1.41
unkda_base ep8 F hdbscan_raw(옛다이얼) | 0.5312 | 0.3538 | 56.91 | 0.9547 | 0.8795 | 0.799 | 0.6354 | 22/32/3 | 0.69
#### unkda_base ep8 projection
unkda_base ep8 Z finch_p1(k217) | 1.0 | 0.6875 | 0.0 | 0.5282 | 0.779 | 0.2151 | 0.1249 | 217/32/42 | 6.78
unkda_base ep8 Z finch_p2(k52) | 0.9062 | 0.5453 | 0.0 | 0.6139 | 0.6538 | 0.3626 | 0.1884 | 52/32/4 | 1.62
unkda_base ep8 Z louvain_res6 | 0.9688 | 0.6156 | 0.0 | 0.6233 | 0.7097 | 0.4435 | 0.2608 | 60/32/10 | 1.88
unkda_base ep8 Z hdbscan_raw(옛다이얼) | 0.5 | 0.2831 | 63.39 | 0.8886 | 0.825 | 0.6604 | 0.6769 | 23/32/2 | 0.72
#### unkda_base ep10 backbone
unkda_base ep10 F finch_p1(k173) | 1.0 | 0.7225 | 0.0 | 0.5821 | 0.8031 | 0.3327 | 0.0999 | 173/32/31 | 5.41
unkda_base ep10 F finch_p2(k44) | 0.8438 | 0.6041 | 0.0 | 0.7141 | 0.736 | 0.5026 | 0.2499 | 44/32/5 | 1.38
unkda_base ep10 F louvain_res6 | 0.9375 | 0.6438 | 0.0 | 0.7089 | 0.7578 | 0.5587 | 0.302 | 47/32/8 | 1.47
unkda_base ep10 F hdbscan_raw(옛다이얼) | 0.5312 | 0.3047 | 65.04 | 0.9144 | 0.9074 | 0.8528 | 0.6286 | 24/32/3 | 0.75
#### unkda_base ep10 projection
unkda_base ep10 Z finch_p1(k207) | 1.0 | 0.6466 | 0.0 | 0.507 | 0.7368 | 0.1807 | 0.1306 | 207/32/44 | 6.47
unkda_base ep10 Z finch_p2(k54) | 0.8438 | 0.5247 | 0.0 | 0.6037 | 0.6453 | 0.3191 | 0.1333 | 54/32/11 | 1.69
unkda_base ep10 Z louvain_res6 | 0.9375 | 0.5672 | 0.0 | 0.5934 | 0.6848 | 0.3864 | 0.2646 | 62/32/6 | 1.94
unkda_base ep10 Z hdbscan_raw(옛다이얼) | 0.4062 | 0.2516 | 67.07 | 0.8314 | 0.8172 | 0.587 | 0.6472 | 23/32/3 | 0.72

### unkda_nv090 (defect-aware strict novel + NV 0.90)
### unkda_nv090 (defect-aware strict novel + NV 0.90)
### unkda_nv090 (defect-aware strict novel + NV 0.90)
### unkda_nv090 (defect-aware strict novel + NV 0.90)
### unkda_nv090 (defect-aware strict novel + NV 0.90)
#### unkda_nv090 ep1 backbone
unkda_nv090 ep1 F finch_p1(k188) | 0.9688 | 0.8175 | 0.0 | 0.646 | 0.8911 | 0.3664 | 0.1244 | 188/32/46 | 5.88
unkda_nv090 ep1 F finch_p2(k50) | 0.875 | 0.68 | 0.0 | 0.7803 | 0.7969 | 0.5418 | 0.2822 | 50/32/13 | 1.56
unkda_nv090 ep1 F louvain_res6 | 0.9375 | 0.7853 | 0.0 | 0.8078 | 0.855 | 0.6747 | 0.3266 | 53/32/12 | 1.66
unkda_nv090 ep1 F hdbscan_raw(옛다이얼) | 0.5938 | 0.3812 | 55.32 | 0.9494 | 0.9152 | 0.8339 | 0.6607 | 29/32/6 | 0.91
#### unkda_nv090 ep2 backbone
unkda_nv090 ep2 F finch_p1(k197) | 1.0 | 0.8009 | 0.0 | 0.613 | 0.8642 | 0.3019 | 0.1118 | 197/32/47 | 6.16
unkda_nv090 ep2 F finch_p2(k50) | 0.9375 | 0.6959 | 0.0 | 0.7566 | 0.7791 | 0.5202 | 0.1942 | 50/32/11 | 1.56
unkda_nv090 ep2 F louvain_res6 | 0.9375 | 0.7366 | 0.0 | 0.7576 | 0.8161 | 0.6231 | 0.3206 | 54/32/12 | 1.69
unkda_nv090 ep2 F hdbscan_raw(옛다이얼) | 0.5312 | 0.3462 | 60.46 | 0.9263 | 0.9112 | 0.8356 | 0.6641 | 28/32/6 | 0.88
#### unkda_nv090 ep3 backbone
unkda_nv090 ep3 F finch_p1(k193) | 1.0 | 0.7753 | 0.0 | 0.5972 | 0.8386 | 0.2969 | 0.115 | 193/32/44 | 6.03
unkda_nv090 ep3 F finch_p2(k49) | 0.8125 | 0.6553 | 0.0 | 0.7242 | 0.7519 | 0.4555 | 0.2615 | 49/32/8 | 1.53
unkda_nv090 ep3 F louvain_res6 | 0.9688 | 0.7209 | 0.0 | 0.7094 | 0.7931 | 0.5503 | 0.2998 | 61/32/10 | 1.91
unkda_nv090 ep3 F hdbscan_raw(옛다이얼) | 0.5312 | 0.3053 | 61.7 | 0.884 | 0.8738 | 0.7366 | 0.6265 | 27/32/6 | 0.84
#### unkda_nv090 ep3 projection
unkda_nv090 ep3 Z finch_p1(k213) | 1.0 | 0.67 | 0.0 | 0.525 | 0.766 | 0.1978 | 0.097 | 213/32/37 | 6.66
unkda_nv090 ep3 Z finch_p2(k58) | 0.9688 | 0.5656 | 0.0 | 0.6117 | 0.6727 | 0.354 | 0.1804 | 58/32/8 | 1.81
unkda_nv090 ep3 Z louvain_res6 | 0.9375 | 0.6147 | 0.0 | 0.6175 | 0.6997 | 0.3996 | 0.2457 | 61/32/9 | 1.91
unkda_nv090 ep3 Z hdbscan_raw(옛다이얼) | 0.4375 | 0.2431 | 65.74 | 0.8448 | 0.794 | 0.6064 | 0.6753 | 23/32/4 | 0.72
#### unkda_nv090 ep4 backbone
unkda_nv090 ep4 F finch_p1(k179) | 1.0 | 0.7344 | 0.0 | 0.5811 | 0.8018 | 0.3309 | 0.0912 | 179/32/38 | 5.59
unkda_nv090 ep4 F finch_p2(k52) | 0.9375 | 0.6347 | 0.0 | 0.6941 | 0.7298 | 0.4829 | 0.21 | 52/32/9 | 1.62
unkda_nv090 ep4 F louvain_res6 | 1.0 | 0.6909 | 0.0 | 0.6866 | 0.7623 | 0.5452 | 0.2704 | 57/32/8 | 1.78
unkda_nv090 ep4 F hdbscan_raw(옛다이얼) | 0.5 | 0.3287 | 57.45 | 0.9 | 0.8579 | 0.6992 | 0.6264 | 25/32/5 | 0.78
#### unkda_nv090 ep6 backbone
unkda_nv090 ep6 F finch_p1(k165) | 1.0 | 0.6703 | 0.0 | 0.5615 | 0.7705 | 0.2788 | 0.0961 | 165/32/26 | 5.16
unkda_nv090 ep6 F finch_p2(k39) | 0.8125 | 0.5731 | 0.0 | 0.7342 | 0.7131 | 0.4347 | 0.198 | 39/32/5 | 1.22
unkda_nv090 ep6 F louvain_res6 | 0.9062 | 0.6231 | 0.0 | 0.6681 | 0.7465 | 0.5005 | 0.278 | 56/32/9 | 1.75
unkda_nv090 ep6 F hdbscan_raw(옛다이얼) | 0.5 | 0.3194 | 56.18 | 0.9143 | 0.8294 | 0.6396 | 0.5654 | 25/32/3 | 0.78
#### unkda_nv090 ep6 projection
unkda_nv090 ep6 Z finch_p1(k214) | 1.0 | 0.6225 | 0.0 | 0.4972 | 0.7263 | 0.1688 | 0.0809 | 214/32/40 | 6.69
unkda_nv090 ep6 Z finch_p2(k60) | 0.875 | 0.5234 | 0.0 | 0.5948 | 0.6511 | 0.3499 | 0.1324 | 60/32/9 | 1.88
unkda_nv090 ep6 Z louvain_res6 | 0.9062 | 0.5672 | 0.0 | 0.5875 | 0.6802 | 0.3675 | 0.2695 | 63/32/8 | 1.97
unkda_nv090 ep6 Z hdbscan_raw(옛다이얼) | 0.5 | 0.2828 | 62.18 | 0.88 | 0.8205 | 0.682 | 0.6889 | 23/32/3 | 0.72
#### unkda_nv090 ep8 backbone
unkda_nv090 ep8 F finch_p1(k190) | 1.0 | 0.71 | 0.0 | 0.559 | 0.7941 | 0.2586 | 0.1222 | 190/32/36 | 5.94
unkda_nv090 ep8 F finch_p2(k48) | 0.9062 | 0.5856 | 0.0 | 0.7168 | 0.7195 | 0.3875 | 0.2026 | 48/32/6 | 1.5
unkda_nv090 ep8 F louvain_res6 | 0.9375 | 0.6509 | 0.0 | 0.6665 | 0.7524 | 0.4842 | 0.263 | 58/32/9 | 1.81
unkda_nv090 ep8 F hdbscan_raw(옛다이얼) | 0.5 | 0.32 | 56.68 | 0.8927 | 0.8273 | 0.6361 | 0.5335 | 25/32/5 | 0.78
#### unkda_nv090 ep8 projection
unkda_nv090 ep8 Z finch_p1(k207) | 1.0 | 0.6244 | 0.0 | 0.4954 | 0.724 | 0.1819 | 0.0967 | 207/32/34 | 6.47
unkda_nv090 ep8 Z finch_p2(k50) | 0.8125 | 0.4487 | 0.0 | 0.5411 | 0.5749 | 0.2846 | 0.03 | 50/32/4 | 1.56
unkda_nv090 ep8 Z louvain_res6 | 1.0 | 0.5444 | 0.0 | 0.5784 | 0.6558 | 0.3485 | 0.1698 | 58/32/9 | 1.81
unkda_nv090 ep8 Z hdbscan_raw(옛다이얼) | 0.4688 | 0.2134 | 74.66 | 0.8722 | 0.8872 | 0.7655 | 0.6721 | 21/32/2 | 0.66
#### unkda_nv090 ep10 backbone
unkda_nv090 ep10 F finch_p1(k219) | 1.0 | 0.6625 | 0.0 | 0.5265 | 0.7726 | 0.1843 | 0.112 | 219/32/37 | 6.84
unkda_nv090 ep10 F finch_p2(k58) | 0.875 | 0.5622 | 0.0 | 0.6542 | 0.7057 | 0.376 | 0.2077 | 58/32/9 | 1.81
unkda_nv090 ep10 F louvain_res6 | 0.9688 | 0.595 | 0.0 | 0.6324 | 0.7168 | 0.4295 | 0.2602 | 59/32/9 | 1.84
unkda_nv090 ep10 F hdbscan_raw(옛다이얼) | 0.5312 | 0.3241 | 58.49 | 0.8856 | 0.8586 | 0.67 | 0.6272 | 27/32/5 | 0.84
#### unkda_nv090 ep10 projection
unkda_nv090 ep10 Z finch_p1(k239) | 1.0 | 0.6219 | 0.0 | 0.4902 | 0.7302 | 0.1671 | 0.1058 | 239/32/45 | 7.47
unkda_nv090 ep10 Z finch_p2(k62) | 0.9375 | 0.5094 | 0.0 | 0.5737 | 0.6375 | 0.3268 | 0.1507 | 62/32/7 | 1.94
unkda_nv090 ep10 Z louvain_res6 | 0.9375 | 0.5441 | 0.0 | 0.5847 | 0.6692 | 0.3612 | 0.2296 | 60/32/9 | 1.88
unkda_nv090 ep10 Z hdbscan_raw(옛다이얼) | 0.4688 | 0.1966 | 76.56 | 0.8232 | 0.8941 | 0.6983 | 0.7623 | 23/32/3 | 0.72

### unkda_nv050 (defect-aware strict novel + NV 0.50)
#### unkda_nv050 ep1 backbone
unkda_nv050 ep1 F finch_p1(k236) | 1.0 | 0.8794 | 0.0 | 0.64 | 0.9257 | 0.3304 | 0.1871 | 236/32/64 | 7.38
unkda_nv050 ep1 F finch_p2(k66) | 1.0 | 0.8134 | 0.0 | 0.7955 | 0.8836 | 0.64 | 0.3753 | 66/32/18 | 2.06
unkda_nv050 ep1 F louvain_res6 | 1.0 | 0.8162 | 0.0 | 0.8131 | 0.8914 | 0.6731 | 0.436 | 62/32/16 | 1.94
unkda_nv050 ep1 F hdbscan_raw(옛다이얼) | 0.7188 | 0.4853 | 43.57 | 0.9675 | 0.9319 | 0.8571 | 0.6572 | 31/32/6 | 0.97
#### unkda_nv050 ep2 backbone
unkda_nv050 ep2 F finch_p1(k211) | 1.0 | 0.8928 | 0.0 | 0.6542 | 0.9343 | 0.3372 | 0.134 | 211/32/48 | 6.59
unkda_nv050 ep2 F finch_p2(k60) | 0.9688 | 0.8225 | 0.0 | 0.8175 | 0.8823 | 0.6729 | 0.304 | 60/32/14 | 1.88
unkda_nv050 ep2 F louvain_res6 | 0.9688 | 0.8394 | 0.0 | 0.8649 | 0.9032 | 0.7571 | 0.4509 | 47/32/9 | 1.47
unkda_nv050 ep2 F hdbscan_raw(옛다이얼) | 0.6562 | 0.5406 | 24.55 | 0.9834 | 0.8125 | 0.5123 | 0.6301 | 28/32/7 | 0.88
#### unkda_nv050 ep3 backbone
unkda_nv050 ep3 F finch_p1(k219) | 1.0 | 0.8734 | 0.0 | 0.6287 | 0.9136 | 0.3014 | 0.1575 | 219/32/48 | 6.84
unkda_nv050 ep3 F finch_p2(k60) | 0.9688 | 0.7847 | 0.0 | 0.7713 | 0.8449 | 0.5916 | 0.2843 | 60/32/12 | 1.88
unkda_nv050 ep3 F louvain_res6 | 0.9375 | 0.8159 | 0.0 | 0.8226 | 0.8696 | 0.7312 | 0.3878 | 51/32/10 | 1.59
unkda_nv050 ep3 F hdbscan_raw(옛다이얼) | 0.6562 | 0.5334 | 21.5 | 0.979 | 0.791 | 0.4364 | 0.609 | 31/32/8 | 0.97
#### unkda_nv050 ep3 projection
unkda_nv050 ep3 Z finch_p1(k242) | 1.0 | 0.6769 | 0.0 | 0.5161 | 0.7714 | 0.1684 | 0.0999 | 242/32/60 | 7.56
unkda_nv050 ep3 Z finch_p2(k54) | 0.8438 | 0.5347 | 0.0 | 0.6224 | 0.6699 | 0.3534 | 0.2164 | 54/32/14 | 1.69
unkda_nv050 ep3 Z louvain_res6 | 0.9062 | 0.5659 | 0.0 | 0.5964 | 0.6879 | 0.3698 | 0.3271 | 63/32/14 | 1.97
unkda_nv050 ep3 Z hdbscan_raw(옛다이얼) | 0.375 | 0.1848 | 28.55 | 0.9658 | 0.4004 | 0.1118 | 0.7468 | 16/32/4 | 0.5
#### unkda_nv050 ep4 backbone
unkda_nv050 ep4 F finch_p1(k222) | 1.0 | 0.8538 | 0.0 | 0.6273 | 0.9035 | 0.2968 | 0.1483 | 222/32/56 | 6.94
unkda_nv050 ep4 F finch_p2(k59) | 0.9062 | 0.7597 | 0.0 | 0.7819 | 0.8461 | 0.6192 | 0.3248 | 59/32/11 | 1.84
unkda_nv050 ep4 F louvain_res6 | 0.9375 | 0.805 | 0.0 | 0.8193 | 0.8681 | 0.722 | 0.3929 | 52/32/11 | 1.62
unkda_nv050 ep4 F hdbscan_raw(옛다이얼) | 0.6562 | 0.5315 | 22.58 | 0.9799 | 0.7941 | 0.4431 | 0.6091 | 31/32/8 | 0.97
#### unkda_nv050 ep6 backbone
unkda_nv050 ep6 F finch_p1(k211) | 1.0 | 0.9091 | 0.0 | 0.6541 | 0.941 | 0.3467 | 0.1294 | 211/32/44 | 6.59
unkda_nv050 ep6 F finch_p2(k57) | 1.0 | 0.8216 | 0.0 | 0.8312 | 0.8894 | 0.6742 | 0.3122 | 57/32/14 | 1.78
unkda_nv050 ep6 F louvain_res6 | 1.0 | 0.8391 | 0.0 | 0.8609 | 0.9058 | 0.7718 | 0.4161 | 49/32/10 | 1.53
unkda_nv050 ep6 F hdbscan_raw(옛다이얼) | 0.6562 | 0.5328 | 26.83 | 0.9842 | 0.8293 | 0.5825 | 0.601 | 26/32/4 | 0.81
#### unkda_nv050 ep6 projection
unkda_nv050 ep6 Z finch_p1(k263) | 1.0 | 0.7525 | 0.0 | 0.5436 | 0.8318 | 0.1905 | 0.1999 | 263/32/55 | 8.22
unkda_nv050 ep6 Z finch_p2(k60) | 0.9688 | 0.6122 | 0.0 | 0.6465 | 0.7317 | 0.3919 | 0.1159 | 60/32/11 | 1.88
unkda_nv050 ep6 Z louvain_res6 | 0.9688 | 0.6669 | 0.0 | 0.6513 | 0.7543 | 0.3946 | 0.2807 | 62/32/9 | 1.94
unkda_nv050 ep6 Z hdbscan_raw(옛다이얼) | 0.375 | 0.2606 | 43.16 | 0.9721 | 0.661 | 0.4417 | 0.6249 | 13/32/1 | 0.41
#### unkda_nv050 ep8 backbone
unkda_nv050 ep8 F finch_p1(k213) | 1.0 | 0.7719 | 0.0 | 0.6042 | 0.8578 | 0.3218 | 0.1123 | 213/32/45 | 6.66
unkda_nv050 ep8 F finch_p2(k53) | 0.8125 | 0.6759 | 0.0 | 0.7603 | 0.7914 | 0.5497 | 0.2704 | 53/32/11 | 1.66
unkda_nv050 ep8 F louvain_res6 | 0.9375 | 0.7597 | 0.0 | 0.7841 | 0.8293 | 0.6924 | 0.3939 | 47/32/7 | 1.47
unkda_nv050 ep8 F hdbscan_raw(옛다이얼) | 0.5625 | 0.4869 | 23.82 | 0.9848 | 0.7867 | 0.5315 | 0.5815 | 23/32/3 | 0.72
#### unkda_nv050 ep8 projection
unkda_nv050 ep8 Z finch_p1(k271) | 1.0 | 0.6422 | 0.0 | 0.4801 | 0.7486 | 0.1363 | 0.2126 | 271/32/55 | 8.47
unkda_nv050 ep8 Z finch_p2(k59) | 0.9062 | 0.495 | 0.0 | 0.5702 | 0.6334 | 0.3015 | 0.2324 | 59/32/11 | 1.84
unkda_nv050 ep8 Z louvain_res6 | 1.0 | 0.5562 | 0.0 | 0.5758 | 0.6674 | 0.3242 | 0.3044 | 64/32/12 | 2.0
unkda_nv050 ep8 Z hdbscan_raw(옛다이얼) | 0.1875 | 0.1284 | 10.67 | 0.9766 | 0.3675 | 0.1376 | 0.7366 | 6/32/1 | 0.19
#### unkda_nv050 ep10 backbone
unkda_nv050 ep10 F finch_p1(k208) | 1.0 | 0.7956 | 0.0 | 0.5975 | 0.859 | 0.2912 | 0.1132 | 208/32/42 | 6.5
unkda_nv050 ep10 F finch_p2(k51) | 0.9062 | 0.6494 | 0.0 | 0.756 | 0.7824 | 0.5737 | 0.2886 | 51/32/11 | 1.59
unkda_nv050 ep10 F louvain_res6 | 0.9688 | 0.7494 | 0.0 | 0.7845 | 0.8238 | 0.6757 | 0.3832 | 46/32/8 | 1.44
unkda_nv050 ep10 F hdbscan_raw(옛다이얼) | 0.625 | 0.4762 | 31.03 | 0.9788 | 0.8199 | 0.6298 | 0.5937 | 25/32/4 | 0.78
#### unkda_nv050 ep10 projection
unkda_nv050 ep10 Z finch_p1(k298) | 1.0 | 0.6221 | 0.0 | 0.4704 | 0.7389 | 0.1245 | 0.2141 | 298/32/68 | 9.31
unkda_nv050 ep10 Z finch_p2(k71) | 0.9688 | 0.4995 | 0.0 | 0.5499 | 0.6402 | 0.2648 | 0.1678 | 71/32/14 | 2.22
unkda_nv050 ep10 Z louvain_res6 | 0.9688 | 0.5224 | 0.0 | 0.5563 | 0.6509 | 0.2751 | 0.3345 | 65/32/13 | 2.03
unkda_nv050 ep10 Z hdbscan_raw(옛다이얼) | 0.2188 | 0.1597 | 10.89 | 0.9781 | 0.4021 | 0.152 | 0.6452 | 9/32/3 | 0.28

### unkda_nv060 (defect-aware strict novel + NV 0.60)
#### unkda_nv060 ep1 backbone
unkda_nv060 ep1 F finch_p1(k203) | 1.0 | 0.8678 | 0.0 | 0.6317 | 0.9086 | 0.3149 | 0.1333 | 203/32/44 | 6.34
unkda_nv060 ep1 F finch_p2(k53) | 0.9062 | 0.7772 | 0.0 | 0.7942 | 0.8434 | 0.6279 | 0.3156 | 53/32/10 | 1.66
unkda_nv060 ep1 F louvain_res6 | 0.9375 | 0.8272 | 0.0 | 0.8311 | 0.882 | 0.7444 | 0.3857 | 50/32/11 | 1.56
unkda_nv060 ep1 F hdbscan_raw(옛다이얼) | 0.5625 | 0.4541 | 34.3 | 0.9876 | 0.7719 | 0.4109 | 0.5981 | 23/32/5 | 0.72
#### unkda_nv060 ep2 backbone
unkda_nv060 ep2 F finch_p1(k206) | 1.0 | 0.7831 | 0.0 | 0.59 | 0.8504 | 0.2578 | 0.1127 | 206/32/44 | 6.44
unkda_nv060 ep2 F finch_p2(k58) | 0.9688 | 0.705 | 0.0 | 0.7154 | 0.789 | 0.5004 | 0.2957 | 58/32/11 | 1.81
unkda_nv060 ep2 F louvain_res6 | 0.9688 | 0.7131 | 0.0 | 0.7457 | 0.7974 | 0.6008 | 0.3583 | 50/32/11 | 1.56
unkda_nv060 ep2 F hdbscan_raw(옛다이얼) | 0.5938 | 0.3528 | 59.61 | 0.9704 | 0.9225 | 0.9017 | 0.7002 | 23/32/5 | 0.72
#### unkda_nv060 ep3 backbone
unkda_nv060 ep3 F finch_p1(k216) | 1.0 | 0.7894 | 0.0 | 0.5872 | 0.8574 | 0.2481 | 0.1149 | 216/32/48 | 6.75
unkda_nv060 ep3 F finch_p2(k58) | 0.9688 | 0.6937 | 0.0 | 0.7153 | 0.7915 | 0.4995 | 0.2487 | 58/32/8 | 1.81
unkda_nv060 ep3 F louvain_res6 | 0.9688 | 0.7244 | 0.0 | 0.7407 | 0.8052 | 0.5982 | 0.3364 | 54/32/11 | 1.69
unkda_nv060 ep3 F hdbscan_raw(옛다이얼) | 0.5312 | 0.3637 | 58.46 | 0.9738 | 0.9242 | 0.9069 | 0.7232 | 23/32/5 | 0.72
#### unkda_nv060 ep3 projection
unkda_nv060 ep3 Z finch_p1(k263) | 1.0 | 0.6462 | 0.0 | 0.4852 | 0.7454 | 0.1598 | 0.173 | 263/32/56 | 8.22
unkda_nv060 ep3 Z finch_p2(k58) | 0.875 | 0.5478 | 0.0 | 0.5997 | 0.6673 | 0.3687 | 0.2755 | 58/32/11 | 1.81
unkda_nv060 ep3 Z louvain_res6 | 0.9375 | 0.59 | 0.0 | 0.6004 | 0.6839 | 0.4072 | 0.3621 | 60/32/10 | 1.88
unkda_nv060 ep3 Z hdbscan_raw(옛다이얼) | 0.4062 | 0.2709 | 33.82 | 0.9687 | 0.5247 | 0.1568 | 0.5682 | 15/32/3 | 0.47
#### unkda_nv060 ep4 backbone
unkda_nv060 ep4 F finch_p1(k212) | 1.0 | 0.7909 | 0.0 | 0.5898 | 0.8645 | 0.2566 | 0.146 | 212/32/45 | 6.62
unkda_nv060 ep4 F finch_p2(k50) | 1.0 | 0.6972 | 0.0 | 0.7421 | 0.7802 | 0.5164 | 0.259 | 50/32/11 | 1.56
unkda_nv060 ep4 F louvain_res6 | 0.9688 | 0.7616 | 0.0 | 0.7756 | 0.8301 | 0.6544 | 0.361 | 49/32/9 | 1.53
unkda_nv060 ep4 F hdbscan_raw(옛다이얼) | 0.625 | 0.4162 | 45.41 | 0.9731 | 0.8273 | 0.5779 | 0.6602 | 26/32/6 | 0.81
#### unkda_nv060 ep6 backbone
unkda_nv060 ep6 F finch_p1(k220) | 1.0 | 0.7859 | 0.0 | 0.5823 | 0.8573 | 0.2641 | 0.1454 | 220/32/43 | 6.88
unkda_nv060 ep6 F finch_p2(k58) | 0.9062 | 0.6953 | 0.0 | 0.7142 | 0.7908 | 0.5178 | 0.2763 | 58/32/11 | 1.81
unkda_nv060 ep6 F louvain_res6 | 0.9688 | 0.7213 | 0.0 | 0.7462 | 0.812 | 0.6121 | 0.3316 | 53/32/12 | 1.66
unkda_nv060 ep6 F hdbscan_raw(옛다이얼) | 0.2188 | 0.2184 | 0.22 | 1.0 | 0.3245 | 0.0675 | 0.4201 | 9/32/2 | 0.28
#### unkda_nv060 ep6 projection
unkda_nv060 ep6 Z finch_p1(k282) | 1.0 | 0.6297 | 0.0 | 0.4704 | 0.7366 | 0.1273 | 0.1473 | 282/32/54 | 8.81
unkda_nv060 ep6 Z finch_p2(k73) | 0.9062 | 0.4978 | 0.0 | 0.5309 | 0.6241 | 0.2601 | 0.1291 | 73/32/5 | 2.28
unkda_nv060 ep6 Z louvain_res6 | 0.875 | 0.5116 | 0.0 | 0.5533 | 0.6403 | 0.3123 | 0.281 | 62/32/13 | 1.94
unkda_nv060 ep6 Z hdbscan_raw(옛다이얼) | 0.2812 | 0.155 | 26.87 | 0.9504 | 0.3242 | 0.0835 | 0.5005 | 11/32/2 | 0.34
#### unkda_nv060 ep8 backbone
unkda_nv060 ep8 F finch_p1(k228) | 1.0 | 0.8122 | 0.0 | 0.5986 | 0.8861 | 0.2565 | 0.1387 | 228/32/47 | 7.12
unkda_nv060 ep8 F finch_p2(k60) | 0.9688 | 0.7347 | 0.0 | 0.7509 | 0.829 | 0.5658 | 0.2846 | 60/32/13 | 1.88
unkda_nv060 ep8 F louvain_res6 | 0.9688 | 0.7594 | 0.0 | 0.798 | 0.8516 | 0.6536 | 0.4085 | 49/32/10 | 1.53
unkda_nv060 ep8 F hdbscan_raw(옛다이얼) | 0.2812 | 0.2706 | 1.43 | 0.9981 | 0.4466 | 0.1305 | 0.5789 | 11/32/3 | 0.34
#### unkda_nv060 ep8 projection
unkda_nv060 ep8 Z finch_p1(k284) | 1.0 | 0.7016 | 0.0 | 0.5153 | 0.8026 | 0.1692 | 0.2152 | 284/32/66 | 8.88
unkda_nv060 ep8 Z finch_p2(k64) | 0.9062 | 0.6016 | 0.0 | 0.6381 | 0.7247 | 0.4093 | 0.2024 | 64/32/10 | 2.0
unkda_nv060 ep8 Z louvain_res6 | 1.0 | 0.5975 | 0.0 | 0.6246 | 0.7186 | 0.37 | 0.3252 | 61/32/12 | 1.91
unkda_nv060 ep8 Z hdbscan_raw(옛다이얼) | 0.4062 | 0.3053 | 16.51 | 0.985 | 0.5805 | 0.2386 | 0.731 | 13/32/1 | 0.41
#### unkda_nv060 ep10 backbone
unkda_nv060 ep10 F finch_p1(k225) | 1.0 | 0.835 | 0.0 | 0.6069 | 0.8875 | 0.2856 | 0.1861 | 225/32/49 | 7.03
unkda_nv060 ep10 F finch_p2(k60) | 0.9375 | 0.7441 | 0.0 | 0.7492 | 0.8146 | 0.5725 | 0.3452 | 60/32/14 | 1.88
unkda_nv060 ep10 F louvain_res6 | 0.9375 | 0.7659 | 0.0 | 0.7756 | 0.8415 | 0.644 | 0.4385 | 55/32/11 | 1.72
unkda_nv060 ep10 F hdbscan_raw(옛다이얼) | 0.125 | 0.125 | 0.03 | 1.0 | 0.211 | 0.0398 | 0.1695 | 5/32/1 | 0.16
#### unkda_nv060 ep10 projection
unkda_nv060 ep10 Z finch_p1(k277) | 1.0 | 0.7034 | 0.0 | 0.5161 | 0.7938 | 0.1706 | 0.1869 | 277/32/61 | 8.66
unkda_nv060 ep10 Z finch_p2(k60) | 0.9062 | 0.5294 | 0.0 | 0.6123 | 0.6854 | 0.349 | 0.1646 | 60/32/10 | 1.88
unkda_nv060 ep10 Z louvain_res6 | 0.9375 | 0.6209 | 0.0 | 0.6212 | 0.7263 | 0.3914 | 0.3311 | 64/32/9 | 2.0
unkda_nv060 ep10 Z hdbscan_raw(옛다이얼) | 0.2812 | 0.2003 | 12.48 | 0.9913 | 0.3964 | 0.1063 | 0.6943 | 10/32/2 | 0.31

### unkda_nv070 (defect-aware strict novel + NV 0.70)
#### unkda_nv070 ep1 backbone
unkda_nv070 ep1 F finch_p1(k213) | 1.0 | 0.8816 | 0.0 | 0.6494 | 0.9249 | 0.3381 | 0.1331 | 213/32/53 | 6.66
unkda_nv070 ep1 F finch_p2(k54) | 0.9062 | 0.79 | 0.0 | 0.8271 | 0.883 | 0.6504 | 0.335 | 54/32/12 | 1.69
unkda_nv070 ep1 F louvain_res6 | 0.9688 | 0.8516 | 0.0 | 0.8557 | 0.9048 | 0.7593 | 0.4066 | 51/32/12 | 1.59
unkda_nv070 ep1 F hdbscan_raw(옛다이얼) | 0.6875 | 0.4672 | 51.32 | 0.9787 | 0.9859 | 0.9667 | 0.673 | 29/32/6 | 0.91
#### unkda_nv070 ep2 backbone
unkda_nv070 ep2 F finch_p1(k211) | 1.0 | 0.9106 | 0.0 | 0.6544 | 0.9435 | 0.3372 | 0.1601 | 211/32/45 | 6.59
unkda_nv070 ep2 F finch_p2(k61) | 0.9688 | 0.8537 | 0.0 | 0.8271 | 0.9072 | 0.6462 | 0.3308 | 61/32/13 | 1.91
unkda_nv070 ep2 F louvain_res6 | 1.0 | 0.875 | 0.0 | 0.8626 | 0.9237 | 0.7489 | 0.4479 | 53/32/12 | 1.66
unkda_nv070 ep2 F hdbscan_raw(옛다이얼) | 0.75 | 0.4796 | 49.29 | 0.9333 | 0.9734 | 0.893 | 0.6856 | 36/32/7 | 1.12
#### unkda_nv070 ep3 backbone
unkda_nv070 ep3 F finch_p1(k221) | 1.0 | 0.8762 | 0.0 | 0.6428 | 0.9264 | 0.331 | 0.1485 | 221/32/48 | 6.91
unkda_nv070 ep3 F finch_p2(k68) | 1.0 | 0.8188 | 0.0 | 0.7829 | 0.8852 | 0.6077 | 0.2426 | 68/32/14 | 2.12
unkda_nv070 ep3 F louvain_res6 | 0.9688 | 0.8125 | 0.0 | 0.8447 | 0.888 | 0.7041 | 0.347 | 50/32/10 | 1.56
unkda_nv070 ep3 F hdbscan_raw(옛다이얼) | 0.6562 | 0.4205 | 53.57 | 0.9632 | 0.939 | 0.8956 | 0.6788 | 28/32/6 | 0.88
#### unkda_nv070 ep3 projection
unkda_nv070 ep3 Z finch_p1(k267) | 1.0 | 0.7059 | 0.0 | 0.5222 | 0.7949 | 0.1749 | 0.1777 | 267/32/57 | 8.34
unkda_nv070 ep3 Z finch_p2(k64) | 0.9062 | 0.6009 | 0.0 | 0.6364 | 0.7195 | 0.3927 | 0.2803 | 64/32/9 | 2.0
unkda_nv070 ep3 Z louvain_res6 | 0.9062 | 0.6053 | 0.0 | 0.6541 | 0.7308 | 0.461 | 0.3872 | 57/32/9 | 1.78
unkda_nv070 ep3 Z hdbscan_raw(옛다이얼) | 0.5 | 0.3 | 39.89 | 0.973 | 0.5995 | 0.2099 | 0.367 | 19/32/4 | 0.59
#### unkda_nv070 ep4 backbone
unkda_nv070 ep4 F finch_p1(k208) | 1.0 | 0.8097 | 0.0 | 0.6135 | 0.8794 | 0.2942 | 0.1278 | 208/32/51 | 6.5
unkda_nv070 ep4 F finch_p2(k59) | 0.9375 | 0.7447 | 0.0 | 0.7449 | 0.8294 | 0.5473 | 0.2837 | 59/32/10 | 1.84
unkda_nv070 ep4 F louvain_res6 | 1.0 | 0.7675 | 0.0 | 0.7837 | 0.8447 | 0.6468 | 0.3459 | 52/32/12 | 1.62
unkda_nv070 ep4 F hdbscan_raw(옛다이얼) | 0.5938 | 0.4303 | 50.08 | 0.9521 | 0.9164 | 0.8207 | 0.6774 | 27/32/5 | 0.84
#### unkda_nv070 ep6 backbone
unkda_nv070 ep6 F finch_p1(k223) | 1.0 | 0.7903 | 0.0 | 0.5959 | 0.8743 | 0.2577 | 0.1138 | 223/32/43 | 6.97
unkda_nv070 ep6 F finch_p2(k58) | 0.9062 | 0.68 | 0.0 | 0.7354 | 0.8028 | 0.5184 | 0.2661 | 58/32/11 | 1.81
unkda_nv070 ep6 F louvain_res6 | 0.9688 | 0.7341 | 0.0 | 0.7975 | 0.8372 | 0.6339 | 0.3906 | 45/32/7 | 1.41
unkda_nv070 ep6 F hdbscan_raw(옛다이얼) | 0.5938 | 0.4166 | 44.17 | 0.9463 | 0.8392 | 0.6775 | 0.6215 | 26/32/5 | 0.81
#### unkda_nv070 ep6 projection
unkda_nv070 ep6 Z finch_p1(k248) | 1.0 | 0.6931 | 0.0 | 0.5211 | 0.7891 | 0.183 | 0.1605 | 248/32/52 | 7.75
unkda_nv070 ep6 Z finch_p2(k60) | 0.9375 | 0.5922 | 0.0 | 0.6371 | 0.712 | 0.4175 | 0.2719 | 60/32/10 | 1.88
unkda_nv070 ep6 Z louvain_res6 | 0.9062 | 0.6041 | 0.0 | 0.6543 | 0.7218 | 0.4503 | 0.3653 | 53/32/7 | 1.66
unkda_nv070 ep6 Z hdbscan_raw(옛다이얼) | 0.5938 | 0.2966 | 50.33 | 0.9157 | 0.7522 | 0.5624 | 0.7602 | 22/32/4 | 0.69
#### unkda_nv070 ep8 backbone
unkda_nv070 ep8 F finch_p1(k222) | 1.0 | 0.7681 | 0.0 | 0.5755 | 0.8499 | 0.2365 | 0.1455 | 222/32/39 | 6.94
unkda_nv070 ep8 F finch_p2(k68) | 0.9688 | 0.7003 | 0.0 | 0.6855 | 0.7965 | 0.4868 | 0.2481 | 68/32/11 | 2.12
unkda_nv070 ep8 F louvain_res6 | 0.9688 | 0.7188 | 0.0 | 0.7504 | 0.8192 | 0.5768 | 0.3547 | 54/32/9 | 1.69
unkda_nv070 ep8 F hdbscan_raw(옛다이얼) | 0.5938 | 0.3962 | 44.4 | 0.9148 | 0.8276 | 0.6466 | 0.6515 | 31/32/7 | 0.97
#### unkda_nv070 ep8 projection
unkda_nv070 ep8 Z finch_p1(k246) | 1.0 | 0.6775 | 0.0 | 0.5193 | 0.7802 | 0.1877 | 0.1336 | 246/32/50 | 7.69
unkda_nv070 ep8 Z finch_p2(k69) | 0.9688 | 0.5469 | 0.0 | 0.6157 | 0.6966 | 0.3591 | 0.2085 | 69/32/11 | 2.16
unkda_nv070 ep8 Z louvain_res6 | 0.9375 | 0.6106 | 0.0 | 0.6369 | 0.726 | 0.4199 | 0.3403 | 63/32/10 | 1.97
unkda_nv070 ep8 Z hdbscan_raw(옛다이얼) | 0.5625 | 0.2831 | 59.51 | 0.8468 | 0.7993 | 0.5433 | 0.7227 | 32/32/6 | 1.0
#### unkda_nv070 ep10 backbone
unkda_nv070 ep10 F finch_p1(k222) | 1.0 | 0.7406 | 0.0 | 0.5645 | 0.8286 | 0.2409 | 0.1706 | 222/32/48 | 6.94
unkda_nv070 ep10 F finch_p2(k60) | 0.9062 | 0.6159 | 0.0 | 0.681 | 0.7489 | 0.4511 | 0.258 | 60/32/8 | 1.88
unkda_nv070 ep10 F louvain_res6 | 0.9688 | 0.6481 | 0.0 | 0.6655 | 0.7625 | 0.4491 | 0.3597 | 65/32/9 | 2.03
unkda_nv070 ep10 F hdbscan_raw(옛다이얼) | 0.5938 | 0.3837 | 48.94 | 0.8518 | 0.8396 | 0.6277 | 0.7199 | 35/32/7 | 1.09
#### unkda_nv070 ep10 projection
unkda_nv070 ep10 Z finch_p1(k256) | 1.0 | 0.6356 | 0.0 | 0.4952 | 0.7536 | 0.1621 | 0.1539 | 256/32/50 | 8.0
unkda_nv070 ep10 Z finch_p2(k69) | 0.9688 | 0.5384 | 0.0 | 0.5784 | 0.6761 | 0.3282 | 0.2503 | 69/32/11 | 2.16
unkda_nv070 ep10 Z louvain_res6 | 0.9688 | 0.5591 | 0.0 | 0.6 | 0.6921 | 0.3587 | 0.3533 | 65/32/11 | 2.03
unkda_nv070 ep10 Z hdbscan_raw(옛다이얼) | 0.5938 | 0.2978 | 53.89 | 0.8043 | 0.7579 | 0.4864 | 0.6327 | 33/32/4 | 1.03

### unkda_nv075 (defect-aware strict novel + NV 0.75)

### unkda_nv075 (defect-aware strict novel + NV 0.75)
### unkda_nv075 (defect-aware strict novel + NV 0.75)
### unkda_nv075 (defect-aware strict novel + NV 0.75)
