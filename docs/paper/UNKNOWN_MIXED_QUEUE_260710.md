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
