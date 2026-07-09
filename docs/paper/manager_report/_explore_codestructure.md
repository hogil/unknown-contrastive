# Contrastive Learning Pipeline Architecture: Structure & Hook Points

**Report Date:** 2026-05-13  
**Scope:** Medium-depth exploration of training loop, eval pipeline, CLI dispatch, result storage, & SOTA config  
**Purpose:** Identify hook points for new techniques

---

## 1. Training Loop Structure (contrastive_unknown_n50.py / contrastive.py)

### 1.1 Loss Combination Point

**File:** /d/project/unknown-contrastive/contrastive.py, lines 880-927

Loss architecture: hardcoded sequential combination

- Line 893: Global InfoNCE loss
- Line 894-900: Optional Queue-based contrastive
- Line 903-915: Local InfoNCE (multi-positive, patch-level)
- Line 917: loss = loss_g + loss_q + loss_l

**Hook Point:** Inject additional terms before line 917. NeCo loss already installed via monkey-patching info_nce_local_multi wrapper in run_contrastive.py (lines 164-194).

### 1.2 Global InfoNCE Functions

- **info_nce_global(z1, z2, t=0.2)** (line 255): Standard contrastive loss without queue
- **info_nce_with_queue(z1, z2, bank, t=0.2, ignore_sim=0.8)** (line 265): Extends global with historical negatives; implements NEG filter via IGNORE_NEG_SIM (threshold 0.72)

### 1.3 Local (Patch-level) InfoNCE

**File:** contrastive.py, lines 286-457, info_nce_local_multi()

- Anchor sampling: grid36_full (36 patches), grid3x3_full (144), grid16_shift4 (multi-shift)
- Positive selection: window (9x9 neighborhood, r=4) or global
- Multi-positive via pos_topk=12 (top-k by cosine similarity)

**Hook Point:** Wrap info_nce_local_multi for DenseCL-style local consistency (NeCo zone-aware + hierarchical already in run_contrastive.py).

### 1.4 Queue Bank (MoCo-style)

**File:** contrastive.py, lines 461-489, class QueueBank

- Init: QueueBank(dim=128, K=16384) FIFO circular buffer
- Enqueue: L2-normalized batch appended; pointer modulo K
- Get: Returns filled portion [0:n] where n <= K

**Hook Point:** Subclass QueueBank for SwAV-style prototypes or k-means clusters.

### 1.5 Embedding Extraction

**File:** contrastive.py, lines 493-547, extract()

- Single-view dataset (no augmentation); outputs L2-normalized embeddings [N, 128] float32
- Per-batch logging at EMBED_LOG_TICKS=20

**Hook Point:** Insert second forward pass with EMA-updated backbone for momentum encoder.

---

## 2. Eval Pipeline (_eval_contrastive_unknown_n50.py)

### 2.1 Pipeline Flow

1. Load run_info.json -> CFG, backbone weights
2. Build model (CL) + load checkpoints/final_infer.pt
3. Extract embeddings -> contrastive.extract() on UNKNOWN_DIR
4. Run HDBSCAN -> fit_predict() with CFG params
5. Compute Tier 1+2 metrics:
   - Tier 1: class_capture_rate (PRIMARY metric)
   - Tier 2: noise_pct (per-class, aggregate)
   - Supplementary: ARI, NMI, silhouette (cosine), homogeneity/completeness
6. Write outputs: eval/{eval_summary.json, cluster_report.parquet, per_class_report.txt, groups/, embeddings/}
7. Rename run_dir -> append _ari{:.2f}_nmi{:.2f} if metric is best

### 2.2 Metric Calculation Points

**File:** lines 160-202

- HDBSCAN params: overridable via CLI args (--min-cluster-size, --min-samples, --hdbscan-metric, --cluster-selection-method, --cluster-selection-epsilon)
- Class capture rate (lines 272-301): per-defect class coverage analysis
- Per-class noise pct (lines 192-201): (cluster_ids==-1).sum() / class_total * 100

**Hook Point for Epsilon Sweep:** Loop HDBSCAN epsilon values pre-fit; store best per metric.

### 2.3 Output Structure

`
run_dir/eval/
  eval_summary.json          <- Tier 1+2 metrics (JSON)
  cluster_report.parquet     <- per-cluster stats (DB-friendly)
  per_class_report.txt       <- human-readable per-defect breakdown
  file_list.parquet          <- [path, group, wafer_basename, ...]
  class_fragmentation.parquet <- class-to-cluster mapping
  groups/
    group_noise/             <- -1 (HDBSCAN noise)
    group_001, group_002, ... <- cluster assignments (image symlinks/copies)
`

---

## 3. CLI Dispatch (_dispatch_iter.py)

### 3.1 Dispatch Mechanism

**File:** /d/project/unknown-contrastive/_dispatch_iter.py

Windows detached subprocess launch (DETACHED_PROCESS flag) -> run_contrastive.py -> contrastive.py

### 3.2 Key Environment Variables

| Env Var | Type | Default | Role |
|---------|------|---------|------|
| DATA_DIR | path | required | Training/eval data directory |
| EPOCHS | int | 10 | Total training epochs |
| BATCH | int | 8 | Batch size |
| IMAGE_SIZE | int | 384 | Input resize |
| WARMUP_EPOCHS | int | 1 | LR warmup steps |
| IGNORE_NEG_SIM | float | 0.72 | NEG filter threshold |
| NCE_TEMP | float | 0.07 | InfoNCE temperature |
| USE_LOCAL | bool | "true" | Enable patch-level loss |
| USE_QUEUE | bool | "true" | Enable queue bank |
| QUEUE_SIZE | int | 4096 | Memory bank size |
| LR_HEAD | float | 1e-3 | Head learning rate |
| SEED | int | 42 | Random seed |
| LOCAL_WEIGHT | float | 0.5 | Local loss weight |
| LOCAL_POS_TOPK | int | 12 | Multi-positive count per anchor |
| NEG_FILTER | str | "fixed" | "fixed" or "perc_pos" (NV-Retriever) |
| FREEZE_BACKBONE | bool | "true" | Freeze ConvNeXtV2 (head only) vs. joint |
| BACKBONE_UNFREEZE_LAST_N | int | 0 | Partial unfreeze: last N stages |
| NECO_WEIGHT | float | 0.0 | NeCo patch-neighbor consistency |
| NECO_ZONE_VERTICAL | int | 0 | Zone-aware NeCo: N horizontal bands (novelty A) |
| NECO_HIER_POOLS | str | "" | Hierarchical NeCo pool factors (novelty B) |
| MIN_CLUSTER_SIZE | int | 12 | HDBSCAN min cluster size |
| MIN_SAMPLES | int | 4 | HDBSCAN min samples (core density) |
| CLUSTER_SELECTION_METHOD | str | "leaf" | "leaf" or "eom" |
| CLUSTER_SELECTION_EPSILON | float | 0.06 | HDBSCAN smoothing |
| GPU_THROTTLE_MS | float | 0.0 | Sleep after optimizer.step (GPU throttle) |

### 3.3 Config Recording

- Boot log: _dispatch_logs/<tag>_<ts>_boot.log (subprocess stdout/stderr)
- Config JSON: _dispatch_logs/<tag>_<ts>_cfg.json (all dispatch args + env vars)

---

## 4. Result Storage & Checkpoints

### 4.1 Output Directory Structure

`
outputs_contrastive_<YYMMDD_HHMMSS>/
  run.log                      <- training loop log
  run_info.json                <- full CFG dict + metadata (torch, device, etc.)
  checkpoints/
    final_infer.pt             <- state_dict for inference
    last_training.pt           <- full checkpoint (epoch, model, CFG, class_to_idx)
  clusters/
    hdbscan/
      cluster_000_size_42/
        <class>_<filename>.png <- images per cluster
        cluster_000_size_42.txt <- per-cluster meta
      cluster_001_size_63/
  cluster_summary/
    cluster_000__<class>__medoid.png <- centroid medoid per cluster
  clusters_summary.txt          <- run_dir top-level summary
  clusters_global_list.txt      <- [clusterno, class, root, step, wafer, yyyymmdd, hhmmss]
`

### 4.2 Configuration Persistence

In run_info.json:
- EPOCHS: 5, BATCH: 8, LR_HEAD: 0.0005
- TEMP: 0.05, IGNORE_NEG_SIM: 0.65
- LOCAL_WEIGHT: 1.0, LOCAL_POS_TOPK: 12
- USE_QUEUE: true, QUEUE_SIZE: 4096
- MIN_CLUSTER_SIZE: 12, MIN_SAMPLES: 4
- CLUSTER_SELECTION_METHOD: leaf
- class_to_idx mapping (43 defect classes)

---

## 5. Current SOTA Configuration (Iter 12, 260506)

### 5.1 Training Hyperparameters

- EPOCHS: 8
- BATCH: 8
- LR_HEAD: 5e-4 (reduced from 1e-3 for stability)
- WARMUP_EPOCHS: 1
- TEMP: 0.07
- IGNORE_NEG_SIM: 0.72 (NEG filter fixed mode)
- USE_LOCAL: true, LOCAL_WEIGHT: 1.0, LOCAL_POS_TOPK: 12
- USE_QUEUE: true, QUEUE_SIZE: 4096
- FREEZE_BACKBONE: true

### 5.2 HDBSCAN Hyperparameters

- MIN_CLUSTER_SIZE: 12
- MIN_SAMPLES: 4 (reduced from 1 for tighter core detection)
- METRIC: euclidean
- CLUSTER_SELECTION_METHOD: leaf (fine-grained) vs. eom (coarse)
- CLUSTER_SELECTION_EPSILON: 0.06

### 5.3 Eval Metrics (Target Without-Normal)

- class_capture_rate: 1.0 (41/42 single-cluster, 1 split-2)
- noise_pct: 0.087% (target: <2%)
- ARI (without-Normal): 0.017 (weak defect discrimination)
- silhouette (cosine): 0.347 (moderate separation)

---

## 6. Hook Points for New Techniques

### 6.1 Iterative Cluster Harvesting

**Hook:** _eval_contrastive_unknown_n50.py, lines 160-176
- Run HDBSCAN at eps_min, eps_max, step
- Score by class_capture_rate + noise_pct
- Store multi-eps results; pick best

### 6.2 RankMe Metric

**Hook:** _eval_contrastive_unknown_n50.py, after line 300
- Per-class rank correlation (within vs. between-cluster distances)
- Add as supplementary metric in eval_summary.json

### 6.3 Soft HDBSCAN Tau-Reassignment

**Hook:** Extend run_hdbscan() in _eval_contrastive_unknown_n50.py
- Soft membership via KNN + temperature scaling

### 6.4 Momentum Encoder

**Hook:** Modify contrastive.py CL class (lines 217-251)
- self.backbone_ema = copy.deepcopy(self.backbone)
- In training loop (889-890), compute z_online and z_momentum
- Average or weight in loss computation

### 6.5 SwAV-style Queue with Prototypes

**Hook:** Extend QueueBank class (lines 461-489)
- Store + update k-means centroids alongside queue
- Prototype clustering instead of raw embeddings

### 6.6 HDBSCAN Epsilon Sweep

**Hook:** Batch eval wrapper around _eval_contrastive_unknown_n50.py
- Loop over eps in [0.03, 0.06, 0.09, 0.12]
- Store metric vs. eps curve; report best

### 6.7 Zone-Aware / Hierarchical NeCo

**Hook:** Already monkey-patched in run_contrastive.py
- Novelty A (zone-aware): lines 197-267
- Novelty B (hierarchical): lines 269-342
- Activate: NECO_WEIGHT=0.2 NECO_ZONE_VERTICAL=3

---

## Summary: Hook Points by Component

| Component | Location | Key Functions | Hook Strategy |
|-----------|----------|----------------|----------------|
| Training Loop | contrastive.py:880-927 | main() | Insert loss term before loss = loss_g + loss_q + loss_l |
| Global Loss | contrastive.py:255-282 | info_nce_global(), info_nce_with_queue() | Wrap/subclass QueueBank; extend loss signatures |
| Local Loss | contrastive.py:286-457 | info_nce_local_multi() | Monkey-patch wrapper (NeCo in run_contrastive.py) |
| Queue | contrastive.py:461-489 | QueueBank | Subclass; register new enqueue(), get() logic |
| Embedding | contrastive.py:493-547 | extract() | Insert second forward pass (momentum) |
| Eval | _eval_contrastive_unknown_n50.py:160-300 | run_hdbscan(), metrics | Wrap HDBSCAN; add post-eval metric computation |
| Dispatch | _dispatch_iter.py:27-175 | main() | Add new CLI args + env var bindings |
| Config | run_contrastive.py:390-523 | CFG.update() | Read new env vars; patch CFG dict |

[OUT] /d/project/unknown-contrastive/docs/paper/manager_report/_explore_codestructure.md
