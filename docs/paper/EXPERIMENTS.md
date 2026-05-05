# Experimental setup

## Hardware

- GPU: NVIDIA (16 GB VRAM)
- RAM: 64 GB
- OS: Windows 11
- CUDA: 12.x

## Software

- Python 3.13
- PyTorch 2.x
- timm (ConvNeXtV2 backbone)
- scikit-learn (clustering metrics)
- hdbscan (clustering)
- numpy, pandas, PIL

## Hyperparameter table (per run)

| Run | tag | EPOCHS | BATCH | IMAGE | TEMP | LR_HEAD | QUEUE_SIZE | USE_LOCAL | n_train | sampling |
|---|---|---|---|---|---|---|---|---|---|---|
| Iter 0 | normal1000_n50_b16_global_e10_resize_reuse | 10 | 16 | 384 | 0.07 | 1e-3 | 4096 | False | 8,357 | uniform 200/class + Normal 1000 |

Per ITERATION 의 detail change history → `ITERATIONS.md`.

## Resource policy

- **GPU**: BATCH=16, IMAGE_SIZE=384 — 사용자 명시 "GPU 작게 써라" (`docs/contrastive-eval/DECISIONS.md` D-9).
- **monitoring (별도)**: alignment + uniformity (label 무관) + (옵션) k-NN top-1.
- **rogue process 보호**: canvas gen / obj_id_maps 자동 spawn 감지 + kill watchdog.

## 코드 위치 (root scripts)

| 파일 | 역할 |
|---|---|
| `contrastive.py` | InfoNCE 학습 engine (수정 금지, wrapper CFG override) |
| `run_contrastive.py` | env wrapper, sister repo backbone state_dict 추출 |
| `_contrastive_n50.py` | small-budget 학습 wrapper (subset hardlink builder) |
| `_eval_contrastive_unknown_n50.py` | 학습 후 평가 — Tier 1+2 + class_fragmentation_summary |
| `eval_align_uniform.py` | post-hoc alignment + uniformity helper |
| `compose_clusters.py` | per-cluster K=20 medoid composite (binary + grademean) |
| `predict_contrastive_daily.py` | production daily inference |

## Output 구조

```
outputs/logs_contrastive/<tag>_<TS>/
├── _init_backbone.pth        # backbone state_dict 사본
├── _wrapper_manifest.json    # wrapper 환경
├── run.log                   # 학습 log (epoch loss / metric)
├── run_info.json             # CFG dump
├── checkpoints/
│   ├── final_infer.pt        # final encoder + head
│   └── last_training.pt      # last training state
├── eval/
│   ├── eval_summary.json     # Tier 1+2 + class_fragmentation_summary
│   ├── align_uniform.json    # Wang & Isola metric (post-hoc)
│   ├── embeddings/           # embedding.npy + files.txt + classes.txt
│   ├── cluster_report.parquet
│   ├── class_fragmentation.parquet
│   ├── retrieval_report.parquet
│   └── plots/                # heatmap / histogram / silhouette
├── clusters/hdbscan/cluster_XXX_size_YYY/  # cluster member 이미지
└── cluster_summary/                          # medoid + composite
```

## Reproducibility

- random seed: 42 (CFG default).
- Augmentation: deterministic with seed.
- HDBSCAN: deterministic (no randomness).

## 검증 protocol

각 run 평가 시 자동:
1. `_eval_contrastive_unknown_n50.py` 실행 — Tier 1+2 + class_fragmentation_summary
2. `eval_align_uniform.py --run <run_dir>` — alignment + uniformity
3. (옵션) `compose_clusters.py --run <run_dir>` — composite map
4. 콘솔 1-2 줄 보고 자동 출력
