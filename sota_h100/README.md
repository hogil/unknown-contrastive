# sota_h100 — single-SOTA seed sweep (H100)

Fully **self-contained** reproduction of the chip multi-label single-SOTA
(iter116J) for H100. Generates its own data (independent synthesis — imports
nothing from `dist_apply` / `chip_multilabel` / `mega_matrix` for image gen),
runs a **seed sweep** of the iter116J recipe, and writes a markdown report.

The SOTA **model engine** is the project's validated trainer
(`chip_multilabel._train_chip_variant`) + evaluator (`chip_multilabel.run_stage1`)
— that recipe *is* the "single SOTA model" under study, so it is reused as-is.
Only data generation + orchestration + reporting are new.

## Files

| file | role |
|---|---|
| `synth.py` | self-contained 200x200 palette-PNG chip synthesis. Palette, grade tiers, Lorentzian/Gaussian alpha profiles, 4 single defects (bank_boundary/fork/scratch/scratch_rot), 6 two-combos (min-blend), 4 OOD (CenterDonut/CrossScratch/DiagonalSmear/Starburst), Normal, Invalid. |
| `gen_data.py` | `--mode train` -> `classification_chips/` (4 single, N/class); `--mode eval` -> `eval_set/` (16 classes, N/class) + manifest.csv + `_preview/`. |
| `run_seed_sweep_1gpu.sh` | single-GPU sequential seed sweep: gen data -> per-seed train+eval -> RESULTS.md. |
| `run_seed_sweep_ddp.sh` | true-DDP (torchrun) seed sweep: each seed trains one model across G GPUs. |
| `make_report.py` | aggregate all seeds -> `RESULTS.md` (per-seed bit_F1 / NI-FAR / OOD-FAR / Total-FAR for I10+I13 + mean±std + best). |

## Metric (CLAUDE.md 260512 rule)

- **bit_F1** = macro-F1 over positive bits (4 single + 6 two-combo).
- **FAR** = false-accept rate over negatives, split **NI** (Normal+Invalid) / **OOD** / **Total**.
- Past frozen best: bit_F1 ≈ 0.9914 (I10) at Total FAR 0%.

## Quickstart (H100)

Put the backbone weights at `models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth`
(ImageNet FCMAE ConvNeXtV2-base). Then:

```bash
# single GPU, 8 seeds (data auto-generated on first run)
CUDA_VISIBLE_DEVICES=0 bash sota_h100/run_seed_sweep_1gpu.sh

# true DDP across 4 GPUs per seed — nproc & per-rank batch derived from the
# device list (4 GPUs -> per-rank batch 4 -> effective 16, matches recipe)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sota_h100/run_seed_sweep_ddp.sh

# 8 GPUs likewise (per-rank batch auto = 16/8 = 2)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash sota_h100/run_seed_sweep_ddp.sh

# override seeds / epochs / data root
SEEDS="1 2 3 4" EPOCHS=24 IMAGES_ROOT=/data/images \
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash sota_h100/run_seed_sweep_ddp.sh
```

`CUDA_VISIBLE_DEVICES` is the GPU selector for both scripts (standard). The 1gpu
script uses the first listed device; the ddp script uses all listed devices and
auto-sets `--nproc_per_node` + per-rank batch.

Result: `outputs/sota_h100_seedsweep_*/RESULTS.md`.

## Recipe (iter116J, fixed)

T7 BCE+LS 0.30, FCM-PM (complement CutMix p=0.25, masked pair, corner fill,
grid 8, n-groups 3, complete-label-scale 0.5), AdamW lr 1e-4, val-criterion
`margin_max`, train on 4 single defects only (`--no-normal`), eval I10 + I13.

## Env knobs

`SEEDS EPOCHS BATCH ACCUM GPU(1gpu) GPUS(ddp) IMAGES_ROOT TRAIN_ROOT EVAL_SET
TRAIN_PER_CLASS EVAL_PER_CLASS EVAL_N_USE WEIGHTS BACKBONE IMG_SIZE SWEEP_ROOT`.

DDP effective batch = `BATCH * GPUS * ACCUM`; keep it 16 to match the SOTA recipe
(the runner warns if it differs).
