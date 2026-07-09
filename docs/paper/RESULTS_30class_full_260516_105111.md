# 30-class 6-cell ablation — n500 full results (polling)

**Generated**: 260516_105111 (polling mode, 10-min cycle)
**Watch target**: `D:/project/unknown-contrastive/outputs_contrastive_*/tier1_*_n500.json`
**Status**: WAITING — no `tier1_*_n500.json` files detected yet.

## Cell config

| cell | USE_LOCAL | LW  | queue | neg_sim | neco |
|---|:-:|:-:|:-:|:-:|:-:|
| B0  | F | 0   | F | 1.00 | 0   |
| B1  | T | 0.5 | F | 1.00 | 0   |
| B3  | T | 0.5 | T | 1.00 | 0   |
| B4  | T | 0.5 | T | 0.72 | 0   |
| B5  | T | 0.5 | T | 0.72 | 0.2 |
| NEW | F | 0   | T | 0.72 | 0.2 |

## Results table (Tier 1 bold, Tier 2 auxiliary)

| cell | config | **Completeness** | AMI | **noise_pct** | **class_capture_rate** | **Homogeneity** | Silhouette | ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0_n500  | local=F LW=0   queue=F neg=1.00 neco=0    | — | — | — | — | — | — | — |
| B1_n500  | local=T LW=0.5 queue=F neg=1.00 neco=0    | — | — | — | — | — | — | — |
| B3_n500  | local=T LW=0.5 queue=T neg=1.00 neco=0    | — | — | — | — | — | — | — |
| B4_n500  | local=T LW=0.5 queue=T neg=0.72 neco=0    | — | — | — | — | — | — | — |
| B5_n500  | local=T LW=0.5 queue=T neg=0.72 neco=0.2  | — | — | — | — | — | — | — |
| NEW_n500 | local=F LW=0   queue=T neg=0.72 neco=0.2  | — | — | — | — | — | — | — |

## Sources

(none yet)

## Polling log

- 260516_105111 — initial scan, 0/6 cells found
- 260516_215300 — paper-recorder invocation start, 0/6 cells found, dispatch chain `_loop_n500_full_260516_124545` already finished at 13:43 with all 6 cells `status: MISSING` (watchdog GPU>40% killed every cell before tier1 eval).
- 260516_220239 — new dispatch chain spawned, B0_n500 entered with watchdog kill cascade BATCH 4 -> 1.
- 260516_225100 — 60-min poll cap reached. Still 0/6. B0_n500 child hung at BATCH=1 GPU=0% for last 25 min, no tier1 JSON produced.
- 260517_083004 — paper-recorder iter 88 resume. 60-min poll budget x 6 cycles started.
  - Polls 1-6 (08:30 / 08:40 / 08:50 / 09:00 / 09:10 / 09:20 KST) all returned `count=0`.
  - Most recent n500 child dispatch boot log unchanged since 260517 01:40:44 (B5).
  - Sample boot log inspection: CUDA `RuntimeError: unknown error` at `CL().to(device)` — memory allocation failed before training began.
  - Dispatch watchdog chain `_loop_n500_full` appears stopped (no fresh attempts in 7.5h).
  - 60-min cap reached at 09:20:14. Final outcome: **0/6 cells**, table rows still TBD.
- 260517_122700 — paper-recorder iter 89 resume (60-min polling, chain BATCH++ patch applied).
  - Watchdog (1 min cadence) ran 12:33:50 → 13:33:18, 60 polls.
  - All 60 polls returned `count=0`.
  - Dispatch chain alive — full cycle B0→B1→B3→B4→B5→NEW observed (NEW first spawn at 13:20:55).
  - Failure mode: every cell crashes with **`MemoryError`** in `PIL.Image.copy()` / `PIL.Image.crop()` inside `ContrastiveDataset.__getitem__` (`contrastive.py:192`, `x1 = self.t(im.copy()); x2 = self.t(im.copy())`). Host RAM exhaustion when materializing 384x384 dual-aug copies on 21,144-item train set with `ratio=0.25 → 5,286 items` × `batch=4` DataLoader.
  - 49 of today's 71 n500 boot logs (260517 00:00→13:33) crashed with `MemoryError`. No tier1 eval reached on any cell.
  - 60-min cap reached at 13:34:18. Final outcome: **0/6 cells**, table rows still TBD.

## Notes

- Tier 1 (bold): Completeness (P3), AMI (aux), noise_pct (P2), class_capture_rate (P1), Homogeneity (P4)
- Tier 2 (auxiliary): Silhouette (cosine), ARI
- No custom metrics. sklearn + HDBSCAN noise_pct only (DECISIONS.md D-1).
- Polling cycle: 10 min. New `tier1_*_n500.json` detection auto-updates table row.
- No training dispatch from this agent.
