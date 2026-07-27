# Repository cleanup manifest — 2026-07-26

## Status

- Repository: `D:\project\unknown-contrastive`
- Initial inventory was recorded before cleanup mutation; this file now records the completed final state.
- Current phase: **completed and verified**
- Final data rule: image data is read directly from `E:\data\images`; repository junctions, symbolic links, and D-side physical image copies are not used.
- Git branch: `main`
- Git state at the first safety snapshot: about 300 status entries, including pre-existing tracked modifications/deletions and many untracked files.
- Rule: never revert, overwrite, move, or delete a tracked/user-modified path as part of this cleanup.

At the latest user authorization, all blocking experiments were declared stopped and failed/temporary runs were explicitly included in cleanup. A live process snapshot still showed some orphaned training/watcher processes, so paths actually referenced by a live PID remain protected. Inactive failed runs and duplicate diagnostics may be deleted permanently after exact path, Git, reparse-point, and process-reference gates.

## Purpose and interpretation

The cleanup had three bounded goals:

1. Reduce the very large number of legacy output folders while preserving referenced, metric-bearing, reproducible, recent, and active results.
2. Consolidate physical images under `E:\data\images`, with path/size/hash checks and no D-side links.
3. Keep one physical **unknown-training master** at `E:\data\images\unknown` and select derived pools from manifests instead of duplicated dataset trees.

“One dataset” means one physical **image dataset tree** for unknown discovery: `E:\data\images\unknown`. `D:\project\unknown-contrastive\data\images` is an intentionally empty compatibility-free directory. Derived train/eval/holdout/anchor pools are exact JSON file lists rooted at the E master.

## Safety snapshot (historical pre-cleanup state)

Everything in this section describes the initial audit, not the final filesystem state. Final state is recorded under “Final canonical data layout.”

### Running processes and active exclusions

The following paths are excluded from cleanup because a running process reads or writes them, or because they are inputs/outputs of an active guard/confirmation chain:

- `D:\project\unknown-contrastive\runs\anchor_xds`
- `D:\project\unknown-contrastive\runs\anchor_xds\run`
- `D:\project\unknown-contrastive\runs\anchor_xds\abl_rc260726_B4_260726_083547`
- `D:\project\unknown-contrastive\runs\anchor_xds\anchor_raw_feats_fcmae.npy`
- `D:\project\unknown-contrastive\runs\anchor_xds\anchor_raw_feats_fcmae_manifest.json`
- `D:\project\unknown-contrastive\runs\clean546\verify_manifest_check_dirmode_260726.json`
- `D:\project\unknown-contrastive\result_grouping\verify_manifest_check_jsonmode_260726`
- `D:\project\unknown-contrastive\data\images\anchor_avg30_repro`
- `D:\project\unknown-contrastive\data\images\mwm38_clean546`
- `D:\project\unknown-contrastive\data\images\wm811k_train_pool_v1_defectonly`
- `D:\project\unknown-contrastive\data\images\unknown_train_all`
- `D:\project\unknown-contrastive\data\images\unknown_train_normal`
- `D:\project\unknown-contrastive\data\pools\unknown_train_normal.json`
- `D:\project\unknown-contrastive\data\pools`
- `D:\project\unknown-contrastive\weights\convnextv2_base.fcmae_ft_in22k_in1k_384.pth`
- `D:\project\unknown-contrastive\scripts\run_fcmae_adapter_temperature_screen_guard.ps1`
- `D:\project\unknown-contrastive\scripts\run_fcmae_adapter_temperature_screen.py`
- `D:\project\unknown-contrastive\runs\fcmae_adapter_temperature_screen_260725`
- `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725.json`
- `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725.csv`
- `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725.md`
- `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725_protocol.json`
- `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725_eval_manifest.json`
- `D:\project\unknown-contrastive\scripts\run_fcmae_adapter_confirmation_chain.ps1`
- `D:\project\unknown-contrastive\runs\fcmae_adapter_ep4_three_seed_260721`
- `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_HOLDOUT_260725`

External input observed but outside this repository and therefore outside cleanup scope:

- `E:\data\images\unknown_pos150_384`

### Concurrent code changes

At 08:41–08:43 KST, another active Claude/user session modified or created the canonical pool-selection implementation:

- `D:\project\unknown-contrastive\scripts\_common.py`
- `D:\project\unknown-contrastive\scripts\make_pool_manifest.py`
- `D:\project\unknown-contrastive\_score_umapfree.py`
- `D:\project\unknown-contrastive\_grouping_eval.py`
- `D:\project\unknown-contrastive\_grouping_deliverable.py`
- `D:\project\unknown-contrastive\_grouping_pipeline.py`
- `D:\project\unknown-contrastive\data\pools`

These paths are treated as pre-existing concurrent user work. This cleanup may inspect and test them read-only, but must not modify, format, stage, revert, or overwrite them. Any failure is recorded as a concurrent-work blocker instead of being patched here.

### Reparse point observed at the initial snapshot

The only reparse point found in the initial repository snapshot was:

- `D:\project\unknown-contrastive\runs\fcmae_adapter_residual_scale_screen_260725`
  - Type: junction
  - Target: `E:\unknown-contrastive-runs\archives\fcmae_adapter_residual_scale_screen_260725`

It was later removed under the final no-links rule. Four temporary/compatibility data junctions were also removed, for five removed junction paths in total:

- `D:\project\unknown-contrastive\data\images\unknown_train_all`
- `D:\project\unknown-contrastive\data\images\hf_flowers102`
- `D:\project\unknown-contrastive\data\images\hf_dtd`
- `D:\project\unknown-contrastive\data\images\hf_resisc45`
- `D:\project\unknown-contrastive\runs\fcmae_adapter_residual_scale_screen_260725`

Final repository-wide reparse-point count is 0; consumers reference E storage directly.

### Strong references

Protected existing references:

- `D:\project\unknown-contrastive\result_grouping\posthoc_cluster_merge_260611.csv`
- `D:\project\unknown-contrastive\runs\fcmae_adapter_temperature_screen_260725`
- `D:\project\unknown-contrastive\runs\fcmae_adapter_residual_scale_screen_260725`
- `D:\project\unknown-contrastive\runs\fcmae_adapter_ep4_three_seed_260721`
- all source, configuration, README, paper, final summary, and evaluation-metric files

The following skill-listed strong references were already absent at the pre-cleanup snapshot. This cleanup must not be reported as deleting them:

- `D:\project\unknown-contrastive\runs\260611_145131_simclr_cross_disjoint_cnninit_laststage_ep2_q2048_lrb1e6_local003_temp0045\contrastive\best_model.pt`
- `D:\project\unknown-contrastive\result_grouping\260611_151739_cross_disjoint_q2048_local003_temp0045_best_ep1_mcs9_ms1_reassign_all_no_tsne`
- `D:\project\unknown-contrastive\result_grouping\260611_182934_cross_disjoint_q2048_adapter128_frozen_local003_temp0045_best_ep1_mcs9_ms1_reassign_all_no_tsne`

## Pre-execution inventory

### Legacy top-level outputs

- Match rule: direct children of `D:\project\unknown-contrastive` named `outputs_contrastive_*`
- Roots: 200
- Files: 511
- Bytes: 19,475,623 (18.57 MiB)
- Model checkpoints: 0
- High-confidence deletion subset under review: 190 roots containing only one legacy `run.log` each, about 0.83 MiB total
- Metric-bearing subset under review: 10 roots, 321 files, about 17.75 MiB

No `outputs_contrastive_*` file is tracked by Git.

Exact deletion list:

- `D:\project\unknown-contrastive\docs\CLEANUP_DELETE_ROOTS_260726.txt`
- 190 absolute roots
- 190 files
- 862,784 bytes
- SHA-256 over the UTF-8 sorted path list with one LF after every path: `7a2fac3fe7655632f277b25d906be7d35468c277de4b69c7640abe8da51a66bb`

Exact preserved metric-bearing roots:

- `D:\project\unknown-contrastive\outputs_contrastive_260511_181441`
- `D:\project\unknown-contrastive\outputs_contrastive_260519_114912`
- `D:\project\unknown-contrastive\outputs_contrastive_260520_204348`
- `D:\project\unknown-contrastive\outputs_contrastive_260521_010837`
- `D:\project\unknown-contrastive\outputs_contrastive_260521_102609`
- `D:\project\unknown-contrastive\outputs_contrastive_260521_141246`
- `D:\project\unknown-contrastive\outputs_contrastive_260521_174701`
- `D:\project\unknown-contrastive\outputs_contrastive_260521_214532`

The two weak structured outputs originally included in the conservative preserve set were later reclassified as non-evidence legacy runs under the user's instruction to include failed/weak results:

- `D:\project\unknown-contrastive\outputs_contrastive_260526_165038`
- `D:\project\unknown-contrastive\outputs_contrastive_260527_064042`

### Data

- `D:\project\unknown-contrastive\data` snapshot: about 245,296 files and 491.47 GiB
- Direct branches: `D:\project\unknown-contrastive\data\_preview`, `data\images`, `data\pools`, `data\positions`, `data\raw`, and `data\wm-811k`
- Canonical image root: `D:\project\unknown-contrastive\data\images`
- Canonical unknown-training master candidate: `D:\project\unknown-contrastive\data\images\unknown_train_all`
  - 21,143 PNG files
  - 264,417,178,347 bytes (246.257 GiB)
- Candidate image moves into the canonical image root:
  - `D:\project\unknown-contrastive\data\_preview`: 129 images, 376,397,603 bytes
  - `D:\project\unknown-contrastive\data\wm-811k`: 11 images, 41,399 bytes
  - Total: 140 images, 376,439,002 bytes (359.00 MiB)

Compact source archives present at the initial snapshot:

- `D:\project\unknown-contrastive\data\raw\wm811k\LSWMD.pkl`
- `D:\project\unknown-contrastive\data\raw\mixedwm38\Wafer_Map_Datasets.npz`

These are historical inventory entries only. `D:\project\unknown-contrastive\data\raw` and `D:\project\unknown-contrastive\data\positions` are absent in the final layout; position metadata is external under `E:\data\positions`.

## Pre-execution gates

No move or deletion may begin until all items below pass:

- [x] Exact legacy-output delete/preserve lists recorded.
- [x] Exact file counts and byte totals recomputed immediately before deletion.
- [x] Every delete target resolves beneath `D:\project\unknown-contrastive` and is not a reparse point.
- [x] Every delete target contains zero Git-tracked files and is outside the active exclusion set.
- [x] Whole-directory data moves used absent destinations, so no same-name collision or overwrite occurred.
- [x] Canonical unknown master versus derived copies was established by relative paths, sizes, generated pool manifests, and targeted content validation.
- [x] Relevant loader/CLI paths were identified before derived dataset trees were removed.

## Planned verification

- Recount files and bytes before and after.
- Verify all moved image source hashes at their destination.
- Confirm no unresolved name collisions.
- Confirm protected strong references and active paths still exist or, if originally absent, remain explicitly marked as pre-existing absence.
- Run loader/path-selection dry runs and focused tests.
- Re-run `git status --short` and distinguish this cleanup’s changes from the pre-existing dirty worktree.

## Execution log

- Root `*.log`, `*.out`, and `*.err`: 290 files / 11,707,694 bytes moved to Recycle Bin by the root coordinator.
- Legacy `outputs_contrastive_*`: 190 roots / 190 files / 862,784 bytes moved to Recycle Bin from the exact list `D:\project\unknown-contrastive\docs\CLEANUP_DELETE_ROOTS_260726.txt`.
- Direct data moves completed by the root coordinator:
  - `D:\project\unknown-contrastive\data\_preview` → `D:\project\unknown-contrastive\data\images\_preview` (129 files / 376,397,603 bytes)
  - `D:\project\unknown-contrastive\data\wm-811k` → `D:\project\unknown-contrastive\data\images\wm-811k` (11 files / 41,399 bytes)
  - `D:\project\unknown-contrastive\data\images_384` → `D:\project\unknown-contrastive\data\images\images_384` (1,632 files / 184,293,599 bytes)
  - These were transitional moves; the non-canonical D image directories were later deleted, leaving `D:\project\unknown-contrastive\data\images` empty.
- Large permanent-delete manifests frozen before execution:
  - `D:\project\unknown-contrastive\docs\CLEANUP_RUN_DELETE_ROOTS_260726.txt`
  - `D:\project\unknown-contrastive\docs\CLEANUP_LARGE_IMAGE_TARGETS_260726.txt`
- Top-level recoverable launcher cleanup manifest frozen before execution:
  - `D:\project\unknown-contrastive\docs\CLEANUP_ROOT_LAUNCHER_DELETE_FILES_260726.txt`

### Final canonical data layout

Physical unknown-image master:

- `E:\data\images\unknown`
- image files: 21,143
- bytes: 264,417,178,347
- zero-byte files: 0

Migration proof collected before deleting the D duplicate:

- every one of the 21,143 D relative paths existed at E with the same size
- total D and E matched at 264,417,178,347 bytes
- an evenly distributed 256-file sample totaling 3,204,945,233 bytes had 0 SHA-256 mismatches
- the former D physical duplicate, `D:\project\unknown-contrastive\data\images\unknown_train_all`, was then permanently deleted: 21,143 files / 264,417,178,347 bytes

One E-only corrupt placeholder was not part of any pool. It was moved, not discarded:

- source: `E:\data\images\unknown\Full_invalid_main\CCH016_00C_08_20260501_010000_67.0_0_PT_ENGINEER.png`
- quarantine: `E:\unknown-contrastive-archive\data_corrupt_260726\CCH016_00C_08_20260501_010000_67.0_0_PT_ENGINEER.png.zero-byte`
- size: 0 bytes

Repository-side final state:

- `D:\project\unknown-contrastive\data\images`: direct items 0; files 0; reparse points 0
- repository-wide reparse points: 0
- repository-wide hard-link paths: 0
- links/reparse points across `E:\data\images\unknown`, `E:\data\images\hf_flowers102`, `E:\data\images\hf_dtd`, and `E:\data\images\hf_resisc45`: 0
- five repository junction paths were removed under the final no-links rule
- no compatibility junction or symbolic link remains
- `D:\project\unknown-contrastive\data` has only `data\images` and `data\pools`
- `D:\project\unknown-contrastive\data\raw` and `D:\project\unknown-contrastive\data\positions` no longer exist
- position metadata is external under `E:\data\positions`; it is not an image pool
- `D:\project\unknown-contrastive\data\pools`: 7 JSON files / 2,835,605 bytes
- nine stale synth split manifests were archived at `D:\project\unknown-contrastive\docs\archive\data_manifests_260726`

Permanent derived-data deletion:

- five canonical-subset/derived roots:
  - `unknown_train_defectaware_260710`: 6,594 image files / 86,386,763,485 logical bytes
  - `unknown_eval100`: 4,149 files / 52,228,037,789 logical bytes
  - `unknown_holdout_100_260713`: 4,100 image files / 51,553,478,353 logical bytes
  - `unknown_train_normal`: 2,998 files / 39,824,791,610 logical bytes
  - `anchor_avg30_repro`: 2,260 files / 29,103,519,213 logical bytes
- remaining 31 non-canonical image directories: 205,664 files / 1,710,398,513 bytes
- combined pre-master data deletion: 225,765 files / 260,806,988,963 logical bytes
- the later 264,417,178,347-byte D physical-master deletion is recorded separately because the earlier logical total includes hard-linked subsets

#### Restored external HF datasets

All restored datasets are physical E directories and passed complete PIL verify-plus-load decoding:

| Physical path | Classes | Images/files | Bytes | Per class | Zero-byte | Non-image | Decode failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E:\data\images\hf_flowers102` | 102 | 1,020 | 81,224,043 | 10 | 0 | 0 | 0 |
| `E:\data\images\hf_dtd` | 47 | 1,880 | 212,327,967 | 40 | 0 | 0 | 0 |
| `E:\data\images\hf_resisc45` | 45 | 1,800 | 45,755,510 | 40 | 0 | 0 | 0 |

Combined: 194 class directories / 4,700 images / 339,307,520 bytes. No D-side links exist for these datasets.

#### Active pool manifests

All five active unknown-master manifests are exact file lists rooted at `E:\data\images\unknown` and have zero missing paths:

| Active manifest | Entries | Missing | Outside E master |
|---|---:|---:|---:|
| `D:\project\unknown-contrastive\data\pools\anchor_avg30_repro.json` | 2,260 | 0 | 0 |
| `D:\project\unknown-contrastive\data\pools\unknown_eval100.json` | 4,149 | 0 | 0 |
| `D:\project\unknown-contrastive\data\pools\unknown_holdout_100_260713.json` | 4,100 | 0 | 0 |
| `D:\project\unknown-contrastive\data\pools\unknown_train_normal.json` | 2,998 | 0 | 0 |
| `D:\project\unknown-contrastive\data\pools\unknown_train_defectaware_260710.json` | 6,594 | 0 | 0 |

The five unknown-master manifests contain 20,101 entries in total. Two additional active manifests also point directly to physical E datasets and resolve with zero missing paths:

| Additional active manifest | E root | Entries | Missing |
|---|---|---:|---:|
| `D:\project\unknown-contrastive\data\pools\mwm38_clean546.json` | `E:\data\images\mixedwm38\rendered\all` | 546 | 0 |
| `D:\project\unknown-contrastive\data\pools\severstal_pilot260726.json` | `E:\data\images\severstal` | 995 | 0 |

That makes seven active loader manifests / 21,642 entries / zero missing paths. The two `*_source_manifest.json` files are historical provenance, not active loader inputs.

Defect-aware exact-set audit:

- the original 2026-07-10 construction copied/hard-linked every immediate file from 11 specified classes
- it used no first-N cap, random selection, seed, or recursive selection
- active full manifest: entries 6,594; unique 6,594; corresponding E class set 6,594; missing 0; extra 0
- historical source/provenance manifest: `D:\project\unknown-contrastive\data\pools\unknown_train_defectaware_260710_source_manifest.json`
  - SHA-256 `7BD2AA017015F811AD5253B8D90A66BFCBE8EAAEF7DE429C01B0213A837D0DBC`
- active full exact-file manifest: `D:\project\unknown-contrastive\data\pools\unknown_train_defectaware_260710.json`
  - SHA-256 `5A8AAF0C64B84BCFB93303F6FE7DAF187167358975E98FC62326FBBFD3FEED3B`
- retain the source hash for historical provenance; use the full manifest for current input resolution

### Output and run cleanup

Recoverable cleanup:

- 190 legacy `outputs_contrastive_*` roots / 190 files / 862,784 bytes moved to Recycle Bin
- two additional weak legacy output roots and one `__pycache__` snapshot moved to Recycle Bin
- 290 root log/out/err files / 11,707,694 bytes moved to Recycle Bin
- 86 untracked launcher/state files / 126,539 bytes moved to Recycle Bin
- 9 temporary/cache directories / 696 files / 25,210,810 bytes moved to Recycle Bin

Eight evidence-bearing `outputs_contrastive_*` roots remain.

Permanent cleanup:

- `D:\project\unknown-contrastive\docs\CLEANUP_RUN_DELETE_ROOTS_260726.txt`: 16 failed run roots / 16,769 files / 3,026,968,074 bytes
- `D:\project\unknown-contrastive\docs\CLEANUP_LARGE_IMAGE_TARGETS_260726.txt`: 4,789 regenerable diagnostic images / 59,873,655,673 bytes; all associated CSVs were preserved
- four legacy run roots (`hard42_headonly_coordinate_descent`, `may37_coordinate_descent`, `may_repro`, `_incomplete_salvage`): 1,180 files / 12,712,255,942 bytes
- four baseline/reproduction roots (`may37_protocol_control_current2260`, `may37_reproduction`, `may37_original_reproduction`, `may_new_tapt_removed_paired_2260`): 2,655 files / 74,175,458,452 bytes
- remaining non-champion `runs\sweep` content: 498 files / 2,187,094,711 bytes
- remaining `runs\260610_*` roots: 37 roots / 39,612 files / 13,563,645,169 bytes
- `runs\clean546`: 15 resume-only `last_training.pt` plus 300 intermediate `proj_ep*.pt`, 315 files / 6,756,641,306 bytes; 15 final inference checkpoints preserved
- duplicate `runs\archive`: 32 files / 1,316,849,538 bytes after SHA-256 equivalence verification
- nonwinning/intermediate temperature-screen arrays/checkpoints: 33 files / 1,393,305,452 bytes; selected `t=0.1` evidence preserved
- three-seed intermediate arrays: 24 files / 229,426,176 bytes; three final checkpoints and epoch-4 raw/projection arrays preserved

Previously recorded permanent-delete groups sum to 291,672 files and 436,042,289,456 logical bytes. Actual physical recovery is smaller where derived datasets were hard links. The later D physical-master deletion adds 21,143 files / 264,417,178,347 physical bytes and is kept separate from that logical total.

### Final link and hard-link cleanup

A final project-wide audit, excluding `.git`, found 2,780 hard-link paths representing 88,752,572 logical bytes.

Duplicate clean546 cluster visualizations were removed from the `clusters` subdirectories of these seven runs:

- `D:\project\unknown-contrastive\runs\clean546\abl_fc546s1_B4_260724_072733`
- `D:\project\unknown-contrastive\runs\clean546\abl_fc546s2_B4_260724_073004`
- `D:\project\unknown-contrastive\runs\clean546\abl_fc546_B0_260724_071633`
- `D:\project\unknown-contrastive\runs\clean546\abl_fc546_B1_260724_071847`
- `D:\project\unknown-contrastive\runs\clean546\abl_fc546_B3_260724_072050`
- `D:\project\unknown-contrastive\runs\clean546\abl_fc546_B4_260724_072252`
- `D:\project\unknown-contrastive\runs\clean546\adapt_fcmae_260723_130433`

The cleanup eliminated 2,277 hard-linked visualization paths and preserved 520 regular cluster files. Some sibling paths became ordinary files automatically as their remaining hard links were removed; final hard-link count under all seven `clusters` trees is 0.

The duplicate checkpoint directory `D:\project\unknown-contrastive\runs\wm811k_fixed_b4\run\abl_wm811k_s42_B4_260724_074724\eval_sparse` contained eight hard-linked `proj_ep*.pt` files. Every file matched the same-named file under `D:\project\unknown-contrastive\runs\wm811k_fixed_b4\run\abl_wm811k_s42_B4_260724_074724\checkpoints` by SHA-256 before `eval_sparse` was deleted. The `checkpoints` directory remains.

After cluster visualization cleanup, 62 hard-link paths remained in seven `cluster_summary` trees. Thirty-nine were converted to independent regular files with length and SHA-256 verification; the other 23 became regular files as sibling links were removed. No temporary conversion files remain.

Final link checks:

- repository-wide reparse points: 0
- repository-wide hard-link paths: 0
- links/reparse points across the four retained E image datasets: 0
- clean546 regular cluster files preserved: 520
- `D:\project\unknown-contrastive\runs\wm811k_fixed_b4\run\abl_wm811k_s42_B4_260724_074724\eval_sparse`: absent
- `D:\project\unknown-contrastive\runs\wm811k_fixed_b4\run\abl_wm811k_s42_B4_260724_074724\checkpoints`: present

### Compact evidence archives

- Fifty-seven obsolete one-off source/launcher/test files were ZIP-archived before permanent source deletion:
  - exact list: `D:\project\unknown-contrastive\docs\CLEANUP_LEGACY_CODE_ARCHIVE_260726.txt`
  - archive: `D:\project\unknown-contrastive\docs\archive\legacy_code_deleted_paths_260726.zip`
  - 57 source files; 58 ZIP entries including `_archive_manifest.txt`
  - 187,525 bytes
  - SHA-256 `6BC6ED749ADD84B509315D5C32DA6893D07E41D1DC791D7074B98FC6849AC185`
  - listed source files remaining after deletion: 0
- `D:\project\unknown-contrastive\docs\archive\legacy_run_metadata_260726.zip`
  - 535 entries / 887,348 bytes
  - SHA-256 `c37d91bcb7df762468a3979a54759de21d0efd8530a0305596c5077718fba80b`
- `D:\project\unknown-contrastive\docs\archive\legacy_reproduction_metadata_260726.zip`
  - 1,580 entries / 4,028,425 bytes
  - SHA-256 `e3c45c1b0f1aef5d5b85d9193b09e1268d7ec38eefbb247b7167ac4a9b5384c1`
- `D:\project\unknown-contrastive\docs\archive\sweep_metadata_260726.zip`
  - 47 entries / 102,010 bytes
  - SHA-256 `540344d0cb8d3f65d16660c7c6f8fd88e1e6d2fbf35ae319319a63b456b25986`
- `D:\project\unknown-contrastive\docs\archive\simclr_component_260610_metadata.zip`
  - 510 entries / 1,610,869 bytes
  - SHA-256 `97513fdba9b8570504f427c562a7ff49646d236802fa722eb16e83ca4f19a248`
- `D:\project\unknown-contrastive\docs\archive\adapter_duplicate_archive_metadata_260726.zip`
  - 7,844 bytes
  - SHA-256 `e008e4d30aa269689551d9426305968bde1d39300b4f52ea4e3c07c838cce221`
- 15 important root result records moved intact to `D:\project\unknown-contrastive\docs\archive\root_results_260726`

### Preserved current champion

The compact sweep tree now contains exactly 9 files / 14,265,443 bytes: three champion checkpoints, their three `run_info.json` files, and their three logs.

- `D:\project\unknown-contrastive\runs\sweep\abl_sw_t20_B4_260724_102757\checkpoints\proj_ep20.pt`
  - SHA-256 `a776d5595cae5e60c717cf6e9f87509f4c24c1cb59ed3e70425fe9b7a9f56155`
- `D:\project\unknown-contrastive\runs\sweep\abl_best_s1_B4_260724_111053\checkpoints\proj_ep18.pt`
  - SHA-256 `cea08f9255270af594c0db91249610ac8f947ab7ad77bd5b7a958c67f72b0669`
- `D:\project\unknown-contrastive\runs\sweep\abl_best_s2_B4_260724_111604\checkpoints\proj_ep17.pt`
  - SHA-256 `cf3b79290294067034892eb3c418e8af6448c66cf0158848e859e166d5ff2829`

### Code changes and final verification

Code was narrowed to the retained physical data and result layout:

- `D:\project\unknown-contrastive\scripts\_common.py`
  - `E:\data\images` is the direct-only image root
  - no D image fallback or compatibility-link path remains
  - `resolve_pool` supports physical directories and exact JSON manifests
- `D:\project\unknown-contrastive\scripts\make_pool_manifest.py`, grouping entry points, `D:\project\unknown-contrastive\_score_umapfree.py`, `D:\project\unknown-contrastive\grouping_deploy.py`, `D:\project\unknown-contrastive\_ssl_methods.py`, and contrastive loaders now consume manifests/E paths instead of deleted physical subsets.
- FCMAE fixed protocol, adapter holdout, strict-novel rescore, hard-42, and May-37 entry points were changed to active pool manifests and retained run locations.
- Residual-screen consumers reference `E:\unknown-contrastive-runs\archives` directly; the former repository junction is gone.
- `D:\project\unknown-contrastive\_hf_download.py` defaults to `E:\data\images\hf_flowers102`.
- Runtime source/config defaults were audited so relative project `data\images` inputs are no longer used; all image roots point directly to `E:\data\images`.
- All remaining link-creation paths (`os.link`, hard-link, symbolic-link, and junction creation) were removed. Dataset/split utilities use ordinary copying when explicitly invoked, while active unknown pools use manifests.
- Stale output defaults were moved from deleted `result_grouping`/reproduction paths to retained `D:\project\unknown-contrastive\runs` locations.
- Fifty-seven obsolete one-off files whose inputs/outputs were deleted were archived and then removed, as recorded in the compact-evidence section.

Final data checks:

- repository-wide reparse points: 0
- repository-wide hard-link paths: 0
- retained E image-dataset links/reparse points: 0
- `D:\project\unknown-contrastive\data\images`: 0 items / 0 files / 0 bytes
- `E:\data\images\unknown`: 21,143 images / 264,417,178,347 bytes / zero-byte 0
- active unknown-master manifests: 5 / entries 20,101 / missing 0 / outside E unknown master 0
- all active loader manifests: 7 / entries 21,642 / missing 0 / D image roots 0
- defect-aware full set: 6,594 unique / missing 0 / extra 0
- HF datasets: all 4,700 images decoded successfully; corrupt 0

The original 12 changed/critical entry points compiled successfully. A final compile pass after removing relative D image defaults and link creation also passed for 22 affected Python files; eight dataset/split/training `--help` smoke checks exited 0.

- `D:\project\unknown-contrastive\scripts\_common.py`
- `D:\project\unknown-contrastive\scripts\make_pool_manifest.py`
- `D:\project\unknown-contrastive\_score_umapfree.py`
- `D:\project\unknown-contrastive\grouping_deploy.py`
- `D:\project\unknown-contrastive\_ssl_methods.py`
- `D:\project\unknown-contrastive\scripts\train_contrastive_ddp.py`
- `D:\project\unknown-contrastive\scripts\fcmae_fixed_protocol.py`
- `D:\project\unknown-contrastive\scripts\run_fcmae_adapter_holdout_validation.py`
- `D:\project\unknown-contrastive\scripts\rescore_unknown_strict_novel.py`
- `D:\project\unknown-contrastive\scripts\hard42_headonly_common.py`
- `D:\project\unknown-contrastive\scripts\run_may37_ablation.py`
- `D:\project\unknown-contrastive\_hf_download.py`

Focused pytest verification:

- `D:\project\unknown-contrastive\tests\test_fcmae_fixed_protocol.py`
- `D:\project\unknown-contrastive\tests\test_fcmae_adapter_alpha_three_seed.py`
- `D:\project\unknown-contrastive\tests\test_fcmae_adapter_screen_provenance.py`
- `D:\project\unknown-contrastive\tests\test_fcmae_adapter_temperature_resume.py`
- `D:\project\unknown-contrastive\tests\test_fcmae_adapter_temperature_screen_guard.py`
- result: **33 passed in 20.38 seconds**

CLI `--help` smoke checks exited 0 for:

- `D:\project\unknown-contrastive\scripts\make_pool_manifest.py`
- `D:\project\unknown-contrastive\_ssl_methods.py`
- `D:\project\unknown-contrastive\scripts\train_contrastive_ddp.py`
- `D:\project\unknown-contrastive\scripts\fcmae_fixed_protocol.py`
- `D:\project\unknown-contrastive\scripts\run_fcmae_adapter_holdout_validation.py`
- `D:\project\unknown-contrastive\scripts\rescore_unknown_strict_novel.py`
- `D:\project\unknown-contrastive\scripts\run_hard42_headonly_ablation.py`
- `D:\project\unknown-contrastive\scripts\run_may37_ablation.py`
- `D:\project\unknown-contrastive\_grouping_eval.py`
- `D:\project\unknown-contrastive\_grouping_deliverable.py`
- `D:\project\unknown-contrastive\_grouping_pipeline.py`
- `D:\project\unknown-contrastive\_score_umapfree.py`
