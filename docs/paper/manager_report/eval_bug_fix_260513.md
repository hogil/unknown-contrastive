# `without_normal` block bit-identical bug — root cause + fix (260513)

## Symptom

All `eval_summary.json` files produced by `_eval_contrastive_unknown_n50.py`
had `without_normal` block **bit-identical** to `with_normal`. This silently
broke the defect-only metric pipeline across all NEW / B4 / B5 runs.

Reference run: `outputs_contrastive_260512_001719/eval/eval_summary.json`

| field | with_normal | without_normal | match? |
|---|---|---|---|
| n | 2146 | 2146 | == |
| ari | 0.8269 | 0.8269 | == |
| nmi | 0.9349 | 0.9349 | == |
| ami | 0.9257 | 0.9257 | == |
| completeness | 0.9173 | 0.9173 | == |
| silhouette_cosine | 0.3972 | 0.3972 | == |
| per_class_noise contains `Normal*` | False | False | == |
| normal_metrics.normal_noise_pct | None | None | == |

`normal_metrics` was `None/None` — first signal that no sample was being
identified as Normal at all.

## Root cause

The argparser default for `--normal-class` is the historical wafer-level
name `"Normal_bank_boundary"` (`_eval_contrastive_unknown_n50.py:57`). The
current contrastive anchor dataset
(`D:/project/data/contrastive_anchor/avg30_new_260508_123037/`) uses the
plain folder name `"Normal"` (1000 samples).

Consequently:

```python
keep = np.array([c != args.normal_class for c in classes])
#                  c != "Normal_bank_boundary"   # True for every sample
# → keep.sum() == 2146 == len(classes)
# → emb[keep] == emb, classes[keep] == classes
# → without_normal == with_normal exactly
```

Same silent no-op happened in 4 other call sites in `main()`:
`cluster_report_records`, `normal_leakage_metrics`,
`class_fragmentation_records`, `retrieval_report_records`,
and the stress-test `normal_mask`.

## Fix

Added `resolve_normal_class()` helper (lines 424-451) and wired it into
`main()` once after `extract_embeddings()` returns the dataset class list.

Resolution policy:
1. **Exact match** (requested name present in dataset) → use as-is.
2. **`startswith("normal")` (case-insensitive)** fallback → log WARN, use
   first match.
3. **No match anywhere** → return requested (preserve current behaviour),
   log WARN — at least the user sees a warning instead of silent no-op.

All `main()` consumers of `args.normal_class` switched to the resolved
local `normal_class`. The argparser default itself is **unchanged** so
sister script defaults (`predict_contrastive_daily.py`, `_eval_contrastive_n50.py`,
`_contrastive_unknown_n50.py`, `eval_tier1_helper.py`) are unaffected.

### Lines changed

| line(s) | change |
|---|---|
| 424-451 | new helper `resolve_normal_class(classes, requested, logger)` |
| 769-771 | call helper after HDBSCAN: `normal_class = resolve_normal_class(...)` + info log on mismatch |
| 773, 775, 777, 780 | `args.normal_class` → `normal_class` (4 downstream functions) |
| 795 | `keep = np.array([c != normal_class for c in classes])` (without-Normal mask) |
| 813 | `normal_mask = np.array([c == normal_class for c in classes])` (stress-test) |

Total: 1 new function (28 lines) + 7 line tweaks. `with_normal` block
untouched. No new metrics, no removed metrics.

## Dry-run verification

Re-ran the fixed script on `outputs_contrastive_260512_001719/` with
`--eval-name eval_dryrun --no-plots --no-rename --no-overall`. Existing
`eval/` directory preserved as `eval_pre_fix_backup/`.

Log evidence:
```
[normal-resolve] requested 'Normal_bank_boundary' not found in dataset
    (43 classes); falling back to 'Normal' (candidates=['Normal'])
[main] normal_class resolved: 'Normal_bank_boundary' -> 'Normal'
```

Post-fix `eval_summary.json` diff:

| field | with_normal | without_normal | match? |
|---|---|---|---|
| n | 2146 | 1146 | != (exactly 1000 Normal removed) |
| n_clusters | 43 | 39 | != |
| n_noise | 834 | 40 | != |
| noise_pct | 38.86% | 3.49% | != |
| ari | 0.7062 | 0.8616 | != |
| nmi | 0.8925 | 0.9536 | != |
| ami | 0.8762 | 0.9420 | != |
| homogeneity | 0.9300 | 0.9435 | != |
| completeness | 0.8578 | 0.9639 | != |
| cluster_purity | 0.9399 | 0.8970 | != |
| silhouette_cosine | 0.7701 | 0.7911 | != |
| Normal in per_class_noise | True | False | flip |
| normal_metrics.n_normal | — | — | 1000 (was 0) |
| normal_metrics.normal_noise_pct | — | — | 79.4% (was None) |
| normal_metrics.normal_leakage_to_defect_cluster | — | — | 0.0% (was None) |

Note: the HDBSCAN cluster counts in this dryrun (`43 / 39`) differ from
the original recorded `39` because the original used HDBSCAN cfg
overrides not replayed here. **That is orthogonal to the bug.** The fix
itself does not change HDBSCAN params; it only changes which samples
each metric set sees.

## Paper impact

**Zero.** All paper claims use `tier1_*.json` files written by a
separate helper (`eval_tier1_helper.py`) which independently filters
Normal samples and was unaffected. This fix only restores correctness
of the `without_normal` block in future eval runs.

The dry-run `eval_dryrun/` subdir can be deleted at user discretion;
`eval/` and `eval_pre_fix_backup/` are preserved.

## Files

- Edited: `D:/project/unknown-contrastive/_eval_contrastive_unknown_n50.py`
  (helper + 5 wiring tweaks)
- Backup: `D:/project/unknown-contrastive/outputs_contrastive_260512_001719/eval_pre_fix_backup/`
- Dry-run output: `D:/project/unknown-contrastive/outputs_contrastive_260512_001719/eval_dryrun/`
- This report: `D:/project/unknown-contrastive/docs/paper/manager_report/eval_bug_fix_260513.md`

## Recommended follow-up (optional, not in this fix)

- Consider promoting the resolver to `predict_contrastive_daily.py` and
  `_eval_contrastive_n50.py` (same default `Normal_bank_boundary` lives
  there too) if those scripts get re-run on the current anchor data.
- Sister scripts using `Normal_bank_boundary` in plain text (docs,
  REPORT.md, `_build_anchor_subset.py`, `_sample_gen.py`, etc.) are
  documentation/legacy — no immediate action.

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/eval_bug_fix_260513.md
