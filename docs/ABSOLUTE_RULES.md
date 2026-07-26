# Absolute Rules — Unknown Contrastive

This is the canonical, binding record of the user objective and operating
design. It applies to every future agent, plan, experiment, evaluation, and
handoff. Historical documents are evidence only; they do not override this
file. Canonical memory index:
[`D:\project\unknown-contrastive\memory\MEMORY.md`](../memory/MEMORY.md).

## 1. Objective and evidence hierarchy

1. Continuously improve contrastive performance across multiple **approved**
   datasets, using controlled ablations and reproducible comparisons to seek
   the best validated performance. Preserve minority/new-defect behavior in
   every selection decision; aggregate-only gains that erase minority groups
   are not wins.
2. The final product objective is, when future real internal server data is
   available, label-free unknown grouping that detects repeated emergence of
   new defect types and raises low-FAR alerts. Public and synthetic datasets
   are validation instruments only, not the final target. No internal real
   dataset is currently available.
3. Continue this evidence-building loop for days if necessary. Stop only on a
   user stop instruction or a real blocker that prevents meaningful progress;
   do not stop merely because a queue, a local sweep, or one dataset is done.
4. Within this authorized scope, proceed autonomously: do not ask the user to
   choose among routine experiment, analysis, ablation, or scheduling options.

## 2. Deployment-priority ladder

Evaluate and report routes in this strict preference order:

1. Direct prediction using the model made here.
2. Train with the fixed recipe made here, then predict.
3. Train with a recipe sweep made here, then predict.
4. CNN TAPT, then training sweep, then predict.

Lower-priority routes may be used to establish evidence, but cannot displace a
higher-priority route without clear comparative evidence.

## 3. Data boundary (absolute)

- Use only the seven approved active roots/manifests recorded in
  `D:\project\unknown-contrastive\memory\cleanup_state_260726.md`:
  five unknown-master manifests plus `mwm38_clean546.json` and
  `severstal_pilot260726.json`.
- CCA, my-lot, and every other non-approved dataset/root are prohibited for
  new training, validation, selection, or claims.
- Do not move, copy, link, hard-link, symbolic-link, or junction-link original
  images. Use the approved physical roots and manifests directly.

## 4. Agent decision and design protocol

All non-deterministic judgment/design rounds use three same-role independent
agents:

| Agent | Model and effort | Required role |
|---|---|---|
| A | `gpt-5.6-sol`, `max` | independent blind assessment/design |
| B | `gpt-5.6-terra`, `ultra` | independent blind assessment/design |
| C | `gpt-5.6-sol`, `ultra` | independent blind assessment/design and final adjudication |

Run rounds in this order:

- **R1 — blind:** A, B, and C independently inspect the same evidence without
  seeing one another's verdicts.
- **R2 — cross-rebuttal and method redesign:** compare the R1 verdicts,
  challenge unsupported assumptions, and redesign the method/queue where the
  evidence warrants it.
- **R3 — C closure:** C resolves the record into the binding decision, stating
  uncertainty and protecting minority-group/new-defect performance.

For deterministic implementation, extraction, bookkeeping, and mechanical
verification, use `gpt-5.6-terra` at `low` to `medium` effort rather than this
three-agent judgment protocol.

## 5. Selection discipline

- Use controlled baselines, ablations, repeated seeds where needed, and
  cross-approved-dataset evidence before declaring a best recipe/model.
- Treat label-free selection, unknown grouping, repeated-occurrence detection,
  and low FAR as first-class deployment criteria; label-derived metrics remain
  validation evidence, not a substitute for the final operating objective.
- Record negative findings and rejected recipes so later agents do not repeat
  disproven branches.

## 6. GPU resource boundary (absolute)

- Treat this project's GPU allocation as at most **40% of total VRAM**.
- Every project PyTorch GPU process must apply
  `torch.cuda.set_per_process_memory_fraction(0.40)` before model or tensor
  allocation. A GPU command without a verified equivalent hook must fail
  closed.
- Run at most one project GPU training/evaluation process at a time. External
  GPU processes are not authorization to terminate them; wait or reschedule
  when the 40% project allocation cannot be respected.
- `REPRO_GPU_MEMORY_FRACTION=0.40` and the campaign safety value
  `gpu_memory_fraction=0.40` are binding, not sweep variables.
