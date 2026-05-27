# sota_h100 — Run Guide (server one-shot + production)

Self-contained chip multi-label SOTA: synth data + iter116J seed-sweep training +
production inference with NB OOD-reject. Imports nothing from the legacy generators
for synthesis (existing generators untouched).

---

## A. Server (H100) — one shot

```bash
# 1) code
git clone https://github.com/hogil/known-cnn.git && cd known-cnn
#    (already cloned)  git pull

# 2) backbone weights — NOT in git (~340MB). copy from a machine that has it:
#    scp models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth user@server:~/known-cnn/models/
ls models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth

# 3) deps
pip install torch timm numpy pandas pillow scikit-learn pyarrow

# 4) ONE command, NO options — gen data (auto, deterministic) + seed-sweep
#    train + eval + RESULTS.md. DDP auto-detects all GPUs; data path defaults
#    to data/images/sota_h100.
nohup bash sota_h100/run_seed_sweep_ddp.sh > sweep.out 2>&1 &     # multi-GPU (all GPUs)

bash sota_h100/run_seed_sweep_1gpu.sh                            # single GPU (GPU 0)
```

(optional GPU pick: `CUDA_VISIBLE_DEVICES=0,1,2,3 bash sota_h100/run_seed_sweep_ddp.sh`)

Result: `outputs/sota_h100_seedsweep_*/RESULTS.md` (per-seed + mean±std + best;
bit_F1 / NI·OOD·Total FAR for I10+I13). Best checkpoint: `seed_<N>/<tag>_*/best_model.pth`.

Knobs (prefix as env): `SEEDS="1 2 3 4 5 6 7 8"  EPOCHS=24  IMAGES_ROOT=...  SWEEP_ROOT=outputs/run1`.

Notes:
- Data auto-generated on first run (synth is deterministic, seed 20260527 → identical everywhere).
  Or copy `data/images/sota_h100/` to skip gen.
- No watchdog on the server (local-only) → trains freely.

---

## B. Production / real-data inference

The trained model classifies real 200×200 chip PNGs into the 4 defect bits, and
**rejects OOD/Normal via NB on the 4-D probability vector** (per-bit threshold alone
cannot separate OOD from a weak real combo — NB on the joint distribution does;
demonstrated: real fork log-lik ≈ +16 vs OOD/Normal ≈ −90…−478).

```bash
# only --input is needed; --model and --nb-fit auto-resolve to the latest
# sota_h100 seed-sweep checkpoint / eval parquet.
python -m sota_h100.predict --input /path/to/real_chips

# explicit (override auto):
python -m sota_h100.predict --input /path/to/real_chips \
  --model outputs/sota_h100_seedsweep_*/seed_1/<tag>_*/best_model.pth \
  --nb-fit outputs/sota_h100_seedsweep_*/seed_1/eval/eval_*/preds_chip.parquet \
  --nb-tau -40 --out preds_real.csv
```

Output `preds_real.csv` columns:
`chip_path, prob_bank_boundary, prob_fork, prob_scratch, prob_scratch_rot, max_prob, nb_loglik, decision, labels`

Decision logic per chip:
1. near-white ratio ≥ `--invalid-white` (0.95) → `Invalid`
2. NB joint log-lik < `--nb-tau` → `UNKNOWN` (OOD / Normal — no defect)
3. else per-bit threshold (`--thresholds bb,fork,sc,sr`) → `defect` with `labels`

Tuning `--nb-tau`: higher (e.g. −10) = stricter OOD reject (fewer false defects,
may drop weak real combos); lower (e.g. −60) = looser. Sweep on a labeled eval set
with `_proto_threshold_reject.py` to pick the FAR=0 / max-bit_F1 point.

`--no-nb` disables OOD reject (per-bit threshold only — high FAR on OOD, not recommended).

---

## Files
| file | role |
|---|---|
| `synth.py` | independent chip synthesis (4 single / 6 combo / 4 OOD / Normal / Invalid) |
| `gen_data.py` | `--mode train|eval` palette-PNG generator |
| `run_seed_sweep_1gpu.sh` / `run_seed_sweep_ddp.sh` | seed-sweep train+eval (iter116J recipe) |
| `make_report.py` | aggregate seeds → RESULTS.md |
| `predict.py` | production inference on real chips + NB OOD-reject |

## Key finding (why NB-reject)
Per-bit threshold cannot separate a real combo's weaker positive bit (~0.49) from an
OOD chip's max prob (~0.59) — they overlap in 1-D. The 4-D joint prob vector does
separate them (NB / GMM). On the new-data model this dropped Total FAR 47.9% → ~0.04%
at bit_F1 0.99+.
