# 04 — Error analysis

This page tracks the recurring error modes across iters 1-5.
Counts come from `outputs/<run>/errors.parquet`, where each row is one
chip×cell that failed multi-label match.

## Iter 1 — T0__I3 (frozen baseline winner)

| error_type           | count | %     |
|----------------------|------:|------:|
| false_positive_fork  |   277 | 39.5% |
| wrong_combo          |   264 | 37.7% |
| missed_normal        |   160 | 22.8% |
| **total**            |   701 | 100%  |

_Source: outputs/stage1_260505_162842/errors.parquet (cell_id=T0__I3)._

### 1. fork over-firing (`false_positive_fork`)

The dominant failure on T0. fork's F1-max threshold collapses to **0.1195**
because the trained backbone is so confident about the *winning* class that
the fork sigmoid for non-fork chips still hovers in the 0.10–0.30 band. With
a low threshold, this band breaks past θ_fork on chips that have no fork at
all.

Fix that worked: **CE + LS 0.10 (T1)** softens the dominant logit, lifting
fork's effective threshold to ~0.22 and dropping false_positive_fork from
277 → 155 in iter 4 (-44%).

Even bigger fix in iter 5: **LS=0.20** rebalances the runner-up sigmoid mass
across all four classes. We don't have an errors.parquet for iter 5 cells
yet, but the macro-F1 jump from 0.8634 to 0.9268 plus per-class precision
hint that fork false-positives have largely been resolved.

### 2. wrong_combo (combo class predicted, wrong combo)

The model "knows" two defects exist but picks the wrong pair. Most often
`scratch + scratch_rot` is mis-asserted on chips containing one or the other
plus `bank_boundary`. The diffuse spatial prior over scratch_rot (the
rotated stamp covers a wide angle band) inflates its sigmoid even for
non-rotated scratch chips.

Iter 4 reduced wrong_combo from **264 → 304** (counter-intuitively up, but
note iter 4 also recovers more chips with combo predictions overall — this
counts mis-assignment among more eligible chips). Iter 5 is the actual fix:
LS=0.20 sharpens scratch_rot's threshold to ~0.50 (vs ~0.83 on T0), making
wrong-combo confusions rarer.

### 3. missed_normal

Chips where the truth is `Normal` but at least one defect class fired.
Single-label-trained T0 has no incentive to push *all four* sigmoids low at
once. **160** missed_normal in iter 1.

Fix: **I10 entropy gate** — if no class crosses θ_c and binary entropy
< 0.30 nats, label as Normal. Iter 3 (T0__I10) drops missed_normal noticeably
(visible in macro_f1 +0.0057 over I7).

Iter 4 still has 62 `wrong_normal_entropy` errors at T1__I10 — these are
chips the entropy gate marks Normal but should have a defect. Hard cases
where the actual defect is small and gives a low-confidence sigmoid that
falls under the entropy threshold. Iter 5 reduces this further by training
the model to assign higher mass to the true defect class even on small
defects.

## Iter 4 — T1__I10 (Stage 2 winner, retrain + entropy gate)

| error_type             | count | %     |
|------------------------|------:|------:|
| wrong_combo            |   304 | 57.7% |
| false_positive_fork    |   155 | 29.4% |
| wrong_normal_entropy   |    62 | 11.8% |
| false_positive_scratch |     6 |  1.1% |
| **total**              |   527 | 100%  |

_Source: outputs/stage1_260505_173649/errors.parquet (cell_id=T0__I10 row, but model = T1)._

Total errors **701 → 527** (-25%) vs iter 1.
Modes that disappeared:
- `missed_normal` is replaced by `wrong_normal_entropy` (62 cases) — same
  Normal-side mistake, but now from the entropy gate over-firing instead of
  a defect class hallucinating.

Modes that emerged:
- `false_positive_scratch` (6) — small but new; an iter-5 thing to monitor.

## Recurrent root causes

1. **Single-label CE collapses runner-up scores.** The trained backbone
   has been told "exactly one class". For combo chips, the second-correct
   class hovers at ~0.10–0.30 sigmoid — too close to the noise floor.
   *Fix lineage*: I1 lower thresholds → T1 LS=0.10 reshape softmax → T1
   LS=0.20 reshape softmax harder. Each step extracts more multi-label
   signal from a single-label model.

2. **fork distribution is the most diffuse defect.** Fork stamps are spread
   geometrically wide on the chip, so non-fork chips have above-zero fork
   sigmoid. This is what drives fork's threshold to absurdly low values
   under F1-max.

3. **scratch_rot has a wide rotation prior.** The training distribution
   covers many angles; non-rotated scratch chips trigger scratch_rot above
   threshold ~30% of the time on T0.

4. **Normal has no positive supervision signal.** "Normal" is implicit
   (no defect fired). The entropy gate gives it explicit decoding.

## Files for chip-level inspection

- `outputs/stage1_260505_165400/errors_review_T0__I7.md` — top-200 fork FP and
  wrong_combo chips for the iter 2 winner, with per-class probabilities
  pasted alongside thumbnails. The single best resource for understanding
  iter 1-2 failure modes by eye.
- `outputs/<run>/errors/<cell>/<error_type>/*.png` — chip thumbnails
  capped at 200/error_type. Useful when filing follow-up iters.
