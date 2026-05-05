# 2. Related Work

## 2.1 Single-label to multi-label decoding

The general problem of repurposing a softmax-trained classifier for
multi-label outputs is older than deep learning, but stays relevant
because production pipelines often only have single-label
ground-truth budget. Two threads dominate:

- **Threshold tuning.** Lipton, Elkan, and Naryanaswamy
  (arXiv:1402.1892) characterise the F1-maximising operating point
  as a function of class-conditional probability and propose grid
  thresholding. We use this directly in I1 (per-class F1-max),
  I3 (F1-max + top-K rescue), and I7 (F1-max + step-search Δ=0.02).
- **Calibration.** Guo, Pleiss, Sun, and Weinberger (arXiv:1706.04599)
  show that modern CNNs are systematically over-confident and that
  a single learned temperature recovers calibrated probabilities at
  near-zero compute. We apply temperature scaling in I4 and I9 and
  find that it improves ECE (0.0778 → 0.0129) but does *not* improve
  macro-F1, because the threshold sweep already absorbs the rescaling.

## 2.2 Label smoothing

Müller, Kornblith, and Hinton (arXiv:1906.02629) frame label
smoothing as a regulariser that prevents the winning logit from
running away to infinity. In our setting this turns out to be
exactly the lever we need: the single-label CE backbone produces
runner-up logits 1–2 orders of magnitude below the winning one, and
LS softens that gap so the multi-label sigmoid threshold decoder
can read second-class signal. Their paper establishes α∈[0.05, 0.10]
as the typical "safe" range; we observe a peak at α=0.20 in our
small-data, strong-pretrain regime, and discuss the discrepancy in §7.

## 2.3 Multi-label losses

For models trained natively as multi-label, Asymmetric Loss (Ridnik
et al. 2021, arXiv:2009.14119) is the current default for handling
the positive/negative class imbalance via two focusing parameters
(γ₊, γ₋) and a probability shift m. We test ASL with the published
default (γ₊=0, γ₋=4, m=0.05) as our T4 variant and find it loses
0.078 macro-F1 against T1 — we attribute this to the strong TAPT
backbone init plus single-positive train labels, where ASL's
suppression of negative gradients destroys useful single-class
discrimination.

Focal Loss (Lin et al. 2017, arXiv:1708.02002), included in our
codebase as T3, was not run in iters 1–5 (planned in Phase C). We
expect similar concerns to ASL because Focal also down-weights easy
negatives.

Plain BCE (T5) is the simplest multi-label baseline; we run it as
a per-class binary cross-entropy on one-hot targets. It regresses
0.095 macro-F1 from T0__I10, suggesting that the *softmax-shaped*
class structure of the original CE training is a load-bearing
property of the backbone, not noise.

## 2.4 Entropy-based decoding

Our I10 decoder declares `Normal` when the softmax entropy of the
training-class logits exceeds 0.85·log(C) (i.e. ≥85% of the maximum
possible entropy for C training classes). The rationale is closer
to a maximum-entropy abstention rule (Geifman & El-Yaniv 2017) than
to a calibrated probability, but adapted to our setting: the
training set has no `Normal` examples, so there is no positive
supervision signal for `Normal`, and entropy is the cleanest
training-free proxy for "no confident class" available.

## 2.5 Test-time augmentation in defect classification

TTA via rotation is standard for general image classification, but
*violates* our class taxonomy: `scratch` and `scratch_rot` are two
distinct labels that differ only by 90° rotation. A 4-view rotation
TTA (I5) collapses them. Iter 1 confirmed this empirically (−0.018
macro-F1) and TTA is permanently disallowed from iter 2 onward. We
flag this as a class-taxonomy-aware design rule rather than an
inference hyperparameter.

## 2.6 Position relative to prior work

Most of the techniques above were proposed for natural-image
multi-label datasets (MS-COCO, Pascal-VOC) where class counts are in
the tens to thousands and per-class supervision is plentiful. Our
setting is the opposite extreme: 4 trainable classes, 327 single-label
training chips, an expensive synthetic eval set with a class
distribution different from training. We hypothesise that the
combination of (a) strong TAPT backbone, (b) very few training
labels, and (c) eval distribution containing classes the model has
never seen (`Normal`, combos) is what shifts the optimum from the
multi-label-native losses (ASL, BCE) to a CE-with-stronger-LS recipe.
