#!/usr/bin/env bash
# Production inference on real chips. One positional arg = the chip folder.
#   bash sota_h100/predict.sh /path/to/real_chips
# model + NB-reject fit auto-resolve to the latest sota_h100 checkpoint/eval.
# Output: preds_real.csv (chip_path, prob_*, nb_loglik, decision, labels).
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IN="${1:?usage: bash sota_h100/predict.sh <real_chip_dir>}"
python -u -X utf8 -m sota_h100.predict --input "$IN" --nb-tau -40 --out preds_real.csv
