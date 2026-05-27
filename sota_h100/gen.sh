#!/usr/bin/env bash
# Generate the dataset. No options.
#   train: 4 single defects, 200/class -> data/images/sota_h100/classification_chips
#   eval : 16 classes, 2000/class      -> data/images/sota_h100/eval_set
# Usage:  bash sota_h100/gen.sh
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -u -X utf8 -m sota_h100.gen_data --mode train --per-class 200  --out data/images/sota_h100/classification_chips --seed 20260527 --clean-first
python -u -X utf8 -m sota_h100.gen_data --mode eval  --per-class 2000 --out data/images/sota_h100/eval_set          --seed 20260527 --clean-first
