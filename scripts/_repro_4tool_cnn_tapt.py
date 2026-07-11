#!/usr/bin/env python3
"""May-style repro: 4-tool contrastive with CNN/TAPT initialization."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SRC = Path(__file__).with_name("_repro_4tool_nocnn.py")
TAPT_CKPT = "D:/project/known-cnn/models/iter116J_frozen/best_model.pth"

spec = importlib.util.spec_from_file_location("_repro_4tool_nocnn_mod", SRC)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.TAG = "4tool_cnn_tapt"
mod.CNN_RUN_DIR = None
mod.BACKBONE_CKPT = TAPT_CKPT
# The wrapper is loaded dynamically; Windows spawn cannot pickle its dataset
# class with worker processes. Single-process loading preserves the recipe.
mod.NUM_WORKERS = 0

if __name__ == "__main__":
    mod.main()
