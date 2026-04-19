#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ExperimentPreset definitions for contrastive.py sweeps.

Each preset specifies:
    - name: unique id (also used as output sub-directory)
    - description: one-line summary of what changes vs baseline
    - cfg_overrides: plain key/value patches against contrastive.CFG
    - structural markers (input_mode / unfreeze_stages / hard_negatives /
      strong_augment / multicrop / deeper_head). These are consumed by
      experiments.run_experiment which injects them into CFG under
      __INPUT_MODE__ / __UNFREEZE_STAGES__ / __HARD_NEGATIVES__ / ... keys so
      that contrastive.py (Task 4) can pick them up without breaking the
      current hard-coded CFG.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentPreset:
    name: str
    description: str
    cfg_overrides: Dict[str, Any] = field(default_factory=dict)
    # Structural markers (handled by run_experiment, not raw CFG keys)
    input_mode: str = "palette_rgb"       # "palette_rgb" | "idx_gray_x8" | "idx_1ch"
    unfreeze_stages: List[str] = field(default_factory=list)  # e.g. ["stages.3"]
    hard_negatives: bool = False
    strong_augment: bool = False
    multicrop: bool = False
    deeper_head: bool = False


PRESETS: Dict[str, ExperimentPreset] = {
    "baseline": ExperimentPreset(
        name="baseline",
        description="Current CFG unchanged (palette_rgb input, frozen backbone, default InfoNCE+queue+local).",
    ),

    "idx_gray_x8": ExperimentPreset(
        name="idx_gray_x8",
        description="Replace palette RGB with 8x-scaled single-channel palette index replicated to 3 channels; backbone stays frozen.",
        input_mode="idx_gray_x8",
    ),

    "idx_1ch": ExperimentPreset(
        name="idx_1ch",
        description="Use raw palette index as a true 1-channel input; requires conv1 surgery, so stage3 is unfrozen to relearn low/mid features.",
        input_mode="idx_1ch",
        unfreeze_stages=["stages.3"],
    ),

    "unfreeze_s3": ExperimentPreset(
        name="unfreeze_s3",
        description="Unfreeze backbone.stages.3 (highest level stage) while keeping palette_rgb input; gives the head a richer adaptable feature map.",
        unfreeze_stages=["stages.3"],
        cfg_overrides={"LR_HEAD": 5e-4},
    ),

    "unfreeze_s3_gray": ExperimentPreset(
        name="unfreeze_s3_gray",
        description="Combine idx_gray_x8 input with stage3 unfreeze to let the backbone adapt to grayscale palette statistics.",
        input_mode="idx_gray_x8",
        unfreeze_stages=["stages.3"],
        cfg_overrides={"LR_HEAD": 5e-4},
    ),

    "hard_negatives": ExperimentPreset(
        name="hard_negatives",
        description="Replace the queue-based InfoNCE with a hard-negative-mining variant that up-weights the top-K nearest negatives per anchor.",
        hard_negatives=True,
        cfg_overrides={"IGNORE_NEG_SIM": 0.85},
    ),

    "strong_augment": ExperimentPreset(
        name="strong_augment",
        description="Stronger self-supervised augmentation stack (larger rotate/translate/scale, random erasing, heavier noise) on top of baseline.",
        strong_augment=True,
    ),

    "multicrop": ExperimentPreset(
        name="multicrop",
        description="SwAV-style multi-crop: 2 global views (full IMAGE_SIZE) + 4 local crops at ~0.5x size contributing to the global InfoNCE objective.",
        multicrop=True,
        cfg_overrides={"BATCH": 128},  # compensate for extra views
    ),

    "deeper_head": ExperimentPreset(
        name="deeper_head",
        description="Replace the 2-layer projection head with a deeper 3-layer MLP (hidden 2048 -> 512 -> PROJ_DIM) with BN.",
        deeper_head=True,
    ),

    "queue_64k": ExperimentPreset(
        name="queue_64k",
        description="Quadruple the negative queue from 16384 to 65536 for richer negative distributions in InfoNCE.",
        cfg_overrides={"QUEUE_SIZE": 65536},
    ),

    "local_global_search": ExperimentPreset(
        name="local_global_search",
        description="Switch local InfoNCE positive-search from windowed to full-feature-map (global) matching.",
        cfg_overrides={"LOCAL_SEARCH": "global"},
    ),

    "no_local_loss": ExperimentPreset(
        name="no_local_loss",
        description="Disable the local InfoNCE loss term entirely — pure global InfoNCE + queue.",
        cfg_overrides={"USE_LOCAL": False, "LOCAL_WEIGHT": 0.0},
    ),

    "epochs_40": ExperimentPreset(
        name="epochs_40",
        description="Double the training budget to 40 epochs with 4-epoch warmup to check whether baseline is under-trained.",
        cfg_overrides={"EPOCHS": 40, "WARMUP_EPOCHS": 4},
    ),

    "temp_0_05": ExperimentPreset(
        name="temp_0_05",
        description="Lower InfoNCE temperature to 0.05 (sharper similarity distribution, stronger hard-negative pressure).",
        cfg_overrides={"TEMP": 0.05},
    ),

    "temp_0_15": ExperimentPreset(
        name="temp_0_15",
        description="Raise InfoNCE temperature to 0.15 (smoother distribution, reduced gradient magnitude on hard negatives).",
        cfg_overrides={"TEMP": 0.15},
    ),
}


# =========================================================
# Evaluation subset selection rule (used when --use-subset)
# =========================================================
EVAL_SUBSET: Dict[str, Any] = {
    "N_CLASSES": 10,              # top-N classes (by image count) — 10 defect types
    "N_PER_CLASS": 200,           # per-class sample cap (fewer used if class is smaller)
    "SELECTION": "largest",       # "largest" | "random" | "explicit"
    "EXPLICIT_CLASSES": None,     # Optional[List[str]] — overrides SELECTION when set
    "SEED": 42,
}


def get_preset(name: str) -> ExperimentPreset:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {sorted(PRESETS.keys())}"
        )
    return PRESETS[name]


def list_presets() -> List[str]:
    return list(PRESETS.keys())
