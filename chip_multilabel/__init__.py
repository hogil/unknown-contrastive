"""chip multi-label evaluation pipeline (single-train -> multi-predict)."""
from .constants import (
    TRAIN_CLASSES,
    SINGLE_KEYS,
    COMBO_KEYS,
    SPECIAL_KEYS,
    ALL_CLASS_KEYS,
    NUM_TRAIN,
    NUM_ALL,
    INFERENCE_VARIANTS,
    TRAIN_VARIANTS,
    DEFAULT_BACKBONE_CKPT,
    DEFAULT_CLASSIFICATION_CHIPS,
    KEY_TO_LABELS,
    LABELS_TO_KEY,
    sort_label_pair,
    labels_to_class_key,
)
