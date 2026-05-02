#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chip 5-class classifier 학습 entry — cnn_train.py engine 의 thin wrapper.

학습 데이터: chip 200x200 crop (D:/project/data/wm-811k/classification_chips/<obj>/*.png)
학습 결과 : logs_chip/<run>/best_model.pth, history.json, ...
            logs_chip/overall/  (val F1 best 면 자동 갱신)

cnn_train.py 의 모든 인자 그대로 전달 가능 (--epochs, --batch, --subset-config, ...).
"""
# ===================== CONFIG =====================
DEFAULT_DATA_DIR = "D:/project/data/wm-811k/classification_chips"
DEFAULT_LOG_ROOT = "logs_chip"
KIND_LABEL = "chip"
# ==================================================

from cnn_train import main as engine_main


if __name__ == "__main__":
    engine_main(
        default_data_dir=DEFAULT_DATA_DIR,
        default_log_root=DEFAULT_LOG_ROOT,
        kind_label=KIND_LABEL,
    )
