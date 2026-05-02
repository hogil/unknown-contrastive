#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wafer 33-class R-only classifier 학습 entry — cnn_train.py engine 의 thin wrapper.

학습 데이터: wafer fail-bit palette PNG (D:/project/data/wm-811k/unknown/<class>/*.png)
            R 채널만 (palette idx grayscale).
학습 결과 : logs_wafer/<run>/best_model.pth, history.json, ...
            logs_wafer/overall/  (val F1 best 면 자동 갱신)

cnn_train.py 의 모든 인자 그대로 전달 가능 (--epochs, --batch, --subset-config, ...).
"""
# ===================== CONFIG =====================
DEFAULT_DATA_DIR = "D:/project/data/wm-811k/unknown"
DEFAULT_LOG_ROOT = "logs_wafer"
KIND_LABEL = "wafer"
# ==================================================

from cnn_train import main as engine_main


if __name__ == "__main__":
    engine_main(
        default_data_dir=DEFAULT_DATA_DIR,
        default_log_root=DEFAULT_LOG_ROOT,
        kind_label=KIND_LABEL,
    )
