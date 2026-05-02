#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chip 5-class classifier 추론 entry — cnn_predict.py engine 의 thin wrapper.

default 모델: logs_chip/overall/best_model.pth (자동 로드, _overall_meta.json 출력)
default 입력: D:/project/data/wm-811k/classification_chips
출력 폴더 : logs_predict_chip/<TS>_<input_name>/
            ├─ preds.json
            ├─ per_class_report.txt
            └─ wrong/<true>/<pred>/*.png
"""
# ===================== CONFIG =====================
DEFAULT_MODEL_GLOB        = "logs_chip/overall/best_model.pth"
DEFAULT_INPUT             = "D:/project/data/wm-811k/classification_chips"
DEFAULT_PREDICT_ROOT      = "logs_predict_chip"
DEFAULT_PSEUDO_LABEL_OUT  = "D:/project/data/wm-811k/classification_chips"  # pseudo crop 저장 root (옵션 켜야 활성)
KIND_LABEL                = "chip"
# ==================================================

from cnn_predict import main as engine_main


if __name__ == "__main__":
    engine_main(
        default_model_glob=DEFAULT_MODEL_GLOB,
        default_input=DEFAULT_INPUT,
        default_predict_root=DEFAULT_PREDICT_ROOT,
        default_pseudo_label_out=DEFAULT_PSEUDO_LABEL_OUT,
        kind_label=KIND_LABEL,
    )
