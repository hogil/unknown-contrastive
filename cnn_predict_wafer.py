#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wafer 33-class R-only classifier 추론 entry — cnn_predict.py engine 의 thin wrapper.

default 모델: logs_wafer/overall/best_model.pth (자동 로드, _overall_meta.json 출력)
default 입력: D:/project/data/wm-811k/unknown
출력 폴더 : logs_predict_wafer/<TS>_<input_name>/
            ├─ preds.json
            ├─ per_class_report.txt
            └─ wrong/<true>/<pred>/*.png
"""
# ===================== CONFIG =====================
DEFAULT_MODEL_GLOB   = "logs_wafer/overall/best_model.pth"
DEFAULT_INPUT        = "D:/project/data/wm-811k/unknown"
DEFAULT_PREDICT_ROOT = "logs_predict_wafer"
KIND_LABEL           = "wafer"
# ==================================================

from cnn_predict import main as engine_main


if __name__ == "__main__":
    engine_main(
        default_model_glob=DEFAULT_MODEL_GLOB,
        default_input=DEFAULT_INPUT,
        default_predict_root=DEFAULT_PREDICT_ROOT,
        kind_label=KIND_LABEL,
    )
