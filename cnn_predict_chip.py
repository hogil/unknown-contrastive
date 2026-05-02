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
# chip filename: <wafer_key 7-token>_X<gx>_Y<gy>_B<bin>  (10 token after split — wafer_key drops yld/syp)
# 즉 prefix, kind, w_idx, date, time, tester, device, X<n>, Y<n>, B<n>
DEFAULT_BASENAME_SCHEMA   = ["prefix","kind","w_idx","date","time","tester","device","gx_token","gy_token","b_token"]
DEFAULT_JSON_ROOT         = None   # chip 은 wafer JSON 1:1 매칭 안 됨 — JSON read off
DEFAULT_JSON_FIELDS       = []
# ==================================================

from cnn_predict import main as engine_main


if __name__ == "__main__":
    engine_main(
        default_model_glob=DEFAULT_MODEL_GLOB,
        default_input=DEFAULT_INPUT,
        default_predict_root=DEFAULT_PREDICT_ROOT,
        default_pseudo_label_out=DEFAULT_PSEUDO_LABEL_OUT,
        default_basename_schema=DEFAULT_BASENAME_SCHEMA,
        default_json_root=DEFAULT_JSON_ROOT,
        default_json_fields=DEFAULT_JSON_FIELDS,
        kind_label=KIND_LABEL,
    )
