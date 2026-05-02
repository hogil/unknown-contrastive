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
DEFAULT_MODEL_GLOB        = "logs_wafer/overall/best_model.pth"
DEFAULT_INPUT             = "D:/project/data/wm-811k/unknown"
DEFAULT_PREDICT_ROOT      = "logs_predict_wafer"
DEFAULT_PSEUDO_LABEL_OUT  = "D:/project/data/wm-811k/unknown"  # pseudo wafer 저장 root (옵션 켜야 활성)
KIND_LABEL                = "wafer"
# wafer basename: prefix_kind_widx_date_time_yld_syp_tester_device  (9 token, _sample_gen.py:684)
DEFAULT_BASENAME_SCHEMA   = ["prefix","kind","w_idx","date","time","yld","syp","tester","device"]
# wafer 마다 sibling JSON 에서 읽을 추가 필드 (preds.csv 컬럼)
DEFAULT_JSON_ROOT         = "D:/project/data/positions/unknown"
DEFAULT_JSON_FIELDS       = ["partid","pgm","wafer","stime","step","yield","sys","tm","lt","netd","gd"]   # part_id 는 partid 와 동일값이라 제외
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
