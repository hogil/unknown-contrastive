---
name: cnn-inference
description: cnn_predict.py 래퍼 — best_model.pth 로드, 단일/폴더 입력, threshold-based Normal/unknown, threshold sweep, per_class_report 생성.
---

# cnn-inference skill

`cnn_predict.py` 래퍼. CNN 학습된 best_model.pth로 wafer 이미지 분류.

## 가장 먼저 읽기

| 문서 | 용도 |
|---|---|
| `cnn_predict.py` (repo root) | CLI flags + 동작 |
| `.claude/skills/cnn-training/SKILL.md` | 출력 폴더 (best_model.pth 위치) |

## CLI 옵션

| flag | 설명 |
|---|---|
| `--model PATH` | best_model.pth (또는 legacy best.pt) — 둘 다 자동 인식 |
| `--input PATH` | 단일 이미지 또는 폴더 |
| `--output JSON` | per-image 예측 결과 저장 (없으면 stdout head) |
| `--threshold F` | max_prob < F → "Normal/unknown" 라벨링 |
| `--threshold-sweep "lo,hi,step"` | 라벨 추론 가능 시 (folder 구조) threshold별 metric 표 |
| `--report-out PATH` | input이 `{class}/img.png` 폴더면 per_class_report.txt 저장 |
| `--ema` | ckpt에 ema_state 있으면 그것을 model에 load |
| `--batch N` | (default 16) |
| `--workers N` | (default 4) |
| `--no-recursive` | 폴더 비재귀 검색 |

## 사용 패턴

```bash
# 1. 단일 이미지 분류
python cnn_predict.py --model log/<run>/best_model.pth --input wafer.png

# 2. 폴더 일괄 분류 (재귀)
python cnn_predict.py --model best_model.pth --input D:/test_dir --output preds.json

# 3. threshold 적용 (운영 모드)
python cnn_predict.py --model best_model.pth --input <dir> --threshold 0.7 --output preds.json

# 4. threshold sweep — 라벨 추정 가능 시 ({class}/img.png 구조)
python cnn_predict.py --model best_model.pth --input val_dir --threshold-sweep 0.1,0.9,0.05

# 5. per_class_report 동시 생성
python cnn_predict.py --model best_model.pth --input test_dir --report-out report.txt --output preds.json

# 6. EMA shadow 사용
python cnn_predict.py --model best_model.pth --input <dir> --ema --output preds.json
```

## 자동 동작

- **classes**: ckpt['classes']에서 자동 로드
- **img_size**: ckpt['img_size'] (default 384)
- **backbone**: ckpt['backbone'] (default convnextv2_base.fcmae_ft_in22k_in1k_384)
- **EMA load**: ckpt['ema_state'] 있고 `--ema` 지정 시 shadow weights → model

## Output JSON schema

```json
[
  {
    "path": "wafer.png",
    "pred_class": "Center_bank_boundary",        # threshold 미적용 시
    "pred_idx": 0,
    "max_prob": 0.95,
    "is_normal": false,                           # threshold > max_prob일 때 true
    "probs": {"Center_bank_boundary": 0.95, ...},
    "true_idx": 0,                                # 라벨 추정 가능 시
    "true_class": "Center_bank_boundary"
  },
  ...
]
```

## 운영 워크플로우

1. cnn_train으로 학습 → log/<run>/best_model.pth
2. val 폴더에 threshold sweep → 운영 threshold 결정 (예: 0.7)
3. Normal pool (5000장)에 predict + threshold → false-defect rate 확인
4. 새 wafer 들어올 때 단일 predict + threshold → 결과 분류

## 금지

- Normal/Starburst/CommaCluster 등 pre-defined class 추가/삭제 (ckpt 정합성)
- 학습 데이터 수정 (read-only inference)
