---
name: cnn-training
description: ConvNeXtV2 base FCMAE 분류기를 33 class wafer fail-bit 데이터로 학습. cnn_train.py wrapper. 출력 폴더 log/{model}_{TS}_F{f1:.2f}_R{r:.2f}/ 컨벤션, class subset YAML 지원, EMA + class_weight(effective) 기본.
---

# cnn-training skill

이 스킬은 `cnn_train.py`를 실행해 production-grade CNN 분류기를 학습한다.

## 가장 먼저 읽기

| 문서 / 파일 | 용도 |
|---|---|
| `cnn_train.py` (repo root) | 본 스크립트 — CLI flags, helpers (FocalLoss, EMA, FilteredImageFolder 등) |
| `D:\project\anomaly-detection\docs\summary.md` | best practice 차용 원본 (focal/EMA/grad clip BKM) |
| `D:\project\anomaly-detection\train.py` | rename-on-end pattern 원본 |

## 사양 핵심

- **Backbone**: `convnextv2_base.fcmae_ft_in22k_in1k_384` (timm + HF)
- **Excluded class**: `Normal` (open-set; inference 시 `--threshold`로 처리)
- **Split**: per-class stratified 80/10/10 (seed 고정)
- **Loss default**: CE + label_smoothing=0.02 + class_weight=`effective` (Cui et al. β=0.999)
- **EMA default**: ON, decay 0.95, warmup 3 epoch
- **Augmentation**: full rotation (wafer는 round) + flip + affine + ColorJitter
- **AMP**: bf16 (4060 Ti 지원)
- **Best 갱신**: smoothed val F1 (median window 3) + val_loss guard (×2 best 거부)
- **Best 갱신 시**: test set 평가까지 자동 → per_class_report.txt + best_confusion_matrix.png + test_eval.json 갱신

## 출력 폴더 (rename-on-end)

```
log/{model_tag}_{YYYYMMDD_HHMMSS}_F{test_f1:.2f}_R{test_recall:.2f}/
  hparams.yaml                 모든 args + CFG
  hparams.txt                  사람이 읽기 좋은 mirror
  best_model.pth               state_dict + classes + img_size + ema_state(if any)
  best_confusion_matrix.png    best 시점 test confusion matrix
  curves.png                   3-axis: train_loss / val_loss / val_F1
  per_class_report.txt         class별 F1/P/R/FP/FN/Sup + macro/weighted 요약
  history.json                 epoch별 metrics
  test_eval.json               best 시점 test classification_report
  run.log
  predictions/                 (--save-pred-samples) tp/fp/fn × class
```

## CLI 핵심 옵션

| flag | default | 의미 |
|---|---|---|
| `--epochs N` | 30 | total epoch |
| `--batch N` | 16 | batch size |
| `--img-size N` | 384 | resize target (backbone에 맞춤) |
| `--subset-config PATH` | None | YAML로 class별 sample 수 제한 |
| `--loss {ce,focal}` | `ce` | focal: hard sample 가중 |
| `--class-weight {none,inverse,effective}` | `effective` | imbalance 자동 보정 |
| `--label-smoothing F` | 0.02 | overconfidence 방지 |
| `--ema / --no-ema` | ON | shadow weights, val 평가 시 적용 |
| `--ema-decay F` | 0.95 | |
| `--stochastic-depth F` | 0.05 | timm `drop_path_rate` |
| `--grad-clip F` | 0.5 | bf16 안정성 |
| `--warmup-epochs N` | 2 | LinearLR(0.1→1) → CosineAnnealing |
| `--weighted-sampler` | OFF | class_weight와 양자택일 |
| `--val-loss-guard F` | 2.0 | val_loss > guard×best면 best save 차단 |
| `--val-smooth-window N` | 3 | median over last N val F1 |
| `--save-pred-samples` | OFF | TP/FP/FN 샘플 이미지 저장 |
| `--patience N` | 7 | early stop |

## subset YAML 형식

```yaml
classes:
  Donut_scratch: 30
  Center_bank_boundary: 50
  default: 200
```

`default` 미지정 시 해당 class는 전체 sample 사용. minor class 시뮬레이션 + class_weight A/B 비교용.

## 실행 예

```bash
# 기본 학습 (33 class × 200, ~2-3시간)
python cnn_train.py --epochs 30 --batch 16

# subset 학습 (imbalance simulation)
python cnn_train.py --epochs 30 --subset-config subset.yaml

# Loss A/B 비교 (동일 seed)
python cnn_train.py --epochs 30 --class-weight none      --model-tag exp_no_cw
python cnn_train.py --epochs 30 --class-weight effective --model-tag exp_eff_cw
python cnn_train.py --epochs 30 --loss focal --focal-gamma 2 --model-tag exp_focal

# Quick smoke (2 epoch, 660 sample)
python cnn_train.py --epochs 2 --subset-config quick.yaml --batch 8
```

## 결정 트리

- **Class 불균형 큼?** → `--class-weight effective` (default) 또는 `--loss focal`
- **Hard sample 많음?** → `--loss focal --focal-gamma 2.0`
- **Spike epoch 자주?** → `--val-loss-guard 1.5` 더 엄격히
- **Best 너무 자주 갱신 (lucky epoch)?** → `--val-smooth-window 5`로 더 smoothing

## 금지

- `EXCLUDE_CLASSES = {"Normal"}` 상수 변경 금지 (open-set 정책)
- `log/<run>/` 결과 폴더 삭제 금지 (사용자 명시 요청 전)
- `BACKBONE` 상수 변경 시 hparams.yaml 비교 가능하도록 `--model-tag`로 구분
