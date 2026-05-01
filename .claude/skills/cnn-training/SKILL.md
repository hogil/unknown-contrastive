---
name: cnn-training
description: ConvNeXtV2 base FCMAE 분류기를 33 class wafer fail-bit 데이터로 학습. cnn_train.py wrapper. 출력 폴더 log/{model_tag}_{YYMMDD_HHMMSS}_{test_f1:.2f}_{val_f1:.2f}/ 컨벤션, class subset YAML 지원, EMA + class_weight(effective) 기본.
---

# cnn-training skill

이 스킬은 `cnn_train.py`를 실행해 production-grade CNN 분류기를 학습한다.

## 가장 먼저 읽기

| 문서 / 파일 | 용도 |
|---|---|
| `cnn_train.py` (repo root) | 본 스크립트 — CLI flags, helpers (FocalLoss, EMA, FilteredImageFolder 등) |
| `USAGE.md` (repo root) | 명령어 예시 + 워크플로우 (image-gen → train → predict) |

## 사양 핵심

- **Backbone**: `convnextv2_base.fcmae_ft_in22k_in1k_384` (timm + local mirror via `download_backbone.py`)
- **Excluded class**: `Normal` (open-set; inference 시 `--threshold`로 처리)
- **Split**: per-class stratified 80/10/10 (3-way), 또는 `--train-val-only`로 80/20
- **Resize**: BICUBIC (사용자 결정 — 1024 BICUBIC이 chip 격자 + line/blob 보존 최소 size)
- **Loss default**: CE + label_smoothing=0.02 + class_weight=`effective` (Cui et al. β=0.999)
- **EMA default**: ON, decay 0.95, warmup 3 epoch
- **Augmentation (도메인-safe, position+palette+angle safe)**:
  - ✅ ±15° rotation (stage 회전 오차 모사)
  - ✅ small affine ±3% (alignment / magnification)
  - ✅ Gaussian noise σ=0.01 (sensor pixel noise)
  - ❌ **HFlip 금지** — scratch_21deg 등 angle 자체가 클래스 정체성 (21° → -21°)
  - ❌ VFlip / 180° rotation (Edge-Top ↔ Edge-Bottom)
  - ❌ ColorJitter (palette grade 의미 손상)
  - ❌ MixUp / CutMix / Cutout (palette pixel 평균이 무의미한 grade)
- **AMP**: bf16
- **Best 갱신**: smoothed val F1 (median window 3) + val_loss guard (×2 best 거부)
- **Best 갱신 시**: test set 자동 평가 → 모든 결과 파일 갱신

## 출력 폴더 (rename-on-end)

```
log/{model_tag}_{YYMMDD_HHMMSS}_{test_f1:.2f}_{val_f1:.2f}/   ← 3-way split
log/{model_tag}_{YYMMDD_HHMMSS}_{val_f1:.2f}/                 ← --train-val-only
  hparams.yaml                 모든 args + CFG
  hparams.txt                  사람이 읽기 좋은 mirror
  best_model.pth               state_dict + classes + img_size + ema_state(if any) + test/val metrics
  best_history.txt             ★ 통합 결과 — 아래 4 sections
  best_confusion_matrix.png    test(위)+val(아래) combined, 셀 숫자 annotation
  curves.png                   3-axis: train_loss / val_loss / val_F1 (매 epoch 갱신)
  history.json                 epoch별 metrics (매 epoch 갱신)
  run.log
  wrong/{val,test}/<true>/<pred>/*.png   오분류 샘플
  predictions/                 (--save-pred-samples) tp/fp/fn × class
```

`best_history.txt` 4-section:
```
[0] ★ BEST OVERALL          한 줄 TEST + VAL aggregate (FINAL 시점)
[1] FINAL BEST per-class    TEST(위)+VAL(아래) 33-class 표
[2] BEST UPDATES SUMMARY    best 갱신 epoch별 한 줄 요약 표
[3] PER-EPOCH PER-CLASS     best #1 → #N 시간순 per-class 상세
```

default `model_tag` = backbone short name (`convnextv2_base`).

폐지된 출력 파일 (이전 버전 호환 필요시 git history `441c532..571861c` 참고):
- `val_per_class_report.txt`, `test_per_class_report.txt`
- `best_confusion_matrix_val.png`, `best_confusion_matrix_test.png`
- `eval_summary.json`

## CLI 핵심 옵션

| flag | default | 의미 |
|---|---|---|
| `--epochs N` | 30 | total epoch |
| `--batch N` | 16 | batch size (1024 → batch 2 권장 on 16GB GPU) |
| `--img-size N` | 384 | BICUBIC resize target |
| `--model-tag T` | `convnextv2_base` | 결과 폴더 prefix |
| `--subset-config PATH` | None | YAML로 class별 sample 수 제한 |
| `--train-val-only` | OFF | test split 없이 80/20 (test 평가 skip) |
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
| `--ram-limit / --gpu-mem-limit` | 80 / 90 | watchdog 한계 (%) |

## subset YAML 형식

```yaml
classes:
  Donut_scratch: 30
  Center_bank_boundary: 50
  default: 200
```

`default` 미지정 시 해당 class는 전체 sample 사용. 기존 yaml:
- `experiments/quick.yaml` — default 20 (smoke)
- `experiments/ablation_size_n50.yaml` — default 50 (size ablation)
- `experiments/imbalance_minor3.yaml` — minor 3 class 줄임

## 실행 예

```bash
# Quick smoke (2 epoch, 660 sample)
python cnn_train.py --epochs 2 --subset-config experiments/quick.yaml --batch 8 --model-tag smoke

# 기본 학습
python cnn_train.py --epochs 30 --batch 16 --model-tag baseline

# subset 학습 (size ablation 384)
python cnn_train.py --epochs 30 --subset-config experiments/ablation_size_n50.yaml \
    --img-size 384 --batch 16 --model-tag sz384_n50

# Loss A/B 비교 (동일 seed)
python cnn_train.py --epochs 30 --class-weight none      --model-tag exp_no_cw
python cnn_train.py --epochs 30 --class-weight effective --model-tag exp_eff_cw
python cnn_train.py --epochs 30 --loss focal --focal-gamma 2 --model-tag exp_focal

# 사이즈 ablation runner (PowerShell)
.\experiments\run_size_ablation_trainval.ps1
```

## 결정 트리

- **Class 불균형 큼?** → `--class-weight effective` (default) 또는 `--loss focal`
- **Hard sample 많음?** → `--loss focal --focal-gamma 2.0`
- **Spike epoch 자주?** → `--val-loss-guard 1.5` 더 엄격히
- **Best 너무 자주 갱신 (lucky epoch)?** → `--val-smooth-window 5`로 더 smoothing
- **OOM at 1024?** → `--batch 2` (4060 Ti 16GB 기준 한계)

## Mid-run 종료 대응

매 epoch 갱신되는 산출물 (= 학습 도중 죽어도 보존):
- `curves.png` (loss/F1 plot)
- `history.json` (전 epoch metrics)

best 갱신 시점 갱신:
- `best_model.pth`, `best_history.txt`, `best_confusion_matrix.png`, `wrong/`

폴더 rename은 학습 종료 시에만 적용 (mid-run 중에는 `_running` suffix 유지).

## 금지

- `EXCLUDE_CLASSES = {"Normal"}` 상수 변경 금지 (open-set 정책)
- `log/<run>/` 결과 폴더 삭제 금지 (사용자 명시 요청 전)
- VFlip / 180° rotation / ColorJitter 추가 금지 (도메인 의미 손상)
- `BACKBONE` 상수 변경 시 비교 가능하도록 `--model-tag`로 구분
