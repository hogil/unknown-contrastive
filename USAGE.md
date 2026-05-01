# Usage Guide — wafer fail-bit synthesis · CNN classifier

전체 파이프라인 사용법. 한 번씩 따라하면 fresh clone에서도 처음부터 끝까지 돌아갑니다.

---

## 1. 파이프라인 개요

```
┌─────────────────────┐     ┌────────────────────┐     ┌─────────────────────┐
│ _sample_gen.py      │     │ cnn_train.py       │     │ cnn_predict.py      │
│ 또는 _sample_gen_gpu│ ──→ │ 33 class supervised│ ──→ │ threshold-based     │
│                     │     │ ConvNeXtV2 base    │     │ Normal/unknown 분류 │
│ 6400×6400 palette   │     │ + EMA + class      │     │                     │
│ PNG + JSON          │     │ weight (effective) │     │ best_model.pth 사용 │
└─────────────────────┘     └────────────────────┘     └─────────────────────┘
        │                            │                          │
        ↓                            ↓                          ↓
  D:/project/data/             log/<run_dir>/             preds.json
  wm-811k/unknown/             best_model.pth             per_class_report.txt
  positions/unknown/           best_history.txt
                               best_confusion_matrix.png
                               curves.png
                               history.json
```

**1회성 사전 단계** (fresh clone 시):

```bash
# backbone weight 한번 다운로드 (≈340 MB, models/ 에 mirror)
python download_backbone.py
```

heatmap (`_dist_heatmaps/`) 은 repo 에 포함돼 있어 별도 작업 불필요.

---

## 2. 이미지 생성

### 2.1. 기본 (multiprocessing CPU)

```bash
# 클래스당 1장 (smoke 36장, 약 2분)
python _sample_gen.py --n 1 --workers 4

# 본 생성 (클래스당 200장 = 약 6800장, 약 30-40분, --workers 8)
python _sample_gen.py --n 200 --workers 8
```

옵션:
- `--n N`: 클래스당 샘플 수 (default 200)
- `--workers W`: 병렬 worker (default 4)

### 2.2. GPU 가속 (single-process + ThreadPool)

```bash
# 클래스당 200장 (CPU 대비 GPU에서 약 3-4배 빠름)
python _sample_gen_gpu.py --n 200 --save-workers 8

# 특정 distribution만 (예: Normal class만 5000장)
python _sample_gen_gpu.py --n 5000 --only-class Normal --save-workers 12

# seed offset — 기존 파일과 충돌 안 나게 추가 생성
python _sample_gen_gpu.py --n 100 --seed-offset 100000
```

옵션:
- `--n N`: 클래스당 샘플 수
- `--save-workers W`: PNG/JSON 저장 thread pool (default 8)
- `--only-class CLS`: 한 distribution만 생성 (Center / Donut / Edge-Ring / Edge-Bottom / Edge-Top / Full / Thick-Edge / Normal / Starburst / CommaCluster)
- `--seed-offset N`: seed 시작 오프셋 (filename collision 회피)

### 2.3. 출력 위치

```
D:/project/data/wm-811k/unknown/<class>/<filename>.png
D:/project/data/positions/unknown/<class>/<filename>.json
```

각 JSON 자동 포함: `partid`/`part_id`/`pgm`/`ftn_keys`(128)/`qtn_keys`(128) + chip별 `f`/`q` (128 dense).

### 2.4. 검증

```bash
# 클래스당 5장 random sampling 검증 (구조 + JSON 스키마 + bin pool + cross-validation)
python _verify.py --sample 5

# strict 모드 — FTN/QTN 분석성 (defect chip hot ratio ≥3x) 추가 검증
python _verify.py --sample 10 --strict

# 특정 클래스만
python _verify.py --class Center_bank_boundary --sample 20

# 전체 (11600장 모두) — 시간 걸림
python _verify.py --sample 0
```

출력에는 클래스별 **defect chip 개수 분포** 표가 포함돼서 generation sanity 체크 가능.

---

## 3. CNN 분류기 학습

### 3.1. Quickstart

```bash
# smoke test — 33 class × 20 sample = 660 (약 5분)
python cnn_train.py \
    --epochs 2 \
    --subset-config experiments/quick.yaml \
    --batch 8 \
    --model-tag smoke

# baseline — 전체 데이터, 30 epoch
python cnn_train.py \
    --epochs 30 \
    --batch 16 \
    --model-tag baseline
```

### 3.2. Subset YAML 활용 (class imbalance 시뮬, 데이터 절약)

`experiments/<plan>.yaml`:
```yaml
classes:
  default: 50                 # 모든 클래스 50장 cap
  Donut_scratch: 30           # 특정 클래스만 30 (override)
  Edge-Bottom_scratch: 30
```

```bash
python cnn_train.py \
    --epochs 30 \
    --subset-config experiments/ablation_size_n50.yaml \
    --img-size 512 \
    --batch 8 \
    --model-tag sz512_n50
```

기존 yaml들:
- `experiments/quick.yaml` — 클래스당 20 (smoke)
- `experiments/ablation_size_n50.yaml` — 클래스당 50 (size ablation)
- `experiments/imbalance_minor3.yaml` — 3 minor class만 30, 나머지 200

### 3.3. 사이즈 ablation (PowerShell runner)

```powershell
# 384 / 512 / 1024 순차 실행 (배치 자동 조절)
.\experiments\run_size_ablation_trainval.ps1
```

### 3.4. 주요 학습 인자

| 인자 | default | 설명 |
|---|---|---|
| `--epochs` | 30 | epoch 수 |
| `--batch` | 16 | batch size (1024×1024는 batch 2 권장) |
| `--img-size` | 384 | 입력 해상도 (BICUBIC resize) |
| `--model-tag` | `convnextv2_base` | 결과 폴더 prefix |
| `--subset-config` | None | YAML subset 경로 |
| `--lr-backbone` / `--lr-head` | 1e-5 / 1e-3 | learning rate |
| `--class-weight` | `effective` | `none` / `inverse` / `effective` |
| `--ema` / `--no-ema` | on | EMA shadow weights |
| `--ema-decay` | 0.95 | EMA decay |
| `--patience` | 7 | early stop |
| `--train-val-only` | off | val/test 미분리 (test 평가 없음) |
| `--save-pred-samples` | off | 끝에 TP/FP/FN 샘플 저장 |
| `--ram-limit` / `--gpu-mem-limit` | 80 / 90 | watchdog 한계 (%) |

### 3.5. 출력 구조

```
log/<model_tag>_<YYMMDD_HHMMSS>_<test_f1>_<val_f1>/
├─ best_history.txt              ← 통합 결과 (4 sections, 아래 참고)
├─ best_confusion_matrix.png     ← test(위)+val(아래) combined, 셀 숫자
├─ best_model.pth                ← state_dict + classes + ema_state
├─ history.json                  ← 모든 epoch metrics (매 epoch 갱신)
├─ curves.png                    ← train_loss/val_loss/val_macro_f1 (매 epoch 갱신)
├─ hparams.{yaml,txt}
├─ run.log
└─ wrong/{val,test}/<true>/<pred>/*.png   ← 오분류 샘플
```

`best_history.txt` 4-section 구조:
```
[0] ★ BEST OVERALL          ← 맨 윗줄 한 줄 요약 (TEST + VAL)
[1] FINAL BEST per-class    ← 최종 best의 TEST(위) + VAL(아래) 33-class 표
[2] BEST UPDATES SUMMARY    ← best 갱신 epoch별 한 줄 요약 표
[3] PER-EPOCH PER-CLASS     ← best #1 → #N 시간순 per-class 상세
```

폴더명에 `_<test_f1>_<val_f1>` suffix는 학습 종료 시 rename 적용.
`--train-val-only` 모드는 test 없이 `_<val_f1>` 한 개만.

---

## 4. CNN 예측 (inference)

### 4.1. 기본 — 폴더 분류

```bash
# 폴더 내 모든 PNG 예측 → JSON 출력
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input D:/project/data/wm-811k/unknown/Center_bank_boundary \
    --output preds.json

# 단일 이미지
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input D:/project/data/wm-811k/unknown/Donut_scratch/abc123_00C_05_..._PE_NORMAL.png \
    --output single_pred.json
```

### 4.2. Threshold 기반 Normal/unknown 분류

`max_prob < threshold` 인 샘플은 `Normal/unknown` 으로 분류.

```bash
# threshold 0.7 — 모델 신뢰도 70% 미만이면 Normal
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input <test_dir> \
    --threshold 0.7 \
    --output preds.json
```

### 4.3. Threshold sweep

입력이 `{class}/img.png` 구조면 ground-truth label 추정 가능 → threshold별 metric 계산.

```bash
# 0.1 ~ 0.9 까지 0.05 step 으로 sweep
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input D:/project/data/wm-811k/unknown \
    --threshold-sweep 0.1,0.9,0.05 \
    --output preds.json
```

### 4.4. per-class report 동시 생성

```bash
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input D:/project/data/wm-811k/unknown \
    --report-out per_class_report.txt \
    --output preds.json
```

### 4.5. EMA shadow weights 사용

```bash
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input <dir> \
    --ema \
    --output preds_ema.json
```

### 4.6. 오분류 샘플 폴더 별도 저장

```bash
python cnn_predict.py \
    --model log/<run>/best_model.pth \
    --input D:/project/data/wm-811k/unknown \
    --save-wrong-out wrong_predict/ \
    --output preds.json
```

### 4.7. 주요 추론 인자

| 인자 | 설명 |
|---|---|
| `--model` | best_model.pth 경로 (필수) |
| `--input` | 단일 이미지 또는 폴더 (필수) |
| `--output` | 예측 결과 JSON (생략 시 stdout) |
| `--threshold T` | max_prob < T 이면 Normal/unknown |
| `--threshold-sweep "lo,hi,step"` | threshold 그리드 sweep |
| `--report-out PATH` | `{class}/img.png` 구조면 per-class report 출력 |
| `--ema` | checkpoint의 ema_state로 모델 가중 swap |
| `--batch` | inference batch (default 16) |
| `--workers` | DataLoader worker (default 4) |
| `--save-wrong-out DIR` | 오분류 샘플 폴더 트리 저장 |
| `--no-recursive` | 폴더 입력 시 하위 폴더 탐색 안 함 |
| `--device` | `cuda` / `cpu` 지정 (default auto) |

---

## 5. 전형적 워크플로우

### A. Fresh clone → 첫 결과까지

```bash
# 0. 사전 (1회성)
python download_backbone.py

# 1. 합성 데이터 생성 (예: 클래스당 200장)
python _sample_gen.py --n 200 --workers 8

# 2. 검증
python _verify.py --sample 5

# 3. CNN 학습
python cnn_train.py --epochs 30 --batch 16 --model-tag baseline

# 4. 결과 확인
cat log/baseline_*/best_history.txt | head -60

# 5. inference (운영 threshold 결정용)
python cnn_predict.py \
    --model log/baseline_*/best_model.pth \
    --input D:/project/data/wm-811k/unknown \
    --threshold-sweep 0.1,0.9,0.05 \
    --report-out sweep_report.txt \
    --output preds_sweep.json
```

### B. 입력 사이즈 ablation

```bash
# subset YAML (클래스당 50)
# experiments/ablation_size_n50.yaml — default: 50

# PowerShell runner: 384/512/1024 순차
.\experiments\run_size_ablation_trainval.ps1

# 결과 폴더 비교
ls log/sz*_n50_*/best_history.txt | foreach { head -5 $_ }
```

### C. Class imbalance 효과 비교

```bash
# 같은 subset, class_weight 옵션만 다르게 두 run 돌려 비교
python cnn_train.py --epochs 30 \
    --subset-config experiments/imbalance_minor3.yaml \
    --class-weight effective \
    --model-tag imbal_eff_cw

python cnn_train.py --epochs 30 \
    --subset-config experiments/imbalance_minor3.yaml \
    --class-weight none \
    --model-tag imbal_no_cw
```

### D. 운영 threshold 선택

```bash
# 1) Normal pool 별도 생성 (또는 inference 전용 분리 데이터)
# 2) sweep 으로 max_prob 분포 확인

python cnn_predict.py \
    --model log/<chosen_run>/best_model.pth \
    --input D:/project/data/wm-811k/unknown/Normal \
    --threshold-sweep 0.5,0.95,0.05 \
    --output normal_pool_preds.json
```

threshold 추천: Normal pool 에서 `max_prob` 90 percentile + ε 정도가 starting point.

---

## 6. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| OOM at `--img-size 1024` | batch 2 또는 1로 낮추기. 4060 Ti 16GB 기준 batch 4 가 한계 |
| heatmap 못 찾음 | `_dist_heatmaps/` 가 repo 에 있는지 확인. fresh clone 후 `git pull` |
| backbone weight 다운로드 실패 | `python download_backbone.py` 재시도, 폐쇄망이면 `models/` 폴더 직접 복사 |
| CRLF / utf-8 경고 | Windows 정상. 무시 가능. |
| `_running` suffix 폴더 남음 | 학습 중간에 죽음 — 결과 일부 보존됨 (`history.json`, `curves.png`, `best_history.txt`). 절대 삭제 금지 |
| `eval_summary.json` 없음 | 의도된 변경 — 모든 결과는 `best_history.txt` 에 통합 |
| FTN/QTN 누락 | 신규 generation은 자동 포함. 다른 데서 들여온 옛 JSON은 git history `441c532..fed8c24` 의 `_backfill_fq_positions.py` 참고 |

---

## 7. 절대 금지

- `log/<run>/` 폴더 무단 삭제 (실험 결과 손실)
- `models/<backbone>.pth` 무단 삭제 (재다운로드 필요)
- `D:/project/data/wm-811k/unknown/`, `D:/project/data/positions/unknown/` 무단 삭제
- 생성된 JSON에 `transparency=31` PNG save (모델 입력 픽셀 손실)

상세 설계 사양은 `docs/image-generation/` 참고.
