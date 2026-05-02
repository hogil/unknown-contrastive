# Usage Guide — wafer fail-bit synthesis · CNN classifier

전체 파이프라인 사용법. 한 번씩 따라하면 fresh clone에서도 처음부터 끝까지 돌아갑니다.

---

## 1. 파이프라인 한 눈에

```
┌─────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│ Stage A: 데이터 합성     │    │ Stage B: 학습             │    │ Stage C: 추론             │
│ _sample_gen.py          │ →  │ cnn_train_chip.py         │ →  │ cnn_predict_chip.py       │
│ _sample_gen_gpu.py      │    │ cnn_train_wafer.py        │    │ cnn_predict_wafer.py      │
│                         │    │ cnn_train_compound.py     │    │ cnn_predict_compound.py   │
│ 36-class palette PNG +  │    │                           │    │                          │
│ positions JSON +        │    │ ConvNeXtV2 base FCMAE     │    │ overall/best_model.pth   │
│ chip 200x200 crops      │    │ 모두 supervised, EMA+CW   │    │ + threshold + sweep      │
└─────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                                          ↓ logs_<kind>/<run>/                ↓ logs_predict_<kind>/<TS>/
                                          best_model.pth                       preds.json
                                          history.json                         per_class_report.txt
                                          curves.png                           wrong/<true>/<pred>/*.png
                                          + overall/ (val F1 best mirror)
```

**1회성 사전 단계** (fresh clone 시):

```bash
python download_backbone.py     # ConvNeXtV2 FCMAE pretrain (~340 MB → models/)
python _dist_learn.py           # WM-811K cca/* heatmap (~10s, → _dist_heatmaps/)
```

---

## 2. 6 entry script — kind 별 학습 + 추론 (정식 진입점)

| 분야 | 학습 entry | 추론 entry | data root | log root | predict root |
|---|---|---|---|---|---|
| **chip** 5-class object | `cnn_train_chip.py` | `cnn_predict_chip.py` | `data/wm-811k/classification_chips/` | `logs_chip/` | `logs_predict_chip/<TS>_<input>/` |
| **wafer** 33-class R-only | `cnn_train_wafer.py` | `cnn_predict_wafer.py` | `data/wm-811k/unknown/` | `logs_wafer/` | `logs_predict_wafer/<TS>_<input>/` |
| **compound** 33-class R+G | `cnn_train_compound.py` | `cnn_predict_compound.py` | `data/wm-811k/unknown/` + `obj_id_maps/` | `logs_compound/` | `logs_predict_compound/<TS>_<input>/` |

각 wrapper 맨 위 `# === CONFIG ===` 섹션에 default 노출. 명시 인자 (`--data-dir`, `--model`, `--input` 등) 로 override 가능.

**추론 자동 동작 (3 predict wrapper 공통):**
- `--model` 생략 → `logs_<kind>/overall/best_model.pth` 자동 로드
- 시작 시 `_overall_meta.json` 의 best_run / val_f1 / seeded_at stderr 출력
- `logs_predict_<kind>/<TS>_<input_name>/` 폴더 자동 (`--no-run-dir` 로 끄기). 내부 산출물:
  - `preds.json` — record per input (path, pred_class, max_prob, probs dict, true_class, is_pseudo, ...)
  - **`preds.csv` — wide table** (한 row = 한 input). cols (왼→오):
    - `path, basename`
    - basename `_` split — wafer/compound: `prefix, kind, w_idx, date, time, yld, syp, tester, device` / chip: `prefix, kind, w_idx, date, time, tester, device, gx_token, gy_token, b_token`
    - sibling JSON (wafer/compound 만): `partid, pgm, wafer, stime, step, yield, sys, tm, lt, netd, gd`  (`part_id` 는 `partid` 와 동일값이라 제외)
    - pred meta: `pred_class, pred_idx, max_prob, is_normal, is_pseudo, [obj_id_npy]`
    - label (있으면): `true_class, true_idx, correct`
    - 클래스별 확률: `prob_<class1>, prob_<class2>, ...`
  - `per_class_report.txt` — sklearn classification_report (label 추정 가능 시)
  - `threshold_sweep.csv` — `--threshold-sweep` 줬을 때만. cols: `threshold, normal_rate, acc_kept, kept_n`
  - `wrong/<true>/<pred>/*.png` — 틀린 예측 갤러리

---

## 3. 파일 호출 관계 (어떤 entry 가 어떤 `_*` 파일 부르나)

```
cnn_train_chip.py         (CONFIG: chip default)
    └── cnn_train.py      (engine — kwargs main())
        └── _resource_guard.py   (assess_start, ResourceMonitor)

cnn_train_wafer.py        (CONFIG: wafer default)
    └── cnn_train.py      (engine, 동일)

cnn_train_compound.py     (독립 — 3채널 dataset 자체 구현)
    └── _resource_guard.py
    └── (training 시) data/.../obj_id_maps/*.npy  +  _meta.json (n_chip_objects 분모)

cnn_predict_chip.py       (CONFIG: chip default)
    └── cnn_predict.py    (engine — kwargs main())

cnn_predict_wafer.py      (CONFIG: wafer default)
    └── cnn_predict.py    (engine, 동일)

cnn_predict_compound.py   (독립 — 3채널 dataset)
    └── (predict 시) data/.../obj_id_maps/*.npy  +  _meta.json

_build_obj_id_maps.py     (preprocessing inference: chip CNN → wafer obj_id 맵)
    └── logs_chip/.../best_model.pth   (학습된 chip 분류기 가중치)
    └── 출력: data/wm-811k/obj_id_maps/<device>_<date>/<basename>.npy + .png + predictions.csv

_orchestrator_compound_chain.py   (build → compound 학습 chain)
    └── _orchestrator_resource_guard.py   (sibling watchdog)
    └── cnn_train_compound.py             (build 끝나면 자동 dispatch)

_sample_gen.py / _sample_gen_gpu.py   (데이터 생성)
    └── _fq_metadata.py    (FTN/QTN keys / values)
    └── _dist_heatmaps/    (WM-811K cca/* 학습 heatmap, _dist_learn.py 산출)
    └── 출력: data/wm-811k/unknown/, classification_chips/, positions/unknown/
```

→ **사용자가 직접 부르는 건 6 entry + `_sample_gen*` + `_verify.py` + `_build_obj_id_maps.py` 정도**. 나머지 `_*.py` 는 위 entry 가 import 하거나 orchestrator 가 sibling 으로 띄우는 helper.

---

## 4. 3-stage compound 파이프라인

`cnn_train_compound.py` 는 chip 분류기 결과를 G 채널로 합쳐 학습한다. 3 단계:

```
Stage 1 │ chip 5-class 분류기 학습     │ python cnn_train_chip.py --epochs 30 --model-tag chip5
        │                              │ → logs_chip/<run>/best_model.pth + overall/
Stage 2 │ chip CNN inference →         │ python _build_obj_id_maps.py \
        │ wafer 마다 obj_id .npy 생성  │     --chip-model logs_chip/overall/best_model.pth \
        │                              │     --batch 128 --device cuda
        │                              │ → data/wm-811k/obj_id_maps/<device>_<date>/*.npy + predictions.csv
Stage 3 │ 3-channel compound CNN 학습  │ python cnn_train_compound.py --epochs 30 --model-tag compound33
        │ R=failbit/31, G=obj_id/N,B=0 │ → logs_compound/<run>/best_model.pth + overall/
```

`stage3-compound` agent + `_orchestrator_compound_chain.py` 가 Stage 2→3 자동 chain. Stage 2 build 끝나면 chain 이 즉시 cnn_train_compound.py dispatch + run_dir 에 산출물 archive (`predictions.csv`, `obj_id_maps_meta.json`, `counts.txt`, `compound_dispatch.log`).

---

## 5. pseudo-label (opt-in)

high-confidence prediction 결과를 다시 학습 데이터로 추가해서 다음 round 학습 시 self-improvement 시도. **default OFF**, 옵션 줘야 활성.

**작동 원리:**
1. 추론 시 각 입력의 max softmax prob (= confidence) 계산
2. confidence ≥ `--pseudo-label-threshold` 이면 (그리고 Normal/unknown 이 아니면)
3. 예측된 class 폴더에 입력 파일 복사: `<out>/<pred_class>/<basename><suffix>.<ext>`
4. `<suffix>` 는 default `_PSEUDO` — 원본 학습 데이터와 grep 으로 구분 가능
5. `--pseudo-label-cap-per-class N` 으로 class 당 최대 개수 제한 (불균형 방지)

**3개 entry 에 다 박혀있음:**

```bash
# chip — chip CNN 으로 chip crop 자기 라벨링 (build 단계에서)
python _build_obj_id_maps.py --chip-model logs_chip/overall/best_model.pth \
    --pseudo-label-threshold 0.97 --pseudo-label-cap-per-class 1000

# chip predict — 기존 chip 데이터셋 위에 high-conf 만 추가 라벨
python cnn_predict_chip.py --pseudo-label-threshold 0.97 --pseudo-label-cap-per-class 500

# wafer R-only — high-conf wafer 추가 라벨링
python cnn_predict_wafer.py --pseudo-label-threshold 0.95 --pseudo-label-cap-per-class 200

# compound — 동일
python cnn_predict_compound.py --pseudo-label-threshold 0.95 --pseudo-label-cap-per-class 200
```

**저장 결과:**
```
classification_chips/scratch/AAU220_..._X12_Y17_B286_PSEUDO.png    ← chip pseudo (build)
classification_chips/scratch/AAU220_..._X12_Y17_B286_PSEUDO.png    ← chip pseudo (predict)
unknown/Center_scratch/<wafer_basename>_PSEUDO.png                 ← wafer pseudo
```

CSV 에 `confidence`, `is_pseudo` 컬럼 추가됨 (`_build_obj_id_maps.py` 의 `predictions.csv`). predict scripts 도 `is_pseudo=true` 가 결과 JSON record 에 박힘.

**주의:**
- threshold 너무 낮으면 (e.g. 0.7) 모델이 자기 실수에 confidence 만 더 부여 → bias 누적. 권장 0.95+.
- `--pseudo-label-cap-per-class` 안 주면 쉬운 class (e.g. bank_boundary) 만 폭증 → class imbalance 왜곡.
- 다음 학습 round 시 `_PSEUDO` 만 골라 down-weight 하거나 별도 set 으로 분리 권장.

---

## 6. logs_obj/build_<TS>_<status>/ 컨벤션

`_build_obj_id_maps.py` 산출물의 archive 폴더. status suffix 로 한눈에 결과 식별:

```
logs_obj/
├─ build_260502_063934_ABORTED/    ← build 가 abort/사용자 kill 로 끝남
├─ build_260502_070940_ABORTED/    ← 동일
└─ build_260502_075725_OK/         ← 성공 — chain 이 _OK rename
```

각 폴더 내:
- `_meta.json` (orchestration meta — chip_model / batch / started_at)
- `build.log/err`, `guard.log/err`, `chain.log/err`
- 성공 시 추가: `obj_id_maps_meta.json`, `predictions.csv` 사본, `counts.txt`, `compound_dispatch.log`

`_orchestrator_compound_chain.py` 가 시작 시 status suffix 없는 옛 폴더 자동 sweep → `_ABORTED`.

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
