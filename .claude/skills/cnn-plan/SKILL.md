---
name: cnn-plan
description: CNN 학습 계획 수립 — hyperparameter, subset YAML, ablation 실험 설계, 이전 실험 history 참고해서 다음 run 구성 제안.
---

# cnn-plan skill

이 스킬은 사용자 학습 목표에 맞춰 `cnn_train.py` 실행 plan을 설계한다.

## 가장 먼저 읽기

| 문서 | 용도 |
|---|---|
| `.claude/skills/cnn-training/SKILL.md` | CLI 옵션·기본값 spec (변경하지 말 것) |
| `cnn_train.py` (repo root) | 실제 구현 — 옵션 동작 확인 |
| `log/` 폴더 (있다면) | 이전 실험 결과 참고 |
| `D:\project\anomaly-detection\docs\summary.md` | hparam BKM (focal/EMA/grad clip 등 검증된 default) |

## 입력 → 산출

| 입력 | 산출 |
|---|---|
| 사용자 목표 (예: "baseline 학습", "minor class 시뮬레이션", "EMA on/off A/B") | 1. 실행 명령어 (CLI flag 조합)<br>2. (필요 시) `subset.yaml` 파일<br>3. 예상 시간·디스크<br>4. 비교 plan (다음 run 제안 포함) |

## 학습 plan 패턴

### 1. Baseline 학습

```bash
python cnn_train.py --epochs 30 --batch 16 --model-tag baseline
```
- 모든 default 값 (effective class weight, EMA on, label_smoothing 0.02)
- 기준값 확보용. ~2-3시간.

### 2. Class imbalance 시뮬레이션

```yaml
# experiments/imbalance_v1.yaml
classes:
  Donut_scratch: 30
  Edge-Bottom_scratch: 30
  Loc_invalid_main: 50
  default: 200
```
```bash
python cnn_train.py --epochs 30 --subset-config experiments/imbalance_v1.yaml \
    --model-tag imbal_v1_eff_cw --class-weight effective
```
A/B 짝꿍:
```bash
python cnn_train.py --epochs 30 --subset-config experiments/imbalance_v1.yaml \
    --model-tag imbal_v1_no_cw --class-weight none
```

### 3. Loss A/B/C 비교 (동일 seed, 동일 subset)

```bash
# A: CE + effective class weight (default)
python cnn_train.py --epochs 30 --model-tag loss_ce_eff
# B: focal loss
python cnn_train.py --epochs 30 --loss focal --focal-gamma 2.0 --model-tag loss_focal
# C: weighted sampler
python cnn_train.py --epochs 30 --weighted-sampler --class-weight none --model-tag loss_ws
```

### 4. Quick smoke test (~5분)

```yaml
# experiments/quick.yaml
classes:
  default: 20
```
```bash
python cnn_train.py --epochs 2 --subset-config experiments/quick.yaml --batch 8 --model-tag smoke
```
- 33×20 = 660 sample, 2 epoch
- 모든 산출물 (hparams, confusion matrix, per_class_report) 파일 존재 검증용

### 5. Hyperparameter sweep (manual)

| Run | flag | 가설 |
|---|---|---|
| `lr_high` | `--lr-head 2e-3 --lr-backbone 2e-5` | 더 빠른 수렴 |
| `lr_low`  | `--lr-head 5e-4 --lr-backbone 5e-6` | 안정성 ↑ |
| `aug_strong` | `--label-smoothing 0.05` | regularization ↑ |
| `larger_img` | `--img-size 512` | spatial detail ↑ |

각 run은 `--model-tag`로 폴더 이름 차별화 → 비교 가능.

## 의사결정 트리

| 사용자 발화 → | 권장 plan |
|---|---|
| "처음부터 학습" | Baseline (#1) |
| "minor class 잘 맞추는지" | Imbalance subset + class_weight A/B (#2) |
| "어떤 loss 좋은지" | Loss A/B/C (#3) |
| "빨리 검증" | Quick smoke (#4) |
| "성능 더 짜내야" | Phase 2 옵션 (CutMix, two-stage, larger img) |

## subset YAML 파일 위치

- `experiments/<plan_name>.yaml` (repo root에 `experiments/` 만들기)
- 예: `experiments/imbalance_minor3.yaml`, `experiments/quick.yaml`
- git ignore 안 함 — 실험 spec 보존

## 이전 실험 참고법

```bash
# 결과 폴더 list
ls log/ | sort

# 각 run의 best 성능
for d in log/*/; do
  python -c "import json; d=json.load(open('$d/eval_summary.json'))['test']; print('$d', d['weighted_avg']['f1-score'])"
done | sort -k2 -r
```

## 금지

- spec에 없는 새 CLI flag 임의 추가 금지 — 먼저 `cnn_train.py` 수정 plan 후
- `EXCLUDE_CLASSES = {"Normal"}` 변경 금지
- 이전 run 폴더(`log/<run>/`) 삭제 금지

## 반환 형식

사용자에게 보고:
1. 추천 명령어 (정확한 flag set)
2. 생성한 subset YAML path (있다면)
3. 예상 시간·디스크
4. 비교용 짝꿍 명령 (필요 시)
5. 결과 분석 단계 → `cnn-analyze` skill 호출 권장
