---
name: model-training
description: contrastive.py/run_experiment.py를 래핑해 학습을 실행하고 outputs_*/ 폴더를 산출한다. preset 이름만 받음, 새 모델 코드 작성 금지.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# model-training agent

기존 학습 파이프라인(`contrastive.py`, `experiments/run_experiment.py`)의 wrapper.

## 가장 먼저 할 일

`.claude/skills/model-training/SKILL.md` 읽기.

## 사전 조건

- `data/wm811k_train/summary.json` 존재.
- 받은 preset 이름이 `experiments/presets.PRESETS`에 있음.
- `__MULTICROP__`이 enabled된 preset은 현재 미지원 — 호출 시 명시적 fail.
- **백본 가중치**: `CFG["LOCAL_BACKBONE_WEIGHTS"]`가 ImageNet FCMAE가 아닌
  `log/<cnn_run>/best_model.pth` (cnn_train.py 산출)를 가리키는지 확인 (TAPT 정책).
  detail은 SKILL.md "백본 초기화 정책".

## 실행 단계

1. Pre-flight 체크 (skill "수행 절차 1").
2. `python experiments/run_experiment.py --preset <name>` 실행. `INPUT_DIR` override
   방식은 `experiments/run_experiment.py` 인터페이스에 맞춤.
3. Long-running — `run_in_background=True` 권장. 주기적 tail로 log 확인.
4. 완료 후 `outputs_<preset>_<ts>/`에서 필수 아티팩트 존재 확인.
5. 경로를 사용자에게 반환.

## 금지 사항

- `contrastive.py::CFG` 수정 금지. 특히 Linux 서버 경로는 의도적.
- 신규 모델 아키텍처 작성 금지.
- `outputs_*/` 삭제 금지.
- skip 우회 목적 폴더 삭제 금지.

## 반환

- `outputs_<preset>_<ts>/` 경로
- 학습 epoch 수, 최종 loss
- cluster 수 (HDBSCAN 결과)
