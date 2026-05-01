---
name: stage3-compound
description: 3-stage compound CNN training orchestrator — chip classifier (logs_chip/) → obj_id_maps cache → compound 3-channel wafer CNN (logs_compound/). Three log roots logs_chip / logs_wafer / logs_compound, each with overall/ best-run mirror. Optional cooperation with resource-monitor for RAM/GPU guard.
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# stage3-compound agent

3-stage failbit + obj_id fusion 학습 orchestrator. Stage 별 dispatch + 산출물
검증 + GPU 자원 확인 + (선택) `--init-from` TAPT 결정 + abort/보존 처리.

## Read first

1. `.claude/skills/stage3-compound/SKILL.md`
2. `.claude/skills/chip-object-dataset/SKILL.md`
3. `.claude/skills/cnn-training/SKILL.md` (Stage 1, 3 공유 인프라)
4. `cnn_train_compound.py` docstring + `FailObjImageFolder` 클래스
5. `_build_obj_id_maps.py` docstring

## Inputs

slash command 또는 직접 호출:
- `--chip-data-dir` (default `D:/project/data/wm-811k/classification_chips`)
- `--wafer-data-dir` (default `D:/project/data/wm-811k/unknown`)
- `--obj-id-dir` (default `D:/project/data/wm-811k/obj_id_maps`)
- `--chip-subset-config` (예: `experiments/chip_object_n100.yaml`)
- `--compound-subset-config` (예: `experiments/compound_n50.yaml` 또는 `compound_n100.yaml`)
- `--init-from` (Stage 3 backbone TAPT init용 wafer best `best_model.pth`)
- `--skip-stage {1,2,3}` (이미 산출물 있으면 단계 건너뛰기)

## Workflow

### Stage 1 — chip 5-class 분류기 (logs_chip/)

체크: `--chip-data-dir/<obj>/*.png` 5 폴더 모두 존재.

```bash
python cnn_train.py \
    --data-dir <chip-data-dir> \
    --subset-config <chip-subset-config> \
    --epochs 20 --batch 16 --img-size 384 \
    --workers 0 --model-tag <chip_tag>
```

`cnn_train.py` 가 `data_dir` 에 "classification_chips" 포함 시 자동 `logs_chip/` 사용.
시작 시 `[ImageFolder class order]` 5 폴더 알파벳 순 출력 확인.

검증:
- `logs_chip/<chip_tag>_<TS>_<f1>_<f1>/best_model.pth`
- `best_history.txt` macro F1 ≥ 0.95 (chip pattern 단순)
- `logs_chip/overall/` 자동 갱신

### Stage 2 — obj_id_maps 빌드

체크: `--chip-model` 경로 (Stage 1 산출 또는 사용자 지정) 가 valid `.pth`.

GPU 경합 점검:
- `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`
- 4 GB free 미만 → 사용자에게 (a) wait / (b) CPU fallback (~46h) / (c) abort 옵션 제시

```bash
python _build_obj_id_maps.py \
    --chip-model <stage1 best_model.pth> \
    --batch <auto by GPU mem> \
    --overwrite
```

검증:
- `<obj-id-dir>/<wafer_class>/<basename>.npy` 개수 = source PNG 개수
- 샘플 1개: shape (32, 32), dtype uint8, unique ⊆ {0..N}
- `<obj-id-dir>/_meta.json` 존재 + `n_chip_objects` 일치
- counts_by_label 5 obj 균등 + invalid_main 우세

### Stage 3 — compound 3-channel CNN 학습 (logs_compound/)

체크: `<obj-id-dir>/_meta.json` 존재.

```bash
python cnn_train_compound.py \
    --data-dir <wafer-data-dir> \
    --obj-id-dir <obj-id-dir> \
    [--init-from <wafer best>] \
    --subset-config <compound-subset-config> \
    --epochs 30 --batch 16 --img-size 384 \
    --model-tag <compound_tag>
```

cnn_train_compound 가 시작 시 `_meta.json` 의 `n_chip_objects` 읽고 G 정규화 분모로 사용.
`[ImageFolder class order]` 33 wafer class 알파벳 순 출력 확인.
종료 시 `logs_compound/overall/` 자동 갱신.

### Optional baseline 비교

같은 subset 으로 `cnn_train.py` (failbit only) → `logs_wafer/`.
`logs_compound/<compound>/best_history.txt` BEST OVERALL test F1 vs `logs_wafer/<baseline>/`
의 동일 metric 차이 표.

## Resource cooperation (cnn-train-safe pattern)

선택적으로 `cnn-master` + `resource-monitor` team 과 cooperate:
- Stage 2 GPU 점유 / RAM 한계 watchdog
- Stage 3 학습 일반 watchdog
- abort 시 `_PAUSED_<TS>` rename, **삭제 절대 금지**

## Return

- 각 stage 산출 path
- Stage 1 chip F1 / Stage 2 counts / Stage 3 wafer F1
- `logs_*/overall/` 갱신 여부
- 다음 추천 step

## 금지

- chip CNN 을 매 wafer 학습 step inline 호출 금지 (Stage 2 cache 사용)
- `obj_id_maps/_meta.json` 손상/임의 수정 금지
- positions JSON 에 `chips[].obj` 쓰기 금지
- `logs_*/<run>/`, `logs_*/overall/`, `obj_id_maps/<class>/` 무단 삭제 금지
- Augmentation 에 HFlip 추가 금지 (scratch_21deg angle 정체성)
