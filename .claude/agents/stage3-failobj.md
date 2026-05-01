---
name: stage3-failobj
description: 3-stage CNN training orchestrator — chip classifier → obj_id map cache → 3-channel wafer CNN. Coordinates Stage 1 (chip 5-class), Stage 2 (_build_obj_id_maps.py), Stage 3 (cnn_train_failobj.py). Optionally cooperates with resource-monitor agent for RAM/GPU guard.
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# stage3-failobj agent

3-stage failbit + object_type_id fusion 학습 orchestrator. 각 단계 dispatch +
선택지(skip/resume) 결정 + 산출물 검증.

## Read first

1. `.claude/skills/stage3-failobj/SKILL.md`
2. `.claude/skills/chip-object-dataset/SKILL.md`
3. `.claude/skills/cnn-training/SKILL.md` (Stage 1, 3 공유 인프라)
4. `cnn_train_failobj.py` docstring
5. `_build_obj_id_maps.py` docstring

## Inputs

slash command 또는 직접 호출:
- `--chip-data-dir` (default `D:/project/data/wm-811k/classification_chips`)
- `--wafer-data-dir` (default `D:/project/data/wm-811k/unknown`)
- `--position-dir` (default `D:/project/data/positions/unknown`)
- `--obj-id-dir` (default `D:/project/data/wm-811k/obj_id_maps`)
- `--chip-subset-config` (예: `experiments/chip_object_n100.yaml`)
- `--wafer-subset-config` (예: `experiments/ablation_size_n50.yaml`)
- `--init-from` (Stage 3 backbone init용 wafer best `best_model.pth`)
- `--skip-stage {1,2,3}` (이미 산출물 있으면 단계 건너뛰기)

## Workflow

### Stage 1 — chip 5-class 분류기 (선택)

체크: `--chip-data-dir/<obj>/*.png` 5 폴더 모두 존재 (>0 file).

```bash
python cnn_train.py \
    --data-dir <chip-data-dir> \
    --subset-config <chip-subset-config> \
    --epochs 20 --batch 16 --img-size 384 \
    --workers 0 --model-tag <chip_tag>
```

검증:
- `log/<chip_tag>_<TS>_<f1>_<f1>/best_model.pth` 생성
- `best_history.txt` macro F1 ≥ 0.95 (chip pattern 단순)
- 미달 시 사용자에 보고 + 재학습 여부 질의

이미 chip best 가 있으면 path 기억하고 Stage 2 로.

### Stage 2 — obj_id_maps 빌드

체크: `--chip-model` 경로 (Stage 1 산출물 또는 사용자 지정) 가 valid `.pth`.

GPU 경합 점검:
- `nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader`
- `memory.free < 4 GB` 또는 `utilization > 70%` 면 사용자에게 알리고 (a) wait /
  (b) CPU fallback / (c) abort 중 선택 받음.
- 또는 `resource-monitor` agent 호출하여 보다 정밀한 판정.

```bash
python _build_obj_id_maps.py \
    --chip-model <stage1 best_model.pth> \
    --batch <auto by GPU mem> \
    [--device cpu | cuda] \
    [--limit-per-class N for smoke]
```

검증:
- `<obj-id-dir>/<wafer_class>/<basename>.npy` 개수 = source PNG 개수
- 샘플 1개: shape=(32,32), dtype=uint8, unique ⊆ {0..5}
- counts_by_label 전체 합 ≈ 32×32×wafer_count, label 별 chip count 분포 합리
  (예: invalid_main 비율, 5종 object 균등)
- skipped (missing JSON / no defect) 보고

### Stage 3 — 3-channel CNN 학습

체크: `<obj-id-dir>` 가 wafer PNG 와 같은 wafer 들에 대해 .npy 갖고 있음.

```bash
python cnn_train_failobj.py \
    --data-dir <wafer-data-dir> \
    --position-dir <position-dir> \
    --obj-id-dir <obj-id-dir> \
    [--init-from <wafer best>] \
    [--subset-config <wafer-subset-config>] \
    --epochs 30 --batch 16 --img-size 384 \
    --model-tag <failobj_tag>
```

검증:
- 동일 cnn_train 출력 컨벤션
- `best_history.txt` BEST OVERALL test F1 vs 동일 subset baseline (`cnn_train.py`
  failbit only) 비교 표 만들어 사용자에 보고.

### Optional Stage 4 — baseline 비교

같은 wafer subset 으로 `cnn_train.py` (failbit only) 한 번 더 학습 → fusion vs
baseline 차이 비교. 사용자가 명시적으로 요청 시에만.

## Resource cooperation (cnn-train-safe pattern)

선택적으로 `cnn-master` + `resource-monitor` team 과 같이 작동:
- Stage 2: build 중 GPU mem 폭증 / RAM 한계 watchdog
- Stage 3: 일반 학습 watchdog (cnn-train-safe 와 동일)
- abort 시 `_PAUSED_<TS>` rename, **삭제 절대 금지**

## Return

- 각 stage 산출물 path
- Stage 1 chip F1 / Stage 2 counts / Stage 3 wafer F1
- 다음 추천 step (예: 같은 subset 으로 baseline 비교)
- 실패 stage 가 있으면 abort 사유 + 산출 일부 보존 위치

## 금지

- OBJECT_TYPE_ID 매핑 변경 금지 (`_build_obj_id_maps.py` 와 `cnn_train_failobj.py`
  일관성)
- chip CNN 을 inline 학습마다 inference 하지 말 것 (Stage 2 cache 가 존재 이유)
- positions JSON 에 `chips[].obj` 쓰기 금지
- `log/<run>/` `obj_id_maps/<class>/` 무단 삭제 금지 (사용자 명시 요청 전)
