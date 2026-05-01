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

## Adaptive resource policy (한도 + 자동 batch tuning)

**한도 (사용자 정책)**:
- CPU ≤ 90%
- RAM ≤ 80%
- GPU mem ≤ 90%
- GPU util — soft target 90% (낮으면 batch 더 키워 활용 ↑)

**자동 batch tuning (각 stage 시작 시)**:
1. nvidia-smi 로 GPU mem.free / util 측정 + psutil 로 RAM/CPU
2. 모든 자원 한도 안이면 batch 상향 시도:
   - GPU mem free > 4 GB → batch ×2 가능
   - GPU util < 60% → workers 늘리거나 batch 키움
3. 한도 초과 시 batch ↓:
   - GPU OOM → batch /2 retry
   - RAM > 80% → workers 줄이거나 prefetch 끔
4. 각 stage 별 출발 batch:
   - chip 학습 (Stage 1) — batch 16, 384 input
   - obj_id build (Stage 2) — batch 128 (RTX 4060 Ti 16 GB 기준)
   - compound 학습 (Stage 3) — batch 16 (3-channel + augment + train)
   - wafer baseline 학습 (Stage 4) — batch 16 (cnn_train.py 동일)

## Auto-progression (stage 순차 자동 dispatch)

이 run 의 표준 순서 (사용자 결정):

```
[Stage 2]  obj_id_maps 전체 빌드 (~수 시간)
[Stage 3a] compound n=50  → logs_compound/compound_n50_*
[Stage 4a] wafer baseline n=50 (cnn_train.py 동일 subset, R only)
                              → logs_wafer/wafer_baseline_n50_*
[Stage 3b] compound n=100 → logs_compound/compound_n100_*
[Stage 4b] wafer baseline n=100 → logs_wafer/wafer_baseline_n100_*
```

각 stage 완료 후 자동:
- `logs_*/overall/` 갱신 확인 (val F1 best 면 통째 복사됨)
- `git add` (코드/docs 변경 있으면) + `commit -m "[Stage X] <model_tag> result: test_f1=...,val_f1=..."` + `push`
- 다음 stage 시작 전 자원 재점검 → batch 동적 결정

비교 표 (최종 보고): n=50 / n=100 각각 compound vs wafer baseline test/val F1 차이.

## Git commit/push policy

Stage 별 산출은 logs_*/ 가 gitignored 라 push 안 됨. 코드/docs 변경만 push.

각 stage 완료 시:
- 산출 path 확인 + `best_history.txt` 의 BEST OVERALL 추출
- commit message:
  ```
  [Stage 3a] compound_n50 → test F1 0.xx / val F1 0.yy
  
  logs_compound/compound_n50_<TS>_<f1>_<f1>/best_history.txt:
    TEST  acc=...  f1=...  p=...  r=...
    VAL   acc=...  f1=...  p=...  r=...
  ```
- 코드/docs 수정 없으면 commit 자체 skip (산출 외 변경 없음)

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
