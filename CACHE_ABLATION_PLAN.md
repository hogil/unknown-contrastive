# Cache + Ablation 실행 plan (재부팅 후 이어가기)

작성: 2026-05-01. PC 재부팅 직전 상태 저장.

## 컨텍스트 요약

`cnn_train.py` 학습 시 GPU 활용률이 낮은 원인을 진단했고 (CPU data loading 병목,
6400×6400 PNG decode + resize = 219ms/img 측정), 해결책으로 입력 사이즈를 미리
다운샘플한 cache PNG를 디스크에 저장하기로 결정.

여러 후보 (NEAREST/BILINEAR/BICUBIC/LANCZOS/BOX/chip-aware) × 사이즈
(384/512/768/1024)를 시각 비교한 결과:

- **1024 BICUBIC** 채택 — chip 격자 + line/blob 모양 모두 보존, ORIG와 시각 거의 동일
- **chip-aware 폐기** — chip border 색은 분류 라벨에 이미 들어있어 신호로 불필요
- 384/512/768은 line 모양이 chip 안에 표현 불가능한 size라 불충분 (사용자 시각 확인)

다만 사용자 요청으로 **size ablation**(384/512/1024 BICUBIC)을 660 sample 짧은 학습으로
먼저 비교한 후 본 학습 size 확정 예정.

## 시각 비교 자료 (보존, untracked)

- `_resize_compare.py`           — 1 sample × 6 방식 (384)
- `_resize_compare4.py`          — 3 sample × {384,768} × {BOX, chip-aware}
- `_resize_compare1024.py`       — 3 sample × 6 방식 (1024)
- 결과 폴더 (각 grid.png + zoom_grid.png 들어있음):
  - `_resize_compare/`
  - `_resize_compare4/`
  - `_resize_compare1024/`

## 다음 세션 실행 단계

### Step 1. Cache 빌드 (1회, ~23분, ~8.3 GB 디스크)

slash command:
```
/make-cache 384,512,1024
```

또는 자연어로 "사이즈별로 샘플 만들어" / "1024 만들어" 식으로 요청.

raw 호출이 필요하면:
```bash
cd D:/project/unknown-contrastive
python _make_cache.py --sizes 384,512,1024 --interp bicubic --workers 12
```

출력:
- `D:/project/data/wm-811k/unknown_384_bicubic/<class>/*.png`  (~1.2 GB, ~3분)
- `D:/project/data/wm-811k/unknown_512_bicubic/<class>/*.png`  (~2.1 GB, ~5분)
- `D:/project/data/wm-811k/unknown_1024_bicubic/<class>/*.png` (~5.0 GB, ~15분)

스크립트는 이미 존재하는 PNG는 skip하므로 중간에 끊겨도 다시 실행하면 이어짐.

### Step 2. Ablation 학습 3회 (epoch 10, 각 660 sample)

먼저 GPU/RAM 상태 확인 (`nvidia-smi`, `free -h` 또는 `wmic`).

```bash
# 384 cache, batch 16
python cnn_train.py \
    --data-dir D:/project/data/wm-811k/unknown_384_bicubic \
    --img-size 384 --batch 16 --epochs 10 \
    --subset-config experiments/ablation_size.yaml \
    --model-tag ablation_384

# 512 cache, batch 12
python cnn_train.py \
    --data-dir D:/project/data/wm-811k/unknown_512_bicubic \
    --img-size 512 --batch 12 --epochs 10 \
    --subset-config experiments/ablation_size.yaml \
    --model-tag ablation_512

# 1024 cache, batch 4 (sweep 필요할 수 있음, OOM 안 나면 6/8 시도)
python cnn_train.py \
    --data-dir D:/project/data/wm-811k/unknown_1024_bicubic \
    --img-size 1024 --batch 4 --epochs 10 \
    --subset-config experiments/ablation_size.yaml \
    --model-tag ablation_1024
```

예상 시간: 384 ~3분, 512 ~5분, 1024 ~20분. 총 ~30분.

### Step 3. 비교

각 run의 `log/<run>/eval_summary.json` 비교:
- macro F1 (val + test)
- per-class F1 — 특히 line/blob 의존 클래스 (scratch, scratch_21deg, particle_blast, bank_boundary, invalid_main)
- 어떤 size에서 어떤 클래스가 가장 차이 나는지

### Step 4. 본 학습 (size 확정 후)

```bash
python cnn_train.py \
    --data-dir D:/project/data/wm-811k/unknown_<SIZE>_bicubic \
    --img-size <SIZE> --batch <BATCH> --epochs 30 \
    --model-tag baseline
```

예상 시간: 384 ~50분, 512 ~1.5시간, 1024 ~5-6시간 (full 11600 sample).

## 결정 보류 사항 (다음 세션에서 사용자 확인 필요)

- ablation epoch 수 (현재 plan: 10, 사용자 confirm 필요. 5나 15도 옵션)
- 1024 batch sweep — 4부터 시작해 OOM 없으면 6/8 시도. ConvNeXtV2-base @1024
  + channels_last + bf16 AMP 가정 (cnn_train.py default)
- BICUBIC 외 보간 (사용자 BICUBIC 선택했으나, ablation 결과 따라 BOX/LANCZOS 재고 가능)

## 시스템 사양 (재부팅 전 확인)

- GPU: RTX 4060 Ti 16GB (nvidia-smi 확인)
- RAM: 64 GB
- 데이터: 11600장 (33 class × ~350장 평균)
- Python/torch/timm: cnn_train.py에서 자동 사용
- backbone weights: `models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth`에 mirrored

## 파일 인벤토리 (재부팅 후 점검)

생성된 파일:
- `_make_cache.py`                                      — cache 빌드 스크립트
- `experiments/ablation_size.yaml`                       — 33×20 subset config
- `_resize_compare.py / _resize_compare4.py / _resize_compare1024.py` — 시각 비교 스크립트
- `_resize_compare* / _resize_compare4* / _resize_compare1024*` 폴더 — 비교 결과 PNG
- `CACHE_ABLATION_PLAN.md` — 이 문서
- memory:
  - `~/.claude/projects/D--project-unknown-contrastive/memory/feedback_input_resolution_decision.md` (신규)
  - `~/.claude/projects/D--project-unknown-contrastive/memory/MEMORY.md` (index 업데이트)

이 파일들은 어떤 경우에도 삭제 금지 (CLAUDE.md absolute rule).

## 진행 중인 background 작업

없음. 모든 작업 동기 완료. 재부팅해도 손실 없음.
