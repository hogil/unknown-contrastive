---
name: gpu-acceleration
description: GPU(CUDA/torch) 활용해서 합성 wafer 데이터셋 생성 가속. 아키텍처 결정(multiprocess vs single proc), batch sizing, GPU 활용률 검증 책임. nvidia-smi로 실측 검증.
---

# gpu-acceleration skill

이 스킬은 GPU 자원을 실제로 효과적으로 사용하도록 코드를 설계·검증한다.
"torch import"만 한다고 GPU 가속이 되는 게 아니라는 점을 인정하고, **활용률을 실측**해
효과를 검증한다.

## 가장 먼저 읽기

- `docs/image-generation/PIPELINE.md` — 어느 단계가 GPU-able 한지
- `docs/image-generation/SPEC.md` — 데이터 크기·형상 (메모리 계획용)
- 이 스킬

## 핵심 원칙

### 1. 절대 규칙: 단일 프로세스 + GPU
- multiprocessing + CUDA = **각 프로세스마다 CUDA context 별도 init** → 매우 느림 + 메모리 낭비 + 활용률 저하
- 정답: **단일 프로세스에서 GPU 순차 compute** + ThreadPool로 I/O (PNG save) overlap
- 측정: `nvidia-smi -l 1` 또는 `nvidia-smi dmon -s u`로 utilization 확인

### 2. GPU 활용률 목표
- **단일 프로세스 GPU 50%+ 활용**이 정상
- 5-15% = GPU가 거의 idle = 잘못된 구조 (per-call overhead가 dominant)
- 90%+ = 좋음 (compute-bound)

### 3. Pipeline 패턴
```
[GPU compute (sequential)] → [canvas to CPU] → [submit to ThreadPool] → [PIL save (concurrent)]
```
GPU compute는 직렬. 저장만 thread pool로 비동기. backpressure로 메모리 폭주 방지.

### 4. 무엇을 GPU로 옮기나
| 작업 | GPU 가치 |
|---|---|
| `torch.rand((6400, 6400))` (40M floats) | ★★★ 매우 큼 (CPU 500ms → GPU 10ms, 50x) |
| `torch.searchsorted(cum, u)` | ★★★ |
| per-chip alpha 계산 (200×200 ops) | ★★ medium (CPU 2ms → GPU 0.5ms, 4x but small ops) |
| per-chip mixed sampling | ★★ |
| outside-wafer fill (mask op) | ★ 작음 |
| **PNG deflate 압축** | **GPU 안 됨** (sequential algorithm, CPU only) |
| ImageDraw text | GPU 안 됨 (PIL only) |

→ **PNG save가 새 bottleneck**. ThreadPool로 overlap이 필수.

### 5. 메모리 계획
RTX 4060 Ti = 8GB GPU memory. 주요 텐서:
- `u` (random uniform): 6400×6400 float32 = **160 MB**
- `canvas`: 6400×6400 uint8 = 40 MB
- per-chip alpha: 200×200 float32 = 0.16 MB × 50 chips = 8 MB
- per-chip cum_mixed: 200×200×8 float32 = 1.3 MB × 50 = 65 MB

단일 wafer 처리 시 ~250 MB. 여유 충분.

배치 처리 (B 개 wafer 동시): u 텐서 B×160MB 가 dominant. B=16이면 2.5GB → 가능.

### 6. 검증 절차

```bash
# 1. GPU 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. 단순 baseline 벤치마크 (CPU vs GPU)
python -c "
import time, numpy as np, torch
SIZE = 6400
CUM = np.cumsum([0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001])
DEV = torch.device('cuda')
CUM_T = torch.tensor(CUM, device=DEV, dtype=torch.float32)
torch.rand((100,100), device=DEV)  # warmup

t0 = time.time()
for _ in range(10):
    u = np.random.random((SIZE, SIZE))
    c = np.searchsorted(CUM, u).astype(np.uint8)
print(f'CPU 10x: {time.time()-t0:.2f}s')

torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    u = torch.rand((SIZE, SIZE), device=DEV)
    c = torch.searchsorted(CUM_T, u).to(torch.uint8).cpu().numpy()
torch.cuda.synchronize()
print(f'GPU 10x: {time.time()-t0:.2f}s')
"
# 기대: GPU >> CPU (50x정도)

# 3. 실제 generator 실행 + GPU 활용률 모니터링
# Terminal 1: python _sample_gen_gpu.py --n 50 --save-workers 8
# Terminal 2: nvidia-smi dmon -s u -c 60
# 활용률이 50%+ 안정적으로 보이면 OK
```

### 7. 흔한 실수 (안티패턴)

| 안티패턴 | 결과 |
|---|---|
| 8 worker process × torch GPU 각자 init | CUDA 컨텍스트 8개, 활용률 5%, 시작 overhead 큼 |
| GPU 호출 후 즉시 `.cpu()` 동기화 | 파이프라인 깨짐, 직렬화됨 |
| numpy→torch→numpy 왕복 매 iteration | 메모리 전송 overhead |
| 작은 텐서(e.g., 200×200) 단일 op | launch overhead가 compute보다 큼 |
| GPU compute 후 `time.time()` 직접 | 비동기 실행 미반영, 실제 시간 다름 (synchronize 필요) |

## 현재 프로젝트 상태

- `_sample_gen.py`: **CPU multiprocessing 8 workers + 작은 GPU baseline** (안티패턴 1: GPU 활용률 5%)
- `_sample_gen_gpu.py`: **단일 프로세스 GPU pipeline + ThreadPool save** (정답 구조, 검증 필요)

## 실행 옵션

```bash
# 권장: 단일 프로세스 GPU pipeline
python _sample_gen_gpu.py --n 200 --save-workers 8

# 안티패턴 (참고용, 사용 X): multiprocess
python _sample_gen.py --n 200 --workers 8
```

## 금지

- multiprocessing + GPU 조합 금지 (단일 프로세스 GPU가 정답)
- 활용률 안 재고 "GPU로 했어요" 주장 금지 (반드시 nvidia-smi 검증)
- `torch.rand` warmup 안 하고 첫 호출 시간 측정 금지 (CUDA init 포함됨)
