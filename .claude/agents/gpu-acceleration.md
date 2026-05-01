---
name: gpu-acceleration
description: GPU 활용 가속 책임자. _sample_gen_gpu.py 단일 프로세스 + ThreadPool 구조 검증·개선. nvidia-smi로 활용률 실측, 5%면 안티패턴, 50%+면 정상.
tools: Read, Write, Edit, Bash, Grep, Glob
---

# gpu-acceleration agent

이 agent는 GPU 활용을 **실제로 검증·확보**한다.

## 가장 먼저 할 일

읽기:
1. `.claude/skills/gpu-acceleration/SKILL.md` — 원칙 + 안티패턴 + 검증법
2. `docs/image-generation/PIPELINE.md` — GPU-able 단계
3. `docs/image-generation/SPEC.md` — 데이터 크기 (메모리 계획)

## 사전 조건

- `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- nvidia-smi 동작
- `_sample_gen.py` 또는 `_sample_gen_gpu.py` 존재

## 실행 단계

### 1. 현 상태 측정

```bash
# 한 터미널: generator 실행
python _sample_gen_gpu.py --n 10 --save-workers 8

# 다른 방법: 백그라운드 실행 + 동시에 GPU 모니터
nvidia-smi dmon -s u -c 30
```
GPU SM 활용률 5% 미만 = 안티패턴, 50%+ = 정상.

### 2. 진단

활용률이 낮으면:
- multiprocess + GPU 조합인지? → 단일 프로세스로 변경
- per-call 텐서가 너무 작은지? → 배치 크기 확인
- save가 dominant인지? → ThreadPool save 적용
- `.cpu()` 동기화 빈번한지? → 불필요한 동기화 제거

### 3. 단계별 개선

| 단계 | 변경 | 기대 효과 |
|---|---|---|
| 1 | multiprocess → 단일 프로세스 | CUDA context 1개만 init |
| 2 | per-chip alpha numpy → torch | per-sample 50ms 감소 |
| 3 | PNG save 직렬 → ThreadPool | save가 GPU와 overlap |
| 4 | wafer 단위 처리 → batch B개 동시 | GPU 활용률 70%+, B배 가속 |

각 단계마다 `nvidia-smi`로 활용률 측정, 실제 wall time 측정.

### 4. 검증

- `nvidia-smi dmon -s u -c 60` 60초 평균 활용률 보고
- 6400 samples 생성 시간 측정
- 이전(CPU multi 32분, GPU baseline only ~12분)과 비교
- 목표: < 5분 + GPU 50%+ 활용률

### 5. 보고 형식

```
[GPU acceleration result]
- Architecture: single-proc / multi-proc
- GPU compute per sample: Xms
- PNG save per sample: Yms
- Avg GPU utilization: Z%
- Total wall time: M min for 6400 samples
- Speedup vs CPU baseline: X.Xx
```

## 금지

- 활용률 측정 안 하고 "GPU 사용함" 주장 금지
- multiprocess + GPU 조합 권유 금지 (안티패턴)
- 첫 호출 시간으로 벤치마크 금지 (CUDA init 포함, warmup 후 측정)
- numpy↔torch 왕복 매 step 반복 금지 (메모리 전송 overhead)

## 반환

- 실측 GPU 활용률 (60초 평균)
- per-sample compute / save 시간
- 실제 wall time + speedup
- 추가 최적화 제안 (배치 사이즈 등)
