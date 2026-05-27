# unknown-contrastive

Wafer defect grouping via **CNN backbone supervised + Contrastive head (frozen backbone) + HDBSCAN clustering**.

학습 데이터 / 평가 데이터 / class 가 완전 disjoint — fair zero-shot transfer evaluation.

---

## 0. Setup (한 번만)

**환경**:
- **Python 3.10+** 필요 (`str | Path` type hint 등 3.10 문법 사용)
- 개발/테스트 환경: Python **3.13.7**, PyTorch **2.6.0+cu124**, CUDA **12.4**

```bash
# 1. Clone
git clone https://github.com/hogil/unknown-contrastive.git
cd unknown-contrastive

# 2. (권장) 가상환경 — Python 3.10+
python --version          # 3.10+ 확인
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# 또는: .venv\Scripts\activate      # Windows cmd / PowerShell
# 또는: source .venv/bin/activate   # Linux / Mac

# 3. PyTorch 먼저 (H100/H200 호환 — CUDA 12.4+)

# ★ 권장 (H100 + H200 둘 다 OK, tested):
pip install torch==2.6.0 torchvision==0.21.0 \
            --index-url https://download.pytorch.org/whl/cu124

# 또는 최신 (H100 + H200, CUDA 12.6):
# pip install torch==2.7.0 torchvision==0.22.0 \
#             --index-url https://download.pytorch.org/whl/cu126

# CPU only (학습 X, 코드 확인용):
# pip install torch==2.6.0 torchvision==0.21.0

# 4. 나머지 dependency
pip install -r requirements.txt
```

---

### 한 줄 setup (H100 / H200 둘 다)

```bash
git clone https://github.com/hogil/unknown-contrastive.git && cd unknown-contrastive
bash setup.sh                 # CUDA 12.4 + PyTorch 2.6 (★ tested)
# bash setup.sh --cuda 126    # CUDA 12.6 + PyTorch 2.7
# bash setup.sh --cpu         # CPU only
```

→ Python 3.10+ check + venv + PyTorch + requirements + CUDA/NCCL 검증 자동.

---

### Server 환경 별 setup (Ubuntu 24.04 + H200 / RHEL 9 + H100)

#### Ubuntu 24.04 LTS + H200

```bash
# 1. Python 확인 (Ubuntu 24.04 기본 = 3.12 ★)
python3 --version
sudo apt update && sudo apt install -y python3-venv python3-pip

# 2. NVIDIA driver + CUDA 12.4+ (이미 설치돼 있다고 가정)
nvidia-smi   # H200 인식 + driver 535+ 확인

# 3. venv + setup
git clone https://github.com/hogil/unknown-contrastive.git
cd unknown-contrastive
python3 -m venv .venv && source .venv/bin/activate

# 4. PyTorch (H200 권장)
pip install torch==2.6.0 torchvision==0.21.0 \
            --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 5. NCCL check (DDP 위해)
python -c "import torch; print(f'cuda={torch.cuda.is_available()} nccl={torch.distributed.is_nccl_available()}')"
# 출력: cuda=True nccl=True
```

#### RHEL 9 + H100

```bash
# 1. Python 3.11 install (RHEL 9 기본 = 3.9 — 우리 최소 3.10 부족)
sudo dnf module install -y python3.11
python3.11 --version    # 3.11.x 확인

# 2. NVIDIA driver + CUDA (NVIDIA repo 추가 후)
#    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/
nvidia-smi   # H100 인식

# 3. venv + setup
git clone https://github.com/hogil/unknown-contrastive.git
cd unknown-contrastive
python3.11 -m venv .venv && source .venv/bin/activate

# 4. PyTorch (H100 권장)
pip install torch==2.6.0 torchvision==0.21.0 \
            --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 5. NCCL check
python -c "import torch; print(f'cuda={torch.cuda.is_available()} nccl={torch.distributed.is_nccl_available()}')"
```

#### 8 GPU DDP 학습 (H100/H200 server 표준)

```bash
# 8 GPU 한방에
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train_pipeline_ddp.py
# → world_size=8 자동, total_batch = BATCH_PER_GPU × 8
# Linux NCCL backend (Windows 는 gloo fallback)
```

---

**GPU 호환 표** (Hopper architecture):

| GPU | CUDA min | 권장 wheel | PyTorch |
|---|---|---|---|
| H100 | 11.8 | cu124 (12.4) | 2.4 ~ 2.7 |
| H200 | 12.2 | cu124 (12.4) | 2.4 ~ 2.7 |
| A100 | 11.0 | cu124 | 2.4 ~ 2.7 |
| RTX 40xx | 11.8 | cu124 | 2.4 ~ 2.7 |

**torch ↔ torchvision 짝** (절대 misversion 안 됨):

| torch | torchvision |
|---|---|
| 2.4 | 0.19 |
| 2.5 | 0.20 |
| **2.6** ★ tested | **0.21** ★ tested |
| 2.7 | 0.22 |

**Python 버전 확인 표**:

| Python | 지원 | 비고 |
|---|:-:|---|
| 3.9 | ✗ | `str \| Path` type hint 안 됨 |
| 3.10 | ✓ | 최소 지원 |
| 3.11 | ✓ | |
| 3.12 | ✓ | |
| 3.13 | ✓ | 개발 환경 |

**weights** (ConvNeXtV2 FCMAE backbone) 는 **자동 다운로드** — 첫 실행 시 `weights/` 폴더에 한 번만 받음.

---

## 1. 이미지 생성 (실제 데이터 없으면)

```bash
python scripts/generate_data.py
# → data/images/unknown/<class>/wafer_*.png (43 class × 100 + Normal × 200 ≈ 4400 장)
```

각 class 별 distinct synthetic pattern. real WM-811K 데이터 있으면 그걸 `data/images/unknown/<class>/` 에 두면 됨.

## 2. 데이터 폴더 분리 (한 번만)

이미지 생성 단계부터 폴더 분리 (사용자 정책: "class 보는 건 치팅"):

```bash
python scripts/_split_data.py
```

산출 3 폴더 (wafer 단위 disjoint, hard link 으로 디스크 절약):

| 폴더 | 형식 | 용도 | class 정보 |
|---|---|---|---|
| `cnn_train/<class>/*.png` | ImageFolder | CNN supervised (Split A 21 class) | ✓ 보존 |
| `contrastive_train/*.png` | flat | Contrastive 학습 | ✗ **숨김** |
| `contrastive_eval/<class>/*.png` | ImageFolder | Contrastive metric 측정 (Split B 22 class) | ✓ 보존 |

---

## 3. 한방에 학습 (pipeline)

```bash
# scripts/train_pipeline.py 의 CONFIG block 확인 후
python scripts/train_pipeline.py
```

진행:
1. **CNN backbone** (Split A 21 class, supervised, ConvNeXtV2 FCMAE) — 30 epoch, early stop
2. CNN best 를 → **Contrastive head** init backbone (frozen)
3. **Contrastive head** (Split B 22 class, InfoNCE + MoCo Queue + NEG filter + NeCo) — 5 epoch
4. **HDBSCAN clustering eval** — tier1 metric (AMI/ARI/capture/noise%)

산출 폴더 (시간 prefix):
```
runs/<YYMMDD_HHMMSS>_pipeline/
├── config.yaml                              # 실행 hyperparam snapshot
├── metrics.json                             # 단계별 성능 누적
├── report.md                                # 표 자동 생성
├── cnn/
│   ├── best_model.pth                       # CNN backbone (val_macro_f1 best)
│   ├── history.json
│   ├── classes.json
│   └── wrong/val/<true_class>/              # 틀린 이미지
│       └── <true>_<pred>_<pct>%_<basename>.png
└── contrastive/
    ├── best_model.pt                        # Contrastive head
    ├── embeddings.npy                       # (N, 128)
    ├── clusters_global_list.txt             # cluster_id  true_class  path
    ├── tier1.json                           # AMI/ARI/capture/noise%
    └── wrong/<true_class>/                  # cluster mismatch + noise
        └── <true>_<cluster_dominant_class>_<pct>%_<basename>.png
```

### 단독 학습 (필요 시)

```bash
python scripts/train_cnn.py            # CNN 만
python scripts/train_contrastive.py    # Contrastive 만 (backbone path 별도 지정)
```

### Multi-GPU (DDP) 학습 — 별도 파일 (option 아님)

```bash
# 4 GPU CNN 학습
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cnn_ddp.py

# 4 GPU Contrastive 학습
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_contrastive_ddp.py

# 4 GPU 한방에 (CNN_DDP → Contrastive_DDP sequential)
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_pipeline_ddp.py
```

- `CUDA_VISIBLE_DEVICES` 의 device 수 = `world_size` 자동
- `mp.spawn` 으로 worker 자동 launch (torchrun 불필요)
- 단일 GPU 면 single-GPU fallback (DDP wrap skip)
- Windows: NCCL 미지원 → gloo backend 자동 fallback
- 각 script CONFIG 의 `BATCH_PER_GPU` × world_size = total batch
- rank=0 만 save/print/metric log, 나머지 rank 는 동기화만

---

## 4. 현업 데이터 grouping

**폴더 구조**: `<image_base>/<product>/<line>/<date>/*.png`

예시:
```
E:/prod/
├── AA/
│   ├── K1AA/
│   │   └── 20260502/wafer_*.png
│   └── K1AB/
│       └── 20260502/wafer_*.png
└── BB/
    └── K1BA/
        └── 20260502/wafer_*.png
```

**CONFIG** (`scripts/predict_grouping_prod.py` 최상단 수정):

```python
# 옵션 A — 단일 폴더
IMAGE_ROOT  = "E:/prod/AA/K1AA/20260502"
IMAGE_BASE  = None

# 옵션 B — 자동 walk (모든 제품/라인/날짜)
IMAGE_ROOT  = None
IMAGE_BASE  = "E:/prod"
PRODUCT_FILTER = ["AA", "BB"]      # None=all
LINE_FILTER    = None
DATE_FILTER    = ["20260502"]       # 특정 날짜만

MODEL_PATH  = "runs/<TS>_pipeline/contrastive/best_model.pt"   # ← step 2 산출 그대로
```

**실행**:
```bash
python scripts/predict_grouping_prod.py
```

**산출**:
```
result_grouping/<YYMMDD_HHMMSS>_grouping/
├── config.yaml
├── all_summaries.json
├── AA/K1AA/20260502/
│   ├── clusters.parquet            # path | group_id | product | line | date  (DB ingestion)
│   ├── clusters.csv                # 같은 내용 CSV
│   ├── embeddings.npy
│   └── summary.json                # n_images, n_clusters, n_noise, groups: {id: count}
└── AA/K1AB/20260502/
    └── ...
```

`COPY_PNG_TO_GROUPS = True` 시 추가:
```
└── AA/K1AA/20260502/groups/
    ├── 0/   wafer_001.png, wafer_002.png, ...
    ├── 1/   ...
    └── -1/  (noise)
```

---

## 5. CONFIG 정책

**모든 script** 의 hyperparam + path 는 파일 최상단 `# === CONFIG ===` block 에 있음.
실행 시 그 부분만 수정.

```python
# === CONFIG (실행 시 이 부분만 수정) ===
DATA_DIR             = "E:/data/images/unknown"
WEIGHTS_DIR          = "weights"
OUTPUT_ROOT          = "runs"
TAG                  = "cnn"

BACKBONE             = "convnextv2_base.fcmae_ft_in22k_in1k_384"
EPOCHS               = 30
BATCH                = 16
LR_BACKBONE          = 2e-5
LR_HEAD              = 2e-4
WEIGHT_DECAY         = 0.01
# ...
```

**적용된 best 기법** (anomaly-detection 정책 매치):
- ConvNeXtV2 base FCMAE pretrained backbone
- AdamW (wd 0.01) + warmup 5 epoch (start factor 0.05) + Cosine annealing
- Label smoothing 0.02
- Grad clip 1.0
- EMA / Mixup / AMP **OFF** (사용자 명시)
- Stochastic depth 0
- Corrupted PNG skip (dummy black image fallback)

---

**모든 path 는 프로젝트 상대경로** (PROJECT_ROOT 자동 detect via `_common.resolve_path`).
absolute path 도 OK (예: `E:/prod/...`) — 그 경우 그대로 사용.

## 6. 단계별 성능 기록 (paper-friendly)

각 stage 끝나면 `metrics.json` 에 자동 append + `report.md` 표 자동 생성:

```json
{
  "stages": [
    {"stage": "cnn_setup",         "ts": "...", "metric": {...}},
    {"stage": "cnn_train_done",    "ts": "...", "metric": {"best_val_macro_f1": 0.989, "test_acc": 0.998}},
    {"stage": "contrastive_setup", "ts": "...", "metric": {...}},
    {"stage": "contrastive_eval",  "ts": "...", "metric": {"ami": 0.886, "noise_pct": 0.0, "capture": 0.962}}
  ]
}
```

기법 1개 추가될 때마다 `log_stage_metric()` 한 줄로 기록 — `runs/<TS>/report.md` 표 즉시 업데이트.

---

## 7. 폴더 구조 요약

```
unknown-contrastive/
├── README.md                        ← 이 파일
├── requirements.txt
├── scripts/
│   ├── _common.py                   공통 utility (weights download, metric log, report.md)
│   ├── _split_data.py               data reorganize
│   ├── train_cnn.py                 ① CNN 단독
│   ├── train_contrastive.py         ② Contrastive 단독
│   ├── train_pipeline.py            ③ CNN → Contrastive 한방에
│   └── predict_grouping_prod.py     ④ 현업 grouping
├── experiments/
│   ├── split_a_cnn_21.yaml          # CNN class list
│   └── split_b_contrastive_22.yaml  # Contrastive class list
├── weights/                         # backbone 자동 다운로드 (gitignored)
└── runs/                            # 학습 산출 (gitignored)
    └── <TS>_<tag>/
```

---

## 8. Quick reference

| 작업 | 명령 |
|---|---|
| 첫 setup | `pip install -r requirements.txt` |
| 이미지 생성 (없으면) | `python scripts/generate_data.py` |
| 데이터 분리 (한 번만) | `python scripts/_split_data.py` |
| 한방에 학습 | `python scripts/train_pipeline.py` |
| CNN 만 | `python scripts/train_cnn.py` |
| Contrastive 만 | `python scripts/train_contrastive.py` |
| 4-GPU CNN | `CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cnn_ddp.py` |
| 4-GPU Contrastive | `CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_contrastive_ddp.py` |
| 4-GPU 한방에 | `CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_pipeline_ddp.py` |
| 현업 grouping | `python scripts/predict_grouping_prod.py` |
