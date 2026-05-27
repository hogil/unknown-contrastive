#!/usr/bin/env bash
# ============================================================
#  unknown-contrastive setup — H100 (RHEL 9) / H200 (Ubuntu 24.04) 공용
#  사내 폐쇄망 호환 — 외부 PyPI URL (--index-url) 사용 안 함.
#  사내 pip mirror 또는 로컬 wheelhouse 가 wheel 을 제공한다고 가정.
#
#  사용:
#    bash setup.sh                       # 사내 mirror 사용 (default)
#    bash setup.sh --cuda 126            # torch 2.7 pin (mirror 가 cu126 wheel 제공)
#    bash setup.sh --cpu                 # CPU only torch (학습 X)
#    bash setup.sh --offline             # wheelhouse/ 로컬 wheel
#    bash setup.sh --wheelhouse /path    # 다른 wheelhouse 폴더
# ============================================================
set -e

# ★ script 어디서 실행하든 항상 setup.sh 위치 (= repo root) 기준 — 프로젝트 상대경로 보장
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "[cwd] $SCRIPT_DIR"

CUDA_VER="124"
TORCH_VER="2.6.0"
TVIS_VER="0.21.0"
CPU_ONLY=0
WHEELHOUSE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --cuda)
      CUDA_VER="$2"; shift 2
      if [[ "$CUDA_VER" == "126" ]]; then
        TORCH_VER="2.7.0"; TVIS_VER="0.22.0"
      fi
      ;;
    --cpu)
      CPU_ONLY=1; shift
      ;;
    --offline)
      WHEELHOUSE="wheelhouse"; shift
      ;;
    --wheelhouse)
      WHEELHOUSE="$2"; shift 2
      ;;
    *)
      echo "Unknown option: $1"; exit 1
      ;;
  esac
done

echo "============================================================"
echo "  unknown-contrastive setup"
echo "  CUDA: cu${CUDA_VER}   torch: ${TORCH_VER}   torchvision: ${TVIS_VER}"
echo "  CPU only: ${CPU_ONLY}"
if [[ -n "$WHEELHOUSE" ]]; then
  echo "  ★ Offline (wheelhouse): $WHEELHOUSE"
fi
echo "============================================================"

# ---------- Python version check ----------
if ! command -v python3 &> /dev/null; then
  echo "[ERR] python3 not found"; exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_OK=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)")

if [[ "$PY_OK" != "1" ]]; then
  echo "[ERR] Python ${PY_VER} < 3.10. 다음 중 하나 필요:"
  echo "  - Ubuntu 24.04: python3 (기본 3.12)"
  echo "  - RHEL 9: 'sudo dnf module install -y python3.11' 후 python3.11 사용"
  echo "  - conda: 'conda create -n contrastive python=3.11'"
  exit 1
fi
echo "[ok] Python ${PY_VER}"

# ---------- venv ----------
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] venv 생성: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[ok] venv activated ($VENV_DIR)"

# ---------- pip ----------
# 사내 폐쇄망 가정 — 외부 PyPI URL (--index-url) 사용 X.
# 사내 pip mirror 또는 로컬 wheelhouse 가 wheel 제공.
if [[ -n "$WHEELHOUSE" ]]; then
  # wheelhouse 가 relative 면 SCRIPT_DIR 기준
  if [[ "$WHEELHOUSE" != /* ]]; then
    WHEELHOUSE="$SCRIPT_DIR/$WHEELHOUSE"
  fi
  if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "[ERR] wheelhouse 폴더 없음: $WHEELHOUSE"; exit 1
  fi
  echo "[offline] wheelhouse: $WHEELHOUSE"
  pip install --no-index --find-links "$WHEELHOUSE" \
              "torch==${TORCH_VER}" "torchvision==${TVIS_VER}"
  pip install --no-index --find-links "$WHEELHOUSE" -r "$SCRIPT_DIR/requirements.txt"
else
  pip install --upgrade pip
  pip install "torch==${TORCH_VER}" "torchvision==${TVIS_VER}"
  pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# ---------- 검증 ----------
echo ""
echo "============================================================"
echo "  검증"
echo "============================================================"
python -c "
import torch, torchvision, timm, hdbscan, sklearn, numpy, scipy
print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  torch:        {torch.__version__}')
print(f'  torchvision:  {torchvision.__version__}')
print(f'  timm:         {timm.__version__}')
print(f'  numpy:        {numpy.__version__}')
print(f'  scipy:        {scipy.__version__}')
print(f'  sklearn:      {sklearn.__version__}')
print()
print(f'  CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:     {torch.version.cuda}')
    print(f'  GPU count:        {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    [{i}] {torch.cuda.get_device_name(i)}')
    print(f'  NCCL available:   {torch.distributed.is_nccl_available()}')
"

echo ""
echo "============================================================"
echo "  setup 완료"
echo "============================================================"
echo "다음 단계:"
echo "  source .venv/bin/activate"
echo "  python scripts/generate_data.py    # synthetic wafer (or 실제 data 가져옴)"
echo "  python scripts/_split_data.py      # train/eval 분리"
echo "  python scripts/train_pipeline.py   # single GPU 한방에"
echo ""
echo "  # multi-GPU (H100/H200 8장)"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train_pipeline_ddp.py"
