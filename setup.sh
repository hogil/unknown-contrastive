#!/usr/bin/env bash
# ============================================================
#  unknown-contrastive setup — H100 (RHEL 9) / H200 (Ubuntu 24.04) 공용
#
#  사용:
#    bash setup.sh           # default: PyTorch 2.6.0 + CUDA 12.4
#    bash setup.sh --cuda 126 # 또는 CUDA 12.6 (torch 2.7)
#    bash setup.sh --cpu      # CPU only
# ============================================================
set -e

CUDA_VER="124"
TORCH_VER="2.6.0"
TVIS_VER="0.21.0"
CPU_ONLY=0

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
    *)
      echo "Unknown option: $1"; exit 1
      ;;
  esac
done

echo "============================================================"
echo "  unknown-contrastive setup"
echo "  CUDA: cu${CUDA_VER}   torch: ${TORCH_VER}   torchvision: ${TVIS_VER}"
echo "  CPU only: ${CPU_ONLY}"
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
if [[ ! -d ".venv" ]]; then
  echo "[setup] venv 생성"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[ok] venv activated"

# ---------- pip upgrade ----------
pip install --upgrade pip

# ---------- PyTorch ----------
if [[ "$CPU_ONLY" == "1" ]]; then
  echo "[setup] PyTorch CPU only"
  pip install "torch==${TORCH_VER}" "torchvision==${TVIS_VER}"
else
  echo "[setup] PyTorch CUDA cu${CUDA_VER}"
  pip install "torch==${TORCH_VER}" "torchvision==${TVIS_VER}" \
              --index-url "https://download.pytorch.org/whl/cu${CUDA_VER}"
fi

# ---------- 나머지 dependency ----------
pip install -r requirements.txt

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
