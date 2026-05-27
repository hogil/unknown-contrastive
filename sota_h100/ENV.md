# Environment setup — RHEL9 / H100 + Ubuntu24 / H200, Python 3.11

H100 and H200 are both compute capability **sm_90** (Hopper), so a single CUDA 12.4
PyTorch build covers both. RHEL9 and Ubuntu24 use the same Python wheels.

## 1. Python 3.11 env

```bash
# conda
conda create -n sota python=3.11 -y && conda activate sota
# OR venv
python3.11 -m venv .venv && source .venv/bin/activate && pip install -U pip
```

## 2. PyTorch (CUDA 12.4 — H100 & H200)

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```
(H200 needs torch ≥2.3 for sm_90; 2.4.1 is safe on both. If your driver is CUDA 12.1
only, use the cu121 wheel + torch==2.4.1 instead.)

## 3. The rest

```bash
pip install -r sota_h100/requirements.txt
```

## 4. Verify

```bash
python - <<'PY'
import torch, timm, numpy, pandas, sklearn, pyarrow, PIL
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "n_gpu", torch.cuda.device_count())
print("dev0", torch.cuda.get_device_name(0))
print("timm", timm.__version__)
PY
nvidia-smi -L          # list GPUs (DDP auto-uses all)
```
Expect `cuda True`, the H100/H200 name, and `timm 1.0+`.

## 5. Backbone weight (offline)

The runners set `HF_HUB_OFFLINE=1`; no download happens. Place the ImageNet-FCMAE
ConvNeXtV2-base weight (not in git, ~340 MB):
```bash
ls models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth   # copy here via scp if missing
```

## 6. Run (no options)

```bash
bash sota_h100/gen.sh                                       # data
CUDA_VISIBLE_DEVICES=0,1,2,3 bash sota_h100/train_ddp.sh    # DDP train (all listed GPUs)
bash sota_h100/predict.sh /path/to/real_chips              # production inference
```

## Notes / gotchas
- All paths are project-relative (scripts `cd` to repo root); no absolute paths to edit.
- `numpy<2.2` pin avoids ABI churn with torch 2.4 / timm. If you hit a numpy-2 error,
  `pip install "numpy<2"`.
- Linux fonts: Invalid-chip text uses DejaVuSans if present, else PIL default — both fine.
- torchrun ships with torch (used by train_ddp.sh); no extra install.
- CPU-only smoke (no GPU): the code falls back to CPU automatically (slow).
