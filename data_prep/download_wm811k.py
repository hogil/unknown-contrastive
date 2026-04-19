"""Download WM-811K LSWMD.pkl with 3-tier fallback.

Order:
    1. Kaggle (kagglehub, then kaggle CLI via subprocess)
    2. HuggingFace Hub (multiple mirror repo_ids)
    3. Manual instructions, exit 1

Output path: ``data_raw/LSWMD.pkl`` (relative to cwd).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tier 1: Kaggle
# ---------------------------------------------------------------------------

def _try_kagglehub(out: Path) -> bool:
    """Try the newer `kagglehub` library first."""
    try:
        import kagglehub  # type: ignore
    except Exception:
        return False

    try:
        # kagglehub.dataset_download returns a directory path with all files.
        path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
        src_dir = Path(path)
        # Look for LSWMD.pkl inside the downloaded directory (recursive).
        candidates = list(src_dir.rglob("LSWMD.pkl"))
        if not candidates:
            print(f"[kagglehub] LSWMD.pkl not found under {src_dir}")
            return False
        src = candidates[0]
        shutil.copy2(src, out)
        print(f"[kagglehub] copied {src} -> {out}")
        return out.exists() and out.stat().st_size > 0
    except Exception as e:
        print(f"[kagglehub] failed: {e}")
        return False


def _try_kaggle_cli(out: Path) -> bool:
    """Fallback: use the `kaggle` CLI via subprocess."""
    if shutil.which("kaggle") is None:
        return False

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "qingyi/wm811k-wafer-map",
            "-p",
            str(out.parent),
            "--unzip",
        ]
        print(f"[kaggle-cli] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        # Search for LSWMD.pkl inside out.parent.
        candidates = list(out.parent.rglob("LSWMD.pkl"))
        if not candidates:
            print("[kaggle-cli] LSWMD.pkl not found after unzip")
            return False
        if candidates[0] != out:
            shutil.copy2(candidates[0], out)
        return out.exists() and out.stat().st_size > 0
    except Exception as e:
        print(f"[kaggle-cli] failed: {e}")
        return False


def try_kaggle(out: Path) -> bool:
    if _try_kagglehub(out):
        return True
    if _try_kaggle_cli(out):
        return True
    return False


# ---------------------------------------------------------------------------
# Tier 2: HuggingFace Hub
# ---------------------------------------------------------------------------

# A list of mirror repo_ids we try in order. We don't know for sure which one
# will be alive, so we try several and swallow errors on each.
_HF_MIRROR_REPOS = [
    "Jaehongjung/WM811K",
    "qingyi/wm811k-wafer-map",
    "WM811K/LSWMD",
    "ywchoi/WM811K",
    "kaist-ai/WM811K",
]

_HF_CANDIDATE_FILENAMES = [
    "LSWMD.pkl",
    "data/LSWMD.pkl",
    "LSWMD/LSWMD.pkl",
]


def try_huggingface(out: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        print("[hf] huggingface_hub not installed, skipping")
        return False

    for repo_id in _HF_MIRROR_REPOS:
        for fname in _HF_CANDIDATE_FILENAMES:
            try:
                print(f"[hf] trying {repo_id}:{fname}")
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    repo_type="dataset",
                )
                src = Path(local)
                if src.exists() and src.stat().st_size > 0:
                    shutil.copy2(src, out)
                    print(f"[hf] copied {src} -> {out}")
                    return True
            except Exception as e:
                # swallow; move to next combination
                print(f"[hf] {repo_id}:{fname} failed: {type(e).__name__}: {e}")
                continue
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out = Path("data_raw/LSWMD.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[skip] already exists: {out.resolve()}")
        return

    # Try 1: Kaggle
    if try_kaggle(out):
        print(f"[ok] downloaded via Kaggle -> {out.resolve()}")
        return

    # Try 2: HuggingFace
    if try_huggingface(out):
        print(f"[ok] downloaded via HuggingFace -> {out.resolve()}")
        return

    # Try 3: Manual instructions
    print("=" * 60)
    print("자동 다운로드 실패. 다음 중 하나로 수동 배치하세요:")
    print("  1. https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map 에서 다운로드 후")
    print(f"     LSWMD.pkl → {out.resolve()} 로 이동")
    print("  2. Kaggle API token 설정 후 이 스크립트 재실행")
    print(f"     - {Path.home() / '.kaggle' / 'kaggle.json'} 생성 (kaggle.com/settings → Create API Token)")
    print("     - pip install kagglehub  또는  pip install kaggle")
    print("  3. HuggingFace 미러 repo 직접 지정:")
    print("     huggingface-cli download <repo_id> LSWMD.pkl --repo-type dataset --local-dir data_raw")
    print("=" * 60)
    sys.exit(1)


if __name__ == "__main__":
    main()
