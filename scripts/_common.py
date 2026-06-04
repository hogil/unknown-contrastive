"""Common utilities — 모든 train/predict 스크립트 공용.

- run dir 생성 (runs/<TS>_<tag>/)
- weights/ 자동 다운로드 (FCMAE backbone)
- metrics.json 단계별 append
- report.md narrative 생성
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


# ===================== Project root =====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts/_common.py → repo root

def project_path(*parts) -> Path:
    """프로젝트 루트 기준 경로 — 어떤 머신에서도 clone 후 동일."""
    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(p) -> Path:
    """absolute → 그대로, relative → PROJECT_ROOT 기준.

    ★ 사용자 directive (260528): 이미지는 로컬에서 E:/data/images/ (절대규).
       서버(Linux 또는 Windows-but-E:없음) 에선 자동으로 PROJECT_ROOT/data/images/
       로 fallback (사용자 directive: "서버에선 알아서 프로젝트 폴더에 만들어서").
       즉 CONFIG 의 'E:/data/images/X' path 가 서버 환경에서 'PROJECT_ROOT/data/images/X'
       로 투명하게 매핑됨 — 코드 환경별 분기 X.
    """
    p = Path(p)
    s = str(p).replace("\\", "/")
    if s.lower().startswith("e:/"):
        # Windows + E: drive 살아있으면 그대로 사용
        if os.name == "nt":
            try:
                if Path("E:/").exists():
                    return p
            except OSError:
                pass
        # 그 외 (Linux 서버 / E: 부재) → PROJECT_ROOT 의 같은 상대 부분으로
        rel = s.split(":", 1)[1].lstrip("/")            # "data/images/unknown"
        return PROJECT_ROOT / rel
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def mask_palette_non_grade_to_white(img, enabled: bool = True):
    """Palette PNG 입력에서 grade 0~7만 색을 유지하고 나머지 index는 흰색 처리.

    원본 파일은 건드리지 않고, Image.open(...) 뒤 convert("RGB") 직전에만 in-memory palette를
    바꾼다. wafer border/background/invalid 계열 색이 학습 shortcut이 되는 것을 막기 위함.
    """
    if not enabled or getattr(img, "mode", None) != "P":
        return img
    palette = img.getpalette()
    if not palette:
        return img
    out = img.copy()
    pal = list(palette)
    if len(pal) < 768:
        pal.extend([0] * (768 - len(pal)))
    for idx in range(8, 256):
        pal[idx * 3:idx * 3 + 3] = [255, 255, 255]
    out.putpalette(pal[:768])
    return out


# ===================== Run directory =====================
def make_run_dir(output_root, tag: str) -> Path:
    """runs/<YYMMDD_HHMMSS>_<tag>/ 폴더 생성. output_root 가 상대면 PROJECT_ROOT 기준."""
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    p = Path(output_root)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    rd = p / f"{ts}_{tag}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def snapshot_config(run_dir: Path, cfg: dict):
    """현재 실행의 hyperparam snapshot 저장."""
    import yaml
    (run_dir / "config.yaml").write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ===================== Metric logging =====================
def init_metrics(run_dir: Path):
    """metrics.json 초기화."""
    p = run_dir / "metrics.json"
    if not p.exists():
        p.write_text(json.dumps({"stages": []}, indent=2, ensure_ascii=False), encoding="utf-8")


def log_stage_metric(run_dir: Path, stage: str, metric: dict[str, Any], notes: str = ""):
    """단계별 성능 한 cell 추가. 기법 1개 돌아갈 때마다 호출."""
    p = run_dir / "metrics.json"
    init_metrics(run_dir)
    data = json.loads(p.read_text(encoding="utf-8"))
    cell = {
        "stage": stage,
        "ts": datetime.now().strftime("%y%m%d_%H%M%S"),
        "metric": metric,
    }
    if notes:
        cell["notes"] = notes
    data["stages"].append(cell)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # report.md 도 자동 업데이트
    update_report_md(run_dir)


def update_report_md(run_dir: Path):
    """metrics.json → report.md 표 자동 생성."""
    p = run_dir / "metrics.json"
    if not p.exists():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    lines = [
        f"# Run Report — {run_dir.name}",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Stage Performance",
        "",
    ]
    stages = data.get("stages", [])
    if not stages:
        lines.append("_(no stage logged yet)_")
    else:
        # collect all metric keys for table header
        all_keys: list[str] = []
        for s in stages:
            for k in s.get("metric", {}).keys():
                if k not in all_keys:
                    all_keys.append(k)
        header = ["stage", "ts"] + all_keys + ["notes"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for s in stages:
            row = [s.get("stage", ""), s.get("ts", "")]
            for k in all_keys:
                v = s.get("metric", {}).get(k)
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                elif v is None:
                    row.append("-")
                else:
                    row.append(str(v))
            row.append(s.get("notes", ""))
            lines.append("| " + " | ".join(row) + " |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ===================== Backbone weights =====================
def ensure_backbone_weights(weights_dir,
                            backbone: str = "convnextv2_base.fcmae_ft_in22k_in1k_384") -> Path:
    """프로젝트의 weights/ 폴더에 backbone 다운로드 (이미 있으면 그대로). 상대경로면 PROJECT_ROOT 기준."""
    weights_dir = Path(weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = PROJECT_ROOT / weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / f"{backbone}.pth"
    if target.exists() and target.stat().st_size > 1024 * 1024:
        print(f"[weights] using cached {target} ({target.stat().st_size/1024/1024:.0f} MB)")
        return target

    # try sister mirror
    sister = Path("D:/project/known-cnn/wafer_train/models") / f"{backbone}.pth"
    if sister.exists():
        shutil.copy2(sister, target)
        print(f"[weights] copied from sister: {sister} → {target}")
        return target

    # download from HF Hub via timm
    print(f"[weights] downloading {backbone} from HF Hub once...")
    import timm
    m = timm.create_model(backbone, pretrained=True)
    torch.save(m.state_dict(), target)
    print(f"[weights] saved {target} ({target.stat().st_size/1024/1024:.0f} MB)")
    return target


# ===================== System info =====================
def system_info(run_dir: Path):
    """학습 시작 시 환경 정보 저장."""
    info = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import timm
        info["timm"] = timm.__version__
    except Exception:
        pass
    (run_dir / "system_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ===================== Resource guard =====================
def gpu_mem_used_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
            creationflags=(0x08000000 if os.name == "nt" else 0),
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


# ===================== Module-level callable for Lambda transforms =====================
class AddGaussianNoise:
    """Picklable transform (lambda 대체 — Windows DataLoader spawn safe)."""
    def __init__(self, std: float = 0.02):
        self.std = std
    def __call__(self, x):
        return (x + torch.randn_like(x) * self.std).clamp(0, 1)
