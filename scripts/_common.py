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
EXTERNAL_IMAGE_ROOT = Path("E:/data/images")

def project_path(*parts) -> Path:
    """프로젝트 루트 기준 경로 — 어떤 머신에서도 clone 후 동일."""
    return PROJECT_ROOT.joinpath(*parts)


def image_dataset_path(name: str) -> Path:
    """E: 드라이브의 단일 물리 데이터셋 경로를 반환한다."""
    return EXTERNAL_IMAGE_ROOT / name


def resolve_path(p) -> Path:
    """absolute → 그대로, relative → PROJECT_ROOT 기준.

    ★ 사용자 directive (260726): 이미지는 E:/data/images/ 원본을 직접 사용한다.
       D:/project/unknown-contrastive/data/images/ fallback 또는 링크는 사용하지 않는다.
    """
    p = Path(p)
    s = str(p).replace("\\", "/")
    if s.lower().startswith("e:/"):
        return p
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def mask_palette_non_grade_to_white(img, enabled: bool = True, keep_background: bool | None = None):
    """Palette PNG 입력에서 grade 0~7와 경계(index 10)만 색을 유지하고 나머지 index는 흰색 처리.

    원본 파일은 건드리지 않고, Image.open(...) 뒤 convert("RGB") 직전에만 in-memory palette를
    바꾼다. wafer border/background/invalid 계열 색이 학습 shortcut이 되는 것을 막기 위함.

    UC_PALETTE_MODE:
        - grade_only: index 0~7 + 경계(10) 유지 (default)
        - grade_bg: index 0~8 + 경계(10) 유지. unknown palette background 보존 실험용.
      - raw: palette를 그대로 둠.
    """
    if not enabled or getattr(img, "mode", None) != "P":
        return img
    mode = os.environ.get("UC_PALETTE_MODE", "grade_only").strip().lower()
    if mode in {"raw", "none", "off"}:
        return img
    if keep_background is None:
        keep_background = mode in {"grade_bg", "grade_background", "background"}
    border_idx = 10
    bg_idx = 8
    palette = img.getpalette()
    if not palette:
        return img
    pal = list(palette)
    if len(pal) < 768:
        pal.extend([0] * (768 - len(pal)))
    keep = set(range(8)) | {border_idx}
    if keep_background:
        keep.add(bg_idx)
    for idx in range(256):
        if idx not in keep:
            pal[idx * 3:idx * 3 + 3] = [255, 255, 255]
    img.putpalette(pal[:768])
    return img


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


# ===================== Pool resolution (single-master manifest selection) =====================
# 사용자 지시(260726): "데이터셋은 1개로 운영하고 코드에서 선택하는 걸로 바꿔라."
# 파생 pool(예: unknown_eval100)을 물리 폴더 복사본이 아니라, E: 마스터
# (E:/data/images/unknown) 위 파일 목록 manifest 로 정의할 수 있게 한다.
# --pool 인자가 디렉토리면 기존 동작 그대로(무변경, 후방호환), .json 이면 manifest 모드.
# manifest 는 scripts/make_pool_manifest.py 로 생성.
POOL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_pool_manifest(manifest_path) -> list:
    """Load a pool manifest JSON: {"root": <str|null>, "files": [{"path": ..., "label": ...}, ...]}.

    Returns a list of (resolved_absolute_path, label_or_None) tuples in
    manifest file order (unsorted — callers apply their own ordering rule).
    ``root``, if given, is resolved via resolve_path() and joined onto each
    relative "path"; an already-absolute "path" is used as-is. "label" may
    be absent or null for unlabeled real data — callers must fall back
    gracefully (e.g. to the parent directory name) rather than assume it is
    present.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest.get("root")
    root_dir = resolve_path(root) if root else None
    entries = []
    for item in manifest["files"]:
        rel = Path(item["path"])
        resolved = (root_dir / rel) if (root_dir is not None and not rel.is_absolute()) else rel
        entries.append((resolved, item.get("label")))
    return entries


def resolve_pool(pool_arg, extensions=POOL_IMAGE_EXTENSIONS):
    """Resolve a --pool CLI argument into (paths, labels).

    ``pool_arg`` is either:
      - a directory: unchanged, pre-existing behavior — sorted immediate
        class-subdirectories, each rglob'd (sorted) for image files; label =
        class subdir name. Byte-for-byte identical to the inline loops this
        replaces in _grouping_eval.py / _grouping_deliverable.py /
        _grouping_pipeline.py.
      - a ``.json`` manifest file: resolved via load_pool_manifest(), then
        sorted by (label, path) — mirroring the two-level directory sort
        above exactly (label first = the sorted-class-dir pass, path second
        = the sorted-rglob-within-dir pass) — so a manifest generated from a
        directory (scripts/make_pool_manifest.py) reproduces the exact same
        (paths, labels) as pointing --pool at that directory would.
    """
    p = Path(pool_arg)
    if p.is_dir():
        paths, labels = [], []
        for cd in sorted(p.iterdir()):
            if cd.is_dir():
                for f in sorted(cd.rglob("*")):
                    if f.suffix.lower() in extensions:
                        paths.append(f)
                        labels.append(cd.name)
        return [str(x) for x in paths], labels
    if p.is_file() and p.suffix.lower() == ".json":
        kept = []
        for path, label in load_pool_manifest(p):
            if path.suffix.lower() not in extensions:
                continue
            kept.append((path, label if label is not None else path.parent.name))
        kept.sort(key=lambda e: (Path(e[1]), e[0]))
        return [str(path) for path, _ in kept], [lab for _, lab in kept]
    raise ValueError(f"--pool must be an existing directory or a .json manifest file, got: {pool_arg}")
