#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compound CNN classifier prediction — wafer 33-class with R(failbit) + G(obj_id) + B(zero).

cnn_predict.py 의 compound 버전. checkpoint 와 obj_id_maps/_meta.json 을 읽어
3-channel 입력을 동일하게 구성하고 분류한다.

사용:
    # 폴더 내 모든 PNG (compound 모델로 추론)
    python cnn_predict_compound.py --model logs_compound/<run>/best_model.pth \\
        --input <wafer_png_dir> --obj-id-root D:/project/data/wm-811k/obj_id_maps \\
        --output preds.json

    # threshold (max_prob < threshold → "Normal/unknown")
    python cnn_predict_compound.py --model logs_compound/<run>/best_model.pth \\
        --input <dir> --obj-id-root <dir> --threshold 0.7 --output preds.json

    # threshold sweep (입력이 {class}/img.png 구조 → label 추론 가능)
    python cnn_predict_compound.py --model logs_compound/<run>/best_model.pth \\
        --input val_dir --obj-id-root <dir> --threshold-sweep 0.1,0.9,0.05

obj_id .npy lookup: --obj-id-root 아래 모든 .npy 를 flat basename → path 로 인덱싱
(서브폴더 구조 무관: <wafer_class>/, <device>_<date>/ 등 어떤 layout 도 호환).
"""
# ===================== CONFIG =====================
DEFAULT_MODEL_GLOB   = "logs_compound/overall/best_model.pth"
DEFAULT_INPUT        = "D:/project/data/wm-811k/unknown"
DEFAULT_OBJ_ID_ROOT  = "D:/project/data/wm-811k/obj_id_maps"
DEFAULT_PREDICT_ROOT = "logs_predict_compound"
KIND_LABEL           = "compound"
PALETTE_IDX_NORM     = 31  # palette idx 31 = invalid_fill — sample_gen 도메인 spec, 고정
# ==================================================

import os, sys, json, argparse, glob, time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm

try:
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _resolve_default_model(default_glob: Optional[str]) -> Optional[str]:
    if not default_glob: return None
    matches = sorted(glob.glob(default_glob))
    return matches[-1] if matches else None


def _print_overall_meta_if_any(model_path: str):
    p = Path(model_path)
    meta_path = p.parent / "_overall_meta.json"
    if not meta_path.exists(): return
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[*] using overall: {model_path}", file=sys.stderr)
        print(f"[*]   sourced from run='{m.get('best_run')}'", file=sys.stderr)
        print(f"[*]   val_f1={m.get('val_f1'):.4f}  seeded_at={m.get('seeded_at')}", file=sys.stderr)
    except Exception as e:
        print(f"[*] using overall: {model_path}  (meta read failed: {e})", file=sys.stderr)


def _make_predict_run_dir(predict_root: str, input_path: str) -> Path:
    ts = time.strftime("%y%m%d_%H%M%S")
    tag = Path(input_path).name or "out"
    run_dir = Path(predict_root) / f"{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class CompoundWaferDataset(Dataset):
    """Build 3-channel feature tensor from (palette PNG + obj_id .npy).

    R = palette_idx / PALETTE_IDX_NORM   (BICUBIC to img_size)
    G = obj_id     / n_chip_objects       (BICUBIC; map shape derived from .npy)
    B = zeros
    Then ImageNet mean/std normalize for backbone-init compat.
    """
    def __init__(self, paths: List[str], img_size: int, n_chip_objects: int,
                 npy_map: Dict[str, Path], labels: Optional[List[int]] = None):
        self.paths = paths
        self.img_size = int(img_size)
        self.n_chip_objects = int(n_chip_objects)
        self.npy_map = npy_map
        self.labels = labels
        self._missing_warned = False

    def __len__(self): return len(self.paths)

    def _load_3ch(self, png_path: str) -> torch.Tensor:
        img = Image.open(png_path)
        if img.mode != "P":
            img = img.convert("P")
        idx = np.asarray(img, dtype=np.uint8)
        idx_pil = Image.fromarray(idx, mode="L")
        idx_resized = idx_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        r = torch.from_numpy(np.asarray(idx_resized, dtype=np.float32) / float(PALETTE_IDX_NORM)) \
                  .clamp_(0.0, 1.0).unsqueeze(0)

        basename = Path(png_path).stem
        obj_path = self.npy_map.get(basename)
        if obj_path is not None and obj_path.exists():
            obj_id = np.load(obj_path).astype(np.uint8)
            obj_pil = Image.fromarray(obj_id, mode="L")
            obj_resized = obj_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
            g = torch.from_numpy(np.asarray(obj_resized, dtype=np.float32) / float(self.n_chip_objects)) \
                      .clamp_(0.0, 1.0).unsqueeze(0)
        else:
            g = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            if not self._missing_warned:
                print(f"[predict] missing obj_id for basename={basename!r} — G=zeros (warn once)",
                      file=sys.stderr, flush=True)
                self._missing_warned = True

        b = torch.zeros_like(r)
        x = torch.cat([r, g, b], dim=0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x

    def __getitem__(self, i):
        x = self._load_3ch(self.paths[i])
        l = self.labels[i] if self.labels is not None else -1
        return x, str(self.paths[i]), l


def collect_images(input_path: str, recursive: bool, classes: Optional[List[str]] = None):
    p = Path(input_path)
    if p.is_file():
        return [str(p)], [-1]
    if not p.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")
    pat = "**/*.png" if recursive else "*.png"
    paths = sorted(set(str(x) for x in p.glob(pat)))
    labels: List[int] = []
    if classes is not None:
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        for path in paths:
            labels.append(cls_to_idx.get(Path(path).parent.name, -1))
    else:
        labels = [-1] * len(paths)
    return paths, labels


def build_npy_map(obj_id_root: Path) -> Dict[str, Path]:
    """flat basename → npy_path (서브폴더 구조 agnostic)."""
    if not obj_id_root.exists():
        print(f"[!] obj-id-root not found: {obj_id_root}", file=sys.stderr)
        return {}
    return {p.stem: p for p in obj_id_root.rglob("*.npy")}


def apply_ema_state(model, ema_state, device):
    p_dict = dict(model.named_parameters())
    for n, v in ema_state.items():
        if n in p_dict:
            p_dict[n].data.copy_(v.to(device))


def per_class_report_txt(labels, preds, classes, out_path: Path, acc: float):
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    rep = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    lines = []
    header = f"{'Class':<32} {'F1':>7} {'P':>7} {'R':>7} {'FP':>5} {'FN':>5} {'Sup':>5}"
    lines.append(header); lines.append("-" * len(header))
    for i, c in enumerate(classes):
        d = rep.get(c, {"precision":0, "recall":0, "f1-score":0, "support":0})
        FP = int(cm[:, i].sum() - cm[i, i])
        FN = int(cm[i, :].sum() - cm[i, i])
        sup = int(d.get("support", 0))
        lines.append(f"{c:<32} {d['f1-score']:7.3f} {d['precision']:7.3f} {d['recall']:7.3f} "
                     f"{FP:5d} {FN:5d} {sup:5d}")
    lines.append("-" * len(header))
    for k in ("macro avg", "weighted avg"):
        d = rep[k]
        lines.append(f"{k:<32} {d['f1-score']:7.3f} {d['precision']:7.3f} {d['recall']:7.3f} "
                     f"{'-':>5} {'-':>5} {int(d['support']):5d}")
    lines.append(f"{'overall acc':<32} {acc:7.3f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_threshold_sweep(s: Optional[str]):
    if not s: return None
    a, b, step = s.split(",")
    return np.arange(float(a), float(b) + 1e-9, float(step)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help=f"compound best_model.pth (default: {DEFAULT_MODEL_GLOB})")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"wafer PNG 폴더 또는 단일 PNG (default: {DEFAULT_INPUT})")
    ap.add_argument("--obj-id-root", default=DEFAULT_OBJ_ID_ROOT,
                    help=f"obj_id .npy 루트 (default: {DEFAULT_OBJ_ID_ROOT})")
    ap.add_argument("--predict-root", default=DEFAULT_PREDICT_ROOT,
                    help=f"logs_predict_compound root for auto run dir (default: {DEFAULT_PREDICT_ROOT})")
    ap.add_argument("--no-run-dir", action="store_true",
                    help="--predict-root 무시, 자동 run dir 생성 안 함")
    ap.add_argument("--output", default=None, help="예측 결과 JSON (default: <run_dir>/preds.json)")
    ap.add_argument("--report-out", default=None,
                    help="per_class_report.txt (default: <run_dir>/per_class_report.txt)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="max_prob < threshold면 'Normal/unknown' 처리")
    ap.add_argument("--threshold-sweep", default=None,
                    help="lo,hi,step (label 추정 가능 시)")
    ap.add_argument("--ema", action="store_true",
                    help="ckpt 의 ema_state 를 model 에 적용")
    ap.add_argument("--n-chip-objects", type=int, default=None,
                    help="G normalization 분모. default: <obj-id-root>/_meta.json 의 n_chip_objects 자동 읽기")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-wrong-out", default=None,
                    help="wrong tree DIR (default: <run_dir>/wrong)")
    args = ap.parse_args()

    if args.model is None:
        args.model = _resolve_default_model(DEFAULT_MODEL_GLOB)
        if not args.model:
            raise SystemExit(f"--model not given and default not found: {DEFAULT_MODEL_GLOB}")
    print(f"[*] cnn_predict kind={KIND_LABEL}", file=sys.stderr)
    _print_overall_meta_if_any(args.model)

    # auto-create predict run dir + route default outputs into it
    if args.predict_root and not args.no_run_dir:
        run_dir = _make_predict_run_dir(args.predict_root, args.input)
        if args.output is None:         args.output = str(run_dir / "preds.json")
        if args.report_out is None:     args.report_out = str(run_dir / "per_class_report.txt")
        if args.save_wrong_out is None: args.save_wrong_out = str(run_dir / "wrong")
        print(f"[*] predict run_dir: {run_dir}", file=sys.stderr)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # === Load checkpoint ===
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    classes = ckpt.get("classes")
    if not classes:
        raise RuntimeError(f"checkpoint에 'classes' 정보 없음: {args.model}")
    img_size = int(ckpt.get("img_size", 384))
    backbone = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))
    state = (ckpt.get("model") if "model" in ckpt and isinstance(ckpt["model"], dict)
             else ckpt.get("state_dict") if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict)
             else ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    if args.ema and "ema_state" in ckpt:
        apply_ema_state(model, ckpt["ema_state"], device)
        print("[*] EMA shadow weights applied", file=sys.stderr)

    # === n_chip_objects ===
    obj_root = Path(args.obj_id_root)
    if args.n_chip_objects is not None:
        n_chip_objects = int(args.n_chip_objects)
    else:
        meta_path = obj_root / "_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                n_chip_objects = int(meta.get("n_chip_objects"))
                print(f"[*] n_chip_objects={n_chip_objects} (from {meta_path})", file=sys.stderr)
            except Exception as e:
                raise RuntimeError(f"failed to read n_chip_objects from {meta_path}: {e}")
        else:
            raise RuntimeError(f"--n-chip-objects 미지정 + {meta_path} 없음. 둘 중 하나 필요.")

    # === npy lookup map ===
    npy_map = build_npy_map(obj_root)
    print(f"[*] obj_id npy indexed: {len(npy_map)} files under {obj_root}", file=sys.stderr)

    # === Collect inputs ===
    paths, labels = collect_images(args.input, recursive=not args.no_recursive, classes=classes)
    if not paths:
        print(f"[!] No images: {args.input}", file=sys.stderr); sys.exit(1)
    has_labels = any(l >= 0 for l in labels)
    print(f"[*] Loaded {len(paths)} images, {len(classes)} classes, "
          f"img_size={img_size}, labels_inferred={has_labels}", file=sys.stderr)

    ds = CompoundWaferDataset(paths, img_size, n_chip_objects, npy_map,
                              labels=labels if has_labels else None)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type == "cuda"))

    results: List[dict] = []
    all_preds: List[int] = []; all_labels: List[int] = []; all_max_probs: List[float] = []

    with torch.no_grad():
        for xb, pb, lb in ld:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb); probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            probs_np = probs.cpu().numpy()
            for i in range(xb.size(0)):
                pi = int(preds[i]); mp = float(confs[i])
                pcls = classes[pi]
                is_normal = (args.threshold is not None and mp < args.threshold)
                rec = {
                    "path": pb[i],
                    "pred_class": "Normal/unknown" if is_normal else pcls,
                    "pred_idx": pi, "max_prob": mp,
                    "is_normal": bool(is_normal),
                    "probs": {classes[k]: float(probs_np[i, k]) for k in range(len(classes))},
                    "obj_id_npy": str(npy_map[Path(pb[i]).stem]) if Path(pb[i]).stem in npy_map else None,
                }
                if has_labels:
                    rec["true_idx"] = int(lb[i])
                    rec["true_class"] = classes[int(lb[i])] if int(lb[i]) >= 0 else None
                results.append(rec)
                all_preds.append(pi); all_max_probs.append(mp)
                if has_labels: all_labels.append(int(lb[i]))

    # === Output JSON ===
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[*] Saved {len(results)} predictions → {args.output}", file=sys.stderr)
    else:
        print(json.dumps(results[:10], indent=2, ensure_ascii=False))
        if len(results) > 10:
            print(f"... ({len(results) - 10} more) — use --output to save full", file=sys.stderr)

    # === per_class_report ===
    if args.report_out and has_labels and HAVE_SKLEARN:
        acc = float(accuracy_score(all_labels, all_preds))
        per_class_report_txt(all_labels, all_preds, classes, Path(args.report_out), acc)
        print(f"[*] per_class_report → {args.report_out}", file=sys.stderr)

    # === Wrong tree ===
    if args.save_wrong_out and has_labels:
        import shutil as _shutil
        out_root = Path(args.save_wrong_out)
        if out_root.exists():
            _shutil.rmtree(out_root, ignore_errors=True)
        out_root.mkdir(parents=True, exist_ok=True)
        n_wrong = 0
        for rec in results:
            li = rec.get("true_idx"); pi = rec.get("pred_idx")
            if li is None or li < 0 or li == pi: continue
            d = out_root / classes[li] / classes[pi]
            d.mkdir(parents=True, exist_ok=True)
            try:
                _shutil.copy2(rec["path"], d / Path(rec["path"]).name)
                n_wrong += 1
            except Exception:
                pass
        print(f"[*] wrong tree: {n_wrong} files -> {out_root}", file=sys.stderr)

    # === Threshold sweep ===
    sweep = parse_threshold_sweep(args.threshold_sweep)
    if sweep and has_labels and HAVE_SKLEARN:
        print(f"\n[Threshold sweep] (label_known={sum(l>=0 for l in all_labels)}/{len(all_labels)})",
              file=sys.stderr)
        print(f"{'thresh':>8} {'normal_rate':>12} {'acc_kept':>9} {'kept_n':>7}", file=sys.stderr)
        for th in sweep:
            kept = [(p, l) for p, l, mp in zip(all_preds, all_labels, all_max_probs) if mp >= th]
            normal_rate = 1.0 - len(kept) / max(1, len(all_preds))
            if not kept:
                print(f"{th:8.3f} {normal_rate:12.3f} {'-':>9} {0:7d}", file=sys.stderr); continue
            preds_k, lbls_k = zip(*kept)
            acc_k = float(accuracy_score(lbls_k, preds_k))
            print(f"{th:8.3f} {normal_rate:12.3f} {acc_k:9.3f} {len(kept):7d}", file=sys.stderr)


if __name__ == "__main__":
    main()
