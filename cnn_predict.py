#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN classifier prediction — best_model.pth (또는 legacy best.pt) 로드 → 입력 이미지 분류.

사용:
    # 폴더 내 모든 PNG 분류
    python cnn_predict.py --model log/<run>/best_model.pth --input <dir> --output preds.json

    # threshold 적용 (max_prob < threshold → "Normal/unknown" 처리)
    python cnn_predict.py --model best_model.pth --input <dir> --threshold 0.7 --output preds.json

    # threshold sweep — 입력이 {class}/img.png 구조일 때 (label 추정 가능)
    python cnn_predict.py --model best_model.pth --input val_dir --threshold-sweep 0.1,0.9,0.05

    # 입력이 {class}/img.png 구조면 per_class_report.txt 동시 생성
    python cnn_predict.py --model best_model.pth --input test_dir --report-out report.txt --output preds.json

    # EMA shadow weights 사용 (체크포인트에 ema_state 있을 때)
    python cnn_predict.py --model best_model.pth --input <dir> --ema --output preds.json
"""
import os, sys, json, argparse, glob
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

try:
    from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                                 classification_report, confusion_matrix)
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def build_eval_transform(img_size: int):
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), norm,
    ])


class ImageList(Dataset):
    def __init__(self, paths, tfm, labels: Optional[List[int]] = None):
        self.paths = paths; self.tfm = tfm; self.labels = labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            x = self.tfm(im.convert("RGB"))
        l = self.labels[i] if self.labels is not None else -1
        return x, str(p), l


def collect_images(input_path: str, recursive: bool = True, classes: Optional[List[str]] = None):
    """Returns (paths, labels). labels are -1 if class folder structure not detected.
    If `classes` provided and parent folder name matches a class, infer label.
    """
    p = Path(input_path)
    if p.is_file():
        return [str(p)], [-1]
    if not p.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")
    pattern = "**/*.png" if recursive else "*.png"
    paths_all: List[str] = []
    for ext in ("png", "jpg", "jpeg"):
        pat = pattern.replace("png", ext) if ext != "png" else pattern
        paths_all.extend(sorted(str(x) for x in p.glob(pat)))
    paths_all = sorted(set(paths_all))

    labels: List[int] = []
    if classes is not None:
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        for path in paths_all:
            rel = Path(path).relative_to(p) if p in Path(path).parents or p == Path(path).parent else Path(path)
            parent_name = Path(path).parent.name
            labels.append(cls_to_idx.get(parent_name, -1))
    else:
        labels = [-1] * len(paths_all)
    return paths_all, labels


def apply_ema_state(model, ema_state, device):
    """체크포인트의 ema_state shadow weights를 model에 직접 주입."""
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
        lines.append(f"{c:<32} {d['f1-score']:7.3f} {d['precision']:7.3f} {d['recall']:7.3f} {FP:5d} {FN:5d} {sup:5d}")
    lines.append("-" * len(header))
    for k in ("macro avg", "weighted avg"):
        d = rep[k]
        lines.append(f"{k:<32} {d['f1-score']:7.3f} {d['precision']:7.3f} {d['recall']:7.3f} {'-':>5} {'-':>5} {int(d['support']):5d}")
    lines.append(f"{'overall acc':<32} {acc:7.3f}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_threshold_sweep(s: Optional[str]):
    if not s: return None
    try:
        a, b, step = s.split(",")
        return np.arange(float(a), float(b) + 1e-9, float(step)).tolist()
    except Exception as e:
        raise ValueError(f"Invalid --threshold-sweep '{s}'. Format: lo,hi,step (e.g. 0.1,0.9,0.05)") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best_model.pth (또는 legacy best.pt)")
    ap.add_argument("--input", required=True, help="이미지 또는 폴더")
    ap.add_argument("--output", default=None, help="예측 결과 JSON")
    ap.add_argument("--report-out", default=None, help="입력이 {class}/img.png 구조면 per_class_report.txt 출력")
    ap.add_argument("--threshold", type=float, default=None,
                    help="max_prob < threshold면 'Normal/unknown' 처리")
    ap.add_argument("--threshold-sweep", default=None,
                    help="lo,hi,step (label 추정 가능 시 threshold별 metrics 비교)")
    ap.add_argument("--ema", action="store_true",
                    help="ckpt에 ema_state 있으면 그것을 model에 load")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-wrong-out", default=None,
                    help="틀린 예측을 <DIR>/<true_class>/<pred_class>/*.png 트리로 저장 (label 추정 가능 시)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load checkpoint (best_model.pth 또는 legacy best.pt) ===
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    classes = ckpt.get("classes")
    if classes is None:
        # legacy fallback
        classes = ckpt.get("class_to_idx") and list(ckpt["class_to_idx"].keys())
    if not classes:
        raise RuntimeError(f"checkpoint에 'classes' 정보 없음: {args.model}")
    img_size = ckpt.get("img_size", 384)
    backbone = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))

    # state_dict key 처리: both `model` and `state_dict` and bare
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    if args.ema and "ema_state" in ckpt:
        apply_ema_state(model, ckpt["ema_state"], device)
        print(f"[*] EMA shadow weights applied", file=sys.stderr)

    # === Collect inputs ===
    paths, labels = collect_images(args.input, recursive=not args.no_recursive, classes=classes)
    if not paths:
        print(f"[!] No images in: {args.input}", file=sys.stderr); sys.exit(1)
    has_labels = any(l >= 0 for l in labels)
    print(f"[*] Loaded {len(paths)} images, {len(classes)} classes, img_size={img_size}, labels_inferred={has_labels}", file=sys.stderr)

    tfm = build_eval_transform(img_size)
    ds = ImageList(paths, tfm, labels=labels if has_labels else None)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"))

    results = []
    all_preds = []; all_labels = []; all_max_probs = []
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
                    "pred_idx": int(pi), "max_prob": mp,
                    "is_normal": bool(is_normal),
                    "probs": {classes[k]: float(probs_np[i, k]) for k in range(len(classes))},
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
        print(json.dumps(results[:10], indent=2, ensure_ascii=False))                  # stdout: head only
        if len(results) > 10:
            print(f"... ({len(results) - 10} more) — use --output to save full", file=sys.stderr)

    # === per_class_report ===
    if args.report_out and has_labels and HAVE_SKLEARN:
        acc = float(accuracy_score(all_labels, all_preds))
        per_class_report_txt(all_labels, all_preds, classes, Path(args.report_out), acc)
        print(f"[*] per_class_report → {args.report_out}", file=sys.stderr)
    elif args.report_out and not has_labels:
        print(f"[!] --report-out: input is not {{class}}/img.png structure, skipping report", file=sys.stderr)

    # === Wrong tree (true_class/pred_class/*.png) ===
    if args.save_wrong_out and has_labels:
        import shutil as _shutil
        out_root = Path(args.save_wrong_out)
        if out_root.exists():
            _shutil.rmtree(out_root, ignore_errors=True)
        out_root.mkdir(parents=True, exist_ok=True)
        n_wrong = 0
        for rec in results:
            li = rec.get("true_idx")
            pi = rec.get("pred_idx")
            if li is None or li < 0 or li == pi:
                continue
            d = out_root / classes[li] / classes[pi]
            d.mkdir(parents=True, exist_ok=True)
            try:
                _shutil.copy2(rec["path"], d / Path(rec["path"]).name)
                n_wrong += 1
            except Exception:
                pass
        print(f"[*] wrong tree: {n_wrong} files -> {out_root}", file=sys.stderr)
    elif args.save_wrong_out and not has_labels:
        print("[!] --save-wrong-out: label 추정 불가 (입력이 {class}/img.png 구조 아님)", file=sys.stderr)

    # === Threshold sweep ===
    sweep = parse_threshold_sweep(args.threshold_sweep)
    if sweep and has_labels and HAVE_SKLEARN:
        print(f"\n[Threshold sweep] (label_known={sum(l>=0 for l in all_labels)}/{len(all_labels)})", file=sys.stderr)
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
