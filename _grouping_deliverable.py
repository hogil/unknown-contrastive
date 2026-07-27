#!/usr/bin/env python3
# ★ 최종 산출물 생성기 (governing plan): 선택된 adapted 모델로 무라벨 grouping 산출.
#   - groups.csv (label-free): group_id, group_size, group_stability, group_coherence, review_status, model_mode
#   - representatives/group_XXX/: centroid-nearest 대표 이미지 복사
#   - composites/group_XXX.png: 그룹 멤버 평균 이미지 (공유 결함 패턴 시각화)
#   - summary.json (label-free)
#   - offline_eval.csv (★별도, 숨긴 라벨): per-group majority + 전체 P1-P4
# 라벨은 offline_eval 에만. 런타임(groups.csv/representatives/composites/summary)엔 라벨 누수 0.
# 사용: python _grouping_deliverable.py --backbone <pth> --proj <proj_ep.pt> --pool <dir> --out <dir> [--tag NAME]
import argparse, sys, csv, json, shutil
from pathlib import Path
import numpy as np
import torch, timm
import torch.nn as nn, torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from _common import resolve_pool  # noqa: E402  (manifest-based --pool selection, backward-compat)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 384
BG = {"Normal", "R", "Random"}
TF = T.Compose([T.Resize((IMG, IMG), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def build_proj():
    return nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024),
                         nn.ReLU(inplace=True), nn.Linear(1024, 128))

def load_backbone(path):
    bb = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=0, global_pool="")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    for p in ("model.", "backbone.", "module."):
        if any(k.startswith(p) for k in sd): sd = {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}
    bb.load_state_dict(sd, strict=False)
    return bb.eval().to(DEV)

def may_hdbscan(z):
    import hdbscan, os
    mcs = int(os.environ.get("DLV_MCS", 12)); ms = int(os.environ.get("DLV_MS", 15))
    meth = os.environ.get("DLV_METHOD", "leaf"); eps = float(os.environ.get("DLV_EPS", 0.06))
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                           cluster_selection_method=meth, cluster_selection_epsilon=eps).fit_predict(z.astype(np.float64)).astype(int)

def per_group_stability(z, base_pred, n_boot=5, frac=0.8, seed=42):
    """group 별 co-assignment stability: 부분표본에서 그 그룹 멤버쌍이 같은 cluster 유지 평균 비율."""
    rng = np.random.RandomState(seed); n = len(z)
    from collections import defaultdict
    keep = defaultdict(list)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        pos = {g: i for i, g in enumerate(idx)}
        p2 = may_hdbscan(z[idx])
        for c in set(base_pred.tolist()):
            if c == -1: continue
            mem = [g for g in np.where(base_pred == c)[0] if g in pos]
            if len(mem) < 2: continue
            lb = np.array([p2[pos[g]] for g in mem])
            same = 0; tot = 0
            for i in range(len(mem)):
                for j in range(i + 1, len(mem)):
                    tot += 1
                    if lb[i] == lb[j] and lb[i] != -1: same += 1
            if tot: keep[c].append(same / tot)
    return {c: float(np.mean(v)) if v else 0.0 for c, v in keep.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--proj", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # --pool: 기존 디렉토리(무변경, 후방호환) 또는 .json manifest (make_pool_manifest.py 생성).
    paths, labels = resolve_pool(a.pool)
    print(f"[data] {len(paths)} imgs", flush=True)

    bb = load_backbone(a.backbone)
    d = torch.load(a.proj, map_location="cpu"); pj = d["proj"] if "proj" in d else d
    pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in pj.items()}
    proj = build_proj(); proj.load_state_dict(pj); proj.eval().to(DEV)

    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i+32]]).to(DEV)
            pool = bb.forward_features(x).mean(dim=(2, 3))       # raw GAP
            embs.append(F.normalize(proj(pool), dim=1).cpu())
    z = F.normalize(torch.cat(embs), dim=1).numpy().astype("float32")

    pred = may_hdbscan(z); n = len(pred)
    clusters = sorted(int(c) for c in set(pred.tolist()) if c != -1)
    stab = per_group_stability(z, pred)
    lab = np.array(labels)

    # --- representatives + composites + groups.csv (label-free) ---
    rep_root = out / "representatives"; comp_root = out / "composites"
    rep_root.mkdir(exist_ok=True); comp_root.mkdir(exist_ok=True)
    rows, off_rows = [], []
    for c in clusters:
        idx = np.where(pred == c)[0]
        center = z[idx].mean(0)
        order = idx[np.argsort(np.linalg.norm(z[idx] - center, axis=1))]
        sim = z[idx] @ z[idx].T; iu = np.triu_indices(len(idx), 1)
        coh = float(sim[iu].mean()) if len(idx) > 1 else 1.0
        over = len(idx) >= 0.20 * n
        # representatives
        gdir = rep_root / f"group_{c:03d}_n{len(idx)}"; gdir.mkdir(exist_ok=True)
        acc = None; cnt = 0
        for rank, i in enumerate(order[:a.reps], 1):
            src = Path(paths[i])
            if src.exists(): shutil.copy2(src, gdir / f"rep{rank:02d}_{src.name}")
        # composite = 멤버 평균 이미지 (공유 패턴)
        for i in idx:
            im = np.asarray(Image.open(paths[i]).convert("L").resize((256, 256)), dtype=np.float32)
            acc = im if acc is None else acc + im; cnt += 1
        if cnt:
            comp = (acc / cnt).clip(0, 255).astype(np.uint8)
            Image.fromarray(comp).save(comp_root / f"group_{c:03d}_n{len(idx)}.png")
        rows.append({"group_id": c, "group_size": len(idx), "group_stability": round(stab.get(c, 0.0), 4),
                     "group_coherence": round(coh, 4),
                     "review_status": "over_merged_review" if over else "candidate",
                     "model_mode": "adapted"})
        # offline (label): majority
        vals, cnts = np.unique(lab[idx], return_counts=True)
        off_rows.append({"group_id": c, "group_size": len(idx),
                         "majority_label": str(vals[cnts.argmax()]),
                         "purity": round(float(cnts.max() / cnts.sum()), 4)})

    with (out / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "group_size", "group_stability", "group_coherence", "review_status", "model_mode"])
        w.writeheader(); w.writerows(rows)

    noise = 100.0 * (pred == -1).sum() / n
    summary = {"n": n, "k": len(clusters), "noise_pct": round(noise, 2),
               "over_merge_groups": [r["group_id"] for r in rows if r["review_status"].startswith("over")],
               "mean_coherence": round(float(np.mean([r["group_coherence"] for r in rows])) if rows else 0, 4),
               "mean_stability": round(float(np.mean([r["group_stability"] for r in rows])) if rows else 0, 4),
               "model_mode": "adapted"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- offline_eval.csv (★별도, 숨긴 라벨) ---
    sys.path.insert(0, "scripts")
    from eval_may37_checkpoints import _summarize_predictions, _expand_ignored, _drop_megaclusters
    ignored = _expand_ignored(lab, set(BG)); measured = ~np.isin(lab, list(ignored))
    fp = _drop_megaclusters(pred.copy(), 0.20)
    r = _summarize_predictions(z[measured], lab[measured], fp[measured], fp, lab, ignored, "deliverable")
    with (out / "offline_eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "group_size", "majority_label", "purity"])
        w.writeheader(); w.writerows(off_rows)
    (out / "offline_summary.json").write_text(json.dumps(
        {"P1_capture": r["P1_capture"], "P2_noise_pct": r["P2_noise_pct"], "P3_completeness": r["P3_completeness"],
         "P4_homogeneity": r["P4_homogeneity"], "Sil_cos": r["Sil_cos"], "fragment_ratio": r["fragment_ratio"],
         "ARI": r["ARI"], "AMI": r["AMI"], "k": r["k"]}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[summary] {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"[offline] P1={r['P1_capture']} P2noise={r['P2_noise_pct']} P4hom={r['P4_homogeneity']} ARI={r['ARI']}", flush=True)
    print(f"[OUT] {out.resolve()}", flush=True)
    print("[DONE_DELIVERABLE]", flush=True)

if __name__ == "__main__":
    main()
