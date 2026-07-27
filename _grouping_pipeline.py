#!/usr/bin/env python3
# ★ 무라벨 신규 웨이퍼 Grouping 파이프라인 (governing plan 구현).
# 한 pool 을 frozen(FCMAE) 또는 adapted(backbone+proj) encoder 로 임베딩 → HDBSCAN grouping →
# 품질게이트(noise% / over-merge≥20% / bootstrap stability / representative coherence) →
# 출력스키마(group_id, group_size, group_stability, review_status, model_mode) + representatives.
# 라벨은 (있으면) review/background 판정·모니터에만. hyperparam 은 고정(신규 데이터마다 재탐색 X).
#
# 사용:
#   python _grouping_pipeline.py --pool <dir[,dir2]> --mode frozen  --backbone <fcmae.pth> --out <dir>
#   python _grouping_pipeline.py --pool <dir[,dir2]> --mode adapted --backbone <bb.pth> --proj <proj_ep.pt> --out <dir>
import argparse, csv, sys, json
from pathlib import Path
import numpy as np
import torch, timm
import torch.nn as nn, torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import predict_grouping_prod as pg   # save_grouping_representatives 재사용
from _common import resolve_pool  # noqa: E402  (manifest-based --pool selection, backward-compat)

# --- 고정 HDBSCAN recipe (canonical, 신규 데이터마다 재탐색 금지) ---
MCS, MS, METHOD = 12, 3, "eom"
IMG = 384
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TF = T.Compose([T.Resize((IMG, IMG), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), NORM])

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

def hdbscan_fit(z):
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, metric="euclidean",
                         cluster_selection_method=METHOD)
    return cl.fit_predict(z.astype(np.float64)).astype(int)   # z 는 L2 정규화됨 → euclidean≈cosine

def list_pool(roots):
    # r: 기존 디렉토리(무변경, 후방호환) 또는 .json manifest (make_pool_manifest.py 생성).
    paths, labels = [], []
    for r in roots:
        p, l = resolve_pool(r)
        paths.extend(p); labels.extend(l)
    return paths, labels

def embed(paths, backbone, proj):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i+32]]).to(DEV)
            pool = backbone.forward_features(x).mean(dim=(2, 3))
            z = proj(pool) if proj is not None else pool
            out.append(F.normalize(z, dim=1).cpu())
    return F.normalize(torch.cat(out), dim=1).numpy().astype("float32")

def bootstrap_stability(z, base_pred, n_boot=5, frac=0.8, seed=42):
    """부분표본 재클러스터 → 공유점 라벨 일치도(ARI) 평균. 높을수록 안정."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(seed)
    n = len(z); aris = []
    for b in range(n_boot):
        idx = rng.choice(n, int(frac * n), replace=False)
        idx.sort()
        p2 = hdbscan_fit(z[idx])
        a, b2 = base_pred[idx], p2
        m = (a != -1) & (b2 != -1)
        if m.sum() > 10:
            aris.append(adjusted_rand_score(a[m], b2[m]))
    return float(np.mean(aris)) if aris else 0.0

def per_group_stability(z, pred, n_boot=5, frac=0.8, seed=42):
    """group 별 안정도: 부분표본에서 같은 group 멤버가 한 cluster 로 유지되는 평균 비율(근사)."""
    # 근사: 각 group 의 intra-cluster 평균 cosine (coherence) 로 대체 표기, 별도 bootstrap 전역값과 병행
    gs = {}
    for c in sorted(set(pred.tolist())):
        if c == -1: continue
        zi = z[pred == c]
        if len(zi) < 2: gs[c] = 1.0; continue
        sim = zi @ zi.T
        iu = np.triu_indices(len(zi), 1)
        gs[c] = float(sim[iu].mean())
    return gs

def analyze(paths, labels, z, out_dir, mode, bg_classes):
    pred = hdbscan_fit(z)
    n = len(pred)
    noise_pct = 100.0 * (pred == -1).sum() / n
    clusters = sorted(int(c) for c in set(pred.tolist()) if c != -1)
    sizes = {c: int((pred == c).sum()) for c in clusters}
    over_merge = [c for c, s in sizes.items() if s >= 0.20 * n]     # ≥20% 과병합
    global_stab = bootstrap_stability(z, pred)
    coh = per_group_stability(z, pred)
    # background: group 내 다수 라벨이 bg_classes 면 background_excluded (라벨 있을 때만; group_id 유지)
    lab = np.array(labels)
    rows = []
    for c in clusters:
        idx = np.where(pred == c)[0]
        maj = ""
        if len(lab) == len(pred):
            vals, cnts = np.unique(lab[idx], return_counts=True)
            maj = str(vals[cnts.argmax()])
        status = "ok"
        if c in over_merge: status = "over_merged"
        elif maj.lower() in {b.lower() for b in bg_classes}: status = "background_excluded"
        rows.append({"group_id": c, "group_size": sizes[c],
                     "group_stability": round(coh.get(c, 0.0), 4),
                     "review_status": status, "model_mode": mode,
                     "majority_label": maj})
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "group_size", "group_stability",
                                          "review_status", "model_mode", "majority_label"])
        w.writeheader(); w.writerows(rows)
    summary = {"mode": mode, "n": n, "k": len(clusters), "noise_pct": round(noise_pct, 2),
               "over_merge_groups": over_merge, "bootstrap_stability": round(global_stab, 4),
               "mean_coherence": round(float(np.mean(list(coh.values()))) if coh else 0.0, 4),
               "n_ok_groups": sum(1 for r in rows if r["review_status"] == "ok")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        pg.SAVE_REFERENCE_COMPOSITES = False   # real 웨이퍼는 palette 아님 → composite 우회
        pg.save_grouping_representatives(out_dir, z, pred, paths, per_cluster=12)
    except Exception as e:
        print(f"[reps skipped] {e}", flush=True)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--mode", required=True, choices=["frozen", "adapted"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--proj", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg-classes", default="Normal,R,Random")
    a = ap.parse_args()
    roots = a.pool.split(",")
    paths, labels = list_pool(roots)
    print(f"[data] {len(paths)} imgs, {len(set(labels))} label-dirs", flush=True)
    bb = load_backbone(a.backbone)
    proj = None
    if a.mode == "adapted":
        assert a.proj, "adapted 모드는 --proj 필요"
        d = torch.load(a.proj, map_location="cpu")
        pj = d["proj"] if "proj" in d else d
        pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in pj.items()}
        proj = build_proj(); proj.load_state_dict(pj); proj.eval().to(DEV)
    z = embed(paths, bb, proj)
    s = analyze(paths, labels, z, a.out, a.mode, set(a.bg_classes.split(",")))
    print(f"\n=== {a.mode} grouping summary ===", flush=True)
    print(json.dumps(s, indent=2, ensure_ascii=False), flush=True)
    print(f"[OUT] {Path(a.out).resolve()}", flush=True)
    print("[DONE_GROUPING]", flush=True)

if __name__ == "__main__":
    main()
