#!/usr/bin/env python3
# ★ 교정 grouping eval (governing plan 2차). 라벨 누수 제거 + 진짜 stability + May HDBSCAN + 무라벨 선택 + 오프라인 P1-P4 분리.
# 한 backbone 에 대해 frozen f / random z0 / adapted ep1..N 을 평가:
#   - 런타임(무라벨): k, noise%, over_merge(≥20%), bootstrap co-assignment stability, coherence, non_noise%
#   - 오프라인(숨긴 라벨): P1(dominant-main /7), P2(defect noise%), P3, P4, Sil, k/7, frag, bg-dom-group수, ARI/AMI(보조)
# 무라벨 선택 ladder: over-merge 탈락 → stability≥0.75 → coherence≥0.80 → (--select-rule 에 따라 선택).
# 사용: python _grouping_eval.py --backbone <pth> --pool <dir> [--proj-dir <ckpt_dir>] [--tag NAME]
# ★ HDBSCAN 다이얼(--mcs/--ms/--eps/--method) 은 optional — default 는 기존 May-dial(12/15/leaf/0.06) 그대로라
#   신규 인자 없이 호출하면 결과가 1비트도 안 바뀐다(후방호환). 다른 다이얼로 재채점할 땐 --out-name 으로
#   출력 파일명을 바꿔 기존 runs/clean546/eval_<tag>.json 을 덮어쓰지 않게 할 것.
# ★ 선택규칙(--select-rule) 도 optional — default "noise" 는 기존 max-non_noise_pct 그대로(후방호환).
#   "rich_noise" = Rule C(260726, `runs/clean546/_selection_rule_260726.md`): gate 통과분을 이 run
#   자신의 k 분포 --k-percentile(기본75) 이상으로 먼저 좁힌 뒤 그 안에서 noise 최소화.
import argparse, sys, glob, re, json, os, hashlib, uuid
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
BG = {"Normal", "R", "Random"}       # 오프라인 배경(런타임엔 미사용)
TF = T.Compose([T.Resize((IMG, IMG), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# May-dial 기본값 (변경 금지 — 사용자 결정 사항). main() 이 --mcs/--ms/--eps/--method 로만 override.
DIAL = {"mcs": 12, "ms": 15, "eps": 0.06, "method": "leaf"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return sha256_file(path)

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
    import hdbscan
    cl = hdbscan.HDBSCAN(min_cluster_size=DIAL["mcs"], min_samples=DIAL["ms"], metric="euclidean",
                         cluster_selection_method=DIAL["method"], cluster_selection_epsilon=DIAL["eps"])
    return cl.fit_predict(z.astype(np.float64)).astype(int)

def bootstrap_stability(z, base_pred, n_boot=5, frac=0.8, seed=42):
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.RandomState(seed); n = len(z); aris = []
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        p2 = may_hdbscan(z[idx]); a = base_pred[idx]
        m = (a != -1) & (p2 != -1)
        if m.sum() > 10: aris.append(adjusted_rand_score(a[m], p2[m]))
    return float(np.mean(aris)) if aris else 0.0

def coherence(z, pred):
    vals = []
    for c in set(pred.tolist()):
        if c == -1: continue
        zi = z[pred == c]
        if len(zi) < 2: continue
        sim = zi @ zi.T; iu = np.triu_indices(len(zi), 1)
        vals.append(float(sim[iu].mean()))
    return float(np.mean(vals)) if vals else 0.0

def label_free(z):
    pred = may_hdbscan(z); n = len(pred)
    clusters = [c for c in set(pred.tolist()) if c != -1]
    sizes = {c: int((pred == c).sum()) for c in clusters}
    over = [c for c, s in sizes.items() if s >= 0.20 * n]
    noise = 100.0 * (pred == -1).sum() / n
    return {"pred": pred, "k": len(clusters), "noise_pct": round(noise, 2),
            "non_noise_pct": round(100 - noise, 2), "over_merge": len(over),
            "stability": round(bootstrap_stability(z, pred), 4),
            "coherence": round(coherence(z, pred), 4)}

def offline(z, labels, pred):
    # ★ 선택에 쓴 그 May-config 클러스터링(pred)으로 P1-P4 계산 (일관성). calculate_metrics 의 내부 재클러스터 미사용.
    sys.path.insert(0, "scripts")
    from eval_may37_checkpoints import _summarize_predictions, _expand_ignored, _drop_megaclusters
    lab = np.asarray(labels)
    ignored = _expand_ignored(lab, set(BG))
    measured = ~np.isin(lab, list(ignored))
    fp = _drop_megaclusters(pred.copy(), 0.20)
    r = _summarize_predictions(z[measured], lab[measured], fp[measured], fp, lab, ignored, "may_grouping")
    bg_dom = 0
    for c in set(pred.tolist()):
        if c == -1: continue
        idx = pred == c
        vals, cnts = np.unique(lab[idx], return_counts=True)
        if str(vals[cnts.argmax()]) in BG: bg_dom += 1
    return {"P1": r["P1_capture"], "P2_noise": r["P2_noise_pct"], "P3_comp": r["P3_completeness"],
            "P4_hom": r["P4_homogeneity"], "Sil": r["Sil_cos"], "frag": r["fragment_ratio"],
            "ARI": r["ARI"], "AMI": r["AMI"], "bg_dom_groups": bg_dom, "cand_groups": r["k"]}

def extract_backbone_features(paths, bb, batch_size=32):
    """Return raw GAP features, matching the projection input used in training."""
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            x = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i+batch_size]]).to(DEV)
            pool = bb.forward_features(x).mean(dim=(2, 3))
            out.append(pool.cpu())
    return torch.cat(out).float()

def embedding_from_features(features, proj):
    """Normalize after projection; never normalize the projection input."""
    with torch.no_grad():
        x = features.to(DEV)
        z = proj(x) if proj is not None else x
        return F.normalize(z, dim=1).cpu().numpy().astype("float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--proj-dir", default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--mcs", type=int, default=12, help="HDBSCAN min_cluster_size (default=May-dial 12)")
    ap.add_argument("--ms", type=int, default=15, help="HDBSCAN min_samples (default=May-dial 15)")
    ap.add_argument("--eps", type=float, default=0.06, help="HDBSCAN cluster_selection_epsilon (default=May-dial 0.06)")
    ap.add_argument("--method", default="leaf", help="HDBSCAN cluster_selection_method (default=May-dial leaf)")
    ap.add_argument("--out-name", default=None,
                    help="output json stem under runs/clean546/ (default=eval_{tag}, unchanged from before). "
                         "Set this when using a non-default dial so the May-dial file is never overwritten.")
    ap.add_argument("--out-dir", default="runs/clean546",
                    help="directory for dial result and pre-offline selection snapshot")
    ap.add_argument("--feat-cache", default=None,
                    help="optional .npy path to cache raw backbone GAP features (paths order-dependent). "
                         "Loaded if present, else computed once and saved. Default off = old behavior.")
    ap.add_argument("--select-rule", default="noise", choices=["noise", "rich_noise"],
                    help="epoch selection tie-break rule among gate-passing epochs. default='noise' = "
                         "original max-non_noise_pct ladder (unchanged, backward-compat). "
                         "'rich_noise' = Rule C (260726, runs/clean546/_selection_rule_260726.md): "
                         "restrict to this run's own gate-passing k >= --k-percentile first (self-normalized "
                         "floor, not an absolute k threshold), then minimize noise within that subset.")
    ap.add_argument("--k-percentile", type=float, default=75,
                    help="percentile (of this run's own gate-passing k distribution) used as the floor for "
                         "--select-rule rich_noise. Default 75; reported stable across 70-85 in the 260726 note. "
                         "Unused when --select-rule=noise.")
    a = ap.parse_args()
    _fraction_raw = os.environ.get("REPRO_GPU_MEMORY_FRACTION")
    if _fraction_raw is not None:
        fraction = float(_fraction_raw)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("REPRO_GPU_MEMORY_FRACTION must be in (0, 1]")
        if DEV != "cuda":
            raise RuntimeError("GPU memory fraction requested but CUDA is unavailable")
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    DIAL.update(mcs=a.mcs, ms=a.ms, eps=a.eps, method=a.method)
    _collapse = os.environ.get("REPRO_COLLAPSE") == "1"   # 폴더명 <Position>_<obj> → Position 만 (위치 grouping)
    # --pool: 기존 디렉토리(무변경, 후방호환) 또는 .json manifest (make_pool_manifest.py 생성).
    paths, raw_labels = resolve_pool(a.pool)
    labels = [lab.split("_")[0] if _collapse else lab for lab in raw_labels]
    print(f"[{a.tag}] {len(paths)} imgs, {len(set(labels))} label-dirs (labels offline-only, collapse={_collapse})", flush=True)
    bb = load_backbone(a.backbone)
    if a.feat_cache and Path(a.feat_cache).exists():
        raw_feats = torch.from_numpy(np.load(a.feat_cache)).float()
        print(f"[{a.tag}] loaded cached raw features from {a.feat_cache}", flush=True)
    else:
        raw_feats = extract_backbone_features(paths, bb)
        if a.feat_cache:
            Path(a.feat_cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(a.feat_cache, raw_feats.numpy())
            print(f"[{a.tag}] cached raw features to {a.feat_cache}", flush=True)

    def z_from(proj):
        return embedding_from_features(raw_feats, proj)

    class Head(nn.Module):   # proj (+ optional residual adapter: f + γ·adapt(f))
        def __init__(self, proj, adapt=None, gamma=None):
            super().__init__(); self.proj = proj; self.adapt = adapt
            self.register_buffer("gamma", gamma if gamma is not None else torch.zeros(1))
        def forward(self, x):
            if self.adapt is not None: x = x + self.gamma * self.adapt(x)
            return self.proj(x)
    def build_adapter():
        return nn.Sequential(nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 1024))

    models = [("frozen_f", None)]
    torch.manual_seed(42); models.append(("random_z0", build_proj().eval().to(DEV)))
    eps = []
    if a.proj_dir:
        for ck in sorted(glob.glob(str(Path(a.proj_dir) / "proj_ep*.pt")),
                         key=lambda s: int(re.search(r"ep(\d+)", s).group(1))):
            d = torch.load(ck, map_location="cpu"); pj = d["proj"] if "proj" in d else d
            pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in pj.items()}
            pr = build_proj(); pr.load_state_dict(pj)
            ad = None
            if "adapt" in d:
                ad = build_adapter(); ad.load_state_dict(d["adapt"])
            head = Head(pr, ad, d.get("gamma")).eval().to(DEV)
            e = int(re.search(r"ep(\d+)", ck).group(1)); eps.append(e); models.append((f"ep{e:02d}", head))

    print(f"\n=== [{a.tag}] label-free runtime - May HDBSCAN mcs{DIAL['mcs']}/ms{DIAL['ms']}/{DIAL['method']}/eps{DIAL['eps']} ===", flush=True)
    hdr = f"{'model':10s} k    noise%  nonNoise% overMrg stab   coh"
    print(hdr, flush=True); print("-"*len(hdr), flush=True)
    rows = {}
    embeddings = {}
    for name, proj in models:
        z = z_from(proj); lf = label_free(z)
        embeddings[name] = z
        rows[name] = {"lf": lf}
        print(f"{name:10s} {lf['k']:<4d} {lf['noise_pct']:<7.1f} {lf['non_noise_pct']:<9.1f} "
              f"{lf['over_merge']:<7d} {lf['stability']:<6.3f} {lf['coherence']:<6.3f}", flush=True)

    # 무라벨 선택 ladder (adapted epoch 중에서).
    # default("noise") = 기존 max-non_noise_pct 그대로 (후방호환, diff 0 검증됨).
    # "rich_noise"(Rule C, 260726 `_selection_rule_260726.md`): gate 통과분을 이 run 자신의 k 분포
    # --k-percentile(기본75) 이상으로 먼저 좁힌 뒤(under-clustered 초기 epoch 배제 — 절대 k 문턱이
    # 아니라 pool마다 스케일 다른 자기-정규화), 그 안에서 non_noise_pct 최대(=noise 최소)를 고른다.
    cand = []
    for e in eps:
        lf = rows[f"ep{e:02d}"]["lf"]
        if lf["over_merge"] == 0 and lf["stability"] >= 0.75 and lf["coherence"] >= 0.80:
            cand.append((e, lf["k"], lf["non_noise_pct"]))
    pool = cand
    k_ref = None
    if a.select_rule == "rich_noise" and cand:
        k_ref = float(np.percentile([c[1] for c in cand], a.k_percentile))
        pool = [c for c in cand if c[1] >= k_ref]
    sel = max(pool, key=lambda c: c[2])[0] if pool else None
    rule_note = f" select_rule={a.select_rule}" + (f" k_percentile={a.k_percentile} k_ref={k_ref:.2f} rich={len(pool)}/{len(cand)}" if a.select_rule == "rich_noise" and cand else "")
    print(f"\n[label-free selection] passed epochs={[c[0] for c in cand]} ->{rule_note} selected={('ep%02d'%sel) if sel else 'none(gate failed)'}", flush=True)
    out_dir = Path(a.out_dir)
    out_name = a.out_name or f"eval_{a.tag}"
    selection_path = out_dir / f"{out_name}.selection.json"
    checkpoint_shas = {}
    if a.proj_dir:
        for checkpoint in sorted(Path(a.proj_dir).glob("proj_ep*.pt")):
            checkpoint_shas[checkpoint.name] = sha256_file(checkpoint)
    selection_snapshot = {
        "tag": a.tag, "dial": dict(DIAL), "select_rule": a.select_rule,
        "k_percentile": a.k_percentile, "selected_ep": sel,
        "passed_epochs": [c[0] for c in cand], "rich_pool_epochs": [c[0] for c in pool],
        "label_free": {
            name: {key: value for key, value in row["lf"].items() if key != "pred"}
            for name, row in rows.items()
        },
        "checkpoint_sha256": checkpoint_shas,
        "offline_labels_evaluated": False,
    }
    selection_sha = atomic_json(selection_path, selection_snapshot)
    print(f"[SELECTION_SNAPSHOT] {selection_path} sha256={selection_sha}", flush=True)

    # Offline labels are touched only after the immutable label-free decision
    # snapshot above is atomically committed and hashed.
    for name in rows:
        rows[name]["off"] = offline(embeddings[name], labels, rows[name]["lf"]["pred"])

    out = {"tag": a.tag, "dial": dict(DIAL), "select_rule": a.select_rule, "k_percentile": a.k_percentile,
           "selected_ep": sel,
           "selection_snapshot_path": str(selection_path.resolve()),
           "selection_snapshot_sha256": selection_sha,
           "frozen": rows["frozen_f"], "z0": rows["random_z0"],
           "selected": rows[f"ep{sel:02d}"] if sel else None,
           "all_eps": {f"ep{e:02d}": rows[f"ep{e:02d}"] for e in eps}}
    result_path = out_dir / f"{out_name}.json"
    atomic_json(result_path, out)
    print(f"[OUT] {result_path}", flush=True)
    print(f"[DONE_EVAL_{a.tag}]", flush=True)

if __name__ == "__main__":
    main()
