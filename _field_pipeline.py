#!/usr/bin/env python3
"""현업 시뮬레이션 파이프라인 — real cca 8-class 불균형 풀(1779장), 라벨·k 없이 이상 그룹핑.

현업 그대로: 라벨 없는 풀 자체로 SSL(transductive SimCLR) 학습 → HDBSCAN(k-free) → 그룹.
라벨은 평가에만 사용 (공식 Tier1: capture_rate / noise_pct / Completeness / Homogeneity + 보조 AMI/ARI).

stages:
  frozen : DINOv3 / FCMAE frozen 임베딩 산출 (baseline 행)
  train  : 풀 자체 transductive SimCLR fine-tune (DINOv3 init, best recipe), 매 epoch 임베딩 저장
  eval   : 모든 임베딩에 HDBSCAN sweep + Tier1 + k̂ 추정 + t-SNE/montage → _field_results.{md,csv}
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import _ssl_methods as S  # Net / aug / TwoView / mask 재사용
from cluster_metrics import capture_metrics

POOL = Path("E:/data/images/cca")          # default — --pool 로 override 가능
ROOT = REPO / "result_grouping/_field_cca"  # default — --root 로 override 가능
EMB = ROOT / "embeddings"
FCMAE_TIMM = "convnextv2_base.fcmae_ft_in22k_in1k_384"


def set_paths(pool: str | None, root: str | None):
    global POOL, ROOT, EMB
    if pool:
        POOL = Path(pool)
    if root:
        ROOT = REPO / root if not Path(root).is_absolute() else Path(root)
    EMB = ROOT / "embeddings"


def pool_imgs():
    e = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    cls_dirs = sorted(d for d in POOL.iterdir() if d.is_dir() and d.name != "labels")
    paths, labs = [], []
    for d in cls_dirs:
        for p in sorted(q for q in d.iterdir() if q.suffix.lower() in e):
            paths.append(p); labs.append(d.name)
    return paths, labs


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def embed_paths(net, paths, dev):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    tf = T.Compose([T.Resize((S.IMG, S.IMG)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            xs = [tf(S.mask_palette_non_grade_to_white(Image.open(p)).convert("RGB")) for p in paths[i:i + 16]]
            f = net(torch.stack(xs).to(dev))
            if isinstance(f, tuple):
                f = f[0]
            out.append(F.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float32)


def stage_frozen():
    import timm, torch
    dev = torch.device("cpu")
    paths, labs = pool_imgs()
    EMB.mkdir(parents=True, exist_ok=True)
    (ROOT / "pool_labels.json").write_text(json.dumps(
        {"files": [str(p) for p in paths], "labels": labs}, ensure_ascii=False), encoding="utf-8")
    print(f"[pool] {len(paths)} imgs, dist={dict(Counter(labs))}", flush=True)
    for name, timm_id in [("DINOv3_frozen", S.TIMM), ("FCMAE_frozen", FCMAE_TIMM)]:
        t0 = time.time()
        net = timm.create_model(timm_id, pretrained=True, num_classes=0, global_pool="avg").to(dev)
        z = embed_paths(net, paths, dev)
        np.save(EMB / f"{name}.npy", z)
        print(f"[frozen] {name} {z.shape} ({time.time()-t0:.0f}s)", flush=True)
        del net
    print(f"[OUT] {EMB}", flush=True)


def stage_train(epochs, batch, lr_bb, lr_head, temp, tag="simclr_tx"):
    import torch
    import torch.nn.functional as F
    dev = torch.device("cpu")
    torch.manual_seed(42)
    paths, labs = pool_imgs()
    print(f"[train] transductive SimCLR on pool {len(paths)} imgs dev=cpu", flush=True)
    dl = torch.utils.data.DataLoader(S.TwoView(paths, S.aug()), batch_size=batch,
                                     shuffle=True, num_workers=0, drop_last=True)
    net = S.Net("simclr").to(dev).train()
    bb_ids = {id(p) for p in net.bb.parameters()}
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) in bb_ids], "lr": lr_bb},
                             {"params": [p for p in params if id(p) not in bb_ids], "lr": lr_head}],
                            weight_decay=1e-6)
    EMB.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        t0 = time.time(); run = 0.0; n = 0
        for x1, x2 in dl:
            x1, x2 = x1.to(dev), x2.to(dev)
            _, z1 = net(x1); _, z2 = net(x2)
            zc = F.normalize(torch.cat([z1, z2]), dim=1); B = z1.size(0)
            sim = zc @ zc.t() / temp
            sim.fill_diagonal_(-1e9)
            pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(dev)
            loss = F.cross_entropy(sim, pos)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); n += 1
        print(f"[tx ep{ep}] loss={run/max(1,n):.4f} ({time.time()-t0:.0f}s)", flush=True)
        z = embed_paths(net, paths, dev)
        np.save(EMB / f"{tag}_ep{ep}.npy", z)
        net.train()
    print(f"[OUT] {EMB}", flush=True)


def tier1(pred, true_idx, labs, classes, excluded=()):
    """Tier1 on target classes while retaining full-pool cluster dominance."""
    from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                                 completeness_score, homogeneity_score)
    pred = np.asarray(pred)
    labs_arr = np.asarray(labs)
    target = ~np.isin(labs_arr, list(excluded))
    pred_target = pred[target]
    true_target = np.asarray(true_idx)[target]
    noise = pred_target == -1
    noise_pct = float(noise.mean() * 100)
    nn = ~noise
    capture = capture_metrics(pred, labs_arr, excluded)
    r = dict(noise_pct=round(noise_pct, 2),
             n_clusters=capture["cluster_count"],
             capture=round(float(capture["capture_rate"]), 4),
             recov=round(float(capture["dominant_image_capture_rate"]), 4),
             capture_count=capture["capture_count"],
             target_class_count=capture["target_class_count"],
             legacy_presence_count=capture["legacy_presence_count"],
             legacy_presence_rate=round(float(capture["legacy_presence_rate"]), 4),
             class_coverage=round(float(capture["class_coverage_rate"]), 4))
    if nn.sum() > 1 and len(set(pred_target[nn])) > 1:
        r.update(completeness=round(float(completeness_score(true_target[nn], pred_target[nn])), 4),
                 homogeneity=round(float(homogeneity_score(true_target[nn], pred_target[nn])), 4),
                 ari=round(float(adjusted_rand_score(true_target[nn], pred_target[nn])), 4),
                 ami=round(float(adjusted_mutual_info_score(true_target[nn], pred_target[nn])), 4))
    else:
        r.update(completeness=0.0, homogeneity=0.0, ari=0.0, ami=0.0)
    captured = set(capture["captured_classes"])
    r["capture_detail"] = {str(cls): float(cls in captured) for cls in sorted(set(labs_arr[target]), key=str)}
    return r


def hdb_sweep(z, true_idx, labs, classes, keep=None):
    """HDBSCAN k-free sweep — 채택규칙 4행:
    default(고정 cfg) / unsup_sel(silhouette 라벨無) / dbcv_sel(DBCV 라벨無, 밀도 세계관) / oracle(AMI 상한).
    keep: 채점 마스크 (Random 등 비정형 클래스는 클러스터링엔 포함하되 채점에서 제외)."""
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score
    try:
        from hdbscan.validity import validity_index
    except Exception:
        validity_index = None
    if keep is None:
        keep = np.ones(len(labs), dtype=bool)
    labs_arr = np.asarray(labs)
    excluded = set(labs_arr[~keep].tolist())
    z64 = np.ascontiguousarray(z, dtype=np.float64)
    rows = []
    for mcs in (5, 10, 12, 20, 40):   # 12 = 옛 트랙 SOTA 레시피 (eom-mcs12-ms3) 연속성
        for ms in (3, 5, 10):
            for method in ("eom", "leaf"):
                for eps in (0.0, 0.1, 0.2):
                    pred = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                                   cluster_selection_method=method,
                                   cluster_selection_epsilon=eps).fit_predict(z)
                    # Keep background rows while deciding each cluster's main
                    # class.  Slicing them first would make a defect minority
                    # inside a Normal-majority group look falsely captured.
                    r = tier1(pred, true_idx, labs, classes, excluded=excluded)
                    # noise 클래스 (Random/Normal — 사용자: "random과 normal은 noise다"):
                    # 정답 배치 = noise(-1). 실제로 noise 로 빠진 비율 (높을수록 좋음).
                    nz = ~keep
                    r["nzcls_to_noise"] = round(float((pred[nz] == -1).mean() * 100), 1) if nz.any() else None
                    nn = pred != -1
                    sil = dbcv = None
                    if nn.sum() > 10 and 1 < len(set(pred[nn])) < nn.sum():
                        sil = float(silhouette_score(z[nn], pred[nn], metric="cosine"))
                        if validity_index is not None:
                            try:
                                dbcv = float(validity_index(z64, np.asarray(pred), metric="euclidean"))
                            except Exception:
                                dbcv = None
                    rows.append((dict(mcs=mcs, ms=ms, method=method, eps=eps), r, sil, pred, dbcv))
    def cfg_eq(c, **kw): return all(c[k] == v for k, v in kw.items())
    default = next(x for x in rows if cfg_eq(x[0], mcs=20, ms=5, method="eom", eps=0.0))
    unsup_pool = [x for x in rows if x[2] is not None and x[1]["noise_pct"] < 40 and x[1]["n_clusters"] >= 2]
    unsup = max(unsup_pool, key=lambda x: x[2]) if unsup_pool else default
    dbcv_pool = [x for x in rows if x[4] is not None and x[1]["n_clusters"] >= 2]
    dbcv_sel = max(dbcv_pool, key=lambda x: x[4]) if dbcv_pool else default
    oracle = max(rows, key=lambda x: x[1]["ami"])
    return default, unsup, dbcv_sel, oracle


def k_hat(z, true_idx, keep=None):
    """라벨 없이 k 추정 (silhouette, 풀 전체) → kmeans(k̂). ARI 채점만 keep 마스크."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    if keep is None:
        keep = np.ones(len(true_idx), dtype=bool)
    best_k, best_s = 2, -1
    for k in range(2, 16):
        lab = KMeans(k, n_init=5, random_state=42).fit_predict(z)
        s = silhouette_score(z, lab, metric="cosine")
        if s > best_s:
            best_k, best_s = k, s
    lab = KMeans(best_k, n_init=10, random_state=42).fit_predict(z)
    return best_k, round(float(adjusted_rand_score(true_idx[keep], lab[keep])), 4), lab


def stage_eval():
    meta = json.loads((ROOT / "pool_labels.json").read_text(encoding="utf-8"))
    paths = [Path(p) for p in meta["files"]]; labs = meta["labels"]
    classes = sorted(set(labs)); c2i = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([c2i[l] for l in labs])
    # 채점 제외 (클러스터링/학습엔 포함): Random/R = 비정형 (특정 모양 아님, 사용자 directive),
    # Normal = defect 아님 (none-class 정책 — metric 은 defect only)
    EXCLUDE_EVAL = {"Random", "R", "Normal"}
    keep = np.array([l not in EXCLUDE_EVAL for l in labs])
    classes_eval = sorted(set(l for l in labs if l not in EXCLUDE_EVAL))
    rows = []
    embs = {}
    for f in sorted(glob.glob(str(EMB / "*.npy"))):
        embs[Path(f).stem] = f
    print(f"[eval] {len(embs)} embeddings, pool={len(labs)}, 채점 {int(keep.sum())}장 (Random 제외), classes={classes_eval}", flush=True)
    best_overall = None
    for name, f in embs.items():
        z_raw = l2(np.load(f).astype(np.float32))
        spaces = [("", z_raw)]
        try:
            import umap
            zu = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
                           metric="cosine", random_state=42).fit_transform(z_raw)
            spaces.append(("umap10+", np.ascontiguousarray(zu, dtype=np.float32)))
        except Exception as e:
            print(f"  [warn] umap skip: {e}", flush=True)
        for sp, z in spaces:
            default, unsup, dbcv_sel, oracle = hdb_sweep(z, true_idx, labs, classes_eval, keep=keep)
            kh, kh_ari, _ = k_hat(z, true_idx, keep=keep)
            disp = f"{sp}{name}"
            for tag, (cfg, r, sil, pred, dbcv) in (("hdb_default", default), ("hdb_unsup_sel", unsup),
                                                   ("hdb_dbcv_sel", dbcv_sel), ("hdb_oracle", oracle)):
                rows.append(dict(embedding=disp, rule=tag,
                                 cfg=f"mcs{cfg['mcs']}_ms{cfg['ms']}_{cfg['method']}_e{cfg['eps']}",
                                 **{k: v for k, v in r.items() if k != "capture_detail"},
                                 capture_detail=json.dumps(r["capture_detail"], ensure_ascii=False)))
                nzs = f"{r.get('nzcls_to_noise')}" if r.get("nzcls_to_noise") is not None else "-"
                print(f"  {disp:28s} {tag:14s} cap={r['capture']:.2f} recov={r['recov']:.3f} noise={r['noise_pct']:5.1f}% "
                      f"nz→noise={nzs:>5s}% k={r['n_clusters']:2d} Comp={r['completeness']:.3f} Hom={r['homogeneity']:.3f} AMI={r['ami']:.3f}", flush=True)
                if tag in ("hdb_unsup_sel", "hdb_dbcv_sel"):  # P1 capture 우선, recov 는 tiebreak
                    score = (r["capture"], r["recov"])
                    if best_overall is None or score > (best_overall[1]["capture"], best_overall[1]["recov"]):
                        best_overall = (disp, r, pred, z_raw)
            rows.append(dict(embedding=disp, rule=f"kmeans_khat(k̂={kh})", cfg="silhouette-k",
                             noise_pct=0.0, n_clusters=kh, capture=None, completeness=None,
                             homogeneity=None, ari=kh_ari, ami=None, capture_detail=""))
            print(f"  {disp:28s} k̂={kh} kmeans ARI={kh_ari}", flush=True)
    cols = ["embedding", "rule", "cfg", "capture", "recov", "noise_pct", "nzcls_to_noise", "n_clusters",
            "completeness", "homogeneity", "ami", "ari", "capture_detail"]
    res_base = REPO / f"_field_results_{ROOT.name.replace('_field_', '')}"  # root 별 파일 (덮어쓰기 방지)
    with Path(str(res_base) + ".csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in cols})
    with Path(str(res_base) + ".md").open("w", encoding="utf-8") as fp:
        fp.write("# 현업 시뮬레이션 — real cca 8-class 불균형 풀, 라벨·k 없이 그룹핑\n\n")
        fp.write(f"pool: {len(labs)}장, 분포 {dict(Counter(labs))}\n\n")
        fp.write("우선순위: P1 capture(불량 회수) > P2 noise% > P3 Completeness > P4 Homogeneity. 라벨은 평가만.\n")
        fp.write("**Random 클래스는 채점 제외** (비정형 — 특정 모양 아님. 클러스터링/학습엔 포함, GT=Random 만 metric 에서 drop).\n")
        fp.write("hdb_unsup_sel = 라벨 없이 silhouette 로 cfg 선택 / hdb_dbcv_sel = 라벨 없이 DBCV(밀도 지표) 로 선택\n")
        fp.write("(둘 다 현업 가능). oracle = AMI 상한 참조 (라벨로 cfg 선택 — 현업 불가, 진단용).\n\n")
        fp.write("**capture (P1)** = 메인(다수) 클래스로 등장한 클래스 수 / 전체 클래스 수. (예: 10종 중 메인 2종류 → 2/10)\n")
        fp.write("recov (보조) = 자기가 메인인 클러스터에 회수된 이미지 비율의 클래스 평균 (잡탕 셋방 불인정).\n")
        fp.write("noise% = defect 가 noise 로 버려진 비율 (낮을수록 좋음) / "
                 "nz→noise% = noise 클래스(Random·Normal)가 올바르게 noise 로 빠진 비율 (높을수록 좋음 — 정답 배치 = noise).\n\n")
        fp.write("| embedding | rule | **capture** | recov | noise% | nz→noise% | k | Comp | Hom | AMI | ARI |\n|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for r in rows:
            fmt = lambda v: "" if v is None else v
            fp.write(f"| {r['embedding']} | {r['rule']} | {fmt(r.get('capture'))} | {fmt(r.get('recov'))} | {fmt(r.get('noise_pct'))} | "
                     f"{fmt(r.get('nzcls_to_noise'))} | {fmt(r.get('n_clusters'))} | {fmt(r.get('completeness'))} | {fmt(r.get('homogeneity'))} | "
                     f"{fmt(r.get('ami'))} | {fmt(r.get('ari'))} |\n")
        # ── best 결과의 3분할 클러스터 인구조사 (도미넌트 / noise 클러스터 / 버림통) ──
        if best_overall is not None:
            bname, br, bpred, _ = best_overall
            labs_np = np.array(labs)
            nzcls = EXCLUDE_EVAL
            fp.write(f"\n## 클러스터 인구조사 — best: {bname}\n\n")
            owners = defaultdict(list); noise_cl = []
            for cl in sorted(set(bpred[bpred >= 0])):
                members = labs_np[bpred == cl]
                cnt = Counter(members); maj, majn = cnt.most_common(1)[0]
                if maj in nzcls:
                    noise_cl.append((cl, cnt, majn / len(members), len(members)))
                else:
                    owners[maj].append(len(members))
            fp.write("### ① 도미넌트(defect) 클러스터\n\n| 주인 클래스 | 클러스터 수 | 크기 분포 |\n|---|--:|---|\n")
            for c in classes_eval:
                sizes = sorted(owners.get(c, []), reverse=True)
                mark = " ⚠️ 형성 실패" if not sizes else ""
                fp.write(f"| {c} | {len(sizes)} | {', '.join(map(str, sizes))}{mark} |\n")
            fp.write("\n### ② noise 클러스터 (Random/Normal 주인 — 존재 자체가 감점, 목표는 -1 강등)\n\n")
            if noise_cl:
                fp.write("| 클러스터 | 크기 | 순도 | 구성 top3 |\n|---|--:|--:|---|\n")
                for cl, cnt, pur, tot in noise_cl:
                    top3 = ", ".join(f"{k} {v}" for k, v in cnt.most_common(3))
                    fp.write(f"| #{cl} | {tot} | {pur:.2f} | {top3} |\n")
            else:
                fp.write("(없음 — noise 클래스 전원 -1 처리됨 ✓)\n")
            n_noise = int((bpred == -1).sum())
            fp.write(f"\n### ③ noise(-1) 버림통: {n_noise}장 ({n_noise/len(bpred)*100:.1f}%)\n")
    print(f"[OUT] {res_base}.md", flush=True)
    if best_overall is not None:
        viz(best_overall, paths, labs, classes)


def viz(best, paths, labs, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib.lines import Line2D
    from PIL import Image
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    name, r, pred, z = best
    vd = ROOT / "viz"; vd.mkdir(parents=True, exist_ok=True)
    zz = PCA(50, random_state=42).fit_transform(z) if z.shape[1] > 50 else z
    z2 = TSNE(2, metric="cosine", init="pca", perplexity=30, random_state=42).fit_transform(zz)
    cmap = plt.get_cmap("tab10").colors
    colors = {c: cmap[i % 10] for i, c in enumerate(classes)}
    labs_np = np.array(labs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=130)
    for c in classes:
        m = labs_np == c
        axes[0].scatter(z2[m, 0], z2[m, 1], s=10, c=[colors[c]], alpha=0.75, label=c)
    axes[0].set_title(f"{name} — true class"); axes[0].legend(fontsize=7); axes[0].set_xticks([]); axes[0].set_yticks([])
    for cl in sorted(set(pred)):
        m = pred == cl
        kw = dict(s=10, alpha=0.75) if cl != -1 else dict(s=6, alpha=0.3, c="lightgray")
        axes[1].scatter(z2[m, 0], z2[m, 1], **kw)
    axes[1].set_title(f"HDBSCAN groups (k={r['n_clusters']}, noise={r['noise_pct']}%)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout(); fig.savefig(vd / "tsne_field.png", bbox_inches="tight"); plt.close(fig)
    for cl in sorted(set(pred)):
        if cl == -1:
            continue
        idx = np.where(pred == cl)[0]
        cent = l2(z[idx].mean(0, keepdims=True))
        order = idx[np.argsort(-(z[idx] @ cent.T).ravel())][:12]
        maj = Counter(labs_np[idx]).most_common(1)[0][0]
        n = len(order); colsn = min(6, n); rowsn = (n + colsn - 1) // colsn
        fig, axs = plt.subplots(rowsn, colsn, figsize=(colsn * 1.7, rowsn * 1.7), dpi=110)
        axs = np.atleast_1d(axs).reshape(-1)
        for ax, i in zip(axs, order):
            try:
                ax.imshow(Image.open(paths[i]).convert("RGB"))
            except Exception:
                pass
            ax.axis("off")
        for ax in axs[n:]: ax.axis("off")
        fig.suptitle(f"group {cl} (n={len(idx)}, 다수={maj})", fontsize=10)
        fig.tight_layout(); fig.savefig(vd / f"group{cl:02d}_maj_{maj}.png", bbox_inches="tight"); plt.close(fig)
    print(f"[OUT] {vd}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["frozen", "train", "eval"])
    ap.add_argument("--pool", default=None, help="풀 디렉토리 override (default cca)")
    ap.add_argument("--root", default=None, help="결과 root override (default result_grouping/_field_cca)")
    ap.add_argument("--tx-tag", default="simclr_tx", help="train 단계 임베딩 tag")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-bb", type=float, default=2e-6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.05)
    a = ap.parse_args()
    set_paths(a.pool, a.root)
    if a.stage == "frozen":
        stage_frozen()
    elif a.stage == "train":
        stage_train(a.epochs, a.batch, a.lr_bb, a.lr_head, a.temp, tag=a.tx_tag)
    else:
        stage_eval()


if __name__ == "__main__":
    main()
