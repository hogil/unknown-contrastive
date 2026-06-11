#!/usr/bin/env python3
"""SSL 변형 무한 sweep supervisor — CPU 순차, 자율 coordinate-descent.

동작:
  1) _field_pipeline.py (transductive train/eval) 가 돌고 있으면 대기 (CPU 단일 정책)
  2) agenda 의 다음 config 로 _ssl_methods.py 학습 (각 run 별 log)
  3) novel_eval k-means(k=3) ARI 자동 평가 → _sweep_leaderboard.csv append
  4) agenda 소진 시 best config 이웃을 자동 생성해 무한 계속
  5) _sweep_STOP.txt 존재 시 현재 run 마치고 정지

정지: `touch _sweep_STOP.txt`
"""
from __future__ import annotations
import csv, json, subprocess, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
EMB = REPO / "result_grouping/_dinov3_ncd_autoloop/ssl_embeddings"
EVAL_DIR = REPO / "data/images/wm811k_novel_disjoint_v1/novel_eval"
LB = REPO / "_sweep_leaderboard.csv"
STOP = REPO / "_sweep_STOP.txt"
RUNLOG_DIR = REPO / "_sweep_runs"; RUNLOG_DIR.mkdir(exist_ok=True)
LOG = REPO / "_sweep_supervisor.log"

BASE = dict(method="simclr", epochs=5, batch=8, lr_bb=2e-6, lr_head=1e-3, temp=0.05,
            queue=0, ignore=0.0, koleo=0.0, neco=0.0, dino_tricks=0)

# ── round 1 agenda (1 atomic change each) ──────────────────────────────────
AGENDA = [
    ("simclr_base_v2",   {}),
    ("simclr_queue4k_v2", dict(queue=4096)),
    ("simclr_ign09_v2",  dict(ignore=0.9)),
    ("simclr_combo_v2",  dict(queue=4096, ignore=0.9, koleo=0.1)),
    ("simclr_t004_v2",   dict(temp=0.04)),
    ("simclr_t006_v2",   dict(temp=0.06)),
    ("simclr_koleo_v2",  dict(koleo=0.1)),
    ("moco_v2",          dict(method="moco")),
    ("simclr_neco_v2",   dict(neco=1.0, batch=4)),
    ("dino_fixed_v2",    dict(method="dino", dino_tricks=1, epochs=10)),
]


def log(msg):
    line = f"[{time.strftime('%m%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def field_busy():
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
             "-and $_.CommandLine -match '_field_pipeline' }).Count"],
            capture_output=True, text=True, timeout=120)
        return int(r.stdout.strip() or 0) > 0
    except Exception:
        return False


def eval_labels():
    e = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in EVAL_DIR.rglob("*") if p.is_file() and p.suffix.lower() in e)
    labs = [p.parent.name for p in paths]
    c2i = {c: i for i, c in enumerate(sorted(set(labs)))}
    return np.array([c2i[l] for l in labs])


def evaluate(tag, y):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    best = (None, -1.0, None)
    for f in sorted(EMB.glob(f"{tag}_ep*.npy")):
        z = np.load(f).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        ari = adjusted_rand_score(y, KMeans(3, n_init=10, random_state=42).fit_predict(z))
        ep = int(f.stem.rsplit("ep", 1)[1])
        if ari > best[1]:
            sil = float(silhouette_score(z, y, metric="cosine"))
            best = (ep, ari, sil)
    return best


def cfg_key(p):
    return json.dumps({k: p[k] for k in sorted(p)}, sort_keys=True)


def load_lb():
    if not LB.exists():
        return []
    with LB.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_lb(row):
    rows = load_lb()
    cols = ["tag", "ari", "best_ep", "sil_true", "params", "key", "ts"]
    with LB.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
        w.writerow(row)


def build_cmd(p, tag):
    cmd = [sys.executable, "-u", "_ssl_methods.py", "--method", p["method"], "--cpu",
           "--epochs", str(p["epochs"]), "--batch", str(p["batch"]),
           "--lr-bb", str(p["lr_bb"]), "--lr-head", str(p["lr_head"]),
           "--temp", str(p["temp"]), "--tag", tag]
    if p["queue"] > 0:
        cmd += ["--use-queue", "--queue-size", str(p["queue"])]
    if p["ignore"] > 0:
        cmd += ["--ignore", str(p["ignore"])]
    if p["koleo"] > 0:
        cmd += ["--koleo", str(p["koleo"])]
    if p["neco"] > 0:
        cmd += ["--neco", str(p["neco"])]
    if p["dino_tricks"]:
        cmd += ["--dino-tricks"]
    return cmd


def neighbors(best_p, tried):
    """best config 주변 이웃 + 이긴 단일 변형들의 조합 생성."""
    cands = []
    for t in (0.03, 0.045, 0.055, 0.065, 0.07):
        cands.append({**best_p, "temp": t})
    if best_p["queue"] > 0:
        for q in (1024, 2048, 8192):
            cands.append({**best_p, "queue": q})
    else:
        cands.append({**best_p, "queue": 4096})
    if best_p["ignore"] > 0:
        for g in (0.85, 0.92, 0.95):
            cands.append({**best_p, "ignore": g})
    else:
        cands.append({**best_p, "ignore": 0.9})
    if best_p["koleo"] > 0:
        for k in (0.05, 0.2):
            cands.append({**best_p, "koleo": k})
    for lr in (1.5e-6, 2.5e-6, 3e-6):
        cands.append({**best_p, "lr_bb": lr})
    cands.append({**best_p, "epochs": 8})
    # 이긴 단일 변형 조합 (leaderboard 의 base 대비 Δ>0 인 simclr 단일들)
    lb = load_lb()
    base = next((float(r["ari"]) for r in lb if r["tag"] == "simclr_base_v2"), None)
    if base is not None:
        winners = [json.loads(r["params"]) for r in lb
                   if float(r["ari"]) > base and json.loads(r["params"]).get("method") == "simclr"]
        for i in range(len(winners)):
            for j in range(i + 1, len(winners)):
                merged = dict(BASE)
                for w in (winners[i], winners[j]):
                    for k, v in w.items():
                        if v != BASE.get(k):
                            merged[k] = v
                cands.append(merged)
    out = []
    for c in cands:
        if cfg_key(c) not in tried:
            out.append(c)
    return out


def next_tag(p, n):
    bits = [p["method"]]
    if p["queue"]: bits.append(f"q{p['queue']}")
    if p["ignore"]: bits.append(f"ig{int(p['ignore']*100)}")
    if p["koleo"]: bits.append(f"kl{p['koleo']}")
    if p["temp"] != 0.05: bits.append(f"t{p['temp']}")
    if p["lr_bb"] != 2e-6: bits.append(f"lr{p['lr_bb']:.1e}")
    if p["epochs"] != 5: bits.append(f"e{p['epochs']}")
    return "auto" + str(n) + "_" + "_".join(bits)


def main():
    y = eval_labels()
    tried = {r["key"] for r in load_lb()}
    log(f"=== sweep supervisor 시작 (leaderboard {len(tried)}건 로드) ===")
    agenda = [(t, {**BASE, **d}) for t, d in AGENDA if cfg_key({**BASE, **d}) not in tried]
    auto_n = 0
    while not STOP.exists():
        while field_busy():
            log("field_pipeline 실행중 — 5분 대기")
            time.sleep(300)
            if STOP.exists():
                log("STOP 감지 — 종료"); return
        if agenda:
            tag, p = agenda.pop(0)
        else:
            lb = load_lb()
            if not lb:
                log("leaderboard 비어있음 — 비정상"); return
            best = max(lb, key=lambda r: float(r["ari"]))
            best_p = {**BASE, **json.loads(best["params"])}
            cands = neighbors(best_p, {r["key"] for r in load_lb()})
            if not cands:
                log("이웃 후보 소진 — 60분 대기 후 재시도"); time.sleep(3600); continue
            auto_n += 1
            p = cands[0]; tag = next_tag(p, auto_n)
            log(f"auto 이웃 생성 (best={best['tag']} ARI={best['ari']}) → {tag}")
        cmd = build_cmd(p, tag)
        log(f"RUN {tag}: {' '.join(cmd[2:])}")
        rl = RUNLOG_DIR / f"{tag}.log"
        t0 = time.time()
        with rl.open("w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO)).returncode
        dur = (time.time() - t0) / 60
        if rc != 0:
            log(f"FAIL {tag} rc={rc} ({dur:.0f}min) — 다음으로"); continue
        ep, ari, sil = evaluate(tag, y)
        if ep is None:
            log(f"WARN {tag}: 임베딩 없음"); continue
        append_lb(dict(tag=tag, ari=round(ari, 4), best_ep=ep, sil_true=round(sil, 4),
                       params=json.dumps(p), key=cfg_key(p), ts=time.strftime("%y%m%d_%H%M")))
        log(f"DONE {tag}: ARI={ari:.4f} (ep{ep}, sil {sil:.3f}, {dur:.0f}min)")
    log("STOP 파일 감지 — supervisor 종료")


if __name__ == "__main__":
    main()
