#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""site/ 공통 유틸 — 사내 서버 무라벨 배포 파이프라인.

★ 경로 규칙: 전부 **프로젝트 루트 기준 상대경로**. 이 파일 위치에서 루트를 derive 하므로
  프로젝트 폴더를 통째로 어디에 두든 그대로 돈다. 절대경로 하드코딩 금지.

★ 설정 규칙: 각 step 스크립트 맨 위 `class Config` 가 단일 소스.
  환경변수가 있으면 그걸 쓰고, 없으면 Config 의 default 를 쓴다.

여기 담긴 판정 규칙은 260726 실측 도출. 근거: docs/GOAL_AND_LADDER.md
  1. 사내엔 라벨이 없다 -> 모든 선택은 label-free.
  2. ★ 불량 종수(k)는 모른다 — 그게 전제다. HDBSCAN 을 쓰는 이유가 k-free 라서다.
     다이얼은 k 가 아니라 **"몇 장 이상 뭉쳐야 그룹으로 볼 것인가"**(MIN_GROUP_SIZE)로 정한다.
     그것도 정하기 싫으면 AUTO 모드가 bootstrap 안정성으로 라벨·k 없이 고른다.
     다른 pool 의 다이얼 값 이식은 금지 — mcs6 을 그대로 썼다가 결론이 뒤집힌 전례가 있다.
  3. 대조군은 frozen 이 아니라 z0(랜덤 head), 용량(head 수)도 맞춘다.
  4. reassign 전/후를 분리 보고. 후처리 후 noise 는 어떤 임베딩에든 나오는 바닥값이다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ── 프로젝트 루트 (이 파일: <root>/site/_site_common.py) ─────────────────────
REPO = Path(__file__).resolve().parent.parent


def rel(*parts) -> Path:
    """루트 기준 상대경로 -> 절대 Path. 이미 절대경로면 그대로 둔다.

    Windows 에서 posix 루트(/data/x)는 is_absolute() 가 False 라 조용히 현재
    드라이브에 붙는다(D:\\data\\x). 에러 없이 틀린 경로가 되므로 명시 처리한다.
    Linux 서버에서는 원래대로 절대경로로 인식되므로 이 분기를 타지 않는다.
    """
    p = Path(*parts)
    if p.is_absolute():
        return p
    head = str(parts[0]) if parts else ""
    if head[:1] in ("/", chr(92)):
        return Path(*parts)          # posix 루트 표기 -> REPO 에 붙이지 않는다
    return REPO / p


def env(name: str, default):
    """환경변수 우선, 없으면 default. default 의 타입으로 캐스팅."""
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if isinstance(default, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


# 측정 잡음폭 (동일 config 4회 반복 실측). 이 안의 차이는 "차이 없음"으로 처리.
BAND = {"noise": 2.28, "ari": 0.019, "hom": 0.005, "comp": 0.033, "p1": 0.0}
# 민감도 순위 (평균 |diff| / 잡음폭). Comp 가 가장 관대 -> 단독 통과 판정 금지.
SENSITIVITY = ("P1", "ARI", "Hom", "noise", "Comp")


def die(msg: str, code: int = 2):
    print(f"\n[FATAL] {msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


def banner(step: str, title: str, ladder: str = ""):
    line = "=" * 78
    print(f"\n{line}\n[{step}] {title}")
    if ladder:
        print(f"  배포 사다리: {ladder}")
    print(line, flush=True)


def show_config(C) -> dict:
    """Config 클래스의 대문자 속성을 표로 출력하고 dict 로 반환 (재현성 기록용)."""
    d = {k: getattr(C, k) for k in dir(C) if k.isupper() and not k.startswith("_")}
    print("[config] (환경변수 > default)")
    for k in sorted(d):
        print(f"    {k:<22} = {d[k]}")
    print(flush=True)
    return d


def check_inputs(backbone: str, image_root: str, projs: list[str] | None = None):
    bb = rel(backbone)
    if not bb.exists():
        die(f"backbone 이 없다: {bb}\n"
            f"  프로젝트 루트 기준 상대경로로 두어라. 필요한 파일은 site/README.md 의 "
            f"'보내야 할 체크포인트' 참조.")
    ir = rel(image_root)
    if not ir.exists():
        die(f"image_root 가 없다: {ir}\n  SITE_IMAGE_ROOT 로 지정하거나 Config 의 default 를 고쳐라.")
    for p in (projs or []):
        if not rel(p).exists():
            die(f"projection head 가 없다: {rel(p)}\n  site/README.md 의 체크포인트 목록 참조.")


# ── 다이얼 ────────────────────────────────────────────────────────────────
def dial_from_min_group(min_group_size: int) -> tuple[int, int]:
    """★ k 를 쓰지 않는다. HDBSCAN 의 min_cluster_size 는 원래 의미 그대로 쓴다:
    **"몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가"** — 운영자가 답할 수 있는 값이다.

    불량 종수(k)는 모른다는 게 이 과제의 전제다. HDBSCAN 을 고른 이유가 바로 그것이고,
    k 를 입력으로 받으면 그 이유가 사라진다. min_cluster_size 는 클래스 수가 아니라
    **보고 가치가 있는 최소 그룹 크기**이므로 k 와 무관하게 정할 수 있다.

    ms = mcs/4 는 실측 승자 조합(mcs20/ms5)에서 왔다.
    """
    mcs = max(5, int(min_group_size))
    return mcs, max(3, mcs // 4)


def dial_scan_range(n: int, min_group_size: int, n_points: int = 8) -> list[int]:
    """운영값 주변 + 데이터 크기 상한으로 스캔 범위. k 를 쓰지 않는다."""
    base = max(5, int(min_group_size))
    cand = {base, max(5, base // 2), max(5, int(base * 0.75)),
            int(base * 1.5), base * 2, base * 3}
    cand |= {max(5, int(n * f)) for f in (0.005, 0.01, 0.02, 0.04)}
    cand = sorted(c for c in cand if 5 <= c <= max(5, n // 4))
    if len(cand) > n_points:
        step = len(cand) / n_points
        cand = [cand[int(i * step)] for i in range(n_points)]
    return sorted(set(cand))


def pick_dial_by_stability(z, scan: list[int], hdbscan_predict, per_group_stability,
                           method: str = "leaf", eps: float = 0.06,
                           min_k: int = 2) -> tuple[int, int, list[dict]]:
    """★ 라벨도 k 도 안 쓰고 다이얼을 고른다 — bootstrap 군집 안정성 최대.

    실측: severstal 에서 아는 정답(mcs20)을 정확히 집어냈다. DBCV 는 15 를 골라 빗나갔고,
    ARI 최대화는 k=2 병합 치팅(mcs60)으로 걸어갔다.
    ⚠ 다이얼이 결과에 거의 영향 없는 pool 도 있다(clean546 은 mcs 5~44 에서 ARI 0.63~0.70 평탄).
      그런 pool 에선 어느 값을 골라도 무방하므로 스캔 표를 함께 보고 판단하라.
    """
    rows = []
    for mcs in scan:
        ms = max(3, mcs // 4)
        pred = hdbscan_predict(z, mcs, ms, method, eps)
        k = len({int(c) for c in pred if c >= 0})
        noise = float((pred == -1).sum()) / len(pred) * 100.0
        if k < min_k:
            rows.append({"mcs": mcs, "ms": ms, "k": k, "noise_pct": round(noise, 2),
                         "stability": 0.0, "skipped": "k<min_k"})
            continue
        st = per_group_stability(z, pred, mcs, ms, method, eps)
        stab = sum(st.values()) / len(st) if st else 0.0
        rows.append({"mcs": mcs, "ms": ms, "k": k, "noise_pct": round(noise, 2),
                     "stability": round(float(stab), 4)})
    ok = [r for r in rows if not r.get("skipped")]
    best = max(ok, key=lambda r: r["stability"]) if ok else rows[0]
    return best["mcs"], best["ms"], rows


# ── 실행 ─────────────────────────────────────────────────────────────────
def run(cmd: list[str], env_extra: dict | None = None, log_path: Path | None = None) -> int:
    """subprocess 실행. cp949 UnicodeEncodeError 회피를 위해 PYTHONIOENCODING 강제.
    (logger 경로는 exit code 0 으로 삼켜지지만 print 경로는 exit 1 을 낸다.)"""
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        e.update({k: str(v) for k, v in env_extra.items()})
    print(f"\n$ {' '.join(cmd)}", flush=True)
    if env_extra:
        print(f"  env: {json.dumps({k: str(v) for k, v in env_extra.items()}, ensure_ascii=False)}",
              flush=True)
    if log_path is None:
        return subprocess.call(cmd, cwd=str(REPO), env=e)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.Popen(cmd, cwd=str(REPO), env=e, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        p.wait()
        return p.returncode


def deploy_cmd(backbone: str, pool: str, out: Path, mcs: int, ms: int,
               projs: list[str] | None, device: str, batch: int,
               reassign: str) -> list[str]:
    """grouping_deploy.py 호출 (label-free 산출: summary.json / groups.csv / representatives)."""
    cmd = [sys.executable, "grouping_deploy.py",
           "--backbone", str(rel(backbone)),
           "--pool", str(rel(pool)),
           "--out", str(rel(out)),
           "--mcs", str(mcs), "--ms", str(ms),
           "--method", "leaf", "--eps", "0.06",
           "--device", device, "--batch", str(batch),
           "--reassign", reassign]
    if projs:
        cmd += ["--proj", *[str(rel(p)) for p in projs]]
    return cmd


def train_env(backbone: str, pool: str, out_dir: Path, seed: int, epochs: int,
              recipe: dict) -> dict:
    """_may_ablation.py 학습 환경변수.

    ★ REPRO_NEG 가 아니라 REPRO_IGNORE_NEG_SIM 이다 — 오타 시 조용히 무시되고
      스윕 셀 3개가 중복 실행된 전례가 있다(260726).
    ★ REPRO_WORKERS 는 넘기지 마라 — Windows spawn 으로 학습이 죽는다.
    """
    return {
        "REPRO_DATA": str(rel(pool)),
        "REPRO_BACKBONE": str(rel(backbone)),
        "REPRO_OUT": str(rel(out_dir)),
        "REPRO_SEED": seed,
        "REPRO_EPOCHS": epochs,
        "REPRO_TEMP": recipe["temp"],
        "REPRO_QUEUE": recipe["queue"],
        "REPRO_IGNORE_NEG_SIM": recipe["ignore_neg"],
        "REPRO_LR": recipe["lr_head"],
        "REPRO_BATCH": recipe["batch"],
        "REPRO_SAMPLING": recipe["sampling"],
        "REPRO_USE_LOCAL": "1" if recipe.get("use_local", True) else "0",
        # ★ residual adapter: GAP 직후 proj 앞. gamma=0 초기화라 시작점이 정확히 frozen.
        "REPRO_ADAPTER": "1" if recipe.get("adapter", False) else "0",
    }


# ── z0 대조군 ─────────────────────────────────────────────────────────────
def write_random_proj(dst: Path, seed: int) -> Path:
    """z0 대조군용 랜덤 projection head 생성.

    grouping_deploy.load_proj() 형식({"proj": state_dict, key prefix "net."})으로 저장.
    ★ z0 는 학습이 아니라 forward 1회짜리 대조군이다. 이게 없으면
      "학습이 개선했다"와 "랜덤 투영의 기하 효과"를 구분할 수 없다 —
      실측에서 랜덤 head 만으로 frozen 대비 ARI +0.048 이 나온 pool 이 있다.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024),
                        nn.ReLU(inplace=True), nn.Linear(1024, 128))
    dst = rel(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"proj": {f"net.{k}": v for k, v in net.state_dict().items()}}, dst)
    return dst


def make_z0_set(out_dir: Path, n_heads: int, seed: int) -> list[str]:
    """champion 과 **용량을 맞춘** z0 세트. champion 이 2-head 앙상블이면 z0 도 2개."""
    paths = []
    for i in range(max(1, n_heads)):
        p = write_random_proj(Path(out_dir) / f"z0_h{i}_s{seed}.pt", seed * 1000 + i)
        paths.append(str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p))
    return paths


# ── 결과 ─────────────────────────────────────────────────────────────────
def read_summary(out_dir) -> dict | None:
    p = rel(out_dir) / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt_row(name: str, s: dict | None) -> str:
    if not s:
        return f"  {name:<26} (결과 없음)"
    g = lambda k, d="?": s.get(k, d)
    return ("  {:<26} n={:<6} k={:<4} noise={:<7} seed_noise={:<7} coh={:<7} stab={:<7}"
            .format(name, g("n"), g("k"), g("noise_pct"),
                    g("seed_noise_pct", g("seed_noise")),
                    g("mean_coherence"), g("mean_stability")))


def save_result(out_root, step: str, payload: dict) -> Path:
    d = rel(out_root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{step}_result.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[OUT] {p}", flush=True)
    return p
