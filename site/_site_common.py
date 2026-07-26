#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""site/ 공통 유틸 — 사내 서버 무라벨 배포 파이프라인.

★ 경로 규칙: 전부 **프로젝트 루트 기준 상대경로**. 이 파일 위치에서 루트를 derive 하므로
  프로젝트 폴더를 통째로 어디에 두든 그대로 돈다. 절대경로 하드코딩 금지.

★ 설정 규칙: 각 step 스크립트 맨 위 `class Config` 가 단일 소스.
  환경변수가 있으면 그걸 쓰고, 없으면 Config 의 default 를 쓴다.

여기 담긴 판정 규칙은 260726 실측 도출. 근거: docs/GOAL_AND_LADDER.md
  1. 사내엔 라벨이 없다 -> 모든 선택은 label-free.
  2. 다이얼(HDBSCAN mcs/ms)은 pool 기하에 맞춘다: mcs ~= (n/k_hat) * 0.10.
     다른 pool 값 이식 금지. mcs6 을 그대로 썼다가 결론이 뒤집힌 전례가 있다.
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
    """루트 기준 상대경로 -> 절대 Path. 이미 절대경로면 그대로 둔다."""
    p = Path(*parts)
    return p if p.is_absolute() else (REPO / p)


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


def check_inputs(backbone: str, image_root: str, k_hat: int, projs: list[str] | None = None):
    bb = rel(backbone)
    if not bb.exists():
        die(f"backbone 이 없다: {bb}\n"
            f"  프로젝트 루트 기준 상대경로로 두어라. 필요한 파일은 site/README.md 의 "
            f"'보내야 할 체크포인트' 참조.")
    ir = rel(image_root)
    if not ir.exists():
        die(f"image_root 가 없다: {ir}\n  SITE_IMAGE_ROOT 로 지정하거나 Config 의 default 를 고쳐라.")
    if int(k_hat) < 2:
        die("K_HAT(예상 불량 종수)은 2 이상이어야 한다.\n"
            "  모르면 현업에 물어라 — 라벨 없이 다이얼을 고를 방법이 없다는 게 실측으로 확인됐다"
            "(무라벨 대리지표 Sil/over_merge/stability 전부 arm 에 따라 부호가 뒤집힘).")
    for p in (projs or []):
        if not rel(p).exists():
            die(f"projection head 가 없다: {rel(p)}\n  site/README.md 의 체크포인트 목록 참조.")


# ── 다이얼 ────────────────────────────────────────────────────────────────
def recommend_dial(n: int, k_hat: int) -> tuple[int, int]:
    """pool 기하 -> HDBSCAN 다이얼.  mcs ~= (n/k) * 0.10

      clean546   546/9  = 60.7  -> mcs 6  (9.9%)   mcs6 정상 작동
      anchor    2260/43 = 52.6  -> mcs 5  (11.4%)  mcs6 정상 작동
      severstal  995/5  = 199.0 -> mcs 20 (10.1%)  <- 168셀 스윕 승자와 일치
    mcs6 을 severstal(3.0%)에 그대로 이식했다가 "적응이 품질을 희생한다"는
    정반대 결론이 나왔던 전례가 있다. ms 는 승자 조합(mcs20/ms5)에서 mcs/4.
    """
    per_class = n / max(1, int(k_hat))
    mcs = max(5, int(round(per_class * 0.10)))
    ms = max(3, int(round(mcs / 4)))
    return mcs, ms


def dial_search_range(n: int, k_hat: int, n_points: int = 4) -> list[int]:
    """좁은 sweep 범위 n/(15k) ~ n/(8k). 단일값으로 확신하지 마라."""
    per_class = n / max(1, int(k_hat))
    lo = max(5, int(round(per_class / 15)))
    hi = max(lo + 1, int(round(per_class / 8)))
    if n_points <= 1:
        return [lo]
    step = max(1, (hi - lo) // (n_points - 1))
    return sorted({min(hi, lo + i * step) for i in range(n_points)})


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
