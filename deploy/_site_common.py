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
     다이얼은 k 가 아니라 **"몇 장 이상 뭉쳐야 그룹으로 볼 것인가"**(Cluster.MCS)로 정한다.
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

# ── 프로젝트 루트 (이 파일: <root>/deploy/_site_common.py) ─────────────────────
REPO = Path(__file__).resolve().parent.parent


def add_path(p) -> None:
    """sys.path 에 **맨 뒤로** 추가한다.

    insert(0, ...) 은 repo 안의 아무 폴더나 site-packages 를 가리게 만든다.
    실제로 폴더명 하나(site)가 stdlib 을 가려 PIL import 가 깨진 전례가 있다.
    표준 경로가 먼저 이기도록 append 한다.
    """
    p = str(p)
    if p not in sys.path:
        sys.path.append(p)

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


_ENV_OVERRIDE = os.environ.get("SITE_ENV_OVERRIDE", "0").strip().lower() in ("1", "true", "yes", "on")
_ENV_USED: dict = {}


def env(name: str, default):
    """★ **config.py 가 권위다.** 환경변수는 `SITE_ENV_OVERRIDE=1` 일 때만 이긴다.

    예전에는 환경변수가 항상 우선이었다. 그래서 쉘에 한 번 `export SITE_MCS=6` 해두면
    **config.py 를 고쳐도 조용히 무시됐다** — 왜 안 바뀌는지 알 방법이 없었다.
    지금은 기본적으로 config 값이 그대로 쓰이고, 환경변수로 덮으려면 명시해야 하며
    덮은 항목은 show_config() 가 `[env]` 로 표시한다.

    일회성 실험은:  SITE_ENV_OVERRIDE=1 SITE_MCS=6 python deploy/step0_prepare.py
    """
    v = os.environ.get(name)
    if v is None or v == "" or not _ENV_OVERRIDE:
        return default
    _ENV_USED[name] = v
    if isinstance(default, bool):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


# ★ 잡음폭/민감도의 정본은 **config.Judge** 다. 여기 있던 사본은 키까지 소문자로
#   달라서(ari vs ARI) 두 표가 조용히 갈릴 수 있었다. 아래 band() 로 통일한다.


def die(msg: str, code: int = 2):
    print(f"\n[FATAL] {msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


# ★ step 이 **실제로 돌기 시작한** 시각. run 폴더 이름은 step0 이 만든 타임스탬프라
#   step1~5 가 언제 돌았는지는 거기서 알 수 없다 (폴더를 쪼개면 step0 의 manifest·캐시를
#   물려 쓰는 체인이 깨지므로 폴더는 공유하고 **시각만 따로 기록**한다, 260730).
_STEP_START: dict = {}


def banner(step: str, title: str, ladder: str = ""):
    import time
    from datetime import datetime
    _STEP_START["name"] = step
    _STEP_START["t0"] = time.time()
    _STEP_START["at"] = datetime.now().strftime("%y%m%d_%H%M%S")
    line = "=" * 78
    print(f"\n{line}\n[{step}] {title}")
    if ladder:
        print(f"  배포 사다리: {ladder}")
    print(f"  시작: {_STEP_START['at']}")
    print(line, flush=True)


def run_root(base, create: bool = False, tag: str = "") -> Path:
    """타임스탬프 run 폴더. **step0 이 만들고 step1~5 가 물려 쓴다.**

        runs/site/                 <- base (Paths.OUT_ROOT)
          ├── _cache/              <- 디코드 캐시. run 마다 다시 만들면 8.7GiB 낭비라 여기 공유
          ├── latest.txt           <- 마지막 step0 이 만든 run 폴더 경로
          └── 260727_183000/       <- 이번 run. step0~5 산출이 전부 여기
               step0_result.json …

    ★ step 마다 새 타임스탬프를 만들면 step1 이 step0 결과를 못 찾는다.
      그래서 step0 만 create=True 이고 나머지는 latest.txt 를 읽는다.
    ★ 이전 run 을 다시 쓰려면 SITE_RUN=260727_183000 (또는 tag 인자).
      기존 결과는 절대 덮어쓰지 않는다 — 새로 돌리면 새 폴더다.
    """
    base = rel(base)
    tag = tag or os.environ.get("SITE_RUN", "").strip()
    ptr = base / "latest.txt"
    if tag:
        d = base / tag
    elif create:
        from datetime import datetime
        d = base / datetime.now().strftime("%y%m%d_%H%M%S")
    elif ptr.exists():
        d = Path(ptr.read_text(encoding="utf-8").strip())
        if not d.is_absolute():
            d = rel(d)
    else:
        die(f"run 폴더를 찾을 수 없다 ({ptr} 없음).\n"
            f"  먼저:  python deploy/step0_prepare.py\n"
            f"  또는 특정 run 을 지정:  SITE_RUN=<폴더명> python deploy/step1_zeroshot.py")
    if create:
        d.mkdir(parents=True, exist_ok=True)
        ptr.write_text(str(d), encoding="utf-8")
    if not d.exists():
        die(f"지정한 run 폴더가 없다: {d}\n  {base} 아래 폴더명을 확인하라.")
    return d


def live_dial() -> tuple[int, int, str, float]:
    """★ 다이얼은 **항상 config.py 에서 그 순간 값을 읽는다.** step0 이 예전에 뭘로
    돌았든 상관없다 — 절대 얼리지 않는다.

    (예전엔 step0 이 dial 을 step0_result.json 에 저장하고 step1~5 가 그 값을
    그대로 물려 썼다. config 를 고쳐도 step0 을 다시 안 돌리면 반영이 안 됐고,
    이게 아무 경고 없이 조용히 무시됐다 — 실측: config MCS=10 으로 고치고
    step1 만 다시 돌렸더니 로그에는 여전히 --mcs 20 이 찍혔다.
    사용자 지시(260728): "무조건 config에서 가져가게" — 얼리는 설계를 버린다.)
    """
    from config import Cluster as C
    return int(C.MCS), int(C.MS), str(C.METHOD), float(C.EPS)


def step_dir(out_root, base_name: str) -> Path:
    """step 산출 폴더를 **그 step 을 돌린 시각으로** 만든다: `<run>/<base>_<YYMMDD_HHMMSS>`.

    ★ 왜: 예전엔 `<run>/step1_zeroshot` 고정이라 step1 을 **다시 돌리면 앞 결과를 덮어썼다.**
      폴더가 안 생기는 게 아니라 같은 자리에 지우고 쓴 것이다 — 프로젝트 절대규칙
      ("새 실험은 반드시 새 폴더명", 결과 삭제/덮어쓰기 금지) 위반이었다 (260730).
    ★ run 폴더 자체는 step0 것이라 공유해야 한다(step1~5 가 manifest·캐시를 물려 쓴다).
      그래서 **run 은 공유, step 산출 폴더만 시각으로 분리**한다.
    ★ `<base>_latest.txt` 에 이번 폴더명을 남겨, 사람이 최신을 바로 찾을 수 있게 한다.
    """
    d = rel(out_root)
    stamp = _STEP_START.get("at") or "nostamp"
    out = d / f"{base_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    try:
        (d / f"{base_name}_latest.txt").write_text(out.name + "\n", encoding="utf-8")
    except Exception:
        pass
    print(f"[out] {base_name} -> {out.name}  (재실행하면 새 폴더가 생긴다)", flush=True)
    return out


def show_config(C) -> dict:
    """Config 클래스의 대문자 속성을 표로 출력하고 dict 로 반환 (재현성 기록용).

    ★ 여기서 `apply_cluster_env()` 도 같이 부른다 — 모든 step 이 main() 맨 앞에서
      이 함수를 부르므로, 한 군데만 걸어두면 step 하나를 빠뜨릴 일이 없다.
      (step 을 빠뜨려서 생기는 조용한 불일치가 바로 260729 감사에서 나온 버그다.)
    """
    apply_cluster_env()
    d = {k: getattr(C, k) for k in dir(C) if k.isupper() and not k.startswith("_")}
    src = "config.py  (SITE_ENV_OVERRIDE=1 이면 환경변수가 이긴다)"
    if _ENV_OVERRIDE:
        src = f"★ SITE_ENV_OVERRIDE=1 — 환경변수 {len(_ENV_USED)}개가 config 를 덮었다"
    print(f"[config] 출처: {src}")
    for k in sorted(d):
        print(f"    {k:<22} = {d[k]}")
    if _ENV_USED:
        print("  [env] config 를 덮은 항목:")
        for k in sorted(_ENV_USED):
            print(f"    {k:<22} = {_ENV_USED[k]}")
    print(flush=True)
    return d


def show_images(image_root: str, manifest: str = "", exts: str = "") -> dict:
    """★ 매 step 시작 시 **이미지 폴더 절대경로와 장수**를 찍는다.

    설정만 보고는 "그래서 어디 걸 몇 장 읽는가"를 알 수 없다 — 상대경로가 어느
    절대경로로 풀렸는지, 하위폴더까지 실제 몇 장이 잡히는지가 사고의 대부분이다.
    manifest 가 지정돼 있으면 그게 실제 입력이므로 그쪽 장수를 우선 보고한다.
    """
    import json
    ex = tuple("." + e.strip().lower().lstrip(".")
               for e in (exts or "png,jpg,jpeg,bmp,tif,tiff").split(",") if e.strip())
    print("[images]")
    info: dict = {"image_root": "", "n_images": 0, "manifest": "", "n_manifest": 0}

    ir = rel(image_root) if image_root else None
    if ir is not None:
        info["image_root"] = str(ir)
        print(f"    image_root       = {ir}")
        if not ir.exists():
            print("                       ^ 없음 (경로를 확인하라)")
        else:
            n, per_dir = 0, {}
            for p in ir.rglob("*"):
                if p.is_file() and p.suffix.lower() in ex:
                    n += 1
                    per_dir[p.parent.name] = per_dir.get(p.parent.name, 0) + 1
            info["n_images"] = n
            info["n_subdirs"] = len(per_dir)
            print(f"    이미지 수        = {n:,} 장  (하위폴더 전부 재귀, 확장자 {','.join(ex)})")
            if per_dir:
                top = sorted(per_dir.items(), key=lambda kv: -kv[1])[:5]
                more = f"  … 외 {len(per_dir)-len(top)}개" if len(per_dir) > len(top) else ""
                print(f"    하위폴더 {len(per_dir)}개    "
                      + ", ".join(f"{k}({v:,})" for k, v in top) + more)
            if n == 0:
                print("                       ^ 0 장이다. 확장자/경로를 확인하라.")

    if manifest:
        mp = rel(manifest)
        info["manifest"] = str(mp)
        print(f"    manifest         = {mp}")
        if not mp.exists():
            print("                       ^ 없음 -> image_root 를 스캔한다")
        else:
            try:
                d = json.loads(mp.read_text(encoding="utf-8"))
                nm = len(d.get("files", []))
                info["n_manifest"] = nm
                print(f"    manifest 이미지  = {nm:,} 장  ★ 실제 입력은 이쪽이다 "
                      f"(root={d.get('root', '?')})")
            except Exception as e:
                print(f"                       ^ 읽기 실패: {type(e).__name__}: {e}")
    print(flush=True)
    return info


def check_inputs(backbone: str, image_root: str, projs: list[str] | None = None):
    bb = rel(backbone)
    if not bb.exists():
        die(f"backbone 이 없다: {bb}\n"
            f"  프로젝트 루트 기준 상대경로로 두어라. 필요한 파일은 deploy/README.md 의 "
            f"'보내야 할 체크포인트' 참조.")
    ir = rel(image_root)
    if not ir.exists():
        die(f"image_root 가 없다: {ir}\n  SITE_IMAGE_ROOT 로 지정하거나 Config 의 default 를 고쳐라.")
    for p in (projs or []):
        if not rel(p).exists():
            die(f"projection head 가 없다: {rel(p)}\n  deploy/README.md 의 체크포인트 목록 참조.")


def run(cmd: list[str], env_extra: dict | None = None, log_path: Path | None = None) -> int:
    """subprocess 실행. cp949 UnicodeEncodeError 회피를 위해 PYTHONIOENCODING 강제.
    (logger 경로는 exit code 0 으로 삼켜지지만 print 경로는 exit 1 을 낸다.)"""
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    # ★ config.Cluster 중 환경변수로만 전달되는 값들을 **항상** 넣는다.
    #   안 넣으면 config 를 고쳐도 서브프로세스에 도달하지 않는다 (6개가 죽어 있었다).
    #   호출자가 env_extra 로 준 값이 이보다 우선한다 (step1 의 UC_PALETTE_MASK 등).
    e.update(cluster_env())
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


def baseline_arms(step1_arms: dict) -> tuple[dict | None, dict | None, str]:
    """step1 arm 표에서 **내 palette 설정과 같은** frozen / z0 기준선을 고른다.

    ★ 왜 필요한가 (260731 실측). step2~5 는 `Cluster.PALETTE_MASK` 로 마스킹을 켜고
      학습하는데, step1 의 `frozen` arm 은 raw 다. 그대로 비교하면:
          frozen        33.67  raw,    학습 X
          frozen_masked  8.83  마스킹, 학습 X   <- step2 와 같은 입력
          step2 seed42  11.33  마스킹, 학습 O
      step2 는 "vs frozen 22.34pp 개선 — 유효" 라고 판정했지만, 그 22.34pp 는 거의
      전부 **마스킹 효과**이고 학습은 오히려 2.50pp 나빴다. 1축 비교가 깨진 것이다.
      마스킹을 켜면 기준선도 `frozen_masked` / `z0_*_masked` 여야 한다.

    반환: (frozen, z0, 설명문). 맞는 arm 이 없으면 raw 로 떨어지고 설명문에 경고를 남긴다.
    """
    try:
        from config import Cluster as C
        masked = bool(C.PALETTE_MASK)
    except Exception:
        masked = False
    if not masked:
        z0k = next((k for k in step1_arms if k.startswith("z0") and not k.endswith("_masked")), None)
        return step1_arms.get("frozen"), (step1_arms.get(z0k) if z0k else None), "raw"

    fz = step1_arms.get("frozen_masked")
    z0k = next((k for k in step1_arms if k.startswith("z0") and k.endswith("_masked")), None)
    z0 = step1_arms.get(z0k) if z0k else None
    note = "masked(unify_border) — 내 학습 입력과 같은 조건의 기준선"
    if fz is None:
        fz = step1_arms.get("frozen")
        note = ("★ 경고: 마스킹을 켜고 학습했는데 step1 에 masked 기준선이 없다. "
                "raw frozen 과 비교하면 마스킹 효과가 학습 효과로 잘못 집계된다 — "
                "step1 을 다시 돌려라 (Cluster.PALETTE_PROBE=True).")
    return fz, z0, note


def band(axis: str = "noise") -> float:
    """판정 잡음폭. **config.Judge.BAND 에서 매번 live 로 읽는다.**

    이 안의 차이는 "차이 없음"이다 — 동일 config 를 4회 반복해서 나온 측정 산포다.
    ★ 260731 이전에는 `Judge` 클래스를 **어디서도 import 하지 않았고**, 2.28 이
      step1/2/4 에 하드코딩돼 있었다. config 를 고쳐도 판정은 옛 값으로 돌았다 —
      `PALETTE_MODE` 가 죽어 있던 것과 똑같은 유형이다.
    """
    try:
        from config import Judge
        return float(Judge.BAND[axis])
    except Exception:
        return {"noise": 2.28, "ARI": 0.019, "Hom": 0.005,
                "Comp": 0.033, "P1": 0.0}.get(axis, 0.0)


def cluster_env() -> dict:
    """config.Cluster 의 값 중 **환경변수로만 전달되는 것**을 모아 준다.

    ★ 이게 없으면 config 에 값을 써도 파이프라인에 도달하지 않는다.
      grouping_deploy 는 이것들을 os.environ 에서 읽는데, step 이 안 넘기면
      config 를 고쳐도 아무 일도 안 일어난다 (실측으로 6개가 죽어 있었다).
    config 를 여기서 import 하면 순환이라(config 가 _site_common 을 import) 지연 import 한다.
    """
    try:
        from config import Cluster as C, Composite as K
    except Exception:
        return {}
    try:
        from config import Runtime as R
    except Exception:
        R = None
    d = {
        "SITE_COMPOSITE_MAX_MEMBERS": str(K.MAX_MEMBERS),
        "UC_PALETTE_MODE": str(C.PALETTE_MODE),
        # ★ MODE 를 켜는 스위치. 이게 빠져 있어서 `unify_border` 가 **죽은 설정**이었다 —
        #   grouping_deploy 기본이 "0" 이라 step2/3/4/5 는 전부 raw 로 채점되고 있었다.
        #   step1 은 arm 마다 env_extra 로 덮어쓰므로 여기 값에 영향받지 않는다.
        "UC_PALETTE_MASK": "1" if getattr(C, "PALETTE_MASK", False) else "0",
        "SITE_METRIC": str(C.METRIC),
        "SITE_OVER_MERGE_FRAC": str(C.OVER_MERGE_FRAC),
        "SITE_HDBSCAN_GPU": "1" if C.HDBSCAN_GPU else "0",
    }
    if R is not None:
        # ★ 260729 감사: 이 3개도 config 에 써도 아무 데도 도달하지 않고 있었다
        #   (cluster_env 가 만들어진 이유와 똑같은 실패가 3건 더 남아 있었다).
        #   grouping_deploy 는 셋 다 os.environ 에서 읽는다.
        d.update({
            "SITE_DECODE_WORKERS": str(R.DECODE_WORKERS),
            "SITE_AMP": str(R.AMP),
            "SITE_CHANNELS_LAST": "1" if R.CHANNELS_LAST else "0",
        })
    return d


def apply_cluster_env() -> dict:
    """`cluster_env()` 를 **이 프로세스의 os.environ 에도** 적용한다.

    ★ run() 은 서브프로세스 env 만 채운다(e = dict(os.environ) 사본). 그런데
      step2 의 Rule C 채점과 step5 의 시간축 격자는 **step 자기 프로세스 안에서**
      grouping_deploy.hdbscan_predict 를 직접 부르고, 그건 SITE_METRIC /
      SITE_HDBSCAN_GPU 를 os.environ 에서 읽는다. 그래서 예전엔 config 로
      METRIC=cosine 을 켜면 **최종 grouping 만 cosine, epoch 선택은 euclidean**
      이라는 split-brain 이 조용히 생겼다 (260729 감사).
      각 step 이 시작할 때 한 번 부르면 in-process 와 subprocess 가 같아진다.
    """
    e = cluster_env()
    os.environ.update(e)
    return e


def deploy_cmd(backbone: str, pool: str, out: Path, mcs: int, ms: int,
               projs: list[str] | None, device: str, batch: int,
               reassign: str, cache: str = "",
               no_composites: bool = False, partition_by: str = "",
               method: str = "", eps=None) -> list[str]:
    """grouping_deploy.py 호출 (label-free 산출: summary.json / groups.csv / representatives).

    ★ cache 를 넘기면 디코드를 건너뛴다. step1 은 arm 이 6개라 캐시가 없으면
      **같은 이미지를 6번 다시 디코드**한다 — 그동안 GPU 는 논다.
      결과는 비트 단위로 동일하다 (검증: groups.csv/summary.json 완전 일치).
    """
    # ★ method/eps 를 하드코딩하면 config.Cluster.METHOD/EPS 가 죽는다.
    if not method or eps is None:
        try:
            from config import Cluster as _C
            method = method or str(_C.METHOD)
            eps = _C.EPS if eps is None else eps
        except Exception:
            method, eps = method or "leaf", 0.06 if eps is None else eps
    cmd = [sys.executable, "grouping_deploy.py",
           "--backbone", str(rel(backbone)),
           "--pool", str(rel(pool)),
           "--out", str(rel(out)),
           "--mcs", str(mcs), "--ms", str(ms),
           "--method", str(method), "--eps", str(eps),
           "--device", device, "--batch", str(batch),
           "--reassign", reassign]
    # ★ 대표 이미지 수. 안 넘기면 grouping_deploy 의 기본값(12)이 그대로 쓰여
    #   config 를 고쳐도 반영이 안 된다 (260731 까지 그랬다).
    try:
        from config import Composite as _K
        cmd += ["--reps", str(int(_K.MAX_REPS))]
    except Exception:
        pass
    if cache:
        cmd += ["--cache", str(rel(cache))]
    if no_composites:
        cmd += ["--no-composites"]
    if partition_by:
        cmd += ["--partition-by", partition_by]
    if projs:
        cmd += ["--proj", *[str(rel(p)) for p in projs]]
    return cmd


def cache_dir_for(out_root, cache_dir: str = "") -> str:
    """캐시 폴더 경로. 비어 있으면 <OUT_ROOT>/_cache."""
    return str(rel(cache_dir)) if cache_dir else str(rel(out_root) / "_cache")


def train_env(backbone: str, pool: str, out_dir: Path, seed: int, epochs: int,
              recipe: dict) -> dict:
    """_may_ablation.py 학습 환경변수.

    ★ REPRO_NEG 가 아니라 REPRO_IGNORE_NEG_SIM 이다 — 오타 시 조용히 무시되고
      스윕 셀 3개가 중복 실행된 전례가 있다(260726).
    ★ REPRO_WORKERS 는 **Windows 에서** 넘기지 마라 — spawn 으로 학습이 죽는다.
      그래서 `Runtime.TRAIN_WORKERS` 기본값이 0(단일 스레드)이고, 0 이면 아예 안 넘긴다.
      사내 Ubuntu24 에서는 >0 으로 켜야 학습이 디코드 병목에서 풀린다 —
      실측 2.65 img/s(anchor)는 GPU 연산이 아니라 6400x6400 단일스레드 디코드 탓이다.
    """
    _w = 0
    try:
        from config import Runtime as _R
        _w = int(getattr(_R, "TRAIN_WORKERS", 0) or 0)
    except Exception:
        _w = 0
    extra = {"REPRO_WORKERS": str(_w)} if _w > 0 else {}
    # ★ device 를 반드시 넘긴다. 안 넘기면 학습기가 혼자 판단해 CUDA 없을 때 **조용히
    #   CPU 로** 떨어진다 — 사내 서버에서 "step2 가 GPU 를 전혀 안 쓴다"의 원인이었다.
    #   넘기면 CUDA 가 없을 때 학습기가 진단을 찍고 **죽는다** (260730).
    try:
        from config import Runtime as _R2
        extra["REPRO_DEVICE"] = str(getattr(_R2, "DEVICE", "cuda"))
    except Exception:
        extra["REPRO_DEVICE"] = "cuda"
    # ★ 학습기 **자체 클러스터링** 다이얼도 config 를 따르게 한다.
    #   `_may_repro_src` 는 자기 CFG(MIN_CLUSTER_SIZE=12 등)로 `cluster_summary/` 와
    #   `clusters/` 를 만드는데, 그동안 config 와 **다른 값**으로 돌고 있었고 바꿀
    #   방법도 없었다 (260731). 이제 deploy 로 돌리면 config.Cluster 와 일치한다.
    try:
        from config import Cluster as _C2
        extra.update({
            "REPRO_MCS": str(_C2.MCS), "REPRO_MS": str(_C2.MS),
            "REPRO_METHOD": str(_C2.METHOD), "REPRO_EPS": str(_C2.EPS),
            "REPRO_METRIC": str(_C2.METRIC),
            # ★ palette 전처리를 **학습에도** 준다. 학습기는 원래 마스킹 코드가 아예
            #   없어서(`Image.open(p).convert("RGB")` 뿐) 항상 raw 로 학습했다.
            #   채점만 마스킹하면 학습 때와 다른 입력이 되어 결과가 무의미해지므로
            #   반드시 cluster_env() 와 **같은 값**이어야 한다 (260731).
            "UC_PALETTE_MASK": "1" if getattr(_C2, "PALETTE_MASK", False) else "0",
            "UC_PALETTE_MODE": str(_C2.PALETTE_MODE),
        })
    except Exception:
        pass
    return {**extra,
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
    return ("  {:<26} n={:<6} k={:<4} noise={:<7} seed_noise={:<7} coh={:<7}"
            .format(name, g("n"), g("k"), g("noise_pct"),
                    g("seed_noise_pct", g("seed_noise")),
                    g("mean_coherence")))


def save_result(out_root, step: str, payload: dict) -> Path:
    """결과 JSON 저장 + **그 step 이 실제로 돌은 시각**을 같이 남긴다.

    ★ run 폴더 이름(`runs/site/260728_160026`)은 step0 이 만든 시각이라, 거기만 보면
      step1~5 가 언제 돌았는지 알 수 없었다. 폴더를 step 마다 쪼개면 step0 의
      manifest·디코드 캐시를 물려 쓰는 체인이 깨지므로(run_root 주석 참조), 폴더는
      공유한 채 **시각을 결과 JSON + `_steps.log` 양쪽에 기록**한다 (260730).
    """
    import time
    from datetime import datetime
    d = rel(out_root)
    d.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    t0 = _STEP_START.get("t0")
    elapsed = (time.time() - t0) if t0 else None
    payload = dict(payload)
    payload["_timing"] = {
        "started_at": _STEP_START.get("at"),
        "finished_at": now.strftime("%y%m%d_%H%M%S"),
        "elapsed_sec": round(elapsed, 1) if elapsed is not None else None,
        "run_dir_stamp": Path(str(d)).name,   # step0 이 만든 폴더 시각 (구분용)
    }
    body = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    # ★ 고정 이름 = **최신 포인터**. downstream step 이 이 이름을 읽으므로 유지해야 한다
    #   (step2 는 step1_result.json 을 기준선으로 읽는다).
    p = d / f"{step}_result.json"
    p.write_text(body, encoding="utf-8")
    # ★ 이력 사본 — 고정 이름만 쓰면 재실행 때 앞 결과가 사라진다. 결과 덮어쓰기 금지
    #   규칙에 맞게 실행 시각을 붙인 사본을 함께 남긴다 (260730).
    stamp = _STEP_START.get("at")
    if stamp:
        (d / f"{step}_result_{stamp}.json").write_text(body, encoding="utf-8")

    # run 루트에 step 별 실행 시각 원장 — 폴더 하나만 보고 전 step 이력을 알 수 있게.
    #   step3 는 partial 을 여러 번 저장하므로 **같은 step 줄은 덮어쓴다.**
    el = f"{elapsed/60:.1f}분" if elapsed and elapsed >= 60 else (f"{elapsed:.0f}초" if elapsed else "?")
    line = (f"{step:<16} {_STEP_START.get('at','?')} -> {payload['_timing']['finished_at']}"
            f"  ({el})")
    ledger = d / "_steps.log"
    # ★ dedupe 키 = (step 이름, **시작시각**). 이유:
    #   - step3 는 한 번 실행 중에 partial 을 수십 번 저장한다 -> 시작시각이 같으니 1줄로 합쳐진다.
    #   - 같은 step 을 **다시 돌리면** 시작시각이 다르니 **새 줄로 쌓인다** = 재실행 이력이 남는다.
    #   step 이름만으로 비교했더니 step1 을 3번 돌려도 마지막 1줄만 남아 이력이 사라졌다 (260730).
    #   (접두사 비교도 안 된다 — "step3" 저장이 "step3_partial" 줄까지 지운다.)
    try:
        prev = ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []
    except Exception:
        prev = []
    key = (step, str(_STEP_START.get("at", "")))

    def _key_of(l: str):
        f = l.split()
        return (f[0], f[1]) if len(f) >= 2 else (l.strip(), "")

    old = [l for l in prev if l.strip() and _key_of(l) != key]
    ledger.write_text("\n".join(old + [line]) + "\n", encoding="utf-8")

    print(f"\n[OUT] {p}")
    print(f"[time] {line}   (run 폴더 시각 {payload['_timing']['run_dir_stamp']} 은 step0 것)",
          flush=True)
    return p
