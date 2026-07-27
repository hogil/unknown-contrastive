#!/usr/bin/env python3
# ★ 계획 Step 4: 재현된 runner(_may_repro_src) 위에서 컴포넌트 leave-one-in ablation.
# 원본 recipe 의 knob 만 조절 (backbone/data/transform/batch 동일 = same-condition):
#   B0 global-only, B1 +local, B3 +local+queue, B4 full(+NEG). (NeCo 는 원본 source 에 없음 → 제외)
# 사용: python _may_ablation.py B0 B1 B3   (인자 없으면 B0 B1 B3)
# 각 cell run_dir = runs/may_repro/abl_<cell>_<TS>/ , proj_ep 저장 → _score_may_repro.py 로 채점.
import sys
import os
from datetime import datetime
import _may_repro_src as M

CELLS = {
    "B0": {"USE_QUEUE": False, "USE_LOCAL": False, "IGNORE_NEG_SIM": None},   # global InfoNCE only
    "B1": {"USE_QUEUE": False, "USE_LOCAL": True,  "IGNORE_NEG_SIM": None},   # +local 0.5
    "B3": {"USE_QUEUE": True,  "USE_LOCAL": True,  "IGNORE_NEG_SIM": None},   # +local +queue (NEG off)
    "B4": {"USE_QUEUE": True,  "USE_LOCAL": True,  "IGNORE_NEG_SIM": 0.72},   # full (= reproduced)
}

def _parse_ignore_neg_sim(raw):
    """Parse the explicit NEG override; None preserves the existing off state."""
    value = raw.strip()
    if value.lower() in {"off", "none", "null", "false", "disabled"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            "REPRO_IGNORE_NEG_SIM must be a float or one of: off, none, null, false, disabled"
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError("REPRO_IGNORE_NEG_SIM must be between 0 and 1, or off")
    return parsed

# ★ Windows multiprocessing(spawn) 은 워커 시작 시 __main__ 모듈을 다시 import 한다.
#   이 가드가 없으면 REPRO_WORKERS>0 일 때 워커마다 아래 driver loop 전체가 재실행되어
#   M.main() 이 중복 호출되고(런 폴더 2개, 학습 2번 죽임) — 260726 관측된 버그.
if __name__ == "__main__":
    BASE = dict(M.CFG)
    _bb = os.environ.get("REPRO_BACKBONE")          # backbone override (예: FCMAE)
    _tag = os.environ.get("REPRO_TAG", "")            # 출력 dir 태그 (예: _fcmae)
    if _bb:
        BASE["LOCAL_BACKBONE_WEIGHTS"] = _bb
        print(f"[ablation] backbone override -> {_bb}", flush=True)
    _seed = os.environ.get("REPRO_SEED")              # seed override (multi-seed 확정)
    if _seed:
        BASE["SEED"] = int(_seed)
        print(f"[ablation] seed override -> {_seed}", flush=True)
    _neg_override_raw = os.environ.get("REPRO_IGNORE_NEG_SIM")
    _neg_override = None
    if _neg_override_raw is not None:
        _neg_override = _parse_ignore_neg_sim(_neg_override_raw)
        print(f"[ablation] IGNORE_NEG_SIM override -> {_neg_override}", flush=True)
    # ★ USE_LOCAL override — cell 템플릿(B0-B4)이 이미 USE_LOCAL 을 정하므로, NEG 와 같은 이유로
    #   BASE 가 아니라 cell_cfg 적용 "이후" override 필요 (recipe-sweep 에서 local term 단독 on/off 축).
    _local_override_raw = os.environ.get("REPRO_USE_LOCAL")
    _local_override = None
    if _local_override_raw is not None:
        _local_override = _local_override_raw.strip().lower() not in {"0", "false", "off", "none", ""}
        print(f"[ablation] USE_LOCAL override -> {_local_override}", flush=True)
    cells = [a for a in sys.argv[1:] if a in CELLS] or ["B0", "B1", "B3"]
    print(f"[ablation] cells={cells} tag='{_tag}'", flush=True)
    for c in cells:
        cell_cfg = dict(CELLS[c])
        if _neg_override_raw is not None:
            # An explicit override is intentional and applies after the cell recipe.
            cell_cfg["IGNORE_NEG_SIM"] = _neg_override
        if _local_override_raw is not None:
            cell_cfg["USE_LOCAL"] = _local_override
        M.CFG.clear(); M.CFG.update(BASE); M.CFG.update(cell_cfg)
        M.CFG["OUTPUT_DIR"] = BASE["OUTPUT_DIR"].rsplit("/run", 1)[0] + f"/abl{_tag}_{c}"
        M.RUN_TS = datetime.now().strftime("%y%m%d_%H%M%S")
        print(f"\n===== CELL {c} : {cell_cfg} -> {M.CFG['OUTPUT_DIR']}_{M.RUN_TS} =====", flush=True)
        M.main()
    print("[ABLATION_DONE]", flush=True)
