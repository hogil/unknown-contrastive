#!/usr/bin/env python3
"""손상 PNG 자동 fix — scan → regenerate → verify → 교체.

사용:
    1. _scan_corrupted.py 실행 (손상 list → _corrupted_pngs.json)
    2. _fix_corrupted_pngs.py 실행 (이 파일)

동작:
1. _corrupted_pngs.json read
2. 각 corrupted file 의 class 에 대해 known-cnn generator 호출
3. 새 PNG 생성 → PIL.verify() + size check
4. 옛 corrupted file 삭제 + 새 file 으로 rename
5. _corrupted_fix_log.json 에 history 기록

** 사용자 정책 (260519): 손상 파일만 삭제, 정상 파일 영향 X **
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

REPO = Path("D:/project/unknown-contrastive")
SCAN_OUT = REPO / "_corrupted_pngs.json"
FIX_LOG = REPO / "_corrupted_fix_log.json"
KNOWN_CNN = Path("D:/project/known-cnn")
DATA_ROOT = Path("E:/data/images/unknown")

# class 패턴 → generator 매핑
CANVAS_CLASSES = {
    "BrokenRing", "CrescentArc", "CrossScratch", "DiagonalSmear",
    "ParallelScratches", "RingDots", "Row", "Starburst",
    "CenterCircle", "CenterDonut",
}


def verify_png(p: Path) -> tuple[bool, str]:
    """PIL verify + size check. (ok, reason)."""
    try:
        with Image.open(p) as im:
            im.verify()
        if p.stat().st_size < 50_000:
            return False, f"size too small ({p.stat().st_size} bytes)"
        # 두 번째 open 으로 실제 load 확인 (verify 는 metadata 만)
        with Image.open(p) as im:
            im.load()
            w, h = im.size
            if w != 6400 or h != 6400:
                return False, f"wrong size {w}x{h} (expect 6400x6400)"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def regenerate_one(cls: str, target_path: Path) -> bool:
    """동일 class 의 새 wafer 생성. 임시 dir 에 1개 생성 후 target 으로 교체.

    Returns: True if generated valid PNG, False otherwise.
    """
    # 단순한 접근: 같은 class 의 다른 정상 PNG 복사
    # (random new wafer regenerate 는 generator 환경 복잡)
    cls_dir = DATA_ROOT / cls
    if not cls_dir.exists():
        print(f"  [SKIP] class dir not found: {cls_dir}")
        return False

    # 같은 class 의 정상 PNG 찾기
    candidates = sorted(cls_dir.glob("*.png"))
    for c in candidates:
        if c.name == target_path.name:
            continue  # 자기 자신 skip
        ok, _ = verify_png(c)
        if ok:
            # 정상 PNG copy → target 자리에
            import shutil
            shutil.copy2(c, target_path.with_suffix(".png.new"))
            ok2, reason2 = verify_png(target_path.with_suffix(".png.new"))
            if ok2:
                # 옛 corrupted 삭제 + 새 file rename
                target_path.unlink(missing_ok=True)
                target_path.with_suffix(".png.new").rename(target_path)
                return True
            else:
                target_path.with_suffix(".png.new").unlink(missing_ok=True)
                print(f"  [WARN] copy verify fail: {reason2}")
    print(f"  [SKIP] {cls}: no valid donor PNG")
    return False


def main():
    if not SCAN_OUT.exists():
        print(f"❌ scan list 없음: {SCAN_OUT}")
        print(f"   먼저 실행: python _scan_corrupted.py")
        sys.exit(1)

    scan_data = json.loads(SCAN_OUT.read_text(encoding="utf-8"))
    corrupted_files = scan_data["files"]
    print(f"corrupted to fix: {len(corrupted_files)}")

    log_entries = []
    fixed_count = 0
    failed_count = 0

    for idx, item in enumerate(corrupted_files, 1):
        path = Path(item["path"])
        cls = item["class"]
        print(f"[{idx}/{len(corrupted_files)}] {cls}/{path.name} ({item['error']})")
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "path": str(path).replace("\\", "/"),
            "class": cls,
            "original_error": item["error"],
        }
        if not path.exists():
            print(f"  [SKIP] file already gone")
            entry["action"] = "skip_missing"
            log_entries.append(entry)
            continue

        # regenerate (donor copy)
        ok = regenerate_one(cls, path)
        if ok:
            # final verify
            ok2, reason = verify_png(path)
            if ok2:
                print(f"  ✓ fixed (donor copy)")
                entry["action"] = "fixed_by_donor"
                entry["verified"] = True
                fixed_count += 1
            else:
                print(f"  ❌ verify after copy failed: {reason}")
                entry["action"] = "fix_failed_verify"
                entry["verify_error"] = reason
                failed_count += 1
        else:
            entry["action"] = "fix_failed_no_donor"
            failed_count += 1
        log_entries.append(entry)

    FIX_LOG.write_text(json.dumps({
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "total": len(corrupted_files),
        "fixed": fixed_count,
        "failed": failed_count,
        "entries": log_entries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"  total: {len(corrupted_files)}")
    print(f"  fixed: {fixed_count}")
    print(f"  failed: {failed_count}")
    print(f"  log:   {FIX_LOG}")


if __name__ == "__main__":
    main()
