#!/usr/bin/env python3
"""E:/data/images/unknown/ 전체 PNG 스캔 → 손상 파일 list 저장."""
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path("E:/data/images/unknown")
OUT = Path("D:/project/unknown-contrastive/_corrupted_pngs.json")

bad = []
total = 0
for cls_dir in sorted(ROOT.iterdir()):
    if not cls_dir.is_dir() or cls_dir.name in ["classification", "classification_chips"]:
        continue
    for p in cls_dir.glob("*.png"):
        total += 1
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append({
                "path": str(p).replace("\\", "/"),
                "class": cls_dir.name,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            })

OUT.write_text(
    json.dumps({
        "scan_ts": datetime.now().isoformat(timespec="seconds"),
        "total_scanned": total,
        "corrupted_count": len(bad),
        "files": bad,
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"scanned {total} PNG, corrupted {len(bad)}")
for b in bad:
    print(f"  {b['class']:30s} {b['path']}")
print(f"saved -> {OUT}")
