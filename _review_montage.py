#!/usr/bin/env python3
# 그룹 리뷰 몬타주: 각 그룹 행 = [composite(평균) | rep1 | rep2 | rep3], 라벨(group_id, size, 검증 majority) 표기.
import sys, csv
from pathlib import Path
from PIL import Image, ImageDraw

D = Path(sys.argv[1])              # deliverable dir
OUT = sys.argv[2]
CELL = 190; PAD = 6; LABELW = 150
comp_dir = D / "composites"; rep_dir = D / "representatives"

# offline majority (검증용 라벨) 읽기
maj = {}
with (D / "offline_eval.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f):
        maj[int(r["group_id"])] = (r["majority_label"], r["purity"])

comps = sorted(comp_dir.glob("group_*.png"), key=lambda p: int(p.stem.split("_")[1]))
rows = []
for cp in comps:
    gid = int(cp.stem.split("_")[1])
    repg = sorted(rep_dir.glob(f"group_{gid:03d}_*"))
    reps = sorted(repg[0].glob("*.png"))[:3] if repg else []
    rows.append((gid, cp, reps))

W = LABELW + 4 * (CELL + PAD) + PAD
H = PAD + len(rows) * (CELL + PAD)
canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
for i, (gid, cp, reps) in enumerate(rows):
    y = PAD + i * (CELL + PAD)
    m, pur = maj.get(gid, ("?", "?"))
    draw.text((6, y + 10), f"g{gid}", fill=(0, 0, 0))
    draw.text((6, y + 30), f"{m}", fill=(180, 0, 0))
    draw.text((6, y + 50), f"pur {pur}", fill=(80, 80, 80))
    imgs = [cp] + reps
    for j, ip in enumerate(imgs):
        try:
            im = Image.open(ip).convert("RGB").resize((CELL, CELL))
        except Exception:
            continue
        x = LABELW + j * (CELL + PAD)
        canvas.paste(im, (x, y))
        tag = "AVG" if j == 0 else f"rep{j}"
        draw.rectangle([x, y, x + CELL, y + 14], fill=(0, 0, 0))
        draw.text((x + 2, y + 2), tag, fill=(255, 255, 255))
canvas.save(OUT)
print(f"[OUT] {OUT}  ({len(rows)} groups)")
