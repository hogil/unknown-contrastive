#!/usr/bin/env python
# HF 고해상도 이미지 데이터셋 다운로드 → ImageFolder (class subdir) 로 저장.
# 사용자: "데이터셋 추가 = HF 고해상도 이미지". no-CNN SSL contrastive + P1~P4 clustering eval 용.
# 기본 = oxford_flowers102 (고해상도 fine-grained 102 class, SSL clustering 벤치 표준).
import os, sys, io, re
from datasets import load_dataset

def sanitize(s):
    # Windows 금지문자 <>:"/\|?* + 제어문자 제거, 공백→_, 끝 . 제거
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', str(s))
    s = s.strip().rstrip('.').replace(" ", "_")
    return s or "unknown"

REPO = sys.argv[1] if len(sys.argv) > 1 else "dpdl-benchmark/oxford_flowers102"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "E:/data/images/hf_flowers102"
PER  = int(sys.argv[3]) if len(sys.argv) > 3 else 40   # class 당 최대 (clustering pool)
os.makedirs(OUT, exist_ok=True)

print(f"[hf] load {REPO}", flush=True)
ds = load_dataset(REPO)
print(f"[hf] splits: {list(ds.keys())}", flush=True)
# 첫 split 사용 (train 우선)
split = "train" if "train" in ds else list(ds.keys())[0]
d = ds[split]
cols = d.column_names
print(f"[hf] split={split} n={len(d)} cols={cols}", flush=True)
img_col = "image" if "image" in cols else next((c for c in cols if "image" in c.lower() or "img" in c.lower()), cols[0])
lab_col = "label" if "label" in cols else next((c for c in cols if "label" in c.lower() or "class" in c.lower()), cols[-1])
print(f"[hf] img_col={img_col} lab_col={lab_col}", flush=True)

# label 이름 매핑
try:
    names = d.features[lab_col].names
except Exception:
    names = None

counts = {}
saved = 0
for i, ex in enumerate(d):
    lab = ex[lab_col]
    name = names[lab] if names and isinstance(lab, int) else str(lab)
    name = sanitize(name)
    if counts.get(name, 0) >= PER:
        continue
    img = ex[img_col]
    if isinstance(img, (bytes, bytearray)):
        from PIL import Image
        img = Image.open(io.BytesIO(img))
    cdir = os.path.join(OUT, name)
    os.makedirs(cdir, exist_ok=True)
    try:
        img.convert("RGB").save(os.path.join(cdir, f"{name}_{counts.get(name,0):03d}.jpg"), quality=95)
        counts[name] = counts.get(name, 0) + 1
        saved += 1
    except Exception as e:
        print(f"[hf] skip {i}: {e}", flush=True)
    if saved % 200 == 0:
        print(f"[hf] saved {saved} ({len(counts)} classes)", flush=True)

print(f"[hf] DONE saved={saved} classes={len(counts)} -> {OUT}", flush=True)
print(f"[OUT] {os.path.abspath(OUT)}", flush=True)
