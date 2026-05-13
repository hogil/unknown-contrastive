"""Convert MixedWM38 npz to ImageFolder format with random stratified train/val/test split.

Output structure:
  D:/dataset/MixedWM38_folder/
    train/{composite_label}/{idx}.png   (~70%)
    val/{composite_label}/{idx}.png     (~15%)
    test/{composite_label}/{idx}.png    (~15%)

Each image: 384x384 RGB PNG (52 -> 384 BICUBIC resize, 0/1/2/3 -> RGB palette).
Stratified split by composite multi-hot label (38 unique).
"""
import os, time
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

NPZ = 'D:/dataset/MixedWM38/Wafer_Map_Datasets.npz'
OUT_ROOT = Path('D:/dataset/MixedWM38_folder')
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remainder

print(f'Loading {NPZ} ...', flush=True)
d = np.load(NPZ)
maps = d['arr_0']
labels = d['arr_1']
N = len(maps)
print(f'  N={N}', flush=True)

# Composite label string
composite = np.array([''.join(str(int(b)) for b in row) for row in labels])
uniq, counts = np.unique(composite, return_counts=True)
print(f'  unique composite: {len(uniq)}', flush=True)
print(f'  min count: {counts.min()}  max count: {counts.max()}', flush=True)

# Stratified split: composite -> train + (val+test); val+test -> val + test
idx_all = np.arange(N)
idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx_all, composite, train_size=TRAIN_FRAC, random_state=SEED, stratify=composite
)
# val:test = 15:15 -> val_frac of temp = 0.5
val_in_temp_frac = VAL_FRAC / (1 - TRAIN_FRAC)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, train_size=val_in_temp_frac, random_state=SEED, stratify=y_temp
)
print(f'  train: {len(idx_train)}  val: {len(idx_val)}  test: {len(idx_test)}', flush=True)

# Palette: 0 blank -> dark, 1 pass -> mid grey, 2 fail -> red, 3 fail-tier2 -> orange
def to_rgb_384(wm):
    r = (wm == 2).astype(np.uint8) * 255 + (wm == 3).astype(np.uint8) * 200
    g = (wm == 1).astype(np.uint8) * 200 + (wm == 3).astype(np.uint8) * 60
    b = (wm == 0).astype(np.uint8) * 50
    rgb52 = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb52)
    return img.resize((384, 384), Image.BICUBIC)

splits = {'train': idx_train, 'val': idx_val, 'test': idx_test}
t0 = time.time()
for split_name, idxs in splits.items():
    print(f'\nWriting {split_name} ({len(idxs)}) ...', flush=True)
    n_done = 0
    for i in idxs:
        cls = composite[i]
        out_dir = OUT_ROOT / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f'{int(i):06d}.png'
        if png.exists():
            n_done += 1
            continue
        img = to_rgb_384(maps[i])
        img.save(png, optimize=False)
        n_done += 1
        if n_done % 2000 == 0:
            el = time.time() - t0
            print(f'  {n_done}/{len(idxs)}  elapsed={el:.0f}s', flush=True)
    print(f'  {split_name} done.  Per-class file counts:', flush=True)
    for c in sorted((OUT_ROOT / split_name).iterdir()):
        if c.is_dir():
            cnt = len(list(c.iterdir()))
            print(f'    {c.name}: {cnt}')

print(f'\n[OUT] {OUT_ROOT.resolve()}')
print(f'  train classes: {len(list((OUT_ROOT/"train").iterdir()))}', flush=True)
print(f'  val   classes: {len(list((OUT_ROOT/"val").iterdir()))}', flush=True)
print(f'  test  classes: {len(list((OUT_ROOT/"test").iterdir()))}', flush=True)
