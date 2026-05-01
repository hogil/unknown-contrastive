"""512×512 cache 후보 6개 방식 시각 비교.

방식:
  1) NEAREST
  2) BILINEAR
  3) BICUBIC
  4) LANCZOS
  5) BOX (= AREA, 영역 평균)
  6) chip-aware (외곽 1px = NEAREST stride, 내부 14×14 BOX 평균)

출력: _resize_compare512/<class>/{method}.png + grid.png + zoom_grid.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = 6400
CHIP_GRID = 32
CHIP_PX = 200
OUT_SIZE = 512
CHIP_OUT = OUT_SIZE // CHIP_GRID                 # 16
VIEW_SIZE = 1024                                  # 512 → 1024 NEAREST upscale view


def chip_aware_resize(rgb: np.ndarray, out_size: int) -> np.ndarray:
    """외곽 1px = NEAREST stride, 내부 = BOX 평균."""
    assert out_size % CHIP_GRID == 0
    chip_out = out_size // CHIP_GRID
    inner_out = chip_out - 2
    out = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    stride = CHIP_PX / chip_out

    for gy in range(CHIP_GRID):
        for gx in range(CHIP_GRID):
            y0, x0 = gy * CHIP_PX, gx * CHIP_PX
            chip = rgb[y0:y0 + CHIP_PX, x0:x0 + CHIP_PX]
            oy0, ox0 = gy * chip_out, gx * chip_out
            ory, ocx = oy0 + chip_out, ox0 + chip_out

            idx = (np.arange(chip_out) * stride + stride / 2).astype(int).clip(0, CHIP_PX - 1)
            out[oy0,     ox0:ocx, :] = chip[1, idx, :]
            out[ory - 1, ox0:ocx, :] = chip[CHIP_PX - 2, idx, :]
            out[oy0:ory, ox0,     :] = chip[idx, 1, :]
            out[oy0:ory, ocx - 1, :] = chip[idx, CHIP_PX - 2, :]

            inner = chip[2:CHIP_PX - 2, 2:CHIP_PX - 2]
            inner_pil = Image.fromarray(inner)
            inner_small = inner_pil.resize((inner_out, inner_out), Image.BOX)
            out[oy0 + 1:ory - 1, ox0 + 1:ocx - 1, :] = np.array(inner_small)
    return out


def find_defect_zoom_anchor(rgb: np.ndarray, n_chip: int = 6):
    chips = rgb.reshape(CHIP_GRID, CHIP_PX, CHIP_GRID, CHIP_PX, 3).astype(np.float32)
    R = chips[..., 0]
    not_white_bg = (R < 200).astype(np.float32)
    score = not_white_bg.mean(axis=(1, 3))
    win = np.zeros((CHIP_GRID - n_chip + 1, CHIP_GRID - n_chip + 1), dtype=np.float64)
    for dy in range(n_chip):
        for dx in range(n_chip):
            win += score[dy:dy + CHIP_GRID - n_chip + 1, dx:dx + CHIP_GRID - n_chip + 1]
    gy, gx = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(gy), int(gx)


def label_image(img: Image.Image, text: str) -> Image.Image:
    canvas = Image.new('RGB', (img.width, img.height + 30), (240, 240, 240))
    canvas.paste(img, (0, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 5), text, fill=(0, 0, 0), font=font)
    return canvas


def make_grid(labeled, cols=3):
    rows = (len(labeled) + cols - 1) // cols
    w, h = labeled[0].width, labeled[0].height
    grid = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(labeled):
        r, c = i // cols, i % cols
        grid.paste(im, (c * w, r * h))
    return grid


def process_one(src_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert('RGB')
    rgb = np.asarray(img)
    print(f"[{src_path.name}] orig={img.size}")

    methods = {
        'NEAREST':    img.resize((OUT_SIZE, OUT_SIZE), Image.NEAREST),
        'BILINEAR':   img.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR),
        'BICUBIC':    img.resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC),
        'LANCZOS':    img.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS),
        'BOX (AREA)': img.resize((OUT_SIZE, OUT_SIZE), Image.BOX),
        'chip-aware': Image.fromarray(chip_aware_resize(rgb, OUT_SIZE)),
    }

    for name, im in methods.items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        im.save(out_dir / f"{safe}.png", optimize=True)

    # ===== Full wafer grid (512 → 1024 NEAREST upscale view) =====
    labeled = []
    for name, im in methods.items():
        view = im.resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)
        labeled.append(label_image(view, f"512 {name}"))
    grid = make_grid(labeled, cols=3)
    grid.save(out_dir / "grid.png", optimize=True)
    print(f"  saved grid: {out_dir/'grid.png'} ({grid.size})")

    # ===== Zoom 비교 (defect 영역 6×6 chip → 1024 view) =====
    n_chip_zoom = 6
    gy_a, gx_a = find_defect_zoom_anchor(rgb, n_chip=n_chip_zoom)
    print(f"  zoom anchor=({gy_a},{gx_a})")

    orig_crop = img.crop((gx_a * CHIP_PX, gy_a * CHIP_PX,
                          (gx_a + n_chip_zoom) * CHIP_PX, (gy_a + n_chip_zoom) * CHIP_PX))
    orig_view = orig_crop.resize((VIEW_SIZE, VIEW_SIZE), Image.BICUBIC)

    zoom_labeled = [label_image(orig_view, "ORIG (6400 BICUBIC view)")]
    for name, im in methods.items():
        x0 = gx_a * CHIP_OUT; y0 = gy_a * CHIP_OUT
        x1 = x0 + n_chip_zoom * CHIP_OUT; y1 = y0 + n_chip_zoom * CHIP_OUT
        cropped = im.crop((x0, y0, x1, y1))
        view = cropped.resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)
        zoom_labeled.append(label_image(view, f"512 {name}  (zoom 6×6 chip)"))
    zoom_grid = make_grid(zoom_labeled, cols=3)
    zoom_grid.save(out_dir / "zoom_grid.png", optimize=True)
    print(f"  saved zoom: {out_dir/'zoom_grid.png'} ({zoom_grid.size})")


def main():
    samples = [
        ('Donut_bank_boundary',     'aby313_00P_01_20260501_010000_94.6_4_EE_PWQ.png'),
        ('Edge-Ring_scratch_21deg', 'ahm609_00C_11_20260501_010000_89.8_8_PT_ENGINEER.png'),
        ('Center_particle_blast',   'acw913_00P_19_20260501_010000_96.0_2_PT_NORMAL.png'),
    ]
    base_in = Path('D:/project/data/wm-811k/unknown')
    base_out = Path('D:/project/unknown-contrastive/_resize_compare512')

    for cls, fname in samples:
        src = base_in / cls / fname
        if not src.exists():
            print(f"[skip] not found: {src}")
            continue
        process_one(src, base_out / cls)


if __name__ == '__main__':
    main()
