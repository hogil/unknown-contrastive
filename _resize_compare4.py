"""4가지 방식 비교: 384/768 × BOX/chip-aware.
한 sample을 4가지로 줄여 동일 view size(768)로 표시한 grid PNG 생성.

방식:
  - 384 BOX        : PIL BOX(=AREA) 영역평균, chip 12×12
  - 384 chip-aware : 외곽 1px = NEAREST stride, 내부 10×10 BOX 평균
  - 768 BOX        : PIL BOX, chip 24×24
  - 768 chip-aware : 외곽 1px = NEAREST stride, 내부 22×22 BOX 평균
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = 6400
CHIP_GRID = 32
CHIP_PX = 200
SIZES = (384, 768)
VIEW_SIZE = 768                                  # 모든 방식을 768 NEAREST upscale로 동일 view


def chip_aware_resize(rgb: np.ndarray, out_size: int) -> np.ndarray:
    """chip 외곽 1px (NEAREST stride) + 내부 BOX 영역평균.
    out_size: 384 -> chip 12, 768 -> chip 24. 32의 배수여야 함.
    """
    assert out_size % CHIP_GRID == 0, f"out_size {out_size} not divisible by {CHIP_GRID}"
    chip_out = out_size // CHIP_GRID
    inner_out = chip_out - 2                                                              # 내부 픽셀 수
    out = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    stride = CHIP_PX / chip_out

    for gy in range(CHIP_GRID):
        for gx in range(CHIP_GRID):
            y0, x0 = gy * CHIP_PX, gx * CHIP_PX
            chip = rgb[y0:y0 + CHIP_PX, x0:x0 + CHIP_PX]
            oy0, ox0 = gy * chip_out, gx * chip_out
            ory, ocx = oy0 + chip_out, ox0 + chip_out

            # 가장자리 1 px = chip 외곽 1-2px stride sample (border 색 보존)
            idx = (np.arange(chip_out) * stride + stride / 2).astype(int).clip(0, CHIP_PX - 1)
            out[oy0,       ox0:ocx, :] = chip[1, idx, :]                                  # top
            out[ory - 1,   ox0:ocx, :] = chip[CHIP_PX - 2, idx, :]                        # bottom
            out[oy0:ory,   ox0,     :] = chip[idx, 1, :]                                  # left
            out[oy0:ory,   ocx - 1, :] = chip[idx, CHIP_PX - 2, :]                        # right

            # 내부 inner_out × inner_out = 입력 [2:198, 2:198] BOX 평균
            inner = chip[2:CHIP_PX - 2, 2:CHIP_PX - 2]                                    # (196,196,3)
            inner_pil = Image.fromarray(inner)
            inner_small = inner_pil.resize((inner_out, inner_out), Image.BOX)
            out[oy0 + 1:ory - 1, ox0 + 1:ocx - 1, :] = np.array(inner_small)
    return out


def find_defect_zoom_anchor(rgb: np.ndarray, n_chip: int = 6):
    """defect 영역 자동 검출."""
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
    canvas = Image.new('RGB', (img.width, img.height + 28), (240, 240, 240))
    canvas.paste(img, (0, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 5), text, fill=(0, 0, 0), font=font)
    return canvas


def make_grid(labeled, cols=2):
    rows = (len(labeled) + cols - 1) // cols
    w, h = labeled[0].width, labeled[0].height
    grid = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(labeled):
        r, c = i // cols, i % cols
        grid.paste(im, (c * w, r * h))
    return grid


def upscale_to_view(im: Image.Image) -> Image.Image:
    """모든 방식을 동일 view 사이즈(VIEW_SIZE)로 NEAREST upscale."""
    if im.size == (VIEW_SIZE, VIEW_SIZE):
        return im
    return im.resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)


def process_one(src_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert('RGB')
    rgb = np.asarray(img)
    print(f"[{src_path.name}] orig={img.size}")

    # 4 방식 생성
    methods = {}
    for sz in SIZES:
        # BOX (PIL 영역평균)
        box_img = img.resize((sz, sz), Image.BOX)
        methods[f"{sz}_BOX"] = box_img
        # chip-aware
        ca_arr = chip_aware_resize(rgb, sz)
        methods[f"{sz}_chip_aware"] = Image.fromarray(ca_arr)

    # 개별 저장 (raw size 그대로)
    for name, im in methods.items():
        im.save(out_dir / f"{name}.png", optimize=True)

    # ===== 1) Full wafer 비교 grid (모두 768 view로 통일) =====
    labeled = []
    for sz in SIZES:
        for kind in ('BOX', 'chip_aware'):
            key = f"{sz}_{kind}"
            view = upscale_to_view(methods[key])
            labeled.append(label_image(view, f"cache {sz}  {kind}  (view 768 NEAREST upscale)"))
    grid = make_grid(labeled, cols=2)
    grid.save(out_dir / "grid4.png", optimize=True)
    print(f"  saved grid: {out_dir/'grid4.png'} ({grid.size})")

    # ===== 2) Zoom 비교 (defect 영역 6×6 chip crop) =====
    n_chip_zoom = 6
    gy_a, gx_a = find_defect_zoom_anchor(rgb, n_chip=n_chip_zoom)
    print(f"  zoom anchor chip=({gy_a},{gx_a})")

    # ORIG zoom (6400→768 view)
    orig_crop = img.crop((gx_a * CHIP_PX, gy_a * CHIP_PX,
                          (gx_a + n_chip_zoom) * CHIP_PX, (gy_a + n_chip_zoom) * CHIP_PX))
    orig_view = orig_crop.resize((VIEW_SIZE, VIEW_SIZE), Image.BICUBIC)

    zoom_labeled = [label_image(orig_view, "ORIG (6400 BICUBIC view)")]
    for sz in SIZES:
        for kind in ('BOX', 'chip_aware'):
            key = f"{sz}_{kind}"
            chip_size_out = sz // CHIP_GRID
            x0 = gx_a * chip_size_out; y0 = gy_a * chip_size_out
            x1 = x0 + n_chip_zoom * chip_size_out; y1 = y0 + n_chip_zoom * chip_size_out
            cropped = methods[key].crop((x0, y0, x1, y1))
            view = cropped.resize((VIEW_SIZE, VIEW_SIZE), Image.NEAREST)
            zoom_labeled.append(label_image(view, f"cache {sz}  {kind}  (zoom 6×6 chip)"))
    zoom_grid = make_grid(zoom_labeled, cols=3)
    zoom_grid.save(out_dir / "grid4_zoom.png", optimize=True)
    print(f"  saved zoom grid: {out_dir/'grid4_zoom.png'} ({zoom_grid.size})")


def main():
    samples = [
        ('Donut_bank_boundary',     'aby313_00P_01_20260501_010000_94.6_4_EE_PWQ.png'),
        ('Edge-Ring_scratch_21deg', 'ahm609_00C_11_20260501_010000_89.8_8_PT_ENGINEER.png'),
        ('Center_particle_blast',   'acw913_00P_19_20260501_010000_96.0_2_PT_NORMAL.png'),
    ]
    base_in = Path('D:/project/data/wm-811k/unknown')
    base_out = Path('D:/project/unknown-contrastive/_resize_compare4')

    for cls, fname in samples:
        src = base_in / cls / fname
        if not src.exists():
            print(f"[skip] not found: {src}")
            continue
        process_one(src, base_out / cls)


if __name__ == '__main__':
    main()
