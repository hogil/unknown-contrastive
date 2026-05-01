"""6400x6400 wafer fail-bit PNG를 6 방식으로 384x384로 줄여 시각 비교.

방식:
  1) NEAREST
  2) BILINEAR  (현재 cnn_train.py 학습이 사용)
  3) BICUBIC
  4) LANCZOS
  5) BOX        (= AREA, 영역 평균)
  6) chip-aware (chip 외곽 1px = NEAREST stride sample / chip 내부 = BOX 평균)

출력: _resize_compare/<sample_basename>/{method}.png + grid.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_SIZE = 384
CANVAS = 6400
CHIP_GRID = 32
CHIP_PX = 200                                    # 6400/32
CHIP_OUT = OUT_SIZE // CHIP_GRID                 # 12 (384/32)
INNER_BLOCK = (CHIP_PX - 2 * 2) // (CHIP_OUT - 2)  # 196/10 = 19.6 -> 19
# 정확히 196 = 10*19 + 6 으로 떨어지지 않음 → numpy reshape용으로 196=14*14 사용 못 함
# 단순화: 내부 196x196 → 10x10 평균 (각 출력 픽셀이 19.6x19.6 입력 평균, AREA 보간)


def chip_aware_resize(rgb: np.ndarray) -> np.ndarray:
    """chip 외곽 보존 + 내부 영역평균. rgb: (6400,6400,3) uint8 -> (384,384,3) uint8."""
    out = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.uint8)
    # chip별 처리
    for gy in range(CHIP_GRID):
        for gx in range(CHIP_GRID):
            y0, x0 = gy * CHIP_PX, gx * CHIP_PX
            chip = rgb[y0:y0 + CHIP_PX, x0:x0 + CHIP_PX]   # (200, 200, 3)
            oy0, ox0 = gy * CHIP_OUT, gx * CHIP_OUT
            ocy = ox0 + CHIP_OUT
            ory = oy0 + CHIP_OUT

            # 가장자리 1 px = chip 외곽 2 px stride sample (border 색 보존)
            # 출력 12x12 가장자리 4면을 위해 입력에서 stride 200/12 ≈ 16.67 sampling
            # border는 chip의 외곽 1-2 px에 있으니 input row[1] / col[1] 사용 (1px border 대응)
            stride = CHIP_PX / CHIP_OUT
            # column index list for sampling (12 points across 200)
            idx = (np.arange(CHIP_OUT) * stride + stride / 2).astype(int).clip(0, CHIP_PX - 1)

            # top edge: 입력 row 1 (border 1px 안쪽)
            out[oy0, ox0:ocy, :] = chip[1, idx, :]
            # bottom: row 198
            out[ory - 1, ox0:ocy, :] = chip[CHIP_PX - 2, idx, :]
            # left: col 1
            out[oy0:ory, ox0, :] = chip[idx, 1, :]
            # right: col 198
            out[oy0:ory, ocy - 1, :] = chip[idx, CHIP_PX - 2, :]

            # 내부 10x10 = 입력 [2:198, 2:198] → 196x196 영역 평균
            inner = chip[2:CHIP_PX - 2, 2:CHIP_PX - 2].astype(np.float32)  # (196,196,3)
            # 196 = 10 * 19.6 → reshape 안 됨. 일단 196x196 → cv2 없이 PIL BOX
            inner_pil = Image.fromarray(inner.astype(np.uint8), mode='RGB')
            inner_small = inner_pil.resize((CHIP_OUT - 2, CHIP_OUT - 2), Image.BOX)
            out[oy0 + 1:ory - 1, ox0 + 1:ocy - 1, :] = np.array(inner_small)
    return out


def label_image(img: Image.Image, text: str) -> Image.Image:
    """이미지 위에 라벨 텍스트 띄우기."""
    canvas = Image.new('RGB', (img.width, img.height + 24), (240, 240, 240))
    canvas.paste(img, (0, 24))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(0, 0, 0), font=font)
    return canvas


def make_grid(labeled: list[Image.Image], cols: int = 3) -> Image.Image:
    """라벨 붙은 이미지들을 grid로 합치기."""
    rows = (len(labeled) + cols - 1) // cols
    w, h = labeled[0].width, labeled[0].height
    grid = Image.new('RGB', (cols * w, rows * h), (255, 255, 255))
    for i, im in enumerate(labeled):
        r, c = i // cols, i % cols
        grid.paste(im, (c * w, r * h))
    return grid


def make_zoom(rgb: np.ndarray, gy: int, gx: int, n_chip: int = 6) -> Image.Image:
    """원본 6400 이미지에서 chip (gy, gx) 부근 n_chip×n_chip 영역 crop."""
    y0, x0 = gy * CHIP_PX, gx * CHIP_PX
    y1, x1 = min(CANVAS, y0 + n_chip * CHIP_PX), min(CANVAS, x0 + n_chip * CHIP_PX)
    return Image.fromarray(rgb[y0:y1, x0:x1])


def find_defect_zoom_anchor(rgb: np.ndarray, n_chip: int = 6):
    """defect/invalid 영역 자동 검출. 각 chip별 'non-white-or-bg-ness' 측정.
    bg는 (220,238,255), white는 (255,255,255). 둘 다 R≥220 + B≥220 → exclude하면 결함만 남음.
    """
    chips = rgb.reshape(CHIP_GRID, CHIP_PX, CHIP_GRID, CHIP_PX, 3).astype(np.float32)
    # 픽셀 단위로 white(>=240,240,240) 또는 bg-light-blue(R<240 & B>=250) 둘 다 'background' 처리
    # 단순화: not_bg = R < 240 (white·bg 모두 R≥220-255이지만 bg는 R≈220, white=255)
    # 결함/border 색은 R<200 인 경우가 많음
    R = chips[..., 0]
    not_white_bg = (R < 200).astype(np.float32)
    # chip별 not_white_bg 비율
    score = not_white_bg.mean(axis=(1, 3))                                                # (32,32)
    # n_chip × n_chip 윈도우 sum
    win = np.zeros((CHIP_GRID - n_chip + 1, CHIP_GRID - n_chip + 1), dtype=np.float64)
    for dy in range(n_chip):
        for dx in range(n_chip):
            win += score[dy:dy + CHIP_GRID - n_chip + 1,
                         dx:dx + CHIP_GRID - n_chip + 1]
    gy, gx = np.unravel_index(int(np.argmax(win)), win.shape)
    return int(gy), int(gx)


def process_one(src_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert('RGB')
    rgb = np.asarray(img)
    print(f"[{src_path.name}] orig size={img.size} mode=RGB")

    # 6 가지 방식
    methods = {
        'NEAREST': img.resize((OUT_SIZE, OUT_SIZE), Image.NEAREST),
        'BILINEAR': img.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR),
        'BICUBIC': img.resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC),
        'LANCZOS': img.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS),
        'BOX (AREA)': img.resize((OUT_SIZE, OUT_SIZE), Image.BOX),
        'chip-aware': Image.fromarray(chip_aware_resize(rgb)),
    }

    # 개별 저장
    for name, im in methods.items():
        safe = name.replace(' ', '_').replace('(', '').replace(')', '')
        im.save(out_dir / f"{safe}.png", optimize=True)

    # grid (4× upscale로 차이 보이게)
    upscale = 2
    labeled = []
    for name, im in methods.items():
        big = im.resize((OUT_SIZE * upscale, OUT_SIZE * upscale), Image.NEAREST)
        labeled.append(label_image(big, f"{name}  ({OUT_SIZE}->upscaled x{upscale} for view)"))
    grid = make_grid(labeled, cols=3)
    grid.save(out_dir / "grid.png", optimize=True)
    print(f"  saved grid: {out_dir/'grid.png'}  ({grid.size})")

    # zoom 비교 (chip border + defect 모양 자세히 보기, defect 영역 자동 검출)
    n_chip_zoom = 6
    gy_anchor, gx_anchor = find_defect_zoom_anchor(rgb, n_chip=n_chip_zoom)
    print(f"  zoom anchor chip=({gy_anchor},{gx_anchor})")
    zoom_orig = make_zoom(rgb, gy_anchor, gx_anchor, n_chip=n_chip_zoom)  # 1200x1200
    zoom_orig_small = zoom_orig.resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)
    zoom_methods = {}
    for name, im in methods.items():
        x0 = int(gx_anchor * CHIP_OUT); y0 = int(gy_anchor * CHIP_OUT)
        x1 = x0 + n_chip_zoom * CHIP_OUT; y1 = y0 + n_chip_zoom * CHIP_OUT
        zoom_methods[name] = im.crop((x0, y0, x1, y1)).resize((OUT_SIZE, OUT_SIZE), Image.NEAREST)

    zoom_labeled = [label_image(zoom_orig_small, "ORIG (6400, BICUBIC view)")]
    for name, im in zoom_methods.items():
        zoom_labeled.append(label_image(im, f"zoom: {name}"))
    zoom_grid = make_grid(zoom_labeled, cols=3)
    zoom_grid.save(out_dir / "zoom_grid.png", optimize=True)
    print(f"  saved zoom grid: {out_dir/'zoom_grid.png'}  ({zoom_grid.size})")


def main():
    samples = [
        # (class, file basename) — defect 패턴 + chip border 둘 다 보여줄 sample
        ('Donut_bank_boundary',     'aby313_00P_01_20260501_010000_94.6_4_EE_PWQ.png'),
        ('Edge-Ring_scratch_21deg', 'ahm609_00C_11_20260501_010000_89.8_8_PT_ENGINEER.png'),
        ('Center_particle_blast',   'acw913_00P_19_20260501_010000_96.0_2_PT_NORMAL.png'),
    ]
    base_in = Path('D:/project/data/wm-811k/unknown')
    base_out = Path('D:/project/unknown-contrastive/_resize_compare')

    for cls, fname in samples:
        src = base_in / cls / fname
        if not src.exists():
            print(f"[skip] not found: {src}")
            continue
        out_dir = base_out / cls
        process_one(src, out_dir)


if __name__ == '__main__':
    main()
