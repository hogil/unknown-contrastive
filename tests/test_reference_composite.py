import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import predict_grouping_prod as pg


def _save_palette_png(path: Path, arr: np.ndarray) -> None:
    arr = arr.astype(np.uint8, copy=False)
    img = Image.frombytes("P", (arr.shape[1], arr.shape[0]), arr.tobytes())
    palette = []
    for i in range(256):
        palette.extend([i, i, i])
    palette[8 * 3:8 * 3 + 3] = [220, 238, 255]
    palette[10 * 3:10 * 3 + 3] = [190, 190, 190]
    img.putpalette(palette)
    img.save(path)


class ReferenceCompositeTest(unittest.TestCase):
    def test_square_weighted_average_uses_positions_and_masks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_dir = root / "images" / "unknown" / "Demo"
            pos_dir = root / "positions" / "unknown" / "Demo"
            ref_dir = root / "out"
            image_dir.mkdir(parents=True)
            pos_dir.mkdir(parents=True)

            arr = np.full((6, 6), 31, dtype=np.uint8)  # invalid outside chip
            arr[1:5, 1:5] = 1
            arr[2:4, 2:4] = 7
            arr[1, 2:4] = 9  # border/bottom index, excluded from grade count
            img_path = image_dir / "wafer.png"
            _save_palette_png(img_path, arr)

            positions = {
                "coord": {"canvas": {"width": 6, "height": 6}},
                "chips": [{"rect": {"x0": 1, "y0": 1, "x1": 5, "y1": 5}}],
            }
            (pos_dir / "wafer.json").write_text(json.dumps(positions), encoding="utf-8")

            pg.REFERENCE_COMPOSITE_MAX_PX = 0
            rows = pg.save_reference_composites(ref_dir, "cluster_000", 1, [img_path])
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["filename"].endswith("_square_weighted_average.png"))

            out = np.asarray(Image.open(rows[0]["path"]))
            self.assertEqual(out[0, 0], 8)       # positions 밖은 background
            self.assertEqual(out[1, 1], 10)      # chip border
            self.assertGreaterEqual(out[2, 2], 24)  # chip 내부 defect는 composite gradient


if __name__ == "__main__":
    unittest.main()
