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
    def _make_positioned_image(self, root: Path, name: str) -> Path:
        image_dir = root / "images" / "unknown" / "Demo"
        pos_dir = root / "positions" / "unknown" / "Demo"
        image_dir.mkdir(parents=True, exist_ok=True)
        pos_dir.mkdir(parents=True, exist_ok=True)

        arr = np.full((6, 6), 31, dtype=np.uint8)
        arr[1:5, 1:5] = 1
        arr[2:4, 2:4] = 7
        arr[1, 2:4] = 9
        img_path = image_dir / f"{name}.png"
        _save_palette_png(img_path, arr)

        positions = {
            "coord": {"canvas": {"width": 6, "height": 6}},
            "chips": [{"rect": {"x0": 1, "y0": 1, "x1": 5, "y1": 5}}],
        }
        (pos_dir / f"{name}.json").write_text(json.dumps(positions), encoding="utf-8")
        return img_path

    def test_linear2_weighted_average_uses_positions_and_masks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_dir = root / "out"
            img_path = self._make_positioned_image(root, "wafer")

            pg.REFERENCE_COMPOSITE_MAX_PX = 0
            rows = pg.save_reference_composites(ref_dir, "cluster_000", 1, [img_path])
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["filename"].endswith("_linear2_weighted_average.png"))
            self.assertEqual(rows[0]["type"], "weighted_linear2_mean")

            out = np.asarray(Image.open(rows[0]["path"]))
            self.assertEqual(out[0, 0], 8)       # positions 밖은 background
            self.assertEqual(out[1, 1], 10)      # chip border
            self.assertGreaterEqual(out[2, 2], 24)  # chip 내부 defect는 composite gradient

    def test_grouping_representatives_writes_composite_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [
                self._make_positioned_image(root, "wafer_a"),
                self._make_positioned_image(root, "wafer_b"),
                self._make_positioned_image(root, "wafer_noise"),
            ]
            embeddings = np.asarray([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]], dtype=np.float32)
            pred = np.asarray([0, 0, -1], dtype=int)
            out_dir = root / "grouping"

            pg.SAVE_REFERENCE_COMPOSITES = True
            pg.REFERENCE_COMPOSITE_MAX_PX = 0
            saved = pg.save_grouping_representatives(out_dir, embeddings, pred, [str(p) for p in paths], 2)

            composite_dir = out_dir / "representatives" / "composite"
            self.assertEqual(saved, 2)
            self.assertTrue(composite_dir.is_dir())
            self.assertTrue((composite_dir / "composite_maps.csv").exists())
            self.assertEqual(len(list(composite_dir.glob("*linear2_weighted_average.png"))), 1)
            self.assertFalse((out_dir / "representatives" / "reference").exists())


if __name__ == "__main__":
    unittest.main()
