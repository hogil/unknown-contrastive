import unittest

import numpy as np
from PIL import Image

from scripts._common import mask_palette_non_grade_to_white


class PaletteMaskTest(unittest.TestCase):
    def test_non_grade_indices_become_white_before_rgb_convert(self):
        arr = np.asarray([[1, 8], [10, 31]], dtype=np.uint8)
        img = Image.frombytes("P", (2, 2), arr.tobytes())
        palette = []
        for i in range(256):
            palette.extend([i, i, i])
        palette[1 * 3:1 * 3 + 3] = [155, 155, 155]
        palette[8 * 3:8 * 3 + 3] = [220, 238, 255]
        palette[10 * 3:10 * 3 + 3] = [190, 190, 190]
        palette[31 * 3:31 * 3 + 3] = [0, 255, 0]
        img.putpalette(palette)

        rgb = np.asarray(mask_palette_non_grade_to_white(img).convert("RGB"))

        self.assertEqual(tuple(rgb[0, 0]), (155, 155, 155))
        self.assertEqual(tuple(rgb[0, 1]), (255, 255, 255))
        self.assertEqual(tuple(rgb[1, 0]), (255, 255, 255))
        self.assertEqual(tuple(rgb[1, 1]), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
