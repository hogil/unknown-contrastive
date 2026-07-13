import unittest

from scripts.cluster_metrics import capture_metrics
from scripts.cluster_scoring import tier1


class CanonicalCaptureMetricTest(unittest.TestCase):
    def test_background_majority_cluster_does_not_capture_defect_minority(self):
        result = capture_metrics(
            pred=[0, 0, 0, 0, 1, 1],
            labels=["Normal", "Normal", "Normal", "A", "B", "B"],
            excluded={"Normal"},
        )

        self.assertEqual(result["capture_count"], 1)
        self.assertEqual(result["target_class_count"], 2)
        self.assertEqual(result["captured_classes"], ["B", "Normal"])
        self.assertEqual(result["legacy_presence_count"], 2)
        self.assertLessEqual(result["capture_count"], result["dominant_cluster_count"])
        self.assertLessEqual(result["dominant_cluster_count"], result["cluster_count"])

    def test_tied_cluster_has_no_dominant_capture(self):
        result = capture_metrics(
            pred=[0, 0, 1, 1],
            labels=["A", "B", "C", "C"],
        )

        self.assertEqual(result["capture_count"], 1)
        self.assertEqual(result["target_class_count"], 3)
        self.assertEqual(result["captured_classes"], ["C"])
        self.assertEqual(result["dominant_cluster_count"], 1)
        self.assertEqual(result["cluster_count"], 2)

    def test_tier1_retains_background_when_scoring_target_classes(self):
        result = tier1(
            pred=[0, 0, 0, 0, 1, 1],
            true_idx=[-1, -1, -1, 0, 1, 1],
            labs=["Normal", "Normal", "Normal", "A", "B", "B"],
            classes=["A", "B"],
            excluded={"Normal"},
        )

        self.assertEqual(result["capture_count"], 1)
        self.assertEqual(result["target_class_count"], 2)
        self.assertEqual(result["noise_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
