import unittest

from scripts.fcmae_fixed_protocol import evaluate_three_seed_gate


def row(
    clusterer,
    seed,
    p1,
    p2=0.0,
    p3=0.8,
    p4=0.9,
    recov=0.85,
    frag=1.5,
    ari=0.7,
    recipe="L0_base_ep4",
):
    return {
        "recipe": recipe,
        "clusterer": clusterer,
        "seed": seed,
        "P1_capture_count": p1,
        "P1_target_class_count": 32,
        "P2_noise_pct": p2,
        "P3_completeness": p3,
        "P4_homogeneity": p4,
        "recov": recov,
        "fragment_ratio": frag,
        "Sil": 0.4,
        "ARI_supporting": ari,
        "AMI_supporting": 0.75,
    }


class FixedProtocolGateTest(unittest.TestCase):
    def frozen(self):
        return [
            row(
                "FINCH-p2", "none", 32, p3=0.80, p4=0.90, recov=0.85, frag=1.8,
                recipe="frozen",
            ),
            row(
                "Louvain-res6", "none", 31, p3=0.82, p4=0.91, recov=0.86, frag=1.7,
                recipe="frozen",
            ),
        ]

    def passing_candidates(self):
        rows = []
        for seed in (1, 3, 5):
            rows.append(row("FINCH-p2", seed, 32, p3=0.82, p4=0.92, recov=0.845, frag=1.6))
            rows.append(row("Louvain-res6", seed, 31, p3=0.84, p4=0.93, recov=0.855, frag=1.5))
        return rows

    def test_balanced_three_seed_candidate_passes(self):
        result = evaluate_three_seed_gate(self.passing_candidates(), self.frozen())
        self.assertTrue(result["accepted"])

    def test_high_ari_cannot_rescue_p1_regression(self):
        candidates = self.passing_candidates()
        candidates[0]["P1_capture_count"] = 31
        candidates[0]["ARI_supporting"] = 1.0
        result = evaluate_three_seed_gate(candidates, self.frozen())
        self.assertFalse(result["accepted"])
        self.assertFalse(result["clusterers"]["FINCH-p2"]["checks"]["P1_all_seeds"])

    def test_single_seed_secondary_gain_is_not_enough(self):
        candidates = self.passing_candidates()
        for candidate in candidates:
            if candidate["clusterer"] == "Louvain-res6" and candidate["seed"] in (3, 5):
                candidate["P3_completeness"] = 0.70
        result = evaluate_three_seed_gate(candidates, self.frozen())
        self.assertFalse(result["accepted"])
        self.assertFalse(
            result["clusterers"]["Louvain-res6"]["checks"]["P3_direction_2of3"]
        )


if __name__ == "__main__":
    unittest.main()
