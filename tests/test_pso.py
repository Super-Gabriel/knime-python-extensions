import unittest

import numpy as np

from src.PSO import PSOFeatureSelection


class TestPSOFeatureSelection(unittest.TestCase):
    def test_decode_position_selects_under_maximum(self):
        pso = PSOFeatureSelection(max_features=3, threshold=0.5)

        mask = pso._decode_position(np.array([0.9, 0.4, 0.8, 0.1, 0.7]))

        np.testing.assert_array_equal(mask, [1, 0, 1, 0, 1])

    def test_decode_position_caps_selected_features_by_position(self):
        pso = PSOFeatureSelection(max_features=2, threshold=0.5)

        mask = pso._decode_position(np.array([0.6, 0.9, 0.8, 0.1]))

        np.testing.assert_array_equal(mask, [0, 1, 1, 0])

    def test_decode_position_selects_at_least_one_feature(self):
        pso = PSOFeatureSelection(max_features=3, threshold=0.5)

        mask = pso._decode_position(np.array([0.2, 0.4, 0.1]))

        np.testing.assert_array_equal(mask, [0, 1, 0])

    def test_zero_maximum_does_not_cap_features(self):
        pso = PSOFeatureSelection(max_features=0, threshold=0.5)

        mask = pso._decode_position(np.array([0.8, 0.7, 0.2, 0.9]))

        np.testing.assert_array_equal(mask, [1, 1, 0, 1])

    def test_preferred_candidate_uses_fewer_features_within_one_standard_error(self):
        pso = PSOFeatureSelection()
        pso._record_candidate(
            np.array([0.8, 0.8, 0.1]), np.array([0.88, 0.90, 0.92])
        )
        pso._record_candidate(
            np.array([0.8, 0.1, 0.1]), np.array([0.89, 0.89, 0.89])
        )

        mask, score = pso._get_preferred_candidate()

        np.testing.assert_array_equal(mask, [1, 0, 0])
        self.assertAlmostEqual(score, 0.89)

    def test_preferred_candidate_does_not_trade_performance_beyond_one_standard_error(self):
        pso = PSOFeatureSelection()
        pso._record_candidate(
            np.array([0.8, 0.8, 0.1]), np.array([0.899, 0.9, 0.901])
        )
        pso._record_candidate(
            np.array([0.8, 0.1, 0.1]), np.array([0.88, 0.88, 0.88])
        )

        mask, score = pso._get_preferred_candidate()

        np.testing.assert_array_equal(mask, [1, 1, 0])
        self.assertAlmostEqual(score, 0.9)

    def test_equal_performance_prefers_fewer_features(self):
        pso = PSOFeatureSelection()
        pso._record_candidate(
            np.array([0.8, 0.8, 0.1]), np.array([0.9, 0.9, 0.9])
        )
        pso._record_candidate(
            np.array([0.8, 0.1, 0.1]), np.array([0.9, 0.9, 0.9])
        )

        mask, score = pso._get_preferred_candidate()

        np.testing.assert_array_equal(mask, [1, 0, 0])
        self.assertAlmostEqual(score, 0.9)


if __name__ == "__main__":
    unittest.main()
