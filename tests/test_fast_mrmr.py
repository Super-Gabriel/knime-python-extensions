import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import fast_mRMR


class TestFastMRMR(unittest.TestCase):
    @patch("src.fast_mRMR.mutual_info_regression")
    @patch("src.fast_mRMR.mutual_info_classif")
    def test_categorical_target_uses_classification_relevance_and_continuous_redundancy(
        self, mutual_info_classif, mutual_info_regression
    ):
        X = pd.DataFrame(
            {
                "first": [0.1, 0.3, 0.5, 0.7],
                "second": [0.8, 0.6, 0.4, 0.2],
            }
        )
        y = np.array(["a", "a", "b", "b"])
        mutual_info_classif.return_value = np.array([0.8, 0.4])
        mutual_info_regression.return_value = np.array([0.1])

        selected = fast_mRMR.fast_mrmr(
            X, y, num_features_wanted=2, target_type="Categorical"
        )

        self.assertEqual(selected, ["first", "second"])
        mutual_info_classif.assert_called_once()
        mutual_info_regression.assert_called_once()
        np.testing.assert_array_equal(
            mutual_info_regression.call_args.args[1], X["first"].to_numpy()
        )


if __name__ == "__main__":
    unittest.main()
