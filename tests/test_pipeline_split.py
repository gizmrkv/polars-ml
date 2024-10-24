import unittest

import polars as pl
from polars import DataFrame, Series

from polars_ml import Pipeline
from polars_ml.testing import assert_component


class TestPipelineSplit(unittest.TestCase):
    def setUp(self):
        self.input_data = DataFrame({"label": [1] * 60 + [2] * 30 + [3] * 10})

    def test_train_valid_split(self):
        assert_component(
            Pipeline()
            .split.train_valid_split(test_size=0.2)
            .group_by("is_valid")
            .len()
            .sort("is_valid"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series("is_valid", [False, True]),
                    Series("len", [80, 20], dtype=pl.UInt32),
                ]
            ),
        )

    def test_k_fold(self):
        assert_component(
            Pipeline()
            .split.k_fold(5, shuffle=False)
            .group_by("fold")
            .len()
            .sort("fold"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series("fold", [0, 1, 2, 3, 4], dtype=pl.UInt32),
                    Series("len", [20, 20, 20, 20, 20], dtype=pl.UInt32),
                ]
            ),
        )

    def test_stratified_k_fold(self):
        assert_component(
            Pipeline()
            .split.stratified_k_fold(5, stratify="label", shuffle=False)
            .group_by("fold", "label")
            .len()
            .sort("fold", "label"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=DataFrame(
                [
                    Series(
                        "fold",
                        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                        dtype=pl.UInt32,
                    ),
                    Series(
                        "label",
                        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                        dtype=pl.Int64,
                    ),
                    Series(
                        "len",
                        [12, 6, 2, 12, 6, 2, 12, 6, 2, 12, 6, 2, 12, 6, 2],
                        dtype=pl.UInt32,
                    ),
                ]
            ),
        )


test = TestPipelineSplit()
test.setUp()
test.test_stratified_k_fold()
