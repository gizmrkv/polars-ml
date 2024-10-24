import unittest

import polars as pl
import polars.selectors as cs

from polars_ml import Pipeline
from polars_ml.generate import generate_all_dtypes
from polars_ml.testing import assert_component


class TestPipelineBase(unittest.TestCase):
    def setUp(self):
        self.input_data = generate_all_dtypes(100, null_rate=0.1, seed=42).select(
            cs.numeric()
        )

    def test_stat_bottom_k(self):
        assert_component(
            Pipeline().stat.bottom_k(5, by="Float32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.bottom_k(5, by="Float32"),
        )

    def test_stat_top_k(self):
        assert_component(
            Pipeline().stat.top_k(5, by="Float32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.top_k(5, by="Float32"),
        )

    def test_stat_count(self):
        assert_component(
            Pipeline().stat.count(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.count(),
        )

    def test_stat_null_count(self):
        assert_component(
            Pipeline().stat.null_count(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.null_count(),
        )

    def test_stat_n_unique(self):
        assert_component(
            Pipeline().stat.n_unique(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.select(pl.all().n_unique()),
        )

    def test_stat_quantile(self):
        assert_component(
            Pipeline().stat.quantile(0.5),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.quantile(0.5),
        )

    def test_stat_max(self):
        assert_component(
            Pipeline().stat.max(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.max(),
        )

    def test_stat_min(self):
        assert_component(
            Pipeline().stat.min(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.min(),
        )

    def test_stat_median(self):
        assert_component(
            Pipeline().stat.median(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.median(),
        )

    def test_stat_sum(self):
        assert_component(
            Pipeline().stat.sum(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.sum(),
        )

    def test_stat_mean(self):
        assert_component(
            Pipeline().stat.mean(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.mean(),
        )

    def test_stat_std(self):
        assert_component(
            Pipeline().stat.std(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.std(),
        )

    def test_stat_var(self):
        assert_component(
            Pipeline().stat.var(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.var(),
        )
