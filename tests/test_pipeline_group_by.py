import unittest

import polars as pl

from polars_ml import Pipeline
from polars_ml.generate import generate_all_dtypes
from polars_ml.testing import assert_component


class TestPipelineGroupBy(unittest.TestCase):
    def setUp(self):
        self.input_data = generate_all_dtypes(100, null_rate=0.1, seed=42)

    def test_group_by_agg(self):
        assert_component(
            Pipeline()
            .group_by("Int32", maintain_order=True)
            .agg((pl.col("Float32") * pl.col("Float32")).mean()),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).agg(
                (pl.col("Float32") * pl.col("Float32")).mean()
            ),
        )

    def test_group_by_all(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).all(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).all(),
        )

    def test_group_by_first(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).first(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by(
                "Int32", maintain_order=True
            ).first(),
        )

    def test_group_by_last(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).last(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).last(),
        )

    def test_group_by_len(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).len(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).len(),
        )

    def test_group_by_head(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).head(2),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).head(
                2
            ),
        )

    def test_group_by_tail(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).tail(2),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).tail(
                2
            ),
        )

    def test_group_by_quantile(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).quantile(0.2),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by(
                "Int32", maintain_order=True
            ).quantile(0.2),
        )

    def test_group_by_max(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).max(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).max(),
        )

    def test_group_by_median(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).median(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by(
                "Int32", maintain_order=True
            ).median(),
        )

    def test_group_by_min(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).min(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).min(),
        )

    def test_group_by_sum(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).sum(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).sum(),
        )

    def test_group_by_mean(self):
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).mean(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.group_by("Int32", maintain_order=True).mean(),
        )

    def test_group_by_n_unique(self):
        input_data = self.input_data.drop("Array")
        assert_component(
            Pipeline().group_by("Int32", maintain_order=True).n_unique(),
            fit_data=input_data,
            input_data=input_data,
            expected_data=input_data.group_by("Int32", maintain_order=True).n_unique(),
        )
