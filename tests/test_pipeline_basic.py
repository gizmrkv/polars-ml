import unittest

import polars as pl

from polars_ml import Pipeline
from polars_ml.generate import generate_all_dtypes
from polars_ml.testing import assert_component


class TestPipelineBase(unittest.TestCase):
    def setUp(self):
        self.input_data = generate_all_dtypes(100, null_rate=0.1, seed=42)

    def test_cast(self):
        assert_component(
            Pipeline().cast({pl.Int32: pl.Float32}),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.cast({pl.Int32: pl.Float32}),
        )

    def test_clear(self):
        assert_component(
            Pipeline().clear(10),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.clear(10),
        )

    def test_clone(self):
        assert_component(
            Pipeline().clone(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.clone(),
        )

    def test_drop(self):
        assert_component(
            Pipeline().drop("Float32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.drop("Float32"),
        )

    def test_drop_nulls(self):
        assert_component(
            Pipeline().drop_nulls(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.drop_nulls(),
        )

    def test_explode(self):
        assert_component(
            Pipeline().explode("List"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.explode("List"),
        )

    def test_fill_nan(self):
        assert_component(
            Pipeline().fill_nan(42),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.fill_nan(42),
        )

    def test_fill_null(self):
        assert_component(
            Pipeline().fill_null(42),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.fill_null(42),
        )

    def test_filter(self):
        assert_component(
            Pipeline().filter(pl.col("Int32") > 5),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.filter(pl.col("Int32") > 5),
        )

    def test_gather_every(self):
        assert_component(
            Pipeline().gather_every(2),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.gather_every(2),
        )

    def test_slice(self):
        assert_component(
            Pipeline().slice(10, 10),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.slice(10, 10),
        )

    def test_head(self):
        assert_component(
            Pipeline().head(10),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.head(10),
        )

    def test_limit(self):
        assert_component(
            Pipeline().limit(10),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.limit(10),
        )

    def test_tail(self):
        assert_component(
            Pipeline().tail(10),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.tail(10),
        )

    def test_interpolate(self):
        assert_component(
            Pipeline().interpolate(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.interpolate(),
        )

    def test_sort(self):
        assert_component(
            Pipeline().sort("Int32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.sort("Int32"),
        )

    def test_set_sorted(self):
        assert_component(
            Pipeline().set_sorted("Int32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.set_sorted("Int32"),
        )

    def test_rename(self):
        assert_component(
            Pipeline().rename({"Int32": "Int32_renamed"}),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.rename({"Int32": "Int32_renamed"}),
        )

    def test_reverse(self):
        assert_component(
            Pipeline().reverse(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.reverse(),
        )

    def test_select(self):
        assert_component(
            Pipeline().select("Int32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.select("Int32"),
        )

    def test_select_seq(self):
        assert_component(
            Pipeline().select_seq("Int32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.select_seq("Int32"),
        )

    def test_shift(self):
        input_data = self.input_data.drop("Array")
        assert_component(
            Pipeline().shift(1),
            fit_data=input_data,
            input_data=input_data,
            expected_data=input_data.shift(1),
        )

    def test_sql(self):
        assert_component(
            Pipeline().sql("SELECT * FROM self"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data,
        )

    def test_unique(self):
        assert_component(
            Pipeline().unique("Int32", maintain_order=True),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.unique("Int32", maintain_order=True),
        )

    def test_unnest(self):
        assert_component(
            Pipeline().unnest("Struct"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.unnest("Struct"),
        )

    def test_unpivot(self):
        assert_component(
            Pipeline().unpivot(on="Int32"),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.unpivot(on="Int32"),
        )

    def test_with_columns(self):
        assert_component(
            Pipeline().with_columns(pl.col("Int32") + 1),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.with_columns(pl.col("Int32") + 1),
        )

    def test_with_columns_seq(self):
        assert_component(
            Pipeline().with_columns_seq(pl.col("Int32") + 1),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.with_columns_seq(pl.col("Int32") + 1),
        )

    def test_with_row_index(self):
        assert_component(
            Pipeline().with_row_index(),
            fit_data=self.input_data,
            input_data=self.input_data,
            expected_data=self.input_data.with_row_index(),
        )
