import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_series_equal

from polars_ml.preprocessing import Discretize


@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


def test_discretize_quantiles(sample_df: DataFrame):
    t = Discretize(exprs="a", quantiles=2, suffix="_bin")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert "a_bin" in output.columns
    assert output["a_bin"].n_unique() == 2

    output = t.fit_transform(sample_df)
    assert "a_bin" in output.columns
    assert output["a_bin"].n_unique() == 2


def test_discretize_labels(sample_df: DataFrame):
    t = Discretize(exprs="a", quantiles=2, labels=["low", "high"], suffix="_label")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert "a_label" in output.columns
    assert_series_equal(
        output["a_label"],
        pl.Series("a_label", ["low"] * 5 + ["high"] * 5, dtype=pl.Categorical),
    )

    output = t.fit_transform(sample_df)
    assert "a_label" in output.columns
    assert_series_equal(
        output["a_label"],
        pl.Series("a_label", ["low"] * 5 + ["high"] * 5, dtype=pl.Categorical),
    )


def test_discretize_fit_transform(sample_df: DataFrame):
    t = Discretize(exprs="a", quantiles=2, labels=["L", "H"])

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert "a_discretized" in output.columns
    assert_series_equal(
        output["a_discretized"],
        pl.Series("a_discretized", ["L"] * 5 + ["H"] * 5, dtype=pl.Categorical),
    )

    output = t.fit_transform(sample_df)
    assert "a_discretized" in output.columns
    assert_series_equal(
        output["a_discretized"],
        pl.Series("a_discretized", ["L"] * 5 + ["H"] * 5, dtype=pl.Categorical),
    )
