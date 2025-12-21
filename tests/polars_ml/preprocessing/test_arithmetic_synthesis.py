import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_series_equal

from polars_ml.preprocessing import ArithmeticSynthesis


@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})


def test_arithmetic_synthesis_additive(sample_df: DataFrame):
    t = ArithmeticSynthesis(
        columns=["a", "b", "c"], order=1, method="additive", show_progress=False
    )
    t.fit(sample_df)
    output = t.transform(sample_df)

    assert "a+b" in output.columns
    assert "a-b" in output.columns
    assert_series_equal(output["a+b"], pl.Series("a+b", [5.0, 7.0, 9.0]))
    assert_series_equal(output["a-b"], pl.Series("a-b", [-3.0, -3.0, -3.0]))

    output = t.fit_transform(sample_df)
    assert "a+b" in output.columns
    assert "a-b" in output.columns
    assert_series_equal(output["a+b"], pl.Series("a+b", [5.0, 7.0, 9.0]))
    assert_series_equal(output["a-b"], pl.Series("a-b", [-3.0, -3.0, -3.0]))


def test_arithmetic_synthesis_multiplicative(sample_df: DataFrame):
    t = ArithmeticSynthesis(
        columns=["a", "b"], order=1, method="multiplicative", show_progress=False
    )
    t.fit(sample_df)
    output = t.transform(sample_df)

    assert "a*b" in output.columns
    assert "a/b" in output.columns
    assert_series_equal(output["a*b"], pl.Series("a*b", [4.0, 10.0, 18.0]))
    assert_series_equal(output["a/b"], pl.Series("a/b", [0.25, 0.4, 0.5]))

    output = t.fit_transform(sample_df)
    assert "a*b" in output.columns
    assert "a/b" in output.columns
    assert_series_equal(output["a*b"], pl.Series("a*b", [4.0, 10.0, 18.0]))
    assert_series_equal(output["a/b"], pl.Series("a/b", [0.25, 0.4, 0.5]))


def test_arithmetic_synthesis_drop_corr(sample_df: DataFrame):
    df = DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.1, 2.1, 3.1]})
    t = ArithmeticSynthesis(
        columns=["a", "b"],
        order=1,
        drop_high_correlation_features_method="pearson",
        threshold=0.9,
        show_progress=False,
    )
    t.fit(df)
    output = t.transform(df)
    assert len(output.columns) > 2

    output = t.fit_transform(df)
    assert len(output.columns) > 2
