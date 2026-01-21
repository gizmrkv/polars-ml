import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.pipeline import Pipeline


def test_discretize_basic():
    df = DataFrame({"a": [1, 2, 3, 4, 10]})

    pipeline = Pipeline().discretize("a", quantiles=2)

    result = pipeline.fit_transform(df)

    assert "a" in result.columns
    assert "a_disc" in result.columns

    assert result["a_disc"].dtype == pl.Categorical

    assert result["a_disc"].n_unique() <= 2 + 1


def test_discretize_custom_labels():
    df = DataFrame({"a": [1, 10, 100]})

    pipeline = Pipeline().discretize("a", quantiles=2, labels=["low", "high"])

    result = pipeline.fit_transform(df)

    assert "a_disc" in result.columns
    assert result["a_disc"].dtype == pl.Categorical

    vals = result["a_disc"].unique().to_list()
    for v in vals:
        assert v in ["low", "high"]


def test_discretize_not_fitted():
    df = DataFrame({"a": [1, 2]})
    pipeline = Pipeline().discretize("a", quantiles=2)

    with pytest.raises(NotFittedError):
        pipeline.transform(df)


def test_discretize_persistence():
    df_fit = DataFrame({"a": range(100)})
    df_transform = DataFrame({"a": [-100, 50, 200]})

    pipeline = Pipeline().discretize("a", quantiles=4)
    pipeline.fit(df_fit)

    result = pipeline.transform(df_transform)

    assert "a_disc" in result.columns
