import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.pipeline import Pipeline


def test_join_agg_basic():
    df = DataFrame({"group": ["a", "a", "b", "b"], "val": [1, 3, 5, 7]})

    pipeline = Pipeline().join_agg("group", pl.col("val").mean(), suffix="_mean_agg")

    result = pipeline.fit_transform(df)

    expected = df.with_columns(val_mean_agg=pl.Series([2.0, 2.0, 6.0, 6.0]))

    assert_frame_equal(result, expected)


def test_join_agg_multiple_aggs():
    df = DataFrame({"group": ["a", "a"], "val": [1, 3]})

    pipeline = Pipeline().join_agg(
        "group",
        pl.col("val").min().alias("min_val"),
        pl.col("val").max().alias("max_val"),
        suffix="",
    )

    result = pipeline.fit_transform(df)

    expected = df.with_columns(min_val=pl.Series([1, 1]), max_val=pl.Series([3, 3]))

    assert_frame_equal(result, expected)


def test_join_agg_custom_args():
    df = DataFrame(
        {"id": [1, 2, 3], "category": ["x", "x", "y"], "value": [10, 20, 30]}
    )

    pipeline = Pipeline().join_agg("category", pl.col("value").sum(), suffix="_custom")

    result = pipeline.fit_transform(df)

    expected = df.with_columns(value_custom=pl.Series([30, 30, 30]))

    assert_frame_equal(result, expected)


def test_join_agg_not_fitted():
    df = DataFrame({"group": ["a"], "val": [1]})
    pipeline = Pipeline().join_agg("group", pl.col("val").mean())

    with pytest.raises(NotFittedError):
        pipeline.transform(df)


def test_join_agg_transform_separate():
    df_fit = DataFrame({"group": ["a", "b"], "val": [10, 20]})

    df_transform = DataFrame(
        {
            "group": ["a", "a", "b", "c"],
            "val": [1, 2, 3, 4],
        }
    )

    pipeline = Pipeline().join_agg(
        "group", pl.col("val").mean(), suffix="_agg", how="left"
    )

    pipeline.fit(df_fit)
    result = pipeline.transform(df_transform)

    expected = df_transform.with_columns(val_agg=pl.Series([10.0, 10.0, 20.0, None]))

    assert_frame_equal(result, expected)
