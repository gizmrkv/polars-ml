import polars as pl
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.pipeline.pipeline import Pipeline


def test_pipeline_getattr_head():
    df = DataFrame({"a": [1, 2, 3, 4, 5]})
    pipeline = Pipeline().head(2)

    result = pipeline.transform(df)
    expected = DataFrame({"a": [1, 2]})
    assert_frame_equal(result, expected)

    result_fit = pipeline.fit_transform(df)
    assert_frame_equal(result_fit, expected)


def test_pipeline_getattr_drop():
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    pipeline = Pipeline().drop("b")

    result = pipeline.transform(df)
    expected = DataFrame({"a": [1, 2]})
    assert_frame_equal(result, expected)

    result_fit = pipeline.fit_transform(df)
    assert_frame_equal(result_fit, expected)


def test_pipeline_getattr_join():
    df1 = DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = DataFrame({"a": [1, 2], "c": [5, 6]})
    pipeline = Pipeline().join(df2, on="a")

    result = pipeline.transform(df1)
    expected = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    assert_frame_equal(result, expected)

    result_fit = pipeline.fit_transform(df1)
    assert_frame_equal(result_fit, expected)


def test_pipeline_groupby_getattr_agg():
    df = DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    pipeline = Pipeline().group_by("g").agg(pl.col("v").sum())

    result = pipeline.transform(df).sort("g")
    expected = DataFrame({"g": ["a", "b"], "v": [3, 3]}).sort("g")
    assert_frame_equal(result, expected)

    result_fit = pipeline.fit_transform(df).sort("g")
    assert_frame_equal(result_fit, expected)


def test_pipeline_groupby_getattr_sum():
    df = DataFrame({"g": ["a", "a", "b"], "v1": [1, 2, 3], "v2": [4, 5, 6]})
    pipeline = Pipeline().group_by("g").sum()

    result = pipeline.transform(df).sort("g")
    expected = DataFrame({"g": ["a", "b"], "v1": [3, 3], "v2": [9, 6]}).sort("g")
    assert_frame_equal(result, expected)

    result_fit = pipeline.fit_transform(df).sort("g")
    assert_frame_equal(result_fit, expected)
