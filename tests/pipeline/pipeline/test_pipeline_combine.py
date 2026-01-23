import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.pipeline import Pipeline


def test_combine_basic():
    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    pipeline = Pipeline().combine(["a", "b", "c"], 2)

    result = pipeline.fit_transform(df)

    expected = df.with_columns(
        pl.struct(["a", "b"]).alias("a_b_comb"),
        pl.struct(["a", "c"]).alias("a_c_comb"),
        pl.struct(["b", "c"]).alias("b_c_comb"),
    )

    assert_frame_equal(result, expected)


def test_combine_custom_args():
    df = DataFrame({"col1": [1], "col2": [2], "col3": [3]})

    pipeline = Pipeline().combine(
        ["col1", "col2", "col3"], 2, delimiter="-", suffix="_res"
    )

    result = pipeline.fit_transform(df)

    expected = df.with_columns(
        pl.struct(["col1", "col2"]).alias("col1-col2_res"),
        pl.struct(["col1", "col3"]).alias("col1-col3_res"),
        pl.struct(["col2", "col3"]).alias("col2-col3_res"),
    )

    assert_frame_equal(result, expected)


def test_combine_not_fitted():
    df = DataFrame({"a": [1], "b": [2]})
    pipeline = Pipeline().combine(["a", "b"], 2)

    with pytest.raises(NotFittedError):
        pipeline.transform(df)


def test_combine_transform_only():
    df_fit = DataFrame({"a": [1], "b": [2]})
    df_transform = DataFrame({"a": [10], "b": [20], "c": [30]})

    pipeline = Pipeline().combine(["a", "b"], 2)
    pipeline.fit(df_fit)

    result = pipeline.transform(df_transform)

    expected = df_transform.with_columns(
        pl.struct(["a", "b"]).alias("a_b_comb"),
    )

    assert_frame_equal(result, expected)
