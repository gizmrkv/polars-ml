import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from polars_ml.pipeline.horizontal import (
    HorizontalAgg,
    HorizontalArgMax,
    HorizontalArgMin,
)


def test_horizontal_agg_basic() -> None:
    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})

    step = HorizontalAgg(
        "a",
        "b",
        aggs=[pl.col("horizontal_agg").sum().alias("sum")],
        maintain_order=True,
    )
    transformed = step.transform(df.lazy()).collect()

    expected = df.with_columns((pl.col("a") + pl.col("b")).alias("sum"))
    assert_frame_equal(transformed, expected)


def test_horizontal_arg_max() -> None:
    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8], "c": [0, 10, 5]})

    step = HorizontalArgMax("a", "b", "c", value_name="best_col")
    transformed = step.transform(df.lazy()).collect()

    assert_series_equal(
        transformed["best_col"].list.get(0), pl.Series("best_col", ["b", "c", "b"])
    )


def test_horizontal_arg_min() -> None:
    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})

    step = HorizontalArgMin(["a", "b"], value_name="worst_col")
    transformed = step.transform(df.lazy()).collect()

    assert_series_equal(
        transformed["worst_col"].list.get(0), pl.Series("worst_col", ["a", "b", "a"])
    )


def test_pipeline_horizontal_agg() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})
    pipe = Pipeline().horizontal_agg(
        ["a", "b"], aggs=[pl.col("horizontal_agg").sum().alias("sum")]
    )
    transformed = pipe.fit_transform(df)
    expected = df.with_columns((pl.col("a") + pl.col("b")).alias("sum"))
    assert_frame_equal(transformed, expected)


def test_pipeline_horizontal_argmax() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8], "c": [0, 10, 5]})
    pipe = Pipeline().horizontal_argmax(["a", "b", "c"], value_name="best_col")
    transformed = pipe.fit_transform(df)
    assert_series_equal(
        transformed["best_col"].list.get(0), pl.Series("best_col", ["b", "c", "b"])
    )


def test_lazy_pipeline_horizontal_argmin() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})
    pipe = LazyPipeline().horizontal_argmin(["a", "b"], value_name="worst_col")
    transformed = pipe.fit_transform(df)
    assert_series_equal(
        transformed["worst_col"].list.get(0), pl.Series("worst_col", ["a", "b", "a"])
    )
