import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline.horizontal import (
    HorizontalAgg,
    HorizontalArgMax,
    HorizontalArgMin,
)


def test_horizontal_agg_basic() -> None:
    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})

    # Sum of a and b for each row
    # HorizontalAgg expects exprs (columns to select) and aggs (aggregation on the unpivoted value)
    # The unpivoted value name defaults to "horizontal_agg"
    step = HorizontalAgg(
        "a",
        "b",
        aggs=[pl.col("horizontal_agg").sum().alias("sum")],
        maintain_order=True,
    )
    transformed = step.transform(df.lazy()).collect()

    # Original columns + result of aggregation
    expected = df.with_columns((pl.col("a") + pl.col("b")).alias("sum"))
    assert_frame_equal(transformed, expected)


def test_horizontal_arg_max() -> None:
    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8], "c": [0, 10, 5]})

    # HorizontalArgMax should return the column name that has the max value
    step = HorizontalArgMax("a", "b", "c", value_name="best_col")
    transformed = step.transform(df.lazy()).collect()

    # For row 0: max(1, 4, 0) is 4 (b)
    # For row 1: max(5, 2, 10) is 10 (c)
    # For row 2: max(2, 8, 5) is 8 (b)
    assert transformed["best_col"].list.get(0).to_list() == ["b", "c", "b"]


def test_horizontal_arg_min() -> None:
    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})

    step = HorizontalArgMin(["a", "b"], value_name="worst_col")
    transformed = step.transform(df.lazy()).collect()

    # Row 0: min(1, 4) is 1 (a)
    # Row 1: min(5, 2) is 2 (b)
    # Row 2: min(2, 8) is 2 (a)
    assert transformed["worst_col"].list.get(0).to_list() == ["a", "b", "a"]


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
    assert transformed["best_col"].list.get(0).to_list() == ["b", "c", "b"]


def test_lazy_pipeline_horizontal_argmin() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"a": [1, 5, 2], "b": [4, 2, 8]})
    pipe = LazyPipeline().horizontal_argmin(["a", "b"], value_name="worst_col")
    transformed = pipe.fit_transform(df)
    assert transformed["worst_col"].list.get(0).to_list() == ["a", "b", "a"]
