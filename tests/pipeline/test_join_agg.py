import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.join_agg import JoinAgg


def test_join_agg_basic() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    # Calculate mean of v per group g and join back
    step = JoinAgg("g", pl.col("v").mean().alias("v_mean"))
    step.fit(df)
    transformed = step.transform(df.lazy()).collect().sort("g", "v")

    expected = df.join(
        df.group_by("g").agg(pl.col("v").mean().alias("v_mean")), on="g", how="left"
    ).sort("g", "v")

    assert_frame_equal(transformed, expected)


def test_join_agg_not_fitted() -> None:
    step = JoinAgg("g", pl.col("v").mean())
    with pytest.raises(NotFittedError):
        step.transform(pl.LazyFrame({"g": ["a"], "v": [1]}))


def test_join_agg_multiple_aggs() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    step = JoinAgg("g", pl.col("v").min().alias("min"), pl.col("v").max().alias("max"))
    step.fit(df)
    transformed = step.transform(df.lazy()).collect().sort("g", "v")

    assert "min" in transformed.columns
    assert "max" in transformed.columns
    assert transformed.filter(pl.col("g") == "a")["min"].to_list() == [1, 1]
    assert transformed.filter(pl.col("g") == "a")["max"].to_list() == [2, 2]


def test_pipeline_join_agg() -> None:
    from polars_ml.pipeline import Pipeline

    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    pipe = Pipeline().join_agg("g", pl.col("v").mean().alias("v_mean"))
    transformed = pipe.fit_transform(df).sort("g", "v")
    expected = df.join(
        df.group_by("g").agg(pl.col("v").mean().alias("v_mean")), on="g", how="left"
    ).sort("g", "v")
    assert_frame_equal(transformed, expected)


def test_lazy_pipeline_join_agg() -> None:
    from polars_ml.pipeline import LazyPipeline

    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    pipe = LazyPipeline().join_agg("g", pl.col("v").mean().alias("v_mean"))
    transformed = pipe.fit_transform(df).sort("g", "v")
    expected = df.join(
        df.group_by("g").agg(pl.col("v").mean().alias("v_mean")), on="g", how="left"
    ).sort("g", "v")
    assert_frame_equal(transformed, expected)
