import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal, assert_series_equal

from polars_ml import Pipeline
from polars_ml.preprocessing import (
    MinMaxScale,
    RobustScale,
    ScaleInverse,
    ScaleInverseContext,
    StandardScale,
)


@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame({"val": [10.0, 20.0, 30.0], "group": ["A", "A", "B"]})


def test_standard_scale(sample_df: DataFrame) -> None:
    t = StandardScale(columns="val")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert_series_equal(output["val"], pl.Series("val", [-1.0, 0.0, 1.0]))


def test_min_max_scale(sample_df: DataFrame) -> None:
    t = MinMaxScale(columns="val")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert_series_equal(output["val"], pl.Series("val", [0.0, 0.5, 1.0]))


def test_robust_scale() -> None:
    df = DataFrame({"val": [1, 2, 3, 4, 5]})
    t = RobustScale(columns="val", quantile_range=(0.25, 0.75))
    t.fit(df)
    output = t.transform(df)
    assert_series_equal(output["val"], pl.Series("val", [-1.0, -0.5, 0.0, 0.5, 1.0]))


def test_scale_by_group() -> None:
    df = DataFrame({"val": [10.0, 20.0, 100.0, 200.0], "group": ["A", "A", "B", "B"]})
    t = StandardScale(columns="val", by="group")
    t.fit(df)
    output = t.transform(df)
    assert_series_equal(
        output.filter(pl.col("group") == "A")["val"],
        pl.Series("val", [-0.707107, 0.707107]),
        abs_tol=1e-5,
    )
    assert_series_equal(
        output.filter(pl.col("group") == "B")["val"],
        pl.Series("val", [-0.707107, 0.707107]),
        abs_tol=1e-5,
    )


def test_scale_inverse(sample_df: DataFrame) -> None:
    t = StandardScale(columns="val")
    t.fit(sample_df)
    scaled = t.transform(sample_df)

    inv = ScaleInverse(t)
    restored = inv.transform(scaled)
    assert_frame_equal(restored, sample_df)


def test_scale_context(sample_df: DataFrame) -> None:
    pipeline = Pipeline()
    t = StandardScale(columns="val")

    with ScaleInverseContext(pipeline, t):
        pass

    output = pipeline.fit_transform(sample_df)
    assert_frame_equal(output, sample_df)
