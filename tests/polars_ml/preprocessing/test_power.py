import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_series_equal

from polars_ml import Pipeline
from polars_ml.preprocessing import (
    BoxCoxTransform,
    PowerTransformInverse,
    PowerTransformInverseContext,
    YeoJohnsonTransform,
)


@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})


def test_box_cox_transform(sample_df: DataFrame):
    t = BoxCoxTransform(columns="val")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert "val" in output.columns
    assert output["val"].n_unique() == 5


def test_yeo_johnson_transform(sample_df: DataFrame):
    t = YeoJohnsonTransform(columns="val")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert "val" in output.columns
    assert output["val"].n_unique() == 5


def test_power_inverse(sample_df: DataFrame):
    t = BoxCoxTransform(columns="val")

    t.fit(sample_df)
    transformed = t.transform(sample_df)

    inv = PowerTransformInverse(t)
    restored = inv.transform(transformed)
    assert_series_equal(restored["val"], sample_df["val"])


def test_power_context(sample_df: DataFrame):
    pipeline = Pipeline()
    t = BoxCoxTransform(columns="val")

    with PowerTransformInverseContext(pipeline, t):
        pass

    pipeline.fit(sample_df)
    output = pipeline.transform(sample_df)
    assert_series_equal(output["val"], sample_df["val"])


def test_power_by_group():
    df = DataFrame(
        {"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "group": ["A", "A", "A", "B", "B", "B"]}
    )
    t = YeoJohnsonTransform(columns="val", by="group")
    t.fit(df)
    output = t.transform(df)

    assert "val" in output.columns
    assert output.filter(pl.col("group") == "A")["val"].n_unique() == 3
    assert output.filter(pl.col("group") == "B")["val"].n_unique() == 3
