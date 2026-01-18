import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal, assert_series_equal

from polars_ml import Pipeline
from polars_ml.preprocessing import (
    LabelEncode,
    LabelEncodeInverse,
    LabelEncodeInverseContext,
)


@pytest.fixture
def sample_df() -> DataFrame:
    return DataFrame({"cat": ["A", "B", "A", "C"], "val": [1, 2, 3, 4]})


def test_label_encode_basic(sample_df: DataFrame) -> None:
    t = LabelEncode(columns="cat")

    t.fit(sample_df)
    output = t.transform(sample_df)
    assert "cat" in output.columns
    assert output["cat"].dtype in [pl.UInt32, pl.Int64, pl.UInt64]
    assert output["cat"].n_unique() == 3


def test_label_encode_inverse(sample_df: DataFrame) -> None:
    t = LabelEncode(columns="cat")
    t.fit(sample_df)
    encoded = t.transform(sample_df)

    inv = LabelEncodeInverse(t)
    decoded = inv.transform(encoded)
    assert_series_equal(decoded["cat"], sample_df["cat"])


def test_label_encode_context(sample_df: DataFrame) -> None:
    pipeline = Pipeline()
    t = LabelEncode(columns="cat")

    with LabelEncodeInverseContext(pipeline, t):
        pass

    assert len(pipeline.steps) == 2

    pipeline.fit(sample_df)
    output = pipeline.transform(sample_df)
    assert_frame_equal(output, sample_df)


def test_label_encode_with_orders() -> None:
    df = DataFrame({"cat": ["A", "B", "C"]})
    t = LabelEncode(columns="cat", orders={"cat": ["C", "B", "A"]})
    t.fit(df)
    output = t.transform(df)
    assert_series_equal(output["cat"], pl.Series("cat", [2, 1, 0], dtype=pl.UInt32))
