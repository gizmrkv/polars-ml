import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.pipeline.getattr import GetAttr, GetAttrPolars


@pytest.fixture
def sample_df() -> DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_getattr_fit_transform(sample_df: DataFrame):
    t = GetAttr("select", pl.col("a") * 2)
    t.fit(sample_df)

    output = t.transform(sample_df)
    expected = sample_df.select(pl.col("a") * 2)
    assert_frame_equal(output, expected)


def test_getattr_polars_fit_transform(sample_df: DataFrame):
    t = GetAttrPolars("concat", [sample_df, sample_df])
    t.fit(sample_df)

    output = t.transform(sample_df)
    expected = pl.concat([sample_df, sample_df])
    assert_frame_equal(output, expected)
