import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.pipeline import Pipeline


@pytest.fixture
def sample_df() -> DataFrame:
    return pl.DataFrame({"group": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})


def test_group_by_agg(sample_df: DataFrame):
    pipeline = Pipeline().group_by("group").agg(pl.col("val").sum().alias("sum_val"))

    pipeline.fit(sample_df)
    output = pipeline.transform(sample_df).sort("group")
    expected = (
        sample_df.group_by("group")
        .agg(pl.col("val").sum().alias("sum_val"))
        .sort("group")
    )
    assert_frame_equal(output, expected)

    output = pipeline.fit_transform(sample_df).sort("group")
    assert_frame_equal(output, expected)
