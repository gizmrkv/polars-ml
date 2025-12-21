import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.pipeline import Pipeline
from polars_ml.pipeline.getattr import GetAttr


@pytest.fixture
def sample_df() -> DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


class TestPipeline:
    def test_pipeline_methods(self, sample_df: DataFrame):
        step1 = GetAttr("with_columns", c=pl.col("a") + pl.col("b"))
        step2 = GetAttr("select", "c")
        pipeline = Pipeline().pipe(step1).pipe(step2)

        pipeline.fit(sample_df)
        output = pipeline.transform(sample_df)
        expected = sample_df.with_columns(c=pl.col("a") + pl.col("b")).select("c")
        assert_frame_equal(output, expected)

        output = pipeline.fit_transform(sample_df)
        assert_frame_equal(output, expected)

    def test_pipeline_shortcut_methods(self, sample_df: DataFrame):
        pipeline = Pipeline().with_columns(c=pl.col("a") + pl.col("b")).select("c")

        pipeline.fit(sample_df)
        output = pipeline.transform(sample_df)
        expected = sample_df.with_columns(c=pl.col("a") + pl.col("b")).select("c")
        assert_frame_equal(output, expected)

        output = pipeline.fit_transform(sample_df)
        assert_frame_equal(output, expected)
