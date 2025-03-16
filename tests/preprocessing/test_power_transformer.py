import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_power_transform_boxcox_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline()
        .with_columns(pl.col("f0").abs() + 1)
        .power_transform("f0", method="boxcox"),
        test_data_medium,
    )


def test_pipeline_power_transform_yeojohnson_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline().power_transform("f0", method="yeojohnson"),
        test_data_medium,
    )
