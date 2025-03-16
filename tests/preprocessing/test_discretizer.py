import polars as pl
from polars import DataFrame, Series

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_discretizer_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline()
        .discretize(pl.col("f0").abs(), quantiles=5, labels=list(map(str, range(5))))
        .group_by("f0_discretized")
        .len()
        .cast({"f0_discretized": pl.UInt32})
        .sort("f0_discretized"),
        test_data_medium,
        DataFrame(
            [
                Series("f0_discretized", list(range(5)), dtype=pl.UInt32),
                Series("len", [200] * 5, dtype=pl.UInt32),
            ]
        ),
    )
