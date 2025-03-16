import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_scale_standard_medium(test_data_medium: DataFrame):
    test_data_medium = test_data_medium.with_columns(pl.col("f0") * 2 + 3)
    assert_component_valid(
        Pipeline()
        .scale("f0", method="standard")
        .select(pl.col("f0").mean().alias("mean"), pl.col("f0").std().alias("std")),
        test_data_medium,
        DataFrame({"mean": 0.0, "std": 1.0}),
    )


def test_pipeline_scale_min_max_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline()
        .scale("f0", method="min-max")
        .select(pl.col("f0").min().alias("min"), pl.col("f0").max().alias("max")),
        test_data_medium,
        DataFrame({"min": 0.0, "max": 1.0}),
    )


def test_pipeline_scale_robust_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline()
        .scale("f0", method="robust")
        .select(
            pl.col("f0").median().alias("median"),
            (pl.col("f0").quantile(0.75) - pl.col("f0").quantile(0.25)).alias("iqr"),
        ),
        test_data_medium,
        DataFrame({"median": 0.0, "iqr": 1.0}),
    )


def test_pipeline_scale_standard_by_medium(test_data_medium: DataFrame):
    test_data_medium = test_data_medium.with_columns(
        pl.when(pl.col("s0") == pl.lit("a"))
        .then(pl.col("f0") * 2 + 3)
        .when(pl.col("s0") == pl.lit("b"))
        .then(pl.col("f0") * 0.5 + 10)
        .when(pl.col("s0") == pl.lit("c"))
        .then(pl.col("f0") * -3 - 5)
    )
    assert_component_valid(
        Pipeline()
        .scale("f0", method="standard", by="s0")
        .group_by("s0")
        .agg(pl.col("f0").mean().alias("mean"), pl.col("f0").std().alias("std"))
        .sort("s0"),
        test_data_medium,
        DataFrame({"s0": ["a", "b", "c"], "mean": [0.0] * 3, "std": [1.0] * 3}),
    )
