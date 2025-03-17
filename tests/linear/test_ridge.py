import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_ridge_simple(test_data_simple_regression: DataFrame):
    assert_component_valid(
        Pipeline()
        .linear.ridge(
            pl.exclude("y"), "y", model_kwargs={"alpha": 1e-4, "fit_intercept": True}
        )
        .select((pl.col("y") - pl.col("ridge")).abs().mean().round(1).alias("mae")),
        test_data_simple_regression,
        DataFrame({"mae": 0.0}),
    )
