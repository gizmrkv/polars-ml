import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_linear_regression_simple(test_data_simple_regression: DataFrame):
    assert_component_valid(
        Pipeline()
        .linear.regression(pl.exclude("y"), "y", model_kwargs={"fit_intercept": True})
        .select(
            (pl.col("y") - pl.col("linear_regression"))
            .abs()
            .mean()
            .round(1)
            .alias("mae")
        ),
        test_data_simple_regression,
        DataFrame({"mae": 0.0}),
    )
