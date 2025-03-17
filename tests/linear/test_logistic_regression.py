import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_logistic_regression_simple(
    test_data_simple_binary_classification: DataFrame,
):
    assert_component_valid(
        Pipeline()
        .linear.logistic_regression(pl.exclude("y"), "y")
        .select(
            (pl.col("y") == pl.col("logistic_regression")).mean().round(2).alias("acc")
        ),
        test_data_simple_binary_classification,
        DataFrame({"acc": 1.0}),
    )
