import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_lightgbm_simple_binary_classification(
    test_data_simple_binary_classification: DataFrame,
):
    assert_component_valid(
        Pipeline()
        .gbdt.lightgbm(
            pl.exclude("y"), "y", {"objective": "binary", "num_iterations": 10}
        )
        .select(
            (pl.col("y") == pl.col("lightgbm").gt(0.5)).mean().round(1).alias("acc")
        ),
        test_data_simple_binary_classification,
        DataFrame({"acc": 1.0}),
    )
