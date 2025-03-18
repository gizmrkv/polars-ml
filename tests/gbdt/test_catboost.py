import tempfile

import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_catboost_simple_binary_classification(
    test_data_simple_binary_classification: DataFrame,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        assert_component_valid(
            Pipeline()
            .gbdt.catboost(
                pl.exclude("y"),
                "y",
                {"objective": "Logloss", "iterations": 10, "train_dir": tmpdir},
            )
            .select(
                (pl.col("y") == pl.col("catboost").gt(0.5)).mean().round(1).alias("acc")
            ),
            test_data_simple_binary_classification,
            DataFrame({"acc": 1.0}),
        )
