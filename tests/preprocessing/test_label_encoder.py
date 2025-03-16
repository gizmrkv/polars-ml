import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_label_encode_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline().label_encode("s2", maintain_order=True),
        test_data_medium,
        test_data_medium.with_columns(
            pl.when(pl.col("s2") == pl.lit("a"))
            .then(0)
            .when(pl.col("s2") == pl.lit("b"))
            .then(1)
            .when(pl.col("s2") == pl.lit("c"))
            .then(2)
            .otherwise(-1)
            .cast(pl.UInt32)
            .alias("s2"),
        ),
    )


def test_pipeline_label_encode_order_medium(test_data_medium: DataFrame):
    assert_component_valid(
        Pipeline().label_encode("s0", orders={"s0": ["a", "b", "c"]}),
        test_data_medium,
        test_data_medium.with_columns(
            pl.when(pl.col("s0") == pl.lit("a"))
            .then(0)
            .when(pl.col("s0") == pl.lit("b"))
            .then(1)
            .when(pl.col("s0") == pl.lit("c"))
            .then(2)
            .otherwise(-1)
            .cast(pl.UInt32)
            .alias("s0"),
        ),
    )


def test_pipeline_label_encode_inverse_medium(test_data_medium: DataFrame):
    pp = Pipeline()
    with pp.label_encode("s2", maintain_order=True, inverse_mapping={"s2": "s2"}):
        pass
    assert_component_valid(pp, test_data_medium, test_data_medium)
