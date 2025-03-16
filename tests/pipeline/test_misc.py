import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_echo_small(test_data_small: DataFrame):
    assert_component_valid(Pipeline().echo(), test_data_small, test_data_small)


def test_pipeline_print_small(test_data_small: DataFrame):
    assert_component_valid(Pipeline().print(), test_data_small, test_data_small)


def test_pipeline_display_small(test_data_small: DataFrame):
    assert_component_valid(Pipeline().display(), test_data_small, test_data_small)


def test_pipeline_sort_columns_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().sort_columns(),
        test_data_small,
        test_data_small.select(
            map(
                lambda x: x[0],
                sorted(test_data_small.schema.items(), key=lambda x: str(x[1])),
            )
        ),
    )
    assert_component_valid(
        Pipeline().sort_columns("name"),
        test_data_small,
        test_data_small.select(sorted(test_data_small.columns)),
    )


def test_pipeline_group_by_then_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().group_by_then("s0", pl.sum("f0").alias("new")).select("s0", "new"),
        test_data_small,
        DataFrame({"s0": ["a", "b", "c", "c"], "new": [-0.1, 0.0, 0.3, 0.3]}),
        DataFrame({"s0": ["b", "c", "a"]}),
        DataFrame({"s0": ["b", "c", "a"], "new": [0.0, 0.3, -0.1]}),
    )


def test_pipeline_impute_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().impute(
            Pipeline().select(pl.lit("x").alias("s2")),
            "s2",
            maintain_order=True,
        ),
        test_data_small.select("i0", "s2"),
        test_data_small.select("i0", "s2").fill_null("x"),
        DataFrame({"s2": ["a", None]}),
        DataFrame({"s2": ["a", "x"]}),
    )
