from typing import Any

import polars as pl
from polars import DataFrame

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def assert_pipeline_group_by_equal(
    data: DataFrame, method: str, *args: Any, **kwargs: Any
):
    assert_component_valid(
        getattr(Pipeline().group_by("s1", maintain_order=True), method)(
            *args, **kwargs
        ),
        data.clone(),
        getattr(data.clone().group_by("s1", maintain_order=True), method)(
            *args, **kwargs
        ),
    )


def test_pipeline_group_by_agg_small(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "agg", pl.sum("f1"))


def test_pipeline_group_by_all_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "all")


def test_pipeline_group_by_first_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "first")


def test_pipeline_group_by_head_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "head", 2)


def test_pipeline_group_by_last_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "last")


def test_pipeline_group_by_len_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "len", "len")


def test_pipeline_group_by_map_groups_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(
        test_data_small,
        "map_groups",
        lambda df: df.sample(2, seed=42, with_replacement=True),  # type: ignore
    )


def test_pipeline_group_by_max_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "max")


def test_pipeline_group_by_mean_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "mean")


def test_pipeline_group_by_median_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "median")


def test_pipeline_group_by_min_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "min")


def test_pipeline_group_by_n_unique_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "n_unique")


def test_pipeline_group_by_quantile_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "quantile", 0.5)


def test_pipeline_group_by_sum_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "sum")


def test_pipeline_group_by_tail_basic(test_data_small: DataFrame):
    assert_pipeline_group_by_equal(test_data_small, "tail", 2)
