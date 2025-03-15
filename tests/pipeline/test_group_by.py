from typing import Any

import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml import Pipeline


@pytest.fixture
def test_data_basic():
    return DataFrame(
        {
            "f0": ["a", "a", "b", "b", "c"],
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f2": [True, True, False, True, False],
        }
    )


def assert_frame_group_by_equal(
    data: DataFrame, method: str, *args: Any, **kwargs: Any
):
    pp: Pipeline = getattr(Pipeline().group_by("f0", maintain_order=True), method)(
        *args, **kwargs
    )
    out = pp.transform(data)
    exp = getattr(data.group_by("f0", maintain_order=True), method)(*args, **kwargs)
    assert_frame_equal(out, exp)


def test_pipeline_group_by_agg_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "agg", pl.sum("f1"))


def test_pipeline_group_by_all_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "all")


def test_pipeline_group_by_first_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "first")


def test_pipeline_group_by_head_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "head", 2)


def test_pipeline_group_by_last_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "last")


def test_pipeline_group_by_len_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "len", "len")


def test_pipeline_group_by_map_groups_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(
        test_data_basic,
        "map_groups",
        lambda df: df.sample(2, seed=42, with_replacement=True),  # type: ignore
    )


def test_pipeline_group_by_max_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "max")


def test_pipeline_group_by_mean_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "mean")


def test_pipeline_group_by_median_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "median")


def test_pipeline_group_by_min_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "min")


def test_pipeline_group_by_n_unique_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "n_unique")


def test_pipeline_group_by_quantile_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "quantile", 0.5)


def test_pipeline_group_by_sum_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "sum")


def test_pipeline_group_by_tail_basic(test_data_basic: DataFrame):
    assert_frame_group_by_equal(test_data_basic, "tail", 2)
