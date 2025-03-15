from typing import Any

import polars as pl
import polars.selectors as cs
import pytest
from polars import DataFrame, Series
from polars.testing import assert_frame_equal

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def assert_pipeline_horizontal_equal(
    data: DataFrame, expected: DataFrame, method: str, *args: Any, **kwargs: Any
):
    assert_component_valid(
        getattr(Pipeline().horizontal, method)(*args, **kwargs, maintain_order=True),
        data.clone(),
        expected.clone(),
    )


def test_pipeline_horizontal_agg_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.agg(
            cs.numeric(), aggs=[pl.all().pow(2).sum().sqrt()], maintain_order=True
        ),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_agg", [1.737814, 1.0, 4.806245, float("nan")])
        ),
    )


def test_pipeline_horizontal_all_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.all(cs.boolean(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_all", [True, False, False, False])
        ),
    )


def test_pipeline_horizontal_count_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.count(cs.all(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_count", [15, 10, 15, 15], dtype=pl.UInt32)
        ),
    )


def test_pipeline_horizontal_max_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.max(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(Series("horizontal_max", [1.0, 1.0, 3.0, 4.0])),
    )


def test_pipeline_horizontal_mean_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.mean(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_mean", [0.111111, 0.166666, 1.044444, float("nan")])
        ),
    )


def test_pipeline_horizontal_median_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.median(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(Series("horizontal_median", [0.0, 0.0, 0.3, 2.0])),
    )


def test_pipeline_horizontal_min_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.min(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(Series("horizontal_min", [-1.0, 0.0, 0.0, 0.0])),
    )


def test_pipeline_horizontal_n_unique_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.n_unique(cs.all(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_n_unique", [8, 7, 11, 10], dtype=pl.UInt32)
        ),
    )


def test_pipeline_horizontal_quantile_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.quantile(cs.numeric(), quantile=0.5, maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_quantile", [0.0, 0.0, 0.3, 2.0])
        ),
    )


def test_pipeline_horizontal_std_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.std(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_std", [0.603001, 0.408248, 1.288517, float("nan")])
        ),
    )


def test_pipeline_horizontal_sum_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.sum(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series("horizontal_sum", [1.0, 1.0, 9.4, float("nan")])
        ),
    )


def test_pipeline_horizontal_argmax_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.argmax(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series(
                "horizontal_argmax", [["i2", "u2"], ["u0"], ["i2", "u2"], ["i2", "u2"]]
            )
        ),
    )


def test_pipeline_horizontal_argmin_small(test_data_small: DataFrame):
    assert_component_valid(
        Pipeline().horizontal.argmin(cs.numeric(), maintain_order=True),
        test_data_small,
        test_data_small.with_columns(
            Series(
                "horizontal_argmin",
                [
                    ["i0"],
                    ["i0", "i1", "u1", "f0", "f1"],
                    ["i1", "u1", "f1"],
                    ["i1", "u1", "f1"],
                ],
            )
        ),
    )
