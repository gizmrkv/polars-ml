from typing import Any

import polars as pl
import polars.selectors as cs
import pytest
from polars import DataFrame, Series
from polars.testing import assert_frame_equal

from polars_ml import Pipeline


@pytest.fixture
def test_data_basic():
    return DataFrame(
        {
            "f0": [1, 2, 3, 4],
            "f1": [1, None, 3, -4],
            "f2": [0.1, 0.2, float("nan"), 0.4],
            "b0": [True, False, True, False],
            "b1": [True, False, True, True],
            "b2": [True, False, False, False],
        }
    )


def assert_frame_horizontal_equal(
    data: DataFrame, expected: DataFrame, method: str, *args: Any, **kwargs: Any
):
    pp: Pipeline = getattr(Pipeline().horizontal, method)(
        *args, **kwargs, maintain_order=True
    )
    out = pp.transform(data.clone())
    assert_frame_equal(out, expected)


def test_pipeline_horizontal_agg_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_agg", [1.417745, 2.009975, float("nan"), 5.670979])
        ),
        "agg",
        cs.numeric(),
        aggs=[pl.all().pow(2).sum().sqrt()],
    )


def test_pipeline_horizontal_all_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_all", [True, False, False, False])
        ),
        "all",
        cs.boolean(),
    )


def test_pipeline_horizontal_count_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_count", [6, 5, 6, 6], dtype=pl.UInt32)
        ),
        "count",
        pl.all(),
    )


def test_pipeline_horizontal_max_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(Series("horizontal_max", [1.0, 2.0, 3.0, 4.0])),
        "max",
        cs.numeric(),
    )


def test_pipeline_horizontal_mean_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_mean", [0.7, 1.1, float("nan"), 0.133333])
        ),
        "mean",
        cs.numeric(),
    )


def test_pipeline_horizontal_median_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(Series("horizontal_median", [1.0, 1.1, 3.0, 0.4])),
        "median",
        cs.numeric(),
    )


def test_pipeline_horizontal_min_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(Series("horizontal_min", [0.1, 0.2, 3.0, -4.0])),
        "min",
        cs.numeric(),
    )


def test_pipeline_horizontal_n_unique_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_n_unique", [2, 4, 4, 5], dtype=pl.UInt32)
        ),
        "n_unique",
        pl.all(),
    )


def test_pipeline_horizontal_quantile_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_quantile", [1.0, 2.0, 3.0, 0.4])
        ),
        "quantile",
        cs.numeric(),
        quantile=0.5,
    )


def test_pipeline_horizontal_std_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_std", [0.519615, 1.272792, float("nan"), 4.006661])
        ),
        "std",
        cs.numeric(),
    )


def test_pipeline_horizontal_sum_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_sum", [2.1, 2.2, float("nan"), 0.4])
        ),
        "sum",
        cs.numeric(),
    )


def test_pipeline_horizontal_argmax_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_argmax", [["f0", "f1"], ["f0"], ["f0", "f1"], ["f0"]])
        ),
        "argmax",
        cs.numeric(),
    )


def test_pipeline_horizontal_argmin_basic(test_data_basic: DataFrame):
    assert_frame_horizontal_equal(
        test_data_basic,
        test_data_basic.with_columns(
            Series("horizontal_argmin", [["f2"], ["f2"], ["f0", "f1"], ["f1"]])
        ),
        "argmin",
        cs.numeric(),
    )
