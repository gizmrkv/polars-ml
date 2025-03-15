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
            "n0": [1, 2, 3, 4],
            "n1": [1, None, 3, -4],
            "n2": [0.1, 0.2, float("nan"), 0.4],
            "b0": [True, False, True, False],
            "b1": [True, False, True, True],
            "b2": [True, False, False, False],
            "s3": ["a", "a", "b", "b", "c"],
        }
    )


def assert_frame_horizontal_equal(
    pipeline: Pipeline, data: DataFrame, expected: DataFrame
):
    out = pipeline.transform(data.clone())
    assert_frame_equal(out, expected)
