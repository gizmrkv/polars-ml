from typing import Self

import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.base import Transformer
from polars_ml.pipeline.pipeline import Pipeline


def test_pipeline_apply():
    df = DataFrame({"a": [1, 2, 3]})
    pipeline = Pipeline().apply(lambda d: d.with_columns(b=pl.col("a") * 2))

    result = pipeline.transform(df)
    expected = DataFrame({"a": [1, 2, 3], "b": [2, 4, 6]})
    assert_frame_equal(result, expected)

    result_fit = pipeline.fit_transform(df)
    assert_frame_equal(result_fit, expected)


def test_pipeline_const():
    df = DataFrame({"a": [1, 2, 3]})
    const_data = DataFrame({"c": [10, 11]})
    pipeline = Pipeline().const(const_data)

    result = pipeline.transform(df)
    assert_frame_equal(result, const_data)


def test_pipeline_echo():
    df = DataFrame({"a": [1, 2, 3]})
    pipeline = Pipeline().echo()

    result = pipeline.transform(df)
    assert_frame_equal(result, df)


def test_pipeline_replay():
    df1 = DataFrame({"a": [1, 2, 3]})
    df2 = DataFrame({"a": [4, 5, 6]})

    pipeline = Pipeline().replay()

    pipeline.fit(df1)

    result = pipeline.transform(df2)
    assert_frame_equal(result, df1)


def test_pipeline_side():
    class MockSideEffect(Transformer):
        def __init__(self):
            self.fit_calls = 0
            self.transform_calls = 0

        def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
            self.fit_calls += 1
            return self

        def transform(self, data: DataFrame) -> DataFrame:
            self.transform_calls += 1
            return data

    spy = MockSideEffect()
    pipeline = Pipeline().side(spy)
    df = DataFrame({"a": [1]})

    result = pipeline.transform(df)
    assert_frame_equal(result, df)
    assert spy.transform_calls == 1
    assert spy.fit_calls == 0

    pipeline.fit(df)
    assert spy.fit_calls == 1

    spy = MockSideEffect()
    pipeline = Pipeline().side(spy)

    result_fit_transform = pipeline.fit_transform(df)
    assert_frame_equal(result_fit_transform, df)
    assert spy.fit_calls == 1
    assert spy.transform_calls == 1
