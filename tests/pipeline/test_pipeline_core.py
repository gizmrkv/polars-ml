from typing import Self

import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.pipeline.pipeline import Pipeline


class Mul(Transformer):
    def __init__(self, factor: int):
        self.factor = factor

    def transform(self, data: DataFrame) -> DataFrame:
        return data.select(pl.all() * self.factor)


class Add(Transformer):
    def __init__(self, value: int):
        self.value = value

    def transform(self, data: DataFrame) -> DataFrame:
        return data.select(pl.all() + self.value)


class FeatureImportanceTransformer(Transformer, HasFeatureImportance):
    def transform(self, data: DataFrame) -> DataFrame:
        return data

    def get_feature_importance(self) -> DataFrame:
        return DataFrame({"feature": ["a"], "importance": [0.5]})


def test_pipeline_init_and_pipe():
    p = Pipeline()
    assert len(p) == 0

    p.pipe(Mul(2))
    assert len(p) == 1

    p2 = Pipeline(Mul(2), Add(1))
    assert len(p2) == 2


def test_pipeline_transform_chaining():
    df = DataFrame({"a": [1, 2, 3]})
    p = Pipeline(Mul(2), Add(1))

    result = p.transform(df)
    expected = DataFrame({"a": [3, 5, 7]})
    assert_frame_equal(result, expected)


def test_pipeline_fit_transform():
    class Statefulstep(Transformer):
        def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
            return self

        def transform(self, data: DataFrame) -> DataFrame:
            return data.with_columns(b=pl.lit(1, dtype=pl.Int64))

    df = DataFrame({"a": [1]})
    p = Pipeline(Statefulstep(), Add(10))

    result = p.fit_transform(df)
    expected = DataFrame({"a": [11], "b": [11]})
    assert_frame_equal(result, expected)


def test_fit_and_transform_separately():
    class FittableStep(Transformer):
        def __init__(self):
            self.val = 0

        def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
            self.val = 10
            return self

        def transform(self, data: DataFrame) -> DataFrame:
            return data.select(pl.all() + self.val)

    p = Pipeline(FittableStep())
    df = DataFrame({"a": [1]})

    res_unfitted = p.transform(df)
    assert_frame_equal(res_unfitted, DataFrame({"a": [1]}))

    p.fit(df)
    res_fitted = p.transform(df)
    assert_frame_equal(res_fitted, DataFrame({"a": [11]}))


def test_feature_importance():
    p = Pipeline(Mul(2), FeatureImportanceTransformer())
    fi = p.get_feature_importance()
    assert_frame_equal(fi, DataFrame({"feature": ["a"], "importance": [0.5]}))

    p_bad = Pipeline(FeatureImportanceTransformer(), Mul(2))
    with pytest.raises(TypeError, match="does not support feature importance"):
        p_bad.get_feature_importance()

    p_empty = Pipeline()
    with pytest.raises(ValueError, match="Pipeline has no steps"):
        p_empty.get_feature_importance()
