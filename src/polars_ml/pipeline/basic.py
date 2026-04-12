from __future__ import annotations

from typing import Callable, Generic, Self, TypeVar

import polars as pl
from polars import DataFrame

from polars_ml import LazyTransformer, Transformer


class Echo(LazyTransformer):
    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data


class Replay(LazyTransformer):
    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.data = data
        return self

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return self.data.lazy()


class Const(Transformer):
    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.data


class LazyConst(LazyTransformer):
    def __init__(self, data: pl.LazyFrame) -> None:
        self.data = data

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return self.data


class Apply(Transformer):
    def __init__(self, func: Callable[[pl.DataFrame], pl.DataFrame]) -> None:
        self.func = func

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return self.func(data)


class LazyApply(LazyTransformer):
    def __init__(self, func: Callable[[pl.LazyFrame], pl.LazyFrame]) -> None:
        self.func = func

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return self.func(data)


TransformerType = TypeVar("TransformerType", bound=Transformer)
LazyTransformerType = TypeVar("LazyTransformerType", bound=LazyTransformer)


class Side(Transformer, Generic[TransformerType]):
    def __init__(self, transformer: TransformerType) -> None:
        self.transformer = transformer

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.transformer.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        self.transformer.fit_transform(data, **more_data)
        return data

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.transformer.transform(data)
        return data


class LazySide(LazyTransformer, Generic[LazyTransformerType]):
    def __init__(self, transformer: LazyTransformerType) -> None:
        self.transformer = transformer

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.transformer.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        self.transformer.fit_transform(data, **more_data)
        return data

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        self.transformer.transform(data)
        return data
