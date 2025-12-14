from __future__ import annotations

from typing import Callable, Self

from polars import DataFrame

from polars_ml import Transformer


class Apply(Transformer):
    def __init__(self, func: Callable[[DataFrame], DataFrame]):
        self.func = func

    def transform(self, data: DataFrame) -> DataFrame:
        return self.func(data)


class Echo(Transformer):
    def __init__(self):
        pass

    def transform(self, data: DataFrame) -> DataFrame:
        return data


class Const(Transformer):
    def __init__(self, data: DataFrame):
        self.data = data

    def transform(self, data: DataFrame) -> DataFrame:
        return self.data


class Parrot(Transformer):
    def __init__(self):
        pass

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.data = data
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return self.data


class Side(Transformer):
    def __init__(self, transformer: Transformer):
        self.transformer = transformer

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.transformer.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        self.transformer.fit_transform(data, **more_data)
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        self.transformer.transform(data)
        return data
