from __future__ import annotations

from typing import Callable, Self, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import ConcatMethod

from polars_ml.base import Transformer


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


class Replay(Transformer):
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


class Concat(Transformer):
    def __init__(
        self,
        items: Sequence[Transformer],
        *,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
    ):
        self.items = items
        self.params = {
            "how": how,
            "rechunk": rechunk,
            "parallel": parallel,
        }

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for item in self.items:
            item.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        data_list = [item.fit_transform(data, **more_data) for item in self.items]
        return pl.concat(data_list, **self.params)

    def transform(self, data: DataFrame) -> DataFrame:
        data_list = [item.transform(data) for item in self.items]
        return pl.concat(data_list, **self.params)
