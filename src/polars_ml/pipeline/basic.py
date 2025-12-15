from __future__ import annotations

from typing import Callable, Self, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import ColumnNameOrSelector, ConcatMethod

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


class ToDummies(Transformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        separator: str = "_",
        drop_first: bool = False,
    ):
        self.columns = columns
        self.separator = separator
        self.drop_first = drop_first

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.columns = data.lazy().select(self.columns).collect_schema().names()
        self.values = {c: data[c].unique().to_list() for c in self.columns}
        if self.drop_first:
            self.values = {c: v[1:] for c, v in self.values.items()}
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return pl.concat(
            [
                data,
                data.select(
                    pl.col(c).eq(v).cast(pl.UInt8).alias(f"{c}{self.separator}{v}")
                    for c, vs in self.values.items()
                    for v in vs
                ),
            ],
            how="horizontal",
        )
