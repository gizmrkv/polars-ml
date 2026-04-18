from __future__ import annotations

from typing import Self, Sequence

import polars as pl
from polars._typing import ConcatMethod

from polars_ml.base import LazyTransformer, Transformer


class Concat(Transformer):
    def __init__(
        self,
        items: Sequence[Transformer],
        *,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
        strict: bool = False,
    ):
        self.items = items
        self.params = {
            "how": how,
            "rechunk": rechunk,
            "parallel": parallel,
            "strict": strict,
        }

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        for item in self.items:
            item.fit(data, **more_data)
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        data_list = [item.fit_transform(data, **more_data) for item in self.items]
        return pl.concat(data_list, **self.params)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        data_list = [item.transform(data) for item in self.items]
        return pl.concat(data_list, **self.params)


class LazyConcat(LazyTransformer):
    def __init__(
        self,
        items: Sequence[LazyTransformer],
        *,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
        strict: bool = False,
    ):
        self.items = items
        self.params = {
            "how": how,
            "rechunk": rechunk,
            "parallel": parallel,
            "strict": strict,
        }

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        for item in self.items:
            item.fit(data, **more_data)
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        data_list = [item.fit_transform(data, **more_data) for item in self.items]
        return pl.concat(data_list, **self.params)

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        data_list = [item.transform(data) for item in self.items]
        return pl.concat(data_list, **self.params)
