from __future__ import annotations

from typing import Self, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import ConcatMethod

from polars_ml.base import Transformer


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
