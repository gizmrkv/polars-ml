from __future__ import annotations

import itertools
from typing import Iterable, Self

import polars as pl
from polars._typing import ColumnNameOrSelector

from polars_ml.base import LazyTransformer
from polars_ml.exceptions import NotFittedError


class Combine(LazyTransformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        n: int,
        *,
        delimiter: str = "_",
    ):
        self._selector = columns
        self._n = n
        self._delimiter = delimiter

        self._combinations: list[tuple[str, ...]] | None = None

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        columns = data.select(self._selector).collect_schema().names()
        self._combinations = list(itertools.combinations(columns, self._n))
        return self

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self._combinations is None:
            raise NotFittedError()

        expressions: list[pl.Expr] = []
        for combination in self._combinations:
            name = self._delimiter.join(combination)
            expressions.append(pl.struct(combination).alias(name))
        return data.with_columns(expressions)
