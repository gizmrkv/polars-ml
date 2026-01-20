from __future__ import annotations

import itertools
from typing import Iterable, Self

import polars as pl
from polars import DataFrame, Expr
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class Combine(Transformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        n: int,
        *,
        delimiter: str = "_",
        suffix: str = "_comb",
    ):
        self._selector = columns
        self._n = n
        self._delimiter = delimiter
        self._suffix = suffix

        self._combinations: list[tuple[str, ...]] | None = None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        columns = data.select(self._selector).collect_schema().names()
        self._combinations = list(itertools.combinations(columns, self._n))
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self._combinations is None:
            raise NotFittedError()

        expressions: list[Expr] = []
        for combination in self._combinations:
            name = self._delimiter.join(combination) + self._suffix
            expressions.append(pl.struct(combination).alias(name))
        return data.with_columns(expressions)
