from __future__ import annotations

import itertools
from typing import Iterable, Self

import polars as pl
from polars import DataFrame, Expr
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class Combine(Transformer):
    """
    Combines n columns into a struct column.
    """

    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        n: int,
        delimiter: str = "_",
        prefix: str = "comb_",
    ):
        self.selector = columns
        self.n = n
        self.delimiter = delimiter
        self.prefix = prefix
        self.combinations_: list[tuple[str, ...]] | None = None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        columns = data.select(self.selector).collect_schema().names()
        self.combinations_ = list(itertools.combinations(columns, self.n))
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self.combinations_ is None:
            raise NotFittedError()

        expressions: list[Expr] = []
        for combination in self.combinations_:
            name = self.prefix + self.delimiter.join(combination)
            expressions.append(pl.struct(combination).alias(name))
        return data.with_columns(expressions)
