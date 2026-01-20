from __future__ import annotations

from typing import Any, Iterable, Self, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class Discretize(Transformer):
    def __init__(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_disc",
    ):
        self._exprs = exprs
        self._more_exprs = more_exprs
        self._quantiles = quantiles
        self._labels = labels
        self._left_closed = left_closed
        self._allow_duplicates = allow_duplicates
        self._suffix = suffix

        self._breakpoints: dict[str, list[Any]] | None = None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        data = data.select(self._exprs, *self._more_exprs)
        self._breakpoints = {
            col: data.select(
                pl.col(col)
                .qcut(
                    self._quantiles,
                    left_closed=self._left_closed,
                    allow_duplicates=self._allow_duplicates,
                    include_breaks=True,
                )
                .struct.field("breakpoint")
                .alias(col)
            )
            .unique()
            .filter(pl.col(col).is_finite())[col]
            .sort()
            .to_list()
            for col in data.columns
        }
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self._breakpoints is None:
            raise NotFittedError()

        discretized = data.select(self._exprs, *self._more_exprs)
        breakpoints = {
            c: bs
            for c, bs in self._breakpoints.items()
            if c in discretized.collect_schema().names()
        }
        discretized = discretized.select(
            pl.col(c)
            .cut(
                bs,
                labels=self._labels,
                left_closed=self._left_closed,
                include_breaks=False,
            )
            .alias(c + self._suffix)
            for c, bs in breakpoints.items()
        )
        return pl.concat([data, discretized], how="horizontal")
