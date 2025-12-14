from __future__ import annotations

from typing import Iterable, Self, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import Transformer


class Discretize(Transformer):
    def __init__(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_discretized",
    ):
        self.exprs = exprs
        self.more_exprs = more_exprs
        self.quantiles = quantiles
        self.labels = labels
        self.left_closed = left_closed
        self.allow_duplicates = allow_duplicates
        self.suffix = suffix

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        data = data.select(self.exprs, *self.more_exprs)
        self.breakpoints = {
            col: data.select(
                pl.col(col)
                .qcut(
                    self.quantiles,
                    left_closed=self.left_closed,
                    allow_duplicates=self.allow_duplicates,
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
        discretized = data.select(self.exprs, *self.more_exprs)
        breakpoints = {
            c: bs
            for c, bs in self.breakpoints.items()
            if c in discretized.collect_schema().names()
        }
        discretized = discretized.select(
            pl.col(c)
            .cut(
                bs,
                labels=self.labels,
                left_closed=self.left_closed,
                include_breaks=False,
            )
            .alias(c + self.suffix)
            for c, bs in breakpoints.items()
        )
        return pl.concat([data, discretized], how="horizontal")
