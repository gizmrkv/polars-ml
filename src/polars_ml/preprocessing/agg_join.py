from __future__ import annotations

from typing import Iterable, Self

import polars as pl
from polars import DataFrame
from polars._typing import ColumnNameOrSelector, IntoExpr, JoinStrategy

from polars_ml.base import Transformer


class AggJoin(Transformer):
    def __init__(
        self,
        by: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *aggs: IntoExpr | Iterable[IntoExpr],
        how: JoinStrategy = "left",
        on: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        suffix: str = "_agg",
    ):
        self.by = by
        self.aggs = aggs
        self.how = how
        self.on = on
        self.suffix = suffix
        self.agg_df_: DataFrame | None = None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.agg_df_ = data.group_by(self.by).agg(self.aggs)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self.agg_df_ is None:
            raise ValueError("AggJoin has not been fitted yet.")

        on = self.on if self.on is not None else self.by
        return data.join(self.agg_df_, on=on, how=self.how, suffix=self.suffix)
