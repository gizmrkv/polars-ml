from __future__ import annotations

from typing import Iterable, Self

import polars as pl
from polars import DataFrame
from polars._typing import ColumnNameOrSelector, IntoExpr, JoinStrategy

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class JoinAgg(Transformer):
    def __init__(
        self,
        by: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *aggs: IntoExpr | Iterable[IntoExpr],
        how: JoinStrategy = "left",
        suffix: str = "_agg",
    ):
        self.by_selector = by
        self.aggs = aggs
        self.how = how
        self.suffix = suffix
        self.agg_df_: DataFrame | None = None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.by = data.lazy().select(self.by_selector).collect_schema().names()
        self.agg_df_ = data.group_by(self.by).agg(self.aggs)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self.agg_df_ is None:
            raise NotFittedError()

        return data.join(self.agg_df_, on=self.by, how=self.how, suffix=self.suffix)
