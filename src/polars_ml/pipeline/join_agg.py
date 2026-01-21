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
        self._by_selector = by
        self._aggs = aggs
        self._how: JoinStrategy = how
        self._suffix = suffix

        self._by: list[str] | None = None
        self._agg_df: DataFrame | None = None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self._by = data.lazy().select(self._by_selector).collect_schema().names()
        self._agg_df = data.group_by(self._by).agg(*self._aggs)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self._agg_df is None:
            raise NotFittedError()

        return data.join(
            self._agg_df,
            on=self._by,
            how=self._how,
            suffix=self._suffix,
        )
