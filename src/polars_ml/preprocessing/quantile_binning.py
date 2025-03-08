from typing import Iterable, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent


class QuantileBinning(PipelineComponent):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_qbin",
    ):
        self.exprs = exprs
        self.quantiles = quantiles
        self.labels = labels
        self.left_closed = left_closed
        self.allow_duplicates = allow_duplicates
        self.suffix = suffix

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        data = data.select(*self.exprs)
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
        return data.with_columns(
            [
                pl.col(col)
                .cut(
                    breaks,
                    labels=self.labels,
                    left_closed=self.left_closed,
                    include_breaks=False,
                )
                .alias(f"{col}{self.suffix}")
                for col, breaks in self.breakpoints.items()
                if col in data.columns
            ]
        )
