from typing import Iterable, Mapping, Sequence

import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.component import Component


class Discretizer(Component):
    def __init__(
        self,
        *columns: str,
        quantiles: Sequence[float] | int | None = None,
        breaks: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_discretized",
    ):
        self.columns = columns
        self.quantiles = quantiles
        self.breaks = breaks
        self.labels = labels
        self.left_closed = left_closed
        self.allow_duplicates = allow_duplicates
        self.suffix = suffix

        assert (self.quantiles is not None) ^ (self.breaks is not None), (
            "Either 'quantiles' or 'breaks' must be provided."
        )

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Discretizer":
        if self.quantiles is not None:
            data = data.select(*self.columns).select(pl.all().name.suffix(self.suffix))
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
        pre = data.select(*self.columns).select(pl.all().name.suffix(self.suffix))
        targets = set(self.breakpoints.keys()) & set(pre.columns)
        return data.with_columns(
            pre.select(
                pl.col(col).cut(
                    self.breakpoints[col],  # type: ignore
                    labels=self.labels,
                    left_closed=self.left_closed,
                    include_breaks=False,
                )
            )[col]
            for col in targets
        )
