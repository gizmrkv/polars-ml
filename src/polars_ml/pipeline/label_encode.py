from __future__ import annotations

from typing import Any, Iterable, Mapping, Self, Sequence

import polars as pl
from polars._typing import ColumnNameOrSelector

from polars_ml.base import LazyTransformer
from polars_ml.exceptions import NotFittedError


class LabelEncode(LazyTransformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
    ):
        self._selector = columns
        self._more_selectors = more_columns
        self._orders = orders or {}
        self._maintain_order = maintain_order

        self._mappings: dict[str, pl.DataFrame] | None = None

    @property
    def mappings(self) -> dict[str, pl.DataFrame]:
        if self._mappings is None:
            raise NotFittedError()
        return self._mappings

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        data = data.select(self._selector, *self._more_selectors)
        self._mappings = {
            col: pl.DataFrame(
                [
                    pl.Series("key", self._orders[col]),
                    pl.Series("value", range(len(self._orders[col])), dtype=pl.UInt32),
                ]
            )
            if col in self._orders
            else (
                data.select(pl.col(col).alias("key"))
                .unique(maintain_order=self._maintain_order)
                .drop_nulls()
                .with_row_index("value")
                .select("key", "value")
            )
            for col in data.columns
        }

        return self

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self._mappings is None:
            raise NotFittedError()

        columns = data.collect_schema().names()
        for col, mapping in self._mappings.items():
            if col not in columns:
                continue

            data = pl.concat(
                [
                    data.drop(col),
                    data.select(col)
                    .join(
                        mapping.lazy().rename({"value": col + "_label"}),
                        left_on=col,
                        right_on="key",
                        how="left",
                    )
                    .select(pl.col(col + "_label").alias(col)),
                ],
                how="horizontal",
            )

        return data.select(*columns)
