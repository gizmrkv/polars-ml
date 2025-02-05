from typing import Any, Iterable, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml import Component


class LabelEncoding(Component):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        orders: dict[str, Sequence[Any]] | None = None,
        maintain_order: bool = False,
    ):
        self.exprs = exprs
        self.orders = orders or {}
        self.maintain_order = maintain_order

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        data = data.select(*self.exprs)
        self.mappings = {
            col: DataFrame(
                [
                    Series(col, self.orders[col]),
                    Series("label", range(len(self.orders[col])), dtype=pl.UInt32),
                ]
            )
            if col in self.orders
            else (
                data.select(col)
                .unique(maintain_order=self.maintain_order)
                .drop_nulls()
                .with_row_index("label")
            )
            for col in data.columns
        }

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return data.with_columns(
            [
                data.select(col).join(mapping, on=col, how="left")["label"].rename(col)
                for col, mapping in self.mappings.items()
                if col in data.columns
            ]
        )

    def inverse_transform(self, data: DataFrame) -> DataFrame:
        return data.with_columns(
            [
                data.select(pl.col(col).alias("label"))
                .join(mapping, on="label", how="left")[col]
                .rename(col)
                for col, mapping in self.mappings.items()
                if col in data.columns
            ]
        )


class InverseLabelEncoding(Component):
    def __init__(
        self, label_encoding: LabelEncoding, mapping: dict[str, str] | None = None
    ):
        self.label_encoding = label_encoding
        self.mapping = mapping or {col: col for col in label_encoding.mappings}

    def transform(self, data: DataFrame) -> DataFrame:
        return data.with_columns(
            [
                data.select(pl.col(col_from).alias("label"))
                .join(self.label_encoding.mappings[col_to], on="label", how="left")[
                    col_to
                ]
                .rename(col_from)
                for col_from, col_to in self.mapping.items()
            ]
        )
