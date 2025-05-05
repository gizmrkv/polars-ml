import uuid
from typing import Any, Iterable, Mapping, Sequence

import polars as pl
from polars import DataFrame, Series

from polars_ml.component import Component


class LabelEncoder(Component):
    def __init__(
        self,
        *columns: str,
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
    ):
        self.columns = columns
        self.orders = orders or {}
        self.maintain_order = maintain_order
        self.suffix = uuid.uuid4().hex

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "LabelEncoder":
        data = data.select(*self.columns)
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
            data.select(col).join(mapping, on=col, how="left")["label"].rename(col)
            for col, mapping in self.mappings.items()
            if col in data.columns
        )

    def inverser(
        self, mapping: Mapping[str, str] | None = None
    ) -> "InverseLabelEncoder":
        return InverseLabelEncoder(self, mapping)


class InverseLabelEncoder(Component):
    def __init__(
        self, label_encoder: LabelEncoder, mapping: Mapping[str, str] | None = None
    ):
        self.label_encoder = label_encoder
        self.mapping = mapping

    def transform(self, data: DataFrame) -> DataFrame:
        mapping = self.mapping or {col: col for col in self.label_encoder.mappings}
        return data.with_columns(
            [
                data.select(pl.col(col_from).alias("label"))
                .join(self.label_encoder.mappings[col_to], on="label", how="left")[
                    col_to
                ]
                .rename(col_from)
                for col_from, col_to in mapping.items()
            ]
        )
