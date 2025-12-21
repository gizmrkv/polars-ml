from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Series
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer

if TYPE_CHECKING:
    from polars_ml import Pipeline


class LabelEncode(Transformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
        suffix: str = "",
    ):
        self.column_selectors = columns
        self.more_column_selectors = more_columns
        self.orders = orders or {}
        self.maintain_order = maintain_order
        self.suffix = suffix

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        data = data.select(self.column_selectors, *self.more_column_selectors)
        self.mappings = {
            col: DataFrame(
                [
                    Series(col, self.orders[col]),
                    Series(
                        col + "_label", range(len(self.orders[col])), dtype=pl.UInt32
                    ),
                ]
            )
            if col in self.orders
            else (
                data.select(col)
                .unique(maintain_order=self.maintain_order)
                .drop_nulls()
                .with_row_index(col + "_label")
                .select(col, col + "_label")
            )
            for col in data.columns
        }

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        columns = data.collect_schema().names()
        for col, mapping in self.mappings.items():
            if col not in columns:
                continue

            data = (
                data.join(mapping, on=col, how="left")
                .with_columns(pl.col(col + "_label").alias(col + self.suffix))
                .drop(col + "_label")
            )

        return data


class LabelEncodeInverse(Transformer):
    def __init__(
        self, label_encode: LabelEncode, mapping: Mapping[str, str] | None = None
    ):
        self.label_encode = label_encode
        self._mapping = mapping

    @property
    def mapping(self) -> Mapping[str, str]:
        if self._mapping is not None:
            return self._mapping
        return {col: col for col in self.label_encode.mappings.keys()}

    def transform(self, data: DataFrame) -> DataFrame:
        import shortuuid

        tmp_suf = shortuuid.ShortUUID().random(length=8)
        for tgt, src in self.mapping.items():
            if tgt not in data.collect_schema().names():
                continue

            if src not in self.label_encode.mappings:
                raise ValueError(f"Column {src} not found in LabelEncode")

            data = (
                data.join(
                    self.label_encode.mappings[src].rename(
                        {src: src + tmp_suf, src + "_label": src}
                    ),
                    left_on=tgt,
                    right_on=src,
                    how="left",
                )
                .with_columns(pl.col(src + tmp_suf).alias(tgt))
                .drop(src + tmp_suf)
            )

        return data


class LabelEncodeInverseContext:
    def __init__(
        self,
        pipeline: Pipeline,
        label_encode: LabelEncode,
        mapping: Mapping[str, str] | None = None,
    ):
        self.pipeline = pipeline
        self.label_encode = label_encode
        self.mapping = mapping

    def __enter__(self) -> Pipeline:
        return self.pipeline.pipe(self.label_encode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipeline.pipe(LabelEncodeInverse(self.label_encode, mapping=self.mapping))
