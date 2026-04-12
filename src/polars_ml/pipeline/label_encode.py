from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Self, Sequence

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
                    pl.Series(col, self._orders[col]),
                    pl.Series(
                        col + "_label", range(len(self._orders[col])), dtype=pl.UInt32
                    ),
                ]
            )
            if col in self._orders
            else (
                data.select(col)
                .unique(maintain_order=self._maintain_order)
                .drop_nulls()
                .with_row_index(col + "_label")
                .select(col, col + "_label")
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

            data = data.update(
                data.select(col)
                .join(mapping.lazy(), on=col, how="left")
                .select(pl.col(col + "_label").alias(col)),
                include_nulls=True,
            )

        return data


class LabelEncodeInverse(LazyTransformer):
    def __init__(
        self, label_encode: LabelEncode, mapping: Mapping[str, str] | None = None
    ):
        self._label_encode = label_encode
        self._mapping = mapping

    @property
    def mapping(self) -> Mapping[str, str]:
        if self._mapping is not None:
            return self._mapping

        return {col: col for col in self._label_encode.mappings.keys()}

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for tgt, src in self.mapping.items():
            if tgt not in data.collect_schema().names():
                continue

            if src not in self._label_encode.mappings:
                raise ValueError(f"Column {src} not found in LabelEncode")

            data = (
                data.join(
                    self._label_encode.mappings[src]
                    .lazy()
                    .rename({src + "_label": src}),
                    left_on=tgt,
                    right_on=src,
                    how="left",
                )
                .with_columns(pl.col(src).alias(tgt))
                .drop(src)
            )

        return data


class Pipeline(Protocol):
    def pipe(self, step: LazyTransformer) -> Self: ...


class LabelEncodeInverseContext:
    def __init__(
        self,
        pipeline: Pipeline,
        label_encode: LabelEncode,
        mapping: Mapping[str, str] | None = None,
    ):
        self._pipeline = pipeline
        self._label_encode = label_encode
        self._label_encode_inverse = LabelEncodeInverse(
            self._label_encode, mapping=mapping
        )
        self._mapping = mapping

    def __enter__(self):
        self._pipeline.pipe(self._label_encode)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._pipeline.pipe(self._label_encode_inverse)
