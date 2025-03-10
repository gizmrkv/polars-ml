import uuid
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from polars_ml import Pipeline


class LabelEncoding(PipelineComponent):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
    ):
        self.exprs = exprs
        self.orders = orders or {}
        self.maintain_order = maintain_order
        self.suffix = uuid.uuid4().hex

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
            data.select(col).join(mapping, on=col, how="left")["label"].rename(col)
            for col, mapping in self.mappings.items()
            if col in data.columns
        )


class LabelEncodingInverse(PipelineComponent):
    def __init__(
        self, label_encoding: LabelEncoding, mapping: Mapping[str, str] | None = None
    ):
        self.label_encoding = label_encoding
        self.mapping = mapping

    def transform(self, data: DataFrame) -> DataFrame:
        mapping = self.mapping or {col: col for col in self.label_encoding.mappings}
        return data.with_columns(
            [
                data.select(pl.col(col_from).alias("label"))
                .join(self.label_encoding.mappings[col_to], on="label", how="left")[
                    col_to
                ]
                .rename(col_from)
                for col_from, col_to in mapping.items()
            ]
        )


class LabelEncodingInverseContext:
    def __init__(
        self,
        pipeline: "Pipeline",
        label_encoding: LabelEncoding,
        mapping: Mapping[str, str] | None = None,
        *,
        component_name: str | None = None,
    ):
        self.pipeline = pipeline
        self.label_encoding = label_encoding
        self.mapping = mapping
        self.component_name = component_name

    def __enter__(self) -> "Pipeline":
        self.pipeline.pipe(self.label_encoding, component_name=self.component_name)
        return self.pipeline

    def __exit__(self, *args: Any, **kwargs: Any):
        self.pipeline.pipe(
            LabelEncodingInverse(self.label_encoding, mapping=self.mapping),
            component_name=self.component_name + "_inverse"
            if self.component_name
            else None,
        )
