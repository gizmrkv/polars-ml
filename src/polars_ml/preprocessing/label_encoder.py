import uuid
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Series
from polars._typing import ColumnNameOrSelector

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from polars_ml import Pipeline


class LabelEncoder(PipelineComponent):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
    ):
        self.columns = columns
        self.more_columns = more_columns
        self.orders = orders or {}
        self.maintain_order = maintain_order
        self.suffix = uuid.uuid4().hex

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        data = data.select(self.columns, *self.more_columns)
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


class LabelEncoderInverse(PipelineComponent):
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


class LabelEncoderInverseContext:
    def __init__(
        self,
        pipeline: "Pipeline",
        label_encoder: LabelEncoder,
        inverse_mapping: Mapping[str, str] | None = None,
        *,
        component_name: str | None = None,
    ):
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.inverse_mapping = inverse_mapping
        self.component_name = component_name

    def __enter__(self) -> "Pipeline":
        self.pipeline.pipe(self.label_encoder, component_name=self.component_name)
        return self.pipeline

    def __exit__(self, *args: Any, **kwargs: Any):
        self.pipeline.pipe(
            LabelEncoderInverse(self.label_encoder, mapping=self.inverse_mapping),
            component_name=self.component_name + "_inverse"
            if self.component_name
            else None,
        )
