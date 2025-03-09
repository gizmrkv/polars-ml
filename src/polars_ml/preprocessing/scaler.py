from typing import TYPE_CHECKING, Any, Literal, Mapping, Self, Sequence, Type

import polars as pl
from polars import DataFrame, Expr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from polars_ml import Pipeline


class Scaler(PipelineComponent):
    def __init__(
        self,
        *column: str,
        by: str | Sequence[str] | None = None,
        method: Literal["standard", "min-max", "robust"] = "standard",
        quantile: tuple[float, float] = (0.25, 0.75),
    ):
        self.columns = column
        self.by = [by] if isinstance(by, str) else list(by) if by is not None else None
        self.method = method
        self.q1, self.q3 = quantile

        assert 0 <= self.q1 < self.q3 <= 1, (
            f"Quantile values must be in the range [0, 1] and q1 < q3. Got {quantile}"
        )

    def move_expr(self, column: str) -> Expr:
        if self.method == "standard":
            return pl.col(column).mean()
        elif self.method == "min-max":
            return pl.col(column).min()
        elif self.method == "robust":
            return pl.col(column).median()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def scale_expr(self, column: str) -> Expr:
        if self.method == "standard":
            return pl.col(column).std()
        elif self.method == "min-max":
            return pl.col(column).max() - pl.col(column).min()
        elif self.method == "robust":
            return pl.col(column).quantile(self.q3) - pl.col(column).quantile(self.q1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if self.by:
            self.move_scale = data.group_by(self.by).agg(
                *[self.move_expr(col).alias(f"{col}_move") for col in self.columns],
                *[self.scale_expr(col).alias(f"{col}_scale") for col in self.columns],
            )
        else:
            self.move_scale = data.select(
                *[self.move_expr(col).alias(f"{col}_move") for col in self.columns],
                *[self.scale_expr(col).alias(f"{col}_scale") for col in self.columns],
            )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        columns = (
            data.lazy()
            .select(set(data.columns) & set(self.columns))
            .collect_schema()
            .names()
        )

        if self.by:
            return (
                data.join(
                    self.move_scale.select(
                        *self.by,
                        *[f"{col}_move" for col in columns],
                        *[f"{col}_scale" for col in columns],
                    ),
                    on=self.by,
                    how="left",
                )
                .with_columns(
                    [
                        (pl.col(col) - pl.col(f"{col}_move")) / pl.col(f"{col}_scale")
                        for col in columns
                    ]
                )
                .select(data.columns)
            )
        else:
            move_scale = self.move_scale.row(0, named=True)
            return data.with_columns(
                [
                    (pl.col(col) - move_scale[f"{col}_move"])
                    / move_scale[f"{col}_scale"]
                    for col in columns
                ]
            )


class InverseScaler(PipelineComponent):
    def __init__(self, scaler: Scaler, mapping: Mapping[str, str] | None = None):
        self.scaler = scaler
        self.mapping = mapping or {col: col for col in scaler.columns}

    def transform(self, data: DataFrame) -> DataFrame:
        columns = (
            data.lazy()
            .select(set(data.columns) & set(self.mapping.keys()))
            .collect_schema()
            .names()
        )
        scales = (
            data.lazy()
            .select(set(data.columns) & set(self.mapping.values()))
            .collect_schema()
            .names()
        )
        if self.scaler.by:
            return (
                data.join(
                    self.scaler.move_scale.select(
                        *self.scaler.by,
                        *[f"{col}_move" for col in scales],
                        *[f"{col}_scale" for col in scales],
                    ),
                    on=self.scaler.by,
                    how="left",
                )
                .with_columns(
                    [
                        (pl.col(col) * pl.col(f"{self.mapping[col]}_scale"))
                        + pl.col(f"{self.mapping[col]}_move")
                        for col in columns
                    ]
                )
                .select(data.columns)
            )
        else:
            move_scale = self.scaler.move_scale.row(0, named=True)
            return data.with_columns(
                [
                    (pl.col(col) * move_scale[f"{self.mapping[col]}_scale"])
                    + move_scale[f"{self.mapping[col]}_move"]
                    for col in columns
                ]
            )


class InverseScalerContext:
    def __init__(
        self,
        pipeline: "Pipeline",
        scaler: Scaler,
        mapping: Mapping[str, str] | None = None,
        *,
        component_name: str | None = None,
    ):
        self.pipeline = pipeline
        self.scaler = scaler
        self.mapping = mapping
        self.component_name = component_name

    def __enter__(self) -> "Pipeline":
        self.pipeline.pipe(self.scaler, component_name=self.component_name)
        return self.pipeline

    def __exit__(self, *args: Any, **kwargs: Any):
        self.pipeline.pipe(
            InverseScaler(self.scaler, mapping=self.mapping),
            component_name=self.component_name + "_inverse"
            if self.component_name
            else None,
        )
