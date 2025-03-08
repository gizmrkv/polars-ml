from abc import ABC, abstractmethod
from typing import Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr

from polars_ml.pipeline.component import PipelineComponent


class BaseScaler(PipelineComponent, ABC):
    def __init__(self, *column: str, by: str | Sequence[str] | None = None):
        self.columns = column
        self.by = [by] if isinstance(by, str) else list(by) if by is not None else None

    @abstractmethod
    def move_expr(self, column: str) -> Expr: ...

    @abstractmethod
    def scale_expr(self, column: str) -> Expr: ...

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

    def inverse_transform(self, data: DataFrame) -> DataFrame:
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
                        (pl.col(col) * pl.col(f"{col}_scale")) + pl.col(f"{col}_move")
                        for col in columns
                    ]
                )
                .select(data.columns)
            )
        else:
            move_scale = self.move_scale.row(0, named=True)
            return data.with_columns(
                [
                    (pl.col(col) * move_scale[f"{col}_scale"])
                    + move_scale[f"{col}_move"]
                    for col in columns
                ]
            )


class InverseScaler(PipelineComponent):
    def __init__(self, scaler: BaseScaler, mapping: Mapping[str, str] | None = None):
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


class StandardScaler(BaseScaler):
    def move_expr(self, column: str) -> Expr:
        return pl.col(column).mean()

    def scale_expr(self, column: str) -> Expr:
        return pl.col(column).std()


class MinMaxScaler(BaseScaler):
    def move_expr(self, column: str) -> Expr:
        return pl.col(column).min()

    def scale_expr(self, column: str) -> Expr:
        return pl.col(column).max() - pl.col(column).min()


class RobustScaler(BaseScaler):
    def __init__(
        self,
        *column: str,
        by: str | Sequence[str] | None = None,
        quantile: tuple[float, float] = (0.25, 0.75),
    ):
        super().__init__(*column, by=by)
        q1, q3 = quantile
        assert 0 <= q1 < q3 <= 1, (
            f"Quantile values must be in the range [0, 1] and q1 < q3. Got {quantile}"
        )
        self.q1 = q1
        self.q3 = q3

    def move_expr(self, column: str) -> Expr:
        return pl.col(column).median()

    def scale_expr(self, column: str) -> Expr:
        return pl.col(column).quantile(self.q3) - pl.col(column).quantile(self.q1)
