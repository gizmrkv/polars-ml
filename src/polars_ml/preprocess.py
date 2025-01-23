import itertools
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars import DataFrame, Expr, Series
from polars._typing import ColumnNameOrSelector, IntoExpr

from polars_ml import Component


class BaseScaler(Component, ABC):
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
                *[pl.col(col).mean().alias(f"{col}_move") for col in self.columns],
                *[pl.col(col).std().alias(f"{col}_scale") for col in self.columns],
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


class InverseScaler(Component):
    def __init__(self, scaler: BaseScaler, mapping: dict[str, str]):
        self.scaler = scaler
        self.mapping = mapping

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


class Binning(Component):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_bin",
    ):
        self.exprs = exprs
        self.quantiles = quantiles
        self.labels = labels
        self.left_closed = left_closed
        self.allow_duplicates = allow_duplicates
        self.suffix = suffix

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        data = data.select(*self.exprs)
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
        return data.with_columns(
            [
                pl.col(col)
                .cut(
                    breaks,
                    labels=self.labels,
                    left_closed=self.left_closed,
                    include_breaks=False,
                )
                .alias(f"{col}{self.suffix}")
                for col, breaks in self.breakpoints.items()
                if col in data.columns
            ]
        )


class Polynomial(Component):
    def __init__(
        self,
        *features: ColumnNameOrSelector,
        degree: int = 2,
    ):
        self.features = features
        self.degree = degree

    def transform(self, data: DataFrame) -> DataFrame:
        columns = data.lazy().select(*self.features).collect_schema().names()
        for a, b in itertools.combinations_with_replacement(columns, r=self.degree):
            data = data.with_columns((pl.col(a) * pl.col(b)).alias(f"{a} * {b}"))

        return data
