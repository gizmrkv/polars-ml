from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Self, Sequence

import polars as pl
from polars._typing import ColumnNameOrSelector

from polars_ml.base import LazyTransformer
from polars_ml.exceptions import NotFittedError


class BaseScale(LazyTransformer, ABC):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ):
        self._selector = columns
        self._more_selectors = more_columns
        self._by = [by] if isinstance(by, str) else [*by] if by is not None else []

        self._columns: list[str] | None = None
        self._stats: pl.DataFrame | None = None

    @property
    def by(self) -> list[str]:
        return self._by

    @property
    def columns(self) -> list[str]:
        if self._columns is None:
            raise NotFittedError()
        return self._columns

    @property
    def stats(self) -> pl.DataFrame:
        if self._stats is None:
            raise NotFittedError()
        return self._stats

    @abstractmethod
    def loc_expr(self, column: str) -> pl.Expr: ...

    @abstractmethod
    def scale_expr(self, column: str) -> pl.Expr: ...

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        data = data.select(self._selector, *self._more_selectors, *self._by)
        self._columns = [c for c in data.columns if c not in self._by]
        exprs = [
            e
            for c in self._columns
            for e in [
                self.loc_expr(c).alias(f"{c}_loc"),
                self.scale_expr(c).alias(f"{c}_scale"),
            ]
        ]
        self._stats = (
            data.group_by(self._by).agg(*exprs) if self._by else data.select(*exprs)
        )
        return self

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self._columns is None or self._stats is None:
            raise NotFittedError()

        input_columns = data.collect_schema().names()
        scale_columns = set(input_columns) & set(self._columns)
        on_args: dict[str, Any] = {"on": self._by if self._by else pl.lit(0)}
        return pl.concat(
            [
                data.drop(*scale_columns),
                data.select(*scale_columns, *self._by)
                .join(
                    self._stats.lazy().select(
                        *self._by,
                        *[pl.col(f"{t}_loc") for t in scale_columns],
                        *[pl.col(f"{t}_scale") for t in scale_columns],
                    ),
                    how="left",
                    **on_args,
                    suffix="",
                )
                .select(
                    (pl.col(c) - pl.col(f"{c}_loc")) / pl.col(f"{c}_scale")
                    for c in scale_columns
                ),
            ],
            how="horizontal",
        ).select(*input_columns)


class StandardScale(BaseScale):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ):
        super().__init__(columns, *more_columns, by=by)

    def loc_expr(self, column: str) -> pl.Expr:
        return pl.col(column).mean()

    def scale_expr(self, column: str) -> pl.Expr:
        return pl.col(column).std()


class MinMaxScale(BaseScale):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ):
        super().__init__(columns, *more_columns, by=by)

    def loc_expr(self, column: str) -> pl.Expr:
        return pl.col(column).min()

    def scale_expr(self, column: str) -> pl.Expr:
        return pl.col(column).max() - pl.col(column).min()


class RobustScale(BaseScale):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
    ):
        super().__init__(columns, *more_columns, by=by)
        self._q_lower, self._q_upper = quantile_range

        if not (0 <= self._q_lower < self._q_upper <= 1):
            raise ValueError(
                "quantile_range must be a tuple of (lower, upper) quantiles in [0, 1]"
            )

    def loc_expr(self, column: str) -> pl.Expr:
        return pl.col(column).median()

    def scale_expr(self, column: str) -> pl.Expr:
        return pl.col(column).quantile(self._q_upper) - pl.col(column).quantile(
            self._q_lower
        )
