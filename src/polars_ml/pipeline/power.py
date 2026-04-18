from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Self, Sequence

import polars as pl
from polars._typing import ColumnNameOrSelector
from scipy import stats

from polars_ml.base import LazyTransformer
from polars_ml.exceptions import NotFittedError


class BasePowerTransform(LazyTransformer, ABC):
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
        self._maxlog: pl.DataFrame | None = None

    @property
    def by(self) -> list[str]:
        return self._by

    @property
    def columns(self) -> list[str]:
        if self._columns is None:
            raise NotFittedError()
        return self._columns

    @property
    def maxlog(self) -> pl.DataFrame:
        if self._maxlog is None:
            raise NotFittedError()
        return self._maxlog

    @abstractmethod
    def calc_maxlog(self, values: pl.Series) -> float: ...

    @abstractmethod
    def power_expr(self, column: str, maxlog: str) -> pl.Expr: ...

    @abstractmethod
    def power_inv_expr(self, column: str, maxlog: str) -> pl.Expr: ...

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        data = data.select(self._selector, *self._more_selectors, *self._by)
        self._columns = [c for c in data.columns if c not in self._by]
        exprs = [
            pl.col(column)
            .map_batches(self.calc_maxlog, return_dtype=pl.Float64, returns_scalar=True)
            .alias(f"{column}_maxlog")
            for column in self._columns
        ]
        if self._by:
            self._maxlog = data.group_by(self._by).agg(*exprs)
        else:
            self._maxlog = data.select(*exprs)
        return self

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self._columns is None or self._maxlog is None:
            raise NotFittedError()

        input_columns = data.collect_schema().names()
        power_columns = set(input_columns) & set(self._columns)
        on_args: dict[str, Any] = {"on": self._by if self._by else pl.lit(0)}
        return pl.concat(
            [
                data.drop(*power_columns),
                data.select(*power_columns, *self._by)
                .join(
                    self._maxlog.lazy().select(
                        *self._by,
                        *[pl.col(f"{c}_maxlog") for c in power_columns],
                    ),
                    how="left",
                    **on_args,
                    suffix="",
                )
                .select(self.power_expr(c, f"{c}_maxlog") for c in power_columns),
            ],
            how="horizontal",
        ).select(*input_columns)


class BoxCoxTransform(BasePowerTransform):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ):
        super().__init__(columns, *more_columns, by=by)

    def calc_maxlog(self, values: pl.Series) -> float:
        return float(stats.boxcox(values.drop_nulls().to_numpy())[1])

    def power_expr(self, column: str, maxlog: str) -> pl.Expr:
        return boxcox(pl.col(column), pl.col(maxlog))

    def power_inv_expr(self, column: str, maxlog: str) -> pl.Expr:
        return boxcox_inv(pl.col(column), pl.col(maxlog))


class YeoJohnsonTransform(BasePowerTransform):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ):
        super().__init__(columns, *more_columns, by=by)

    def calc_maxlog(self, values: pl.Series) -> float:
        return float(stats.yeojohnson(values.drop_nulls().to_numpy())[1])  # type: ignore

    def power_expr(self, column: str, maxlog: str) -> pl.Expr:
        return yeojohnson(pl.col(column), pl.col(maxlog))

    def power_inv_expr(self, column: str, maxlog: str) -> pl.Expr:
        return yeojohnson_inv(pl.col(column), pl.col(maxlog))


def boxcox_maxlog(x: pl.Series) -> float:
    return float(stats.boxcox(x.drop_nulls().to_numpy())[1])


def boxcox(x: pl.Expr, lmbda: pl.Expr) -> pl.Expr:
    return pl.when(lmbda != 0).then((x**lmbda - 1) / lmbda).otherwise(x.log())


def boxcox_inv(x: pl.Expr, lmbda: pl.Expr) -> pl.Expr:
    return pl.when(lmbda != 0).then((x * lmbda + 1) ** (1 / lmbda)).otherwise(x.exp())


def yeojohnson_maxlog(x: pl.Series) -> float:
    return float(stats.yeojohnson(x.drop_nulls().to_numpy())[1])  # type: ignore


def yeojohnson(x: pl.Expr, lmbda: pl.Expr) -> pl.Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then(((x + 1) ** lmbda - 1) / lmbda)
        .when((x >= 0) & (lmbda == 0))
        .then((x + 1).log())
        .when((x < 0) & (lmbda != 2))
        .then(((-x + 1) ** (2 - lmbda) - 1) / (lmbda - 2))
        .otherwise(-(-x + 1).log())
    )


def yeojohnson_inv(x: pl.Expr, lmbda: pl.Expr) -> pl.Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then((x * lmbda + 1) ** (1 / lmbda) - 1)
        .when((x >= 0) & (lmbda == 0))
        .then(x.exp() - 1)
        .when((x < 0) & (lmbda != 2))
        .then(1 - (x * lmbda - 2 * x + 1) ** (1 / (2 - lmbda)))
        .otherwise(1 - (x.exp() - 1))
    )
