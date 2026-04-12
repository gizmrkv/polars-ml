from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Protocol, Self, Sequence

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
        on_args: dict[str, Any] = (
            {"on": self._by}
            if self._by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        return data.update(
            data.select(*power_columns, *self._by)
            .join(
                self._maxlog.lazy().select(
                    *self._by,
                    *[pl.col(f"{c}_maxlog") for c in power_columns],
                ),
                how="left",
                **on_args,
            )
            .with_columns(self.power_expr(c, f"{c}_maxlog") for c in power_columns)
            .drop(f"{c}_maxlog" for c in power_columns),
            include_nulls=True,
        )


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


class PowerTransformInverse(LazyTransformer):
    def __init__(
        self,
        power_transform: BasePowerTransform,
        mapping: Mapping[str, str] | None = None,
    ):
        self._power_transform = power_transform
        self._mapping = mapping

    @property
    def mapping(self) -> Mapping[str, str]:
        if self._mapping is not None:
            return self._mapping
        return {col: col for col in self._power_transform.columns}

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        input_columns = data.collect_schema().names()
        sources = set(input_columns) & set(self.mapping.values())
        on_args: dict[str, Any] = (
            {"on": self._power_transform.by}
            if self._power_transform.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        data = data.join(
            self._power_transform.maxlog.lazy().select(
                *self._power_transform.by,
                *[pl.col(f"{c}_maxlog").alias(f"{c}_maxlog") for c in sources],
            ),
            how="left",
            **on_args,
        )
        return data.with_columns(
            self._power_transform.power_inv_expr(s, f"{s}_maxlog") for s in sources
        ).drop(f"{s}_maxlog" for s in sources)


class Pipeline(Protocol):
    def pipe(self, step: LazyTransformer) -> Self: ...


class PowerTransformInverseContext:
    def __init__(
        self,
        pipeline: Pipeline,
        power_transform: BasePowerTransform,
        mapping: Mapping[str, str] | None = None,
    ):
        self._pipeline = pipeline
        self._power_transform = power_transform
        self._power_transform_inverse = PowerTransformInverse(
            self._power_transform, mapping=mapping
        )
        self._mapping = mapping

    def __enter__(self) -> tuple[BasePowerTransform, PowerTransformInverse]:
        self._pipeline.pipe(self._power_transform)
        return self._power_transform, self._power_transform_inverse

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._pipeline.pipe(self._power_transform_inverse)


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
