from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Self,
    Sequence,
)

import polars as pl
from polars import DataFrame, Expr, Series
from polars._typing import ColumnNameOrSelector
from scipy import stats
from shortuuid import ShortUUID

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    from polars_ml import Pipeline


class BasePowerTransform(Transformer, ABC):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> None:
        self.column_selectors = columns
        self.more_column_selectors = more_columns
        self.by = [by] if isinstance(by, str) else [*by] if by is not None else []
        self.suffix = suffix

    @abstractmethod
    def calc_maxlog(self, values: Series) -> float: ...

    @abstractmethod
    def power_expr(self, column: str, maxlog: str) -> Expr: ...

    @abstractmethod
    def power_inv_expr(self, column: str, maxlog: str) -> Expr: ...

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        data = data.select(self.column_selectors, *self.more_column_selectors, *self.by)
        self.columns = [c for c in data.columns if c not in self.by]
        exprs = [
            pl.col(column)
            .map_batches(self.calc_maxlog, return_dtype=pl.Float64, returns_scalar=True)
            .alias(f"{column}_maxlog")
            for column in self.columns
        ]
        if self.by:
            self.maxlog = data.group_by(self.by).agg(*exprs)
        else:
            self.maxlog = data.select(*exprs)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, "maxlog"):
            raise NotFittedError()

        input_columns = data.collect_schema().names()
        power_columns = set(input_columns) & set(self.columns)
        on_args: dict[str, Any] = (
            {"on": self.by}
            if self.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        tmp_suf = ShortUUID().random(length=8)
        return (
            data.join(
                self.maxlog.select(
                    *self.by,
                    *[
                        pl.col(f"{c}_maxlog").name.suffix(f"_{tmp_suf}")
                        for c in power_columns
                    ],
                ),
                how="left",
                **on_args,
            )
            .with_columns(
                self.power_expr(c, f"{c}_maxlog_{tmp_suf}").alias(f"{c}{self.suffix}")
                for c in power_columns
            )
            .drop(*[f"{c}_maxlog_{tmp_suf}" for c in power_columns])
        )


class BoxCoxTransform(BasePowerTransform):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> None:
        super().__init__(columns, *more_columns, by=by, suffix=suffix)

    def calc_maxlog(self, values: Series) -> float:
        return float(stats.boxcox(values.drop_nulls().to_numpy())[1])

    def power_expr(self, column: str, maxlog: str) -> Expr:
        return boxcox(pl.col(column), pl.col(maxlog))

    def power_inv_expr(self, column: str, maxlog: str) -> Expr:
        return boxcox_inv(pl.col(column), pl.col(maxlog))


class YeoJohnsonTransform(BasePowerTransform):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> None:
        super().__init__(columns, *more_columns, by=by, suffix=suffix)

    def calc_maxlog(self, values: Series) -> float:
        return float(stats.yeojohnson(values.drop_nulls().to_numpy())[1])  # type: ignore

    def power_expr(self, column: str, maxlog: str) -> Expr:
        return yeojohnson(pl.col(column), pl.col(maxlog))

    def power_inv_expr(self, column: str, maxlog: str) -> Expr:
        return yeojohnson_inv(pl.col(column), pl.col(maxlog))


class PowerTransformInverse(Transformer):
    def __init__(
        self,
        power_transform: BasePowerTransform,
        mapping: Mapping[str, str] | None = None,
    ) -> None:
        self.power_transform = power_transform
        self._mapping = mapping

    @property
    def mapping(self) -> Mapping[str, str]:
        if self._mapping is not None:
            return self._mapping
        return {col: col for col in self.power_transform.columns}

    def transform(self, data: DataFrame) -> DataFrame:
        if not hasattr(self.power_transform, "maxlog"):
            raise NotFittedError()

        input_columns = data.collect_schema().names()
        sources = set(self.power_transform.columns) & set(self.mapping.values())
        on_args: dict[str, Any] = (
            {"on": self.power_transform.by}
            if self.power_transform.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        tmp_suf = ShortUUID().random(length=8)
        data = data.join(
            self.power_transform.maxlog.select(
                *self.power_transform.by,
                *[
                    pl.col(f"{c}_maxlog").alias(f"{c}_maxlog_{tmp_suf}")
                    for c in sources
                ],
            ),
            how="left",
            **on_args,
        )
        return data.with_columns(
            self.power_transform.power_inv_expr(t, f"{s}_maxlog_{tmp_suf}").alias(t)
            for t, s in self.mapping.items()
            if t in input_columns
        ).drop(*(f"{s}_maxlog_{tmp_suf}" for s in sources))


class PowerTransformInverseContext:
    def __init__(
        self,
        pipeline: Pipeline,
        power_transform: BasePowerTransform,
        mapping: Mapping[str, str] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.power_transform = power_transform
        self.mapping = mapping

    def __enter__(self) -> Pipeline:
        return self.pipeline.pipe(self.power_transform)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.pipeline.pipe(
            PowerTransformInverse(self.power_transform, mapping=self.mapping),
        )


def boxcox_maxlog(x: Series) -> float:
    return float(stats.boxcox(x.drop_nulls().to_numpy())[1])


def boxcox(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(lmbda != 0).then((x**lmbda - 1) / lmbda).otherwise(x.log())


def boxcox_inv(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(lmbda != 0).then((x * lmbda + 1) ** (1 / lmbda)).otherwise(x.exp())


def yeojohnson_maxlog(x: Series) -> float:
    return float(stats.yeojohnson(x.drop_nulls().to_numpy())[1])  # type: ignore


def yeojohnson(x: Expr, lmbda: Expr) -> Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then(((x + 1) ** lmbda - 1) / lmbda)
        .when((x >= 0) & (lmbda == 0))
        .then((x + 1).log())
        .when((x < 0) & (lmbda != 2))
        .then(((-x + 1) ** (2 - lmbda) - 1) / (lmbda - 2))
        .otherwise(-(-x + 1).log())
    )


def yeojohnson_inv(x: Expr, lmbda: Expr) -> Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then((x * lmbda + 1) ** (1 / lmbda) - 1)
        .when((x >= 0) & (lmbda == 0))
        .then(x.exp() - 1)
        .when((x < 0) & (lmbda != 2))
        .then(1 - (x * lmbda - 2 * x + 1) ** (1 / (2 - lmbda)))
        .otherwise(1 - (x.exp() - 1))
    )
