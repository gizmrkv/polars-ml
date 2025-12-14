from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from polars._typing import ColumnNameOrSelector
from shortuuid import ShortUUID

from polars_ml.base import Transformer

if TYPE_CHECKING:
    from polars_ml import Pipeline


class BaseScale(Transformer, ABC):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ):
        self.column_selectors = columns
        self.more_column_selectors = more_columns
        self.by = [by] if isinstance(by, str) else [*by] if by is not None else []
        self.suffix = suffix

    @abstractmethod
    def loc_expr(self, column: str) -> Expr: ...

    @abstractmethod
    def scale_expr(self, column: str) -> Expr: ...

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        data = data.select(self.column_selectors, *self.more_column_selectors, *self.by)
        self.columns = [c for c in data.columns if c not in self.by]
        exprs = [
            e
            for c in self.columns
            for e in [
                self.loc_expr(c).alias(f"{c}_loc"),
                self.scale_expr(c).alias(f"{c}_scale"),
            ]
        ]
        self.stats = (
            data.group_by(self.by).agg(*exprs) if self.by else data.select(*exprs)
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_columns = data.collect_schema().names()
        scale_columns = set(input_columns) & set(self.columns)
        on_args: dict[str, Any] = (
            {"on": self.by}
            if self.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        tmp_suf = ShortUUID().random(length=8)
        return (
            data.join(
                self.stats.select(
                    *self.by,
                    *[f"{t}_loc_{tmp_suf}" for t in scale_columns],
                    *[f"{t}_scale_{tmp_suf}" for t in scale_columns],
                ),
                how="left",
                **on_args,
            )
            .with_columns(
                (
                    (pl.col(c) - pl.col(f"{c}_loc_{tmp_suf}"))
                    / pl.col(f"{c}_scale_{tmp_suf}")
                ).name.suffix(self.suffix)
                for c in scale_columns
            )
            .drop(
                *[f"{c}_loc_{tmp_suf}" for c in scale_columns],
                *[f"{c}_scale_{tmp_suf}" for c in scale_columns],
            )
        )


class StandardScale(BaseScale):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ):
        super().__init__(columns, *more_columns, by=by, suffix=suffix)

    def loc_expr(self, column: str) -> Expr:
        return pl.col(column).mean()

    def scale_expr(self, column: str) -> Expr:
        return pl.col(column).std()


class MinMaxScale(BaseScale):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ):
        super().__init__(columns, *more_columns, by=by, suffix=suffix)

    def loc_expr(self, column: str) -> Expr:
        return pl.col(column).min()

    def scale_expr(self, column: str) -> Expr:
        return pl.col(column).max() - pl.col(column).min()


class RobustScale(BaseScale):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        suffix: str = "",
    ):
        super().__init__(columns, *more_columns, by=by, suffix=suffix)
        self.q_lower, self.q_upper = quantile_range

        if not (0 <= self.q_lower < self.q_upper <= 1):
            raise ValueError(
                "quantile_range must be a tuple of (lower, upper) quantiles in [0, 1]"
            )

    def loc_expr(self, column: str) -> Expr:
        return pl.col(column).median()

    def scale_expr(self, column: str) -> Expr:
        return pl.col(column).quantile(self.q_upper) - pl.col(column).quantile(
            self.q_lower
        )


class ScaleInverse(Transformer):
    def __init__(self, scale: BaseScale, mapping: Mapping[str, str] | None = None):
        self.scale = scale
        self.mapping = mapping or {col: col for col in scale.columns}

    def transform(self, data: DataFrame) -> DataFrame:
        input_columns = data.collect_schema().names()
        targets = set(input_columns) & set(self.mapping.keys())
        sources = set(self.scale.columns) & set(self.mapping.values())
        on_args: dict[str, Any] = (
            {"on": self.scale.by}
            if self.scale.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        tmp_suf = ShortUUID().random(length=8)
        return (
            data.join(
                self.scale.stats.select(
                    *self.scale.by,
                    *[pl.col(f"{col}_loc_{tmp_suf}") for col in sources],
                    *[pl.col(f"{col}_scale_{tmp_suf}") for col in sources],
                ),
                how="left",
                **on_args,
            )
            .with_columns(
                (pl.col(t) * pl.col(f"{s}_scale_{tmp_suf}"))
                + pl.col(f"{s}_loc_{tmp_suf}")
                for t, s in [(t, s) for t, s in self.mapping.items() if t in targets]
            )
            .select(input_columns)
        )


class ScaleInverseContext:
    def __init__(
        self,
        pipeline: "Pipeline",
        scale: BaseScale,
        mapping: Mapping[str, str] | None = None,
    ):
        self.pipeline = pipeline
        self.scale = scale
        self.mapping = mapping

    def __enter__(self) -> Pipeline:
        return self.pipeline.pipe(self.scale)

    def __exit__(self):
        self.pipeline.pipe(ScaleInverse(self.scale, mapping=self.mapping))
