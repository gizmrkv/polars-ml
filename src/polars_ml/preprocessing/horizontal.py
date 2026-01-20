from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from polars_ml import Pipeline

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import Transformer


class HorizontalAgg(Transformer):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_agg",
        variable_name: str | None = None,
        maintain_order: bool = True,
        aggs: Iterable[IntoExpr | Iterable[IntoExpr]] | None = None,
        named_aggs: Mapping[str, IntoExpr] | None = None,
    ):
        self.exprs = expr
        self.more_exprs = more_expr
        self.value_name = value_name
        self.variable_name = variable_name or uuid.uuid4().hex
        self.is_variable_none = variable_name is None
        self.maintain_order = maintain_order
        self.aggs = aggs or []
        self.named_aggs = named_aggs or {}
        self.index_name = uuid.uuid4().hex

    def transform(self, data: DataFrame) -> DataFrame:
        return (
            data.with_row_index(self.index_name)
            .join(
                data.select(self.exprs, *self.more_exprs)
                .with_row_index(self.index_name)
                .unpivot(
                    ~cs.by_name(self.index_name),
                    index=self.index_name,
                    value_name=self.value_name,
                    variable_name=self.variable_name,
                )
                .select(
                    pl.exclude(self.variable_name)
                    if self.is_variable_none
                    else pl.all()
                )
                .group_by(self.index_name, maintain_order=self.maintain_order)
                .agg(*self.aggs, **self.named_aggs),
                on=self.index_name,
            )
            .drop(self.index_name)
        )


class HorizontalAll(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_all",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().all()],
            maintain_order=maintain_order,
        )


class HorizontalCount(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_count",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().count()],
            maintain_order=maintain_order,
        )


class HorizontalMax(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_max",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().max()],
            maintain_order=maintain_order,
        )


class HorizontalMean(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_mean",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().mean()],
            maintain_order=maintain_order,
        )


class HorizontalMedian(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_median",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().median()],
            maintain_order=maintain_order,
        )


class HorizontalMin(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_min",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().min()],
            maintain_order=maintain_order,
        )


class HorizontalNUnique(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_n_unique",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().n_unique()],
            maintain_order=maintain_order,
        )


class HorizontalQuantile(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        quantile: float,
        value_name: str = "horizontal_quantile",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().quantile(quantile)],
            maintain_order=maintain_order,
        )


class HorizontalStd(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_std",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().std()],
            maintain_order=maintain_order,
        )


class HorizontalSum(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_sum",
        maintain_order: bool = True,
    ):
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            aggs=[pl.all().sum()],
            maintain_order=maintain_order,
        )


class HorizontalArgMax(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmax",
        maintain_order: bool = True,
    ):
        self.variable_name = uuid.uuid4().hex
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            variable_name=self.variable_name,
            maintain_order=maintain_order,
            aggs=[
                pl.struct(value_name, self.variable_name).filter(
                    pl.col(value_name) == pl.col(value_name).max()
                )
            ],
        )

    def transform(self, data: DataFrame) -> DataFrame:
        return (
            super()
            .transform(data)
            .with_columns(
                pl.col(self.value_name).list.eval(
                    pl.element().struct.field(self.variable_name)
                )
            )
        )


class HorizontalArgMin(HorizontalAgg):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmin",
        maintain_order: bool = True,
    ):
        self.variable_name = uuid.uuid4().hex
        super().__init__(
            expr,
            *more_expr,
            value_name=value_name,
            variable_name=self.variable_name,
            maintain_order=maintain_order,
            aggs=[
                pl.struct(value_name, self.variable_name).filter(
                    pl.col(value_name) == pl.col(value_name).min()
                )
            ],
        )

    def transform(self, data: DataFrame) -> DataFrame:
        return (
            super()
            .transform(data)
            .with_columns(
                pl.col(self.value_name).list.eval(
                    pl.element().struct.field(self.variable_name)
                )
            )
        )


class HorizontalNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN HorizontalNameSpace

    def all(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_all",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalAll(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def arg_max(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmax",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalArgMax(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def arg_min(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmin",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalArgMin(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def count(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_count",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalCount(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def max(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_max",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalMax(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def mean(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_mean",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalMean(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def median(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_median",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalMedian(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def min(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_min",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalMin(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def n_unique(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_n_unique",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalNUnique(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def quantile(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        quantile: float,
        value_name: str = "horizontal_quantile",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalQuantile(
                expr,
                *more_expr,
                quantile=quantile,
                value_name=value_name,
                maintain_order=maintain_order,
            )
        )

    def std(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_std",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalStd(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def sum(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_sum",
        maintain_order: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            HorizontalSum(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    # --- END INSERTION MARKER IN HorizontalNameSpace
