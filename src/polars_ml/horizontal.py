import uuid
from typing import Iterable, Mapping

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.component import Component


class HorizontalAgg(Component):
    def __init__(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_agg",
        variable_name: str | None = None,
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
        maintain_order: bool = False,
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
