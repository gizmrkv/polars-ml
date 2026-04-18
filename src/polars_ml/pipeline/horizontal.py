from __future__ import annotations

import uuid
from typing import Iterable, Mapping

import polars as pl
import polars.selectors as cs
from polars._typing import IntoExpr

from polars_ml.base import LazyTransformer


class HorizontalAgg(LazyTransformer):
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
        self._exprs = expr
        self._more_exprs = more_expr
        self._value_name = value_name
        self._variable_name = variable_name or uuid.uuid4().hex
        self._is_variable_none = variable_name is None
        self._maintain_order = maintain_order
        self._aggs = aggs or []
        self._named_aggs = named_aggs or {}
        self._index_name = uuid.uuid4().hex

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return (
            data.with_row_index(self._index_name)
            .join(
                data.select(self._exprs, *self._more_exprs)
                .with_row_index(self._index_name)
                .unpivot(
                    ~cs.by_name(self._index_name),
                    index=self._index_name,
                    value_name=self._value_name,
                    variable_name=self._variable_name,
                )
                .select(
                    pl.exclude(self._variable_name)
                    if self._is_variable_none
                    else pl.all()
                )
                .group_by(self._index_name, maintain_order=self._maintain_order)
                .agg(*self._aggs, **self._named_aggs),
                on=self._index_name,
                how="left",
            )
            .drop(self._index_name)
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

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return (
            super()
            .transform(data)
            .with_columns(
                pl.col(self._value_name).list.eval(
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

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return (
            super()
            .transform(data)
            .with_columns(
                pl.col(self._value_name).list.eval(
                    pl.element().struct.field(self.variable_name)
                )
            )
        )
