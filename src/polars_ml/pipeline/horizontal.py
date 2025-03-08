import uuid
from typing import Iterable, Mapping

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent


class HorizontalAgg(PipelineComponent):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_agg",
        variable_name: str | None = None,
        maintain_order: bool = False,
        aggs: Iterable[IntoExpr | Iterable[IntoExpr]] | None = None,
        named_aggs: Mapping[str, IntoExpr] | None = None,
    ):
        self.exprs = expr
        self.value_name = value_name
        self.variable_name = variable_name
        self.maintain_order = maintain_order
        self.aggs = aggs or []
        self.named_aggs = named_aggs or {}

    def transform(self, data: DataFrame) -> DataFrame:
        index_name = uuid.uuid4().hex
        data = data.with_row_index(index_name)
        return data.join(
            data.select(*self.exprs, index_name)
            .unpivot(
                ~cs.by_name(index_name),
                index=index_name,
                value_name=self.value_name,
                variable_name=self.variable_name,
            )
            .group_by(index_name, maintain_order=self.maintain_order)
            .agg(*self.aggs, **self.named_aggs),
            on=index_name,
        ).drop(index_name)


class HorizontalAll(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_all",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().all()],
        )


class HorizontalCount(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_count",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().count()],
        )


class HorizontalMax(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_max",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().max()],
        )


class HorizontalMean(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_mean",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().mean()],
        )


class HorizontalMedian(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_median",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().median()],
        )


class HorizontalMin(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_min",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().min()],
        )


class HorizontalNUnique(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_n_unique",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().n_unique()],
        )


class HorizontalQuantile(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        quantile: float,
        value_name: str = "horizontal_quantile",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().quantile(quantile)],
        )


class HorizontalSum(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_sum",
        maintain_order: bool = False,
    ):
        super().__init__(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().sum()],
        )


class HorizontalArgMax(HorizontalAgg):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmax",
        maintain_order: bool = False,
    ):
        self.variable_name = uuid.uuid4().hex
        super().__init__(
            *expr,
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
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmin",
        maintain_order: bool = False,
    ):
        self.variable_name = uuid.uuid4().hex
        super().__init__(
            *expr,
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
