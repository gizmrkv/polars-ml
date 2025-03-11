import uuid
from typing import TYPE_CHECKING, Iterable, Mapping

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from polars_ml import Pipeline


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
                data.select(*self.exprs)
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


class HorizontalNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def agg(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_agg",
        variable_name: str | None = None,
        maintain_order: bool = False,
        aggs: Iterable[IntoExpr | Iterable[IntoExpr]] | None = None,
        named_aggs: Mapping[str, IntoExpr] | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            HorizontalAgg(
                *expr,
                value_name=value_name,
                variable_name=variable_name,
                maintain_order=maintain_order,
                aggs=aggs,
                named_aggs=named_aggs,
            ),
            component_name=component_name,
        )

    def all(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_all",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().all()],
            component_name=component_name,
        )

    def count(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_count",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().count()],
            component_name=component_name,
        )

    def max(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_max",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().max()],
            component_name=component_name,
        )

    def mean(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_mean",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().mean()],
            component_name=component_name,
        )

    def median(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_median",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().median()],
            component_name=component_name,
        )

    def min(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_min",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().min()],
            component_name=component_name,
        )

    def n_unique(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_n_unique",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().n_unique()],
            component_name=component_name,
        )

    def quantile(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        quantile: float,
        value_name: str = "horizontal_quantile",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().quantile(quantile)],
            component_name=component_name,
        )

    def std(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_std",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().std()],
            component_name=component_name,
        )

    def sum(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_sum",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.agg(
            *expr,
            value_name=value_name,
            maintain_order=maintain_order,
            aggs=[pl.all().sum()],
            component_name=component_name,
        )

    def argmax(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmax",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            HorizontalArgMax(
                *expr,
                value_name=value_name,
                maintain_order=maintain_order,
            ),
            component_name=component_name,
        )

    def argmin(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmin",
        maintain_order: bool = False,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            HorizontalArgMin(
                *expr,
                value_name=value_name,
                maintain_order=maintain_order,
            ),
            component_name=component_name,
        )
