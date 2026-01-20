from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from polars import DataFrame
from polars._typing import IntoExpr, QuantileMethod, SchemaDict

from polars_ml.base import Transformer

if TYPE_CHECKING:
    from polars_ml import Pipeline


class GroupByGetAttr(Transformer):
    def __init__(
        self,
        group_by_attr: str,
        agg_attr: str,
        group_by_args: tuple[Any, ...] | None = None,
        group_by_kwargs: Mapping[str, Any] | None = None,
        *agg_args: Any,
        **agg_kwargs: Any,
    ):
        self.group_by_attr = group_by_attr
        self.group_by_args = group_by_args or ()
        self.group_by_kwargs = group_by_kwargs or {}
        self.agg_attr = agg_attr
        self.agg_args = agg_args or ()
        self.agg_kwargs = agg_kwargs or {}

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(
            getattr(data, self.group_by_attr)(
                *self.group_by_args, **self.group_by_kwargs
            ),
            self.agg_attr,
        )(*self.agg_args, **self.agg_kwargs)


class GroupByNameSpace:
    def __init__(self, pipeline: Pipeline, attr: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "agg", self.args, self.kwargs, *aggs, **named_aggs
            )
        )

    def all(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "all",
                self.args,
                self.kwargs,
            )
        )

    def count(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "count",
                self.args,
                self.kwargs,
            )
        )

    def first(self, ignore_nulls: bool = False) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "first", self.args, self.kwargs, ignore_nulls=ignore_nulls
            )
        )

    def head(self, n: int = 5) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(self.attr, "head", self.args, self.kwargs, n)
        )

    def last(self, ignore_nulls: bool = False) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "last", self.args, self.kwargs, ignore_nulls=ignore_nulls
            )
        )

    def len(self, name: str | None = None) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(self.attr, "len", self.args, self.kwargs, name)
        )

    def map_groups(self, function: Callable[[DataFrame], DataFrame]) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(self.attr, "map_groups", self.args, self.kwargs, function)
        )

    def max(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "max",
                self.args,
                self.kwargs,
            )
        )

    def mean(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "mean",
                self.args,
                self.kwargs,
            )
        )

    def median(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "median",
                self.args,
                self.kwargs,
            )
        )

    def min(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "min",
                self.args,
                self.kwargs,
            )
        )

    def n_unique(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "n_unique",
                self.args,
                self.kwargs,
            )
        )

    def quantile(
        self, quantile: float, interpolation: QuantileMethod = "nearest"
    ) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "quantile", self.args, self.kwargs, quantile, interpolation
            )
        )

    def sum(self) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr,
                "sum",
                self.args,
                self.kwargs,
            )
        )

    def tail(self, n: int = 5) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(self.attr, "tail", self.args, self.kwargs, n)
        )


class DynamicGroupByNameSpace:
    def __init__(self, pipeline: Pipeline, attr: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "agg", self.args, self.kwargs, *aggs, **named_aggs
            )
        )

    def map_groups(
        self, function: Callable[[DataFrame], DataFrame], schema: SchemaDict | None
    ) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "map_groups", self.args, self.kwargs, function, schema
            )
        )


class RollingGroupByNameSpace:
    def __init__(self, pipeline: Pipeline, attr: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "agg", self.args, self.kwargs, *aggs, **named_aggs
            )
        )

    def map_groups(
        self, function: Callable[[DataFrame], DataFrame], schema: SchemaDict | None
    ) -> Pipeline:
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.attr, "map_groups", self.args, self.kwargs, function, schema
            )
        )
