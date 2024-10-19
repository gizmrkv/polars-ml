from typing import Any, Callable, Generic, Iterable

from polars import DataFrame
from polars._typing import IntoExpr, RollingInterpolationMethod, SchemaDict

from polars_ml.typing import PipelineType
from polars_ml.utils import LazyGroupByGetAttr


class LazyGroupByNameSpace(Generic[PipelineType]):
    def __init__(self, pipeline: PipelineType, method: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="agg",
                agg_args=aggs,
                agg_kwargs=named_aggs,
            ).set_component_name("GroupByAgg")
        )

    def all(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="all",
            ).set_component_name("GroupByAll")
        )

    def any(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="any",
            ).set_component_name("GroupByAny")
        )

    def first(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="first",
            ).set_component_name("GroupByFirst")
        )

    def last(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="last",
            ).set_component_name("GroupByLast")
        )

    def count(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="count",
            ).set_component_name("GroupByCount")
        )

    def len(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="len",
            ).set_component_name("GroupByLen")
        )

    def head(self, n: int) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="head",
                agg_args=(n,),
            ).set_component_name("GroupByHead")
        )

    def tail(self, n: int) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="tail",
                agg_args=(n,),
            ).set_component_name("GroupByTail")
        )

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="quantile",
                agg_args=(quantile, interpolation),
            ).set_component_name("GroupByQuantile")
        )

    def max(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="max",
            ).set_component_name("GroupByMax")
        )

    def min(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="min",
            ).set_component_name("GroupByMin")
        )

    def sum(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="sum",
            ).set_component_name("GroupBySum")
        )

    def mean(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="mean",
            ).set_component_name("GroupByMean")
        )

    def median(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="median",
            ).set_component_name("GroupByMedian")
        )

    def n_unique(self) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="n_unique",
            ).set_component_name("GroupByNUnique")
        )

    def map_groups(
        self, function: Callable[[DataFrame], DataFrame], schema: SchemaDict | None
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                group_by_method=self.method,
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_method="map_groups",
                agg_args=(function, schema),
            ).set_component_name("GroupByMapGroups")
        )
