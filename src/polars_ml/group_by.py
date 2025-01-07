from typing import TYPE_CHECKING, Any, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr, RollingInterpolationMethod, SchemaDict

from .component import Component

if TYPE_CHECKING:
    from .pipeline import Pipeline


class GroupByGetAttr(Component):
    def __init__(
        self,
        group_by_method: str,
        agg_method: str,
        *,
        group_by_args: tuple[Any, ...] | None = None,
        group_by_kwargs: dict[str, Any] | None = None,
        agg_args: tuple[Any, ...] | None = None,
        agg_kwargs: dict[str, Any] | None = None,
    ):
        self.group_by_method = group_by_method
        self.group_by_args = group_by_args or ()
        self.group_by_kwargs = group_by_kwargs or {}
        self.agg_method = agg_method
        self.agg_args = agg_args or ()
        self.agg_kwargs = agg_kwargs or {}

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(
            getattr(data, self.group_by_method)(
                *self.group_by_args, **self.group_by_kwargs
            ),
            self.agg_method,
        )(*self.agg_args, **self.agg_kwargs)


class GroupBy:
    def __init__(self, pipeline: "Pipeline", method: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "agg",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=aggs,
                agg_kwargs=named_aggs,
            ),
        )

    def all(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "all",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def count(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "count",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def first(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "first",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def head(self, n: int = 5) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "head",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(n,),
            )
        )

    def last(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "last",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def len(self, name: str | None = None) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "len",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(name,),
            )
        )

    def map_groups(self, function: Callable[[DataFrame], DataFrame]) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "map_groups",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(function,),
            )
        )

    def max(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "max",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def mean(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "mean",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def median(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "median",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def min(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "min",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def n_unique(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "n_unique",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "quantile",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(quantile, interpolation),
            )
        )

    def sum(self) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "sum",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
            )
        )

    def tail(self, n: int = 5) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "tail",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(n,),
            )
        )


class DynamicGroupBy:
    def __init__(self, pipeline: "Pipeline", method: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "agg",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=aggs,
                agg_kwargs=named_aggs,
            )
        )

    def map_groups(
        self,
        function: Callable[[DataFrame], DataFrame],
        schema: SchemaDict | None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "map_groups",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(function, schema),
            )
        )


class RollingGroupBy:
    def __init__(self, pipeline: "Pipeline", method: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "agg",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=aggs,
                agg_kwargs=named_aggs,
            )
        )

    def map_groups(
        self,
        function: Callable[[DataFrame], DataFrame],
        schema: SchemaDict | None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            GroupByGetAttr(
                self.method,
                "map_groups",
                group_by_args=self.args,
                group_by_kwargs=self.kwargs,
                agg_args=(function, schema),
            )
        )
