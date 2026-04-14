from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr, QuantileMethod, SchemaDict

from polars_ml.base import LazyTransformer

if TYPE_CHECKING:
    from polars_ml import LazyPipeline


class LazyGroupByGetAttr(LazyTransformer):
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

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return getattr(
            getattr(data, self.group_by_attr)(
                *self.group_by_args, **self.group_by_kwargs
            ),
            self.agg_attr,
        )(*self.agg_args, **self.agg_kwargs)


class LazyGroupByNameSpace:
    def __init__(
        self, pipeline: LazyPipeline, attr: str, *args: Any, **kwargs: Any
    ) -> None:
        self.pipeline = pipeline
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    # --- START INSERTION MARKER IN LazyGroupByNameSpace

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr, "agg", self.args, self.kwargs, *aggs, **named_aggs
            )
        )

    def all(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "all",
                self.args,
                self.kwargs,
            )
        )

    def count(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "count",
                self.args,
                self.kwargs,
            )
        )

    def first(self, ignore_nulls: bool = False) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr, "first", self.args, self.kwargs, ignore_nulls=ignore_nulls
            )
        )

    def head(self, n: int = 5) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(self.attr, "head", self.args, self.kwargs, n)
        )

    def last(self, ignore_nulls: bool = False) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr, "last", self.args, self.kwargs, ignore_nulls=ignore_nulls
            )
        )

    def len(self, name: str | None = None) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(self.attr, "len", self.args, self.kwargs, name)
        )

    def map_groups(
        self, function: Callable[[pl.DataFrame], DataFrame], schema: SchemaDict | None
    ) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr, "map_groups", self.args, self.kwargs, function, schema
            )
        )

    def max(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "max",
                self.args,
                self.kwargs,
            )
        )

    def mean(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "mean",
                self.args,
                self.kwargs,
            )
        )

    def median(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "median",
                self.args,
                self.kwargs,
            )
        )

    def min(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "min",
                self.args,
                self.kwargs,
            )
        )

    def n_unique(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "n_unique",
                self.args,
                self.kwargs,
            )
        )

    def quantile(
        self, quantile: float, interpolation: QuantileMethod = "nearest"
    ) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr, "quantile", self.args, self.kwargs, quantile, interpolation
            )
        )

    def sum(self) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(
                self.attr,
                "sum",
                self.args,
                self.kwargs,
            )
        )

    def tail(self, n: int = 5) -> LazyPipeline:
        return self.pipeline.pipe(
            LazyGroupByGetAttr(self.attr, "tail", self.args, self.kwargs, n)
        )

    # --- END INSERTION MARKER IN LazyGroupByNameSpace
