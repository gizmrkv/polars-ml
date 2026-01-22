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

    # --- START INSERTION MARKER IN GroupByNameSpace

    # --- END INSERTION MARKER IN GroupByNameSpace


class DynamicGroupByNameSpace:
    def __init__(self, pipeline: Pipeline, attr: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    # --- START INSERTION MARKER IN DynamicGroupByNameSpace

    # --- END INSERTION MARKER IN DynamicGroupByNameSpace


class RollingGroupByNameSpace:
    def __init__(self, pipeline: Pipeline, attr: str, *args: Any, **kwargs: Any):
        self.pipeline = pipeline
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    # --- START INSERTION MARKER IN RollingGroupByNameSpace

    # --- END INSERTION MARKER IN RollingGroupByNameSpace
