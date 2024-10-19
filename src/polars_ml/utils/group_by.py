from typing import Any, Dict, Tuple, override

from polars import DataFrame, LazyFrame

from polars_ml.component import Component, LazyComponent


class GroupByGetAttr(Component):
    def __init__(
        self,
        group_by_method: str,
        agg_method: str,
        *,
        group_by_args: Tuple[Any, ...] | None = None,
        group_by_kwargs: Dict[str, Any] | None = None,
        agg_args: Tuple[Any, ...] | None = None,
        agg_kwargs: Dict[str, Any] | None = None,
    ):
        self.group_by_method = group_by_method
        self.agg_method = agg_method
        self.group_by_args = group_by_args or ()
        self.group_by_kwargs = group_by_kwargs or {}
        self.agg_args = agg_args or ()
        self.agg_kwargs = agg_kwargs or {}
        self._is_fitted = True

        self.set_component_name(
            "".join(
                "".join(w[0].capitalize() + w[1:] for w in method.split("_"))
                for method in [group_by_method, agg_method]
            )
        )

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        return getattr(
            getattr(data, self.group_by_method)(
                *self.group_by_args, **self.group_by_kwargs
            ),
            self.agg_method,
        )(*self.agg_args, **self.agg_kwargs)


class LazyGroupByGetAttr(LazyComponent):
    def __init__(
        self,
        group_by_method: str,
        agg_method: str,
        *,
        group_by_args: Tuple[Any, ...] | None = None,
        group_by_kwargs: Dict[str, Any] | None = None,
        agg_args: Tuple[Any, ...] | None = None,
        agg_kwargs: Dict[str, Any] | None = None,
    ):
        self.group_by_method = group_by_method
        self.agg_method = agg_method
        self.group_by_args = group_by_args or ()
        self.group_by_kwargs = group_by_kwargs or {}
        self.agg_args = agg_args or ()
        self.agg_kwargs = agg_kwargs or {}
        self._is_fitted = True

        self.set_component_name(
            "".join(
                "".join(w[0].capitalize() + w[1:] for w in method.split("_"))
                for method in [group_by_method, agg_method]
            )
        )

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return getattr(
            getattr(data, self.group_by_method)(
                *self.group_by_args, **self.group_by_kwargs
            ),
            self.agg_method,
        )(*self.agg_args, **self.agg_kwargs)
