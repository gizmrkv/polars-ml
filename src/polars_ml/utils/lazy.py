from pathlib import Path
from typing import Literal, Self, override

from polars import LazyFrame
from polars._typing import EngineType

from polars_ml.component import Component, LazyComponent


class Lazy(LazyComponent):
    def __init__(
        self,
        component: Component,
        *,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        cluster_with_columns: bool = True,
        collapse_joins: bool = True,
        no_optimization: bool = False,
        streaming: bool = False,
        engine: EngineType = "cpu",
        background: Literal[False] = False,
        _eager: bool = False,
    ):
        self.component = component
        self.collect_kwargs = {
            "type_coercion": type_coercion,
            "predicate_pushdown": predicate_pushdown,
            "projection_pushdown": projection_pushdown,
            "simplify_expression": simplify_expression,
            "slice_pushdown": slice_pushdown,
            "comm_subplan_elim": comm_subplan_elim,
            "comm_subexpr_elim": comm_subexpr_elim,
            "cluster_with_columns": cluster_with_columns,
            "collapse_joins": collapse_joins,
            "no_optimization": no_optimization,
            "streaming": streaming,
            "engine": engine,
            "background": background,
            "_eager": _eager,
        }

    @override
    def is_fitted(self) -> bool:
        return self.component.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        super().set_log_dir(log_dir)
        self.component.set_log_dir(log_dir)
        return self

    @override
    def set_component_name(self, name: str) -> Self:
        super().set_component_name(name)
        self.component.set_component_name(name)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        self.component.fit(data.collect(**self.collect_kwargs))
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return self.component.execute(data.collect(**self.collect_kwargs)).lazy()

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        return self.component.fit_execute(data.collect(**self.collect_kwargs)).lazy()
