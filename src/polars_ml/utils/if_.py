from pathlib import Path
from typing import Self, override

from polars import LazyFrame

from polars_ml.component import Component, LazyComponent

from .lazy import Lazy


class IfFit(LazyComponent):
    def __init__(self, component: Component | LazyComponent):
        self.component = (
            Lazy(component) if isinstance(component, Component) else component
        )
        self._is_fitted = False

    @override
    def is_fitted(self) -> bool:
        return self.component.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        self.component.set_log_dir(log_dir)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        self.component.fit(data)
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return data

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        return self.component.fit_execute(data)


class IfExecute(LazyComponent):
    def __init__(self, component: Component | LazyComponent):
        self.component = (
            Lazy(component) if isinstance(component, Component) else component
        )
        self._is_fitted = False

    @override
    def is_fitted(self) -> bool:
        return self.component.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        self.component.set_log_dir(log_dir)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        self.component.fit(data)
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return self.component.execute(data)

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        return data
