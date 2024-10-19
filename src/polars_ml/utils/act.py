from pathlib import Path
from typing import Self, override

from polars import LazyFrame
from tqdm import tqdm

from polars_ml.component import Component, ComponentList, LazyComponent

from .lazy import Lazy


class Act(LazyComponent):
    def __init__(
        self, *components: Component | LazyComponent, show_progress: bool = True
    ):
        self.components = ComponentList(
            [
                Lazy(component) if isinstance(component, Component) else component
                for component in components
            ]
        )
        self.show_progress = show_progress

    @override
    def is_fitted(self) -> bool:
        return self.components.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        super().set_log_dir(log_dir)
        self.components.set_log_dir(log_dir)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        for component in tqdm(self.components, disable=not self.show_progress):
            component.fit(data)
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        for component in tqdm(self.components, disable=not self.show_progress):
            component.execute(data)
        return data

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        for component in tqdm(self.components, disable=not self.show_progress):
            component.fit_execute(data)
        return data
