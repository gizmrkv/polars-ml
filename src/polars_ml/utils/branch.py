from pathlib import Path
from typing import Self, override

import polars as pl
from polars import LazyFrame
from polars._typing import ConcatMethod
from tqdm import tqdm

from polars_ml.component import Component, ComponentList, LazyComponent

from .lazy import Lazy


class Branch(LazyComponent):
    def __init__(
        self,
        *components: Component | LazyComponent,
        how: ConcatMethod = "horizontal",
        rechunk: bool = False,
        parallel: bool = True,
        show_progress: bool = True,
    ):
        self.components = ComponentList(
            [
                Lazy(component) if isinstance(component, Component) else component
                for component in components
            ]
        )
        self.how: ConcatMethod = how
        self.rechunk = rechunk
        self.parallel = parallel
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
        return pl.concat(
            [
                component.execute(data)
                for component in tqdm(self.components, disable=not self.show_progress)
            ],
            how=self.how,
            rechunk=self.rechunk,
            parallel=self.parallel,
        )

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        return pl.concat(
            [
                component.fit_execute(data)
                for component in tqdm(self.components, disable=not self.show_progress)
            ],
            how=self.how,
            rechunk=self.rechunk,
            parallel=self.parallel,
        )
