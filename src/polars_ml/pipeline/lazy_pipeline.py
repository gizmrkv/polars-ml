from pathlib import Path
from typing import TYPE_CHECKING, Self, override

from polars import LazyFrame
from tqdm import tqdm

from polars_ml.component import LazyComponent

from .base_pipeline import BasePipeline
from .horizontal import LazyHorizontalNameSpace
from .split import LazySplitNameSpace
from .stat import LazyStatNameSpace
from .transform import LazyTransformNameSpace

if TYPE_CHECKING:
    from .pipeline import Pipeline


class LazyPipeline(BasePipeline, LazyComponent):
    def __init__(
        self,
        *,
        log_dir: str | Path | None = None,
        pipeline_name: str | None = None,
        show_progress: bool = False,
    ):
        super().__init__()

        if log_dir := log_dir:
            self.set_log_dir(log_dir)
        if pipeline_name := pipeline_name:
            self.set_component_name(pipeline_name)

        self.show_progress = show_progress

    @staticmethod
    def load(path: Path) -> "LazyPipeline":
        import joblib

        pipe = joblib.load(path)
        if isinstance(pipe, LazyPipeline):
            return pipe
        else:
            raise ValueError(f"Expected a LazyPipeline but got {type(pipe)}")

    @override
    def fit(self, data: LazyFrame) -> Self:
        bar = tqdm(total=len(self.components), disable=not self.show_progress)
        for i, component in enumerate(self.components):
            bar.set_description(component.component_name)
            if i == len(self.components) - 1:
                component.fit(data)
            else:
                data = component.fit_execute(data)
            bar.update()

        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        bar = tqdm(total=len(self.components), disable=not self.show_progress)
        for component in self.components:
            bar.set_description(component.component_name)
            data = component.execute(data)
            bar.update()

        return data

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        bar = tqdm(total=len(self.components), disable=not self.show_progress)
        for component in self.components:
            bar.set_description(component.component_name)
            data = component.fit_execute(data)
            bar.update()

        return data

    def collect(self) -> "Pipeline":
        return Pipeline(log_dir=self.log_dir, show_progress=self.show_progress).pipe(
            self
        )

    @property
    def stat(self) -> LazyStatNameSpace:
        return LazyStatNameSpace(self)

    @property
    def horizontal(self) -> LazyHorizontalNameSpace:
        return LazyHorizontalNameSpace(self)

    @property
    def split(self) -> LazySplitNameSpace:
        return LazySplitNameSpace(self)

    @property
    def transform(self) -> LazyTransformNameSpace:
        return LazyTransformNameSpace(self)
