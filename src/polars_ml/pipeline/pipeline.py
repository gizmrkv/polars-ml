from __future__ import annotations

from typing import Self

from polars import DataFrame

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.gbdt import GBDTNameSpace
from polars_ml.linear import LinearNameSpace
from polars_ml.metrics import MetricsNameSpace
from polars_ml.optimize import OptimizeNameSpace

from .mixin import PipelineMixin


class Pipeline(PipelineMixin, HasFeatureImportance):
    def __init__(self, *steps: Transformer):
        self.steps: list[Transformer] = list(steps)

    def pipe(self, step: Transformer) -> Self:
        self.steps.append(step)
        return self

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for i, step in enumerate(self.steps):
            try:
                if i < len(self.steps) - 1:
                    data = step.fit_transform(data, **more_data)
                    more_data = {k: step.transform(v) for k, v in more_data.items()}
                else:
                    step.fit(data, **more_data)
            except Exception as e:
                if e.args and isinstance(e.args[0], str):
                    e.args = (
                        f"Step {i} ({type(step).__name__}): {e.args[0]}",
                    ) + e.args[1:]
                else:
                    e.args = (f"Step {i} ({type(step).__name__})",) + e.args
                raise
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        for i, step in enumerate(self.steps):
            try:
                data = step.fit_transform(data, **more_data)
                if i < len(self.steps) - 1:
                    more_data = {k: step.transform(v) for k, v in more_data.items()}
            except Exception as e:
                if e.args and isinstance(e.args[0], str):
                    e.args = (
                        f"Step {i} ({type(step).__name__}): {e.args[0]}",
                    ) + e.args[1:]
                else:
                    e.args = (f"Step {i} ({type(step).__name__})",) + e.args
                raise
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        for i, step in enumerate(self.steps):
            try:
                data = step.transform(data)
            except Exception as e:
                if e.args and isinstance(e.args[0], str):
                    e.args = (
                        f"Step {i} ({type(step).__name__}): {e.args[0]}",
                    ) + e.args[1:]
                else:
                    e.args = (f"Step {i} ({type(step).__name__})",) + e.args
                raise
        return data

    def get_feature_importance(self) -> DataFrame:
        if not self.steps:
            raise ValueError("Pipeline has no steps.")

        last_step = self.steps[-1]
        if isinstance(last_step, HasFeatureImportance):
            return last_step.get_feature_importance()

        raise TypeError(
            f"The last step of the pipeline ({type(last_step).__name__}) "
            "does not support feature importance."
        )

    @property
    def gbdt(self) -> GBDTNameSpace:
        return GBDTNameSpace(self)

    @property
    def linear(self) -> LinearNameSpace:
        return LinearNameSpace(self)

    @property
    def metrics(self) -> MetricsNameSpace:
        return MetricsNameSpace(self)

    @property
    def optimize(self) -> OptimizeNameSpace:
        return OptimizeNameSpace(self)
