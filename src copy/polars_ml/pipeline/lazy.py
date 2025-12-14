from typing import Self

from polars import DataFrame

from polars_ml import Transformer

from .mixin import PipelineMixin


class LazyPipeline(PipelineMixin, Transformer):
    def __init__(self, *steps: Transformer):
        self.steps: list[Transformer] = list(*steps)

    def pipe(self, step: Transformer) -> Self:
        self.steps.append(step)
        return self

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for i, step in enumerate(self.steps):
            if i < len(self.steps) - 1:
                data = step.fit_transform(data, **more_data)
                more_data = {
                    k: step.transform(v.lazy()).collect() for k, v in more_data.items()
                }
            else:
                step.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        for i, step in enumerate(self.steps):
            data = step.fit_transform(data, **more_data)
            if i < len(self.steps) - 1:
                more_data = {
                    k: step.transform(v.lazy()).collect() for k, v in more_data.items()
                }
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        for step in self.steps:
            data = step.transform(data)
        return data
