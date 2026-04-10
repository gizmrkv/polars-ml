from __future__ import annotations

from typing import Self

import polars as pl

from polars_ml import LazyTransformer, Transformer


class Pipeline(Transformer):
    def __init__(self, *steps: Transformer) -> None:
        self._steps = list(steps)

    def pipe(self, step: Transformer) -> Self:
        self._steps.append(step)
        return self

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        for i, step in enumerate(self._steps):
            if i < len(self._steps) - 1:
                data = step.fit_transform(data, **more_data)
                more_data = {k: step.transform(v) for k, v in more_data.items()}
            else:
                step.fit(data, **more_data)

        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        for i, step in enumerate(self._steps):
            data = step.fit_transform(data, **more_data)
            if i < len(self._steps) - 1:
                more_data = {k: step.transform(v) for k, v in more_data.items()}

        return data

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        for step in self._steps:
            data = step.transform(data)
        return data


class LazyPipeline(LazyTransformer):
    def __init__(self, *steps: LazyTransformer) -> None:
        self._steps = list(steps)

    def pipe(self, step: LazyTransformer) -> Self:
        self._steps.append(step)
        return self

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        for i, step in enumerate(self._steps):
            if i < len(self._steps) - 1:
                data = step.fit_transform(data, **more_data)
                more_data = {
                    k: step.collect().transform(v) for k, v in more_data.items()
                }
            else:
                step.fit(data, **more_data)

        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        for i, step in enumerate(self._steps):
            data = step.fit_transform(data, **more_data)
            if i < len(self._steps) - 1:
                more_data = {
                    k: step.collect().transform(v) for k, v in more_data.items()
                }

        return data

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for step in self._steps:
            data = step.transform(data)
        return data
