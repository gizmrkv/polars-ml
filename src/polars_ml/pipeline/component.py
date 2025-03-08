from abc import ABC, abstractmethod
from typing import Mapping, Self

from polars import DataFrame


class PipelineComponent(ABC):
    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        return self

    @abstractmethod
    def transform(self, data: DataFrame) -> DataFrame: ...

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        return self.fit(data, validation_data).transform(data)
