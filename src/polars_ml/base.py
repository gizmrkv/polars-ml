from abc import ABC, abstractmethod
from typing import Self

from polars import DataFrame


class Transformer(ABC):
    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        return self.fit(data, **more_data).transform(data)

    @abstractmethod
    def transform(self, data: DataFrame) -> DataFrame: ...
