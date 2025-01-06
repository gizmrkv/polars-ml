from abc import ABC, abstractmethod
from typing import Self

from polars import DataFrame


class Component(ABC):
    def fit(self, data: DataFrame) -> Self:
        return self

    @abstractmethod
    def transform(self, data: DataFrame) -> DataFrame: ...

    def fit_transform(self, data: DataFrame) -> DataFrame:
        return self.fit(data).transform(data)
