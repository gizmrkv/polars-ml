from abc import ABC, abstractmethod
from typing import Self

from polars import DataFrame, LazyFrame


class Transformer(ABC):
    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        return self.fit(data, **more_data).transform(data)

    @abstractmethod
    def transform(self, data: DataFrame) -> DataFrame: ...


class LazyTransformer(ABC):
    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        return self.fit(data, **more_data).transform(data.lazy()).collect()

    @abstractmethod
    def transform(self, data: LazyFrame) -> LazyFrame: ...

    def eager(self) -> "Eager":
        return Eager(self)


class Eager(Transformer):
    def __init__(self, transformer: LazyTransformer):
        self.transformer = transformer

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.transformer.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        return self.transformer.fit_transform(data, **more_data)

    def transform(self, data: DataFrame) -> DataFrame:
        return self.transformer.transform(data.lazy()).collect()
