from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar

import polars as pl

TransformerType = TypeVar("TransformerType", bound="Transformer")
LazyTransformerType = TypeVar("LazyTransformerType", bound="LazyTransformer")


class Transformer(ABC):
    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        return self.fit(data, **more_data).transform(data)

    @abstractmethod
    def transform(self, data: pl.DataFrame) -> pl.DataFrame: ...

    def lazy(self) -> Lazy[Self]:
        return Lazy(self)


class LazyTransformer(ABC):
    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        return self.fit(data, **more_data).transform(data.lazy()).collect()

    @abstractmethod
    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame: ...

    def collect(self) -> Collect[Self]:
        return Collect(self)


class Lazy(LazyTransformer, Generic[TransformerType]):
    def __init__(self, transformer: TransformerType) -> None:
        self._transformer = transformer

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        self._transformer.fit(data, **more_data)
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        return self._transformer.fit_transform(data, **more_data)

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return self._transformer.transform(data.collect()).lazy()


class Collect(Transformer, Generic[LazyTransformerType]):
    def __init__(self, transformer: LazyTransformerType) -> None:
        self._transformer = transformer

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        self._transformer.fit(data, **more_data)
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        return self._transformer.fit_transform(data, **more_data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return self._transformer.transform(data.lazy()).collect()
