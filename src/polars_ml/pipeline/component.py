from abc import ABC, abstractmethod
from pathlib import Path
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

    @property
    def out_dir(self) -> Path | None:
        if hasattr(self, "_out_dir"):
            return self._out_dir
        else:
            return None

    @out_dir.setter
    def out_dir(self, value: str | Path):
        self._out_dir = Path(value)
        return self
