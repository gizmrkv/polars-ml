from abc import ABC, abstractmethod
from typing import Self

from polars_ml import Transformer


class PipelineMixin(ABC):
    @abstractmethod
    def pipe(self, step: Transformer) -> Self: ...
