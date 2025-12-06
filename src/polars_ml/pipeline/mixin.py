from abc import ABC, abstractmethod
from typing import Self

from polars_ml import LazyTransformer


class PipelineMixin(ABC):
    @abstractmethod
    def pipe(self, step: LazyTransformer) -> Self: ...
