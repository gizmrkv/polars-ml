from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

from polars_ml import LazyTransformer

from .basic import Echo, Replay


class PipelineMixin(ABC):
    @abstractmethod
    def pipe(self, step: LazyTransformer) -> Self: ...

    def echo(self) -> Self:
        return self.pipe(Echo())

    def replay(self) -> Self:
        return self.pipe(Replay())
