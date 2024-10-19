from typing import Protocol, Self, TypeVar

from polars_ml.component import LazyComponent


class PipelineMixin(Protocol):
    def pipe(self, *components: LazyComponent) -> Self: ...


PipelineType = TypeVar("PipelineType", bound=PipelineMixin)
