from typing import TYPE_CHECKING, Any, Callable, Iterable

from polars._typing import ColumnNameOrSelector

from .polynomial import Polynomial

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["Polynomial"]


class FeatureEngineeringNameSapce:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def polynomial(
        self,
        *features: ColumnNameOrSelector,
        degree: int = 2,
    ) -> "Pipeline":
        self.pipeline.pipe(Polynomial(*features, degree=degree))
        return self.pipeline
