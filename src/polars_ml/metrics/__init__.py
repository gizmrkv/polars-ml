from typing import TYPE_CHECKING

from .binary_classification import BinaryClassificationMetrics
from .regression import RegressionMetrics

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["BinaryClassificationMetrics", "RegressionMetrics"]


class MetricsNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    # --- BEGIN AUTO-GENERATED METHODS IN MetricsNameSpace ---
    def binary_classification(
        self, y_true: str, y_pred: str, by: str | None = None
    ) -> "Pipeline":
        return self.pipeline.pipe(BinaryClassificationMetrics(y_true, y_pred, by=by))

    def regression(self, y_true: str, y_pred: str, by: str | None = None) -> "Pipeline":
        return self.pipeline.pipe(RegressionMetrics(y_true, y_pred, by=by))

    # --- END AUTO-GENERATED METHODS IN MetricsNameSpace ---
