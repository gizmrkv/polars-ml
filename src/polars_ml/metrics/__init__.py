from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from polars._typing import ColumnNameOrSelector

from .binary_classification import BinaryClassificationMetrics
from .multiclass_classification import MulticlassClassificationMetrics
from .regression import RegressionMetrics

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = [
    "BinaryClassificationMetrics",
    "MulticlassClassificationMetrics",
    "RegressionMetrics",
]


class MetricsNameSpace:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN MetricsNameSpace

    def binary_classification(
        self,
        y_true: str,
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(BinaryClassificationMetrics(y_true, y_preds, by=by))

    def multiclass_classification(
        self,
        y_true: str,
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            MulticlassClassificationMetrics(y_true, y_preds, by=by)
        )

    def regression(
        self,
        y_true: str,
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(RegressionMetrics(y_true, y_preds, by=by))

    # --- END INSERTION MARKER IN MetricsNameSpace
