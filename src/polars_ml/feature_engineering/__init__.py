from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

from polars._typing import ColumnNameOrSelector, CorrelationMethod

if TYPE_CHECKING:
    from polars_ml import Pipeline

from .arithmetic_synthesis import ArithmeticSynthesis

__all__ = ["ArithmeticSynthesis", "FeatureEngineeringNameSpace"]


class FeatureEngineeringNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN FeatureEngineeringNameSpace

    def arithmetic_synthesis(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        order: int,
        method: Literal["additive", "multiplicative"] = "additive",
        drop_high_correlation_features_method: CorrelationMethod | None = None,
        threshold: float = 0.9,
        show_progress: bool = True,
    ) -> Pipeline:
        return self.pipeline.pipe(
            ArithmeticSynthesis(
                columns,
                order=order,
                method=method,
                drop_high_correlation_features_method=drop_high_correlation_features_method,
                threshold=threshold,
                show_progress=show_progress,
            )
        )

    # --- END INSERTION MARKER IN FeatureEngineeringNameSpace
