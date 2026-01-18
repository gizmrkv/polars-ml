from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from polars._typing import IntoExpr

from .logistic import LogisticRegression
from .regression import LinearRegression, ScikitLearnLinearModel

if TYPE_CHECKING:
    from sklearn import linear_model

    from polars_ml import Pipeline


__all__ = ["LinearRegression", "LogisticRegression"]


class LinearNameSpace:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def regression(
        self,
        model: ScikitLearnLinearModel,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str | Sequence[str] = "prediction",
    ) -> Pipeline:
        return self.pipeline.pipe(
            LinearRegression(
                model,
                label,
                features,
                prediction_name=prediction_name,
            )
        )

    def logistic(
        self,
        model: linear_model.LogisticRegression | linear_model.LogisticRegressionCV,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str | Sequence[str] = "prediction",
    ) -> Pipeline:
        return self.pipeline.pipe(
            LogisticRegression(
                model,
                label,
                features,
                prediction_name=prediction_name,
            )
        )
