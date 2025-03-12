from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr

from .logistic_regression import (
    LogisticRegression,
    LogisticRegressionFitArguments,
    LogisticRegressionParameters,
)
from .regression import (
    LinearRegression,
    LinearRegressionFitArguments,
    LinearRegressionParameters,
)

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["LinearRegression"]


class LinearNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def regression(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str,
        include_input: bool,
        model_kwargs: LinearRegressionParameters
        | Callable[[DataFrame], LinearRegressionParameters]
        | None = None,
        fit_kwargs: LinearRegressionFitArguments
        | Callable[[DataFrame], LinearRegressionFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LinearRegression(
                features,
                label,
                prediction_name=prediction_name,
                include_input=include_input,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def logistic_regression(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str,
        include_input: bool,
        model_kwargs: LogisticRegressionParameters
        | Callable[[DataFrame], LogisticRegressionParameters]
        | None = None,
        fit_kwargs: LogisticRegressionFitArguments
        | Callable[[DataFrame], LogisticRegressionFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LogisticRegression(
                features,
                label,
                prediction_name=prediction_name,
                include_input=include_input,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
