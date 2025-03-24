from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from polars import DataFrame
from polars._typing import IntoExpr
from sklearn import linear_model

from .logistic_regression import LogisticRegression
from .regression import LinearRegression

if TYPE_CHECKING:
    from polars_ml import Pipeline


class LinearNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def regression(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        model: linear_model.LinearRegression
        | linear_model.Lasso
        | linear_model.Ridge
        | linear_model.ElasticNet,
        *,
        prediction_name: str = "linear_regression",
        include_input: bool = True,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], Mapping[str, Any]]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LinearRegression(
                features,
                label,
                model,
                prediction_name=prediction_name,
                include_input=include_input,
                fit_kwargs=fit_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def logistic_regression(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        model: linear_model.LogisticRegression,
        *,
        prediction_name: str = "logistic_regression",
        include_input: bool = True,
        predict_proba: bool = False,
        fit_kwargs: Mapping[str, Any]
        | Callable[[DataFrame], Mapping[str, Any]]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LogisticRegression(
                features,
                label,
                model,
                prediction_name=prediction_name,
                include_input=include_input,
                predict_proba=predict_proba,
                fit_kwargs=fit_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
