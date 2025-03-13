from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr

from .elastic_net import ElasticNet, ElasticNetFitArguments, ElasticNetParameters
from .lasso import Lasso, LassoFitArguments, LassoParameters
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
from .ridge import Ridge, RidgeFitArguments, RidgeParameters

if TYPE_CHECKING:
    from polars_ml import Pipeline


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

    def lasso(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str,
        include_input: bool,
        model_kwargs: LassoParameters
        | Callable[[DataFrame], LassoParameters]
        | None = None,
        fit_kwargs: LassoFitArguments
        | Callable[[DataFrame], LassoFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            Lasso(
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

    def ridge(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str,
        include_input: bool,
        model_kwargs: RidgeParameters
        | Callable[[DataFrame], RidgeParameters]
        | None = None,
        fit_kwargs: RidgeFitArguments
        | Callable[[DataFrame], RidgeFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            Ridge(
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

    def elastic_net(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        *,
        prediction_name: str,
        include_input: bool,
        model_kwargs: ElasticNetParameters
        | Callable[[DataFrame], ElasticNetParameters]
        | None = None,
        fit_kwargs: ElasticNetFitArguments
        | Callable[[DataFrame], ElasticNetFitArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ElasticNet(
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
