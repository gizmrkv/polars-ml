from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml import Pipeline

from .catboost_ import CatBoost
from .decomposition import NMF, PCA, TruncatedSVD
from .lightgbm_ import LightGBM
from .linear import ElasticNet, Lasso, LinearRegression, Ridge
from .xgboost_ import XGBoost

if TYPE_CHECKING:
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    from polars_ml import Pipeline

__all__ = ["LightGBM"]


class TreeNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def lightgbm(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        params: dict[str, Any],
        *,
        prediction_name: str = "lightgbm",
        append_prediction: bool = True,
        train_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: dict[str, Any]
        | Callable[[DataFrame, "lgb.Booster"], dict[str, Any]]
        | None = None,
        train_dataset_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        validation_dataset_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        dir: str | Path | None = None,
        plot_importance: bool = False,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LightGBM(
                features,
                label,
                params,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs,
                train_dataset_kwargs=train_dataset_kwargs,
                validation_dataset_kwargs=validation_dataset_kwargs,
            )
        )

    def xgboost(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        params: dict[str, Any],
        *,
        prediction_name: str = "xgboost",
        append_prediction: bool = True,
        train_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: dict[str, Any]
        | Callable[[DataFrame, "xgb.Booster"], dict[str, Any]]
        | None = None,
        train_dmatrix_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        validation_dmatrix_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        dir: str | Path | None = None,
        plot_importance: bool = False,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            XGBoost(
                features,
                label,
                params,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs,
                train_dmatrix_kwargs=train_dmatrix_kwargs,
                validation_dmatrix_kwargs=validation_dmatrix_kwargs,
                dir=dir,
                plot_importance=plot_importance,
            )
        )

    def catboost(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        params: dict[str, Any],
        *,
        cat_features: list[str] | None = None,
        prediction_name: str = "catboost",
        append_prediction: bool = True,
        train_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        predict_kwargs: dict[str, Any]
        | Callable[[DataFrame, "cb.CatBoost"], dict[str, Any]]
        | None = None,
        pool_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        dir: str | Path | None = None,
        plot_importance: bool = False,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CatBoost(
                features,
                label,
                params,
                cat_features=cat_features,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs,
                pool_kwargs=pool_kwargs,
                dir=dir,
                plot_importance=plot_importance,
            )
        )


class LinearNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def regression(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "linear_regression",
        append_prediction: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LinearRegression(
                features,
                label,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )

    def ridge(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "ridge",
        append_prediction: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            Ridge(
                features,
                label,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )

    def lasso(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "lasso",
        append_prediction: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            Lasso(
                features,
                label,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )

    def elastic_net(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        prediction_name: str = "elastic_net",
        append_prediction: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            ElasticNet(
                features,
                label,
                prediction_name=prediction_name,
                append_prediction=append_prediction,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )


class DecompositionNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def pca(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str = "pca",
        append_components: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            PCA(
                features,
                components_name=components_name,
                append_components=append_components,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )

    def nmf(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str = "nmf",
        append_components: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            NMF(
                features,
                components_name=components_name,
                append_components=append_components,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )

    def truncated_svd(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        components_name: str = "truncated_svd",
        append_components: bool = True,
        model_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        fit_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            TruncatedSVD(
                features,
                components_name=components_name,
                append_components=append_components,
                model_kwargs=model_kwargs,
                fit_kwargs=fit_kwargs,
            )
        )
