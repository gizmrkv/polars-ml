from typing import TYPE_CHECKING, Any, Callable, Iterable

import lightgbm as lgb
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml import Pipeline

from .lightgbm_ import LightGBM

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["LightGBM"]


class ModelNameSpace:
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
        | Callable[[DataFrame, lgb.Booster], dict[str, Any]]
        | None = None,
        train_dataset_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
        validation_dataset_kwargs: dict[str, Any]
        | Callable[[DataFrame], dict[str, Any]]
        | None = None,
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
