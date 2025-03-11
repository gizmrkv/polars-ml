from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr

from .catboost_ import CatBoost
from .lightgbm_ import (
    LightGBM,
    LightGBMParameters,
    LightGBMPredictArguments,
    LightGBMTrainArguments,
    LightGBMTrainDatasetArguments,
    LightGBMValidateDatasetArguments,
)
from .xgboost_ import XGBoost

if TYPE_CHECKING:
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    from polars_ml import Pipeline

__all__ = ["CatBoost", "LightGBM", "XGBoost"]


class GBDTNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def lightgbm(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: LightGBMParameters | None = None,
        *,
        prediction_name: str = "prediction",
        append_prediction: bool = True,
        train_kwargs: LightGBMTrainArguments
        | Callable[[DataFrame], LightGBMTrainArguments]
        | None = None,
        predict_kwargs: LightGBMPredictArguments
        | Callable[[DataFrame, "lgb.Booster"], LightGBMPredictArguments]
        | None = None,
        train_dataset_kwargs: LightGBMTrainDatasetArguments
        | Callable[[DataFrame], LightGBMTrainDatasetArguments]
        | None = None,
        validation_dataset_kwargs: LightGBMValidateDatasetArguments
        | Callable[[DataFrame], LightGBMValidateDatasetArguments]
        | None = None,
        save_dir: str | Path | None = None,
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
                save_dir=save_dir,
            )
        )
