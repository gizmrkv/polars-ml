from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr

from .catboost_ import (
    CatBoost,
    CatBoostParameters,
    CatBoostPoolArguments,
    CatBoostPredictArguments,
    CatBoostTrainArguments,
)
from .lightgbm_ import (
    LightGBM,
    LightGBMParameters,
    LightGBMPredictArguments,
    LightGBMTrainArguments,
    LightGBMTrainDatasetArguments,
    LightGBMValidateDatasetArguments,
)
from .xgboost_ import (
    XGBoost,
    XGBoostDMatrixArguments,
    XGBoostParameters,
    XGBoostPredictArguments,
    XGBoostTrainArguments,
)

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
        prediction_name: str = "lightgbm",
        include_input: bool = True,
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
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LightGBM(
                features,
                label,
                params,
                prediction_name=prediction_name,
                include_input=include_input,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs,
                train_dataset_kwargs=train_dataset_kwargs,
                validation_dataset_kwargs=validation_dataset_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def xgboost(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: XGBoostParameters | None = None,
        *,
        prediction_name: str = "xgboost",
        include_input: bool = True,
        train_kwargs: XGBoostTrainArguments
        | Callable[[DataFrame], XGBoostTrainArguments]
        | None = None,
        predict_kwargs: XGBoostPredictArguments
        | Callable[[DataFrame, "xgb.Booster"], XGBoostPredictArguments]
        | None = None,
        dmatrix_kwargs: XGBoostDMatrixArguments
        | Callable[[DataFrame], XGBoostDMatrixArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            XGBoost(
                features,
                label,
                params,
                prediction_name=prediction_name,
                include_input=include_input,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs,
                dmatrix_kwargs=dmatrix_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def catboost(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: CatBoostParameters | None = None,
        *,
        prediction_name: str = "catboost",
        include_input: bool = True,
        train_kwargs: CatBoostTrainArguments
        | Callable[[DataFrame], CatBoostTrainArguments]
        | None = None,
        predict_kwargs: CatBoostPredictArguments
        | Callable[[DataFrame, "cb.CatBoost"], CatBoostPredictArguments]
        | None = None,
        pool_kwargs: CatBoostPoolArguments
        | Callable[[DataFrame], CatBoostPoolArguments]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            CatBoost(
                features,
                label,
                params,
                prediction_name=prediction_name,
                include_input=include_input,
                train_kwargs=train_kwargs,
                predict_kwargs=predict_kwargs,
                pool_kwargs=pool_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
