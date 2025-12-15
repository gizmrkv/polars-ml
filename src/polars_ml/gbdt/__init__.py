from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from polars._typing import IntoExpr

from .lightgbm_ import (
    LGBDatasetParams,
    LGBPredictParams,
    LGBTrainParams,
    LGBTunerCVParams,
    LGBTunerParams,
    LightGBM,
    LightGBMTuner,
    LightGBMTunerCV,
)
from .xgboost_ import XGBDatasetParams, XGBoost, XGBPredictParams, XGBTrainParams

if TYPE_CHECKING:
    from polars_ml import Pipeline


__all__ = ["LightGBM", "LightGBMTuner", "LightGBMTunerCV"]


class GBDTNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    # --- BEGIN AUTO-GENERATED METHODS IN GBDTNameSpace ---
    def lightgbm(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        train_params: LGBTrainParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LightGBM(
                label,
                params,
                features=features,
                dataset_params=dataset_params,
                train_params=train_params,
                predict_params=predict_params,
                prediction_name=prediction_name,
                out_dir=out_dir,
            )
        )

    def xgboost(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: XGBDatasetParams | None = None,
        train_params: XGBTrainParams | None = None,
        predict_params: XGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            XGBoost(
                label,
                params,
                features=features,
                dataset_params=dataset_params,
                train_params=train_params,
                predict_params=predict_params,
                prediction_name=prediction_name,
                out_dir=out_dir,
            )
        )

    def lightgbm_tuner(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        tuner_params: LGBTunerParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LightGBMTuner(
                label,
                params,
                features=features,
                dataset_params=dataset_params,
                tuner_params=tuner_params,
                predict_params=predict_params,
                prediction_name=prediction_name,
                out_dir=out_dir,
            )
        )

    def lightgbm_tuner_cv(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        tuner_params: LGBTunerCVParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            LightGBMTunerCV(
                label,
                params,
                features=features,
                dataset_params=dataset_params,
                tuner_params=tuner_params,
                predict_params=predict_params,
                prediction_name=prediction_name,
                out_dir=out_dir,
            )
        )

    # --- END AUTO-GENERATED METHODS IN GBDTNameSpace ---
