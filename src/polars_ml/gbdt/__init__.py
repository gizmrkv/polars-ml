from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from polars._typing import IntoExpr

from .catboost_ import CatBoost
from .lightgbm_ import (
    LightGBM,
    LightGBMTuner,
    LightGBMTunerCV,
)
from .xgboost_ import XGBoost

if TYPE_CHECKING:
    from polars_ml import Pipeline


__all__ = ["LightGBM", "LightGBMTuner", "LightGBMTunerCV", "XGBoost", "CatBoost"]


class GBDTNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN GBDTNameSpace

    def lightgbm(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBM(
                params,
                label,
                features,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def xgboost(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            XGBoost(
                params,
                label,
                features,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def lightgbm_tuner(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTuner(
                params,
                label,
                features,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def lightgbm_tuner_cv(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTunerCV(
                params,
                label,
                features,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def catboost(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            CatBoost(
                params,
                label,
                features,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    # --- END INSERTION MARKER IN GBDTNameSpace
