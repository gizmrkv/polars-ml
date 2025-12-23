from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from polars._typing import IntoExpr

from .lightgbm_ import (
    LightGBM,
    LightGBMTuner,
    LightGBMTunerCV,
)
from .xgboost_ import XGBoost

if TYPE_CHECKING:
    from polars_ml import Pipeline


__all__ = ["LightGBM", "LightGBMTuner", "LightGBMTunerCV"]


class GBDTNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN GBDTNameSpace

    def lightgbm(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBM(
                label,
                features,
                prediction_name=prediction_name,
                out_dir=out_dir,
                **params,
            )
        )

    def xgboost(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            XGBoost(
                label,
                features,
                prediction_name=prediction_name,
                out_dir=out_dir,
                **params,
            )
        )

    def lightgbm_tuner(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTuner(
                label,
                features,
                prediction_name=prediction_name,
                out_dir=out_dir,
                **params,
            )
        )

    def lightgbm_tuner_cv(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTunerCV(
                label,
                features,
                prediction_name=prediction_name,
                out_dir=out_dir,
                **params,
            )
        )

    # --- END INSERTION MARKER IN GBDTNameSpace
