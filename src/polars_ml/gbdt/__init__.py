from __future__ import annotations

import os
from ctypes import Union
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
)

from polars._typing import ColumnNameOrSelector

from .catboost_ import CatBoost
from .lightgbm_ import (
    LightGBM,
    LightGBMTuner,
    LightGBMTunerCV,
)
from .xgboost_ import XGBoost

if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    import xgboost as xgb
    from optuna import Study
    from optuna.trial import FrozenTrial
    from sklearn.model_selection import BaseCrossValidator

    from polars_ml import Pipeline


__all__ = ["LightGBM", "LightGBMTuner", "LightGBMTunerCV", "XGBoost", "CatBoost"]


class GBDTNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN GBDTNameSpace

    def lightgbm(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        params: Mapping[str, Any],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        fit_dir: str | Path | None = None,
        **train_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBM(
                target,
                prediction,
                params,
                features=features,
                fit_dir=fit_dir,
                **train_params,
            )
        )

    def xgboost(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        params: Mapping[str, Any],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        fit_dir: str | Path | None = None,
        **train_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            XGBoost(
                target,
                prediction,
                params,
                features=features,
                fit_dir=fit_dir,
                **train_params,
            )
        )

    def lightgbm_tuner(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        params: Mapping[str, Any],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        fit_dir: str | Path | None = None,
        **tuner_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTuner(
                target,
                prediction,
                params,
                features=features,
                fit_dir=fit_dir,
                **tuner_params,
            )
        )

    def lightgbm_tuner_cv(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        params: Mapping[str, Any],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        fit_dir: str | Path | None = None,
        **tuner_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTunerCV(
                target,
                prediction,
                params,
                features=features,
                fit_dir=fit_dir,
                **tuner_params,
            )
        )

    def catboost(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        params: Mapping[str, Any] | None = None,
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        fit_dir: str | Path | None = None,
        **fit_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(
            CatBoost(
                target,
                prediction,
                params,
                features=features,
                fit_dir=fit_dir,
                **fit_params,
            )
        )

    # --- END INSERTION MARKER IN GBDTNameSpace
