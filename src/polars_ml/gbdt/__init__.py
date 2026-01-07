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

from polars._typing import IntoExpr

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
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        num_boost_round: int = 100,
        feval: Callable[..., Any] | None = None,
        init_model: Union[str, Path, lgb.Booster] | None = None,
        keep_training_booster: bool = False,
        callbacks: list[Callable] | None = None,
        categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBM(
                params,
                label,
                features,
                num_boost_round=num_boost_round,
                feval=feval,
                init_model=init_model,
                keep_training_booster=keep_training_booster,
                callbacks=callbacks,
                categorical_feature=categorical_feature,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def xgboost(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        num_boost_round: int = 10,
        evals: Sequence[tuple[xgb.DMatrix, str]] | None = None,
        obj: xgb.Objective | None = None,
        maximize: bool | None = None,
        early_stopping_rounds: int | None = None,
        evals_result: xgb.TrainingCallback.EvalsLog | None = None,
        verbose_eval: bool | int | None = True,
        xgb_model: str | os.PathLike | xgb.Booster | bytearray | None = None,
        callbacks: Sequence[xgb.TrainingCallback] | None = None,
        custom_metric: xgb.Metric | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            XGBoost(
                params,
                label,
                features,
                num_boost_round=num_boost_round,
                evals=evals,
                obj=obj,
                maximize=maximize,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=verbose_eval,
                xgb_model=xgb_model,
                callbacks=callbacks,
                custom_metric=custom_metric,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def lightgbm_tuner(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        num_boost_round: int = 1000,
        feval: Callable[..., Any] | None = None,
        categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
        keep_training_booster: bool = False,
        callbacks: list[Callable[..., Any]] | None = None,
        time_budget: int | None = None,
        sample_size: int | None = None,
        study: optuna.study.Study | None = None,
        optuna_callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        model_dir: str | None = None,
        show_progress_bar: bool = True,
        optuna_seed: int | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTuner(
                params,
                label,
                features,
                num_boost_round=num_boost_round,
                feval=feval,
                categorical_feature=categorical_feature,
                keep_training_booster=keep_training_booster,
                callbacks=callbacks,
                time_budget=time_budget,
                sample_size=sample_size,
                study=study,
                optuna_callbacks=optuna_callbacks,
                model_dir=model_dir,
                show_progress_bar=show_progress_bar,
                optuna_seed=optuna_seed,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def lightgbm_tuner_cv(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        num_boost_round: int = 1000,
        folds: Generator[tuple[int, int], None, None]
        | Iterator[tuple[int, int]]
        | BaseCrossValidator
        | None = None,
        nfold: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        feval: Callable[..., Any] | None = None,
        categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
        fpreproc: Callable[..., Any] | None = None,
        seed: int = 0,
        callbacks: list[Callable[..., Any]] | None = None,
        time_budget: int | None = None,
        sample_size: int | None = None,
        study: optuna.study.Study | None = None,
        optuna_callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        model_dir: str | None = None,
        show_progress_bar: bool = True,
        optuna_seed: int | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            LightGBMTunerCV(
                params,
                label,
                features,
                num_boost_round=num_boost_round,
                folds=folds,
                nfold=nfold,
                stratified=stratified,
                shuffle=shuffle,
                feval=feval,
                categorical_feature=categorical_feature,
                fpreproc=fpreproc,
                seed=seed,
                callbacks=callbacks,
                time_budget=time_budget,
                sample_size=sample_size,
                study=study,
                optuna_callbacks=optuna_callbacks,
                model_dir=model_dir,
                show_progress_bar=show_progress_bar,
                optuna_seed=optuna_seed,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    def catboost(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        fit_params: Mapping[str, Any] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            CatBoost(
                params,
                label,
                features,
                fit_params=fit_params,
                prediction_name=prediction_name,
                save_dir=save_dir,
            )
        )

    # --- END INSERTION MARKER IN GBDTNameSpace
