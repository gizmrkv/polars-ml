from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Union

import numpy as np
from sklearn.metrics import mean_squared_error

from .linear_ensemble import LinearEnsemble
from .optuna_ import ModelFunction, ObjectiveFunction, OptunaOptimizer
from .weighted_average import WeightedAverage

if TYPE_CHECKING:
    import optuna
    from polars._typing import ColumnNameOrSelector

    from polars_ml import Pipeline


class OptimizeNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    # --- START INSERTION MARKER IN OptimizeNameSpace

    def optuna(
        self,
        model_fn: ModelFunction,
        objective_fn: ObjectiveFunction,
        search_space: Mapping[str, Mapping[str, Any]],
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        study_name: str | None = None,
        is_higher_better: bool = False,
        load_if_exists: bool = False,
        n_trials: int | None = None,
        timeout: int | None = None,
        n_jobs: int = 1,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
        storage: str | Path | optuna.storages.BaseStorage = "./journal.log",
    ) -> Pipeline:
        return self.pipeline.pipe(
            OptunaOptimizer(
                model_fn,
                objective_fn,
                search_space,
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                is_higher_better=is_higher_better,
                load_if_exists=load_if_exists,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                gc_after_trial=gc_after_trial,
                show_progress_bar=show_progress_bar,
                storage=storage,
            )
        )

    def weighted_average(
        self,
        pred_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        target_column: str,
        metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        is_higher_better: bool = False,
        method: str = "SLSQP",
        sum_to_one: bool = True,
        non_negative: bool = True,
        output_column: str = "weighted_average",
        scipy_kwargs: Mapping[str, Any] | None = None,
    ) -> Pipeline:
        return self.pipeline.pipe(
            WeightedAverage(
                pred_columns,
                target_column,
                metric_fn=metric_fn,
                is_higher_better=is_higher_better,
                method=method,
                sum_to_one=sum_to_one,
                non_negative=non_negative,
                output_column=output_column,
                scipy_kwargs=scipy_kwargs,
            )
        )

    def linear_ensemble(
        self,
        pred_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        target_column: str,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = False,
        positive: bool = True,
        max_iter: int = 1000,
        output_column: str = "ensemble",
    ) -> Pipeline:
        return self.pipeline.pipe(
            LinearEnsemble(
                pred_columns,
                target_column,
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                positive=positive,
                max_iter=max_iter,
                output_column=output_column,
            )
        )

    # --- END INSERTION MARKER IN OptimizeNameSpace


__all__ = [
    "OptunaOptimizer",
    "ModelFunction",
    "ObjectiveFunction",
    "OptimizeNameSpace",
    "WeightedAverage",
    "LinearEnsemble",
]
