from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from .optuna_ import ModelFunction, ObjectiveFunction, OptunaOptimizer

if TYPE_CHECKING:
    import optuna

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

    # --- END INSERTION MARKER IN OptimizeNameSpace


__all__ = ["OptunaOptimizer", "ModelFunction", "ObjectiveFunction", "OptimizeNameSpace"]
