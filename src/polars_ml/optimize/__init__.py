from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from .optuna_ import ModelFunction, ObjectiveFunction, OptunaOptimizer

if TYPE_CHECKING:
    import optuna

    from polars_ml import Pipeline


class OptimizeNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def optuna(
        self,
        model_fn: ModelFunction,
        objective: ObjectiveFunction,
        search_space: Mapping[str, Mapping[str, Any]],
        *,
        sampler: Optional["optuna.samplers.BaseSampler"] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        study_name: str | None = None,
        is_higher_better: bool = False,
        load_if_exists: bool = False,
        n_trials: int | None = None,
        timeout: int | None = None,
        n_jobs: int = 1,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
        storage: Union[str, Path, "optuna.storages.BaseStorage"] = "./journal.log",
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            OptunaOptimizer(
                model_fn,
                objective,
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
            ),
            component_name=component_name,
        )
