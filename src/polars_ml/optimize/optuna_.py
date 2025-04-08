from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Self,
    Union,
)

from polars import DataFrame

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import optuna
    import optuna.storages.journal


class ModelFunction(Protocol):
    def __call__(
        self, *args: Any, trial: Optional["optuna.Trial"] = None, **kwargs: Any
    ) -> PipelineComponent: ...


class ObjectiveFunction(Protocol):
    def __call__(
        self,
        model: PipelineComponent,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
        *,
        trial: Optional["optuna.Trial"] = None,
    ) -> Any: ...


class OptunaOptimizer(PipelineComponent):
    def __init__(
        self,
        model_fn: ModelFunction,
        objective_fn: ObjectiveFunction,
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
    ):
        self.model_fn = model_fn
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.sampler = sampler
        self.pruner = pruner
        self.study_name = study_name
        self.is_higher_better = is_higher_better
        self.load_if_exists = load_if_exists
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.gc_after_trial = gc_after_trial
        self.show_progress_bar = show_progress_bar

        if isinstance(storage, (str, Path)):
            file_path = storage if isinstance(storage, str) else storage.as_posix()
            storage = optuna.storages.journal.JournalStorage(
                optuna.storages.journal.JournalFileBackend(file_path)
            )
        self.storage = storage

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        study = optuna.create_study(
            storage=self.storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name,
            direction="maximize" if self.is_higher_better else "minimize",
            load_if_exists=self.load_if_exists,
        )

        def wrap_objective(
            objective: ObjectiveFunction,
            search_space: Mapping[str, Mapping[str, Any]],
        ) -> Callable[["optuna.Trial"], Any]:
            def _objective(trial: optuna.Trial) -> Any:
                return objective(
                    self.model_fn(
                        **self.suggest_sample(trial, search_space), trial=trial
                    ),
                    deepcopy(data),
                    deepcopy(validation_data),
                    trial=trial,
                )

            return _objective

        study.optimize(
            wrap_objective(self.objective_fn, self.search_space),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            gc_after_trial=self.gc_after_trial,
            show_progress_bar=self.show_progress_bar,
        )

        self.best_params = study.best_params
        self.best_model = self.model_fn(**self.best_params)
        self.best_model.fit(data, validation_data)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return self.best_model.transform(data)

    def is_valid_clause(
        self,
        target: Mapping[str, Any],
        *required: str,
        optional: Iterable[str] | None = None,
    ) -> bool:
        optional_keys = set(optional) if optional is not None else set()

        if not all(key in target for key in required):
            return False

        target_keys = set(target.keys())
        valid_keys = set(required) | optional_keys
        return target_keys.issubset(valid_keys)

    def parse_grid_search_space(
        self, search_space: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, list[Any]]:
        grid_space: dict[str, list[Any]] = {}
        for var_name, var_space in search_space.items():
            if self.is_valid_clause(var_space, "values"):
                values = var_space["values"]
                if isinstance(values, dict):
                    grid_space |= {
                        f"{var_name}_{k}": v
                        for k, v in self.parse_grid_search_space(values).items()
                    }
                elif isinstance(values, (list, tuple)):
                    grid_space[var_name] = list(values)
            elif self.is_valid_clause(var_space, "min", "max"):
                min_v = var_space["min"]
                max_v = var_space["max"]
                if isinstance(min_v, int) and isinstance(max_v, int):
                    grid_space[var_name] = list(range(min_v, max_v + 1))
                else:
                    raise ValueError(
                        f"Invalid configuration for '{var_name}': 'min' and 'max' must be integers."
                    )

        return grid_space

    def suggest_sample(
        self,
        trial: "optuna.Trial",
        search_space: Mapping[str, Mapping[str, Any]],
        prefix: str = "",
    ) -> dict[str, Any]:
        sample: dict[str, Any] = {}
        for var_name, var_space in search_space.items():
            if self.is_valid_clause(var_space, "value"):
                sample[var_name] = var_space["value"]

            elif self.is_valid_clause(var_space, "values"):
                values = var_space["values"]

                if isinstance(values, dict):
                    sample[var_name] = self.suggest_sample(
                        trial, values, prefix=prefix + var_name + "/"
                    )

                elif isinstance(values, (list, tuple)):
                    sample[var_name] = trial.suggest_categorical(
                        prefix + var_name, values
                    )

                else:
                    raise ValueError(
                        f"Invalid configuration for '{var_name}': 'values' must be a list or a dictionary. Got {values}"
                    )

            elif self.is_valid_clause(
                var_space, "min", "max", optional=["log", "step"]
            ):
                min_v = var_space["min"]
                max_v = var_space["max"]

                if isinstance(min_v, int) and isinstance(max_v, int):
                    step = var_space.get("step", 1)
                    sample[var_name] = trial.suggest_int(
                        prefix + var_name, min_v, max_v, step=step
                    )

                elif isinstance(min_v, float) and isinstance(max_v, float):
                    log = var_space.get("log", False)
                    step = var_space.get("step", None)
                    sample[var_name] = trial.suggest_float(
                        prefix + var_name, min_v, max_v, log=log, step=step
                    )

                else:
                    raise ValueError(
                        f"Invalid configuration for '{var_name}': 'min' and 'max' must be of the same type (both int or both float). Got {min_v} and {max_v}"
                    )

            else:
                raise ValueError(
                    f"Invalid configuration for '{var_name}': must contain 'value', 'values', or 'min' and 'max'. Got {var_space}"
                )

        return sample
