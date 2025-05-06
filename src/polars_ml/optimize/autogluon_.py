from typing import TYPE_CHECKING, Any, Literal, Mapping

from numpy.typing import NDArray
from polars import DataFrame, Series

from polars_ml.component import Component

if TYPE_CHECKING:
    from autogluon.tabular import TabularPredictor


class AutoGluon(Component):
    def __init__(
        self,
        predictor: "TabularPredictor",
        *,
        time_limit: float | None = None,
        presets: list[str] | str | None = None,
        hyperparameters: dict[str, Any] | str | None = None,
        feature_metadata: str = "infer",
        infer_limit: float | None = None,
        infer_limit_batch_size: int | None = None,
        fit_weighted_ensemble: bool = True,
        fit_full_last_level_weighted_ensemble: bool = True,
        full_weighted_ensemble_additionally: bool = False,
        dynamic_stacking: bool | str = False,
        calibrate_decision_threshold: bool | str = "auto",
        num_cpus: int | str = "auto",
        num_gpus: int | str = "auto",
        fit_strategy: Literal["sequential", "parallel"] = "sequential",
        memory_limit: float | str = "auto",
        callbacks: list[Any] | None = None,
        prediction_name: str = "autogluon",
        include_input: bool = True,
    ):
        self.predictor = predictor

        self.fit_params = {
            "time_limit": time_limit,
            "presets": presets,
            "hyperparameters": hyperparameters,
            "feature_metadata": feature_metadata,
            "infer_limit": infer_limit,
            "infer_limit_batch_size": infer_limit_batch_size,
            "fit_weighted_ensemble": fit_weighted_ensemble,
            "fit_full_last_level_weighted_ensemble": fit_full_last_level_weighted_ensemble,
            "full_weighted_ensemble_additionally": full_weighted_ensemble_additionally,
            "dynamic_stacking": dynamic_stacking,
            "calibrate_decision_threshold": calibrate_decision_threshold,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "fit_strategy": fit_strategy,
            "memory_limit": memory_limit,
            "callbacks": callbacks,
        }

        self.prediction_name = prediction_name
        self.include_input = include_input

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "AutoGluon":
        data_pd = data.to_pandas()
        if validation_data is None:
            self.predictor.fit(data_pd, **self.fit_params)
        elif isinstance(validation_data, DataFrame):
            valid_data_pd = validation_data.to_pandas()
            self.predictor.fit(data_pd, tuning_data=valid_data_pd, **self.fit_params)
        else:
            raise NotImplementedError(
                "AutoGluon does not support multiple validation datasets."
            )

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        data_pd = data.to_pandas()
        pred: NDArray[Any] = self.predictor.predict(data_pd)  # type: ignore
        if pred.ndim == 1:
            columns = [Series(self.prediction_name, pred)]
        else:
            n = pred.shape[1]
            zero_pad = len(str(n))
            columns = [
                Series(f"{self.prediction_name}_{i:0{zero_pad}d}", pred[:, i])
                for i in range(n)
            ]

        if self.include_input:
            return data.with_columns(columns)
        else:
            return DataFrame(columns)
