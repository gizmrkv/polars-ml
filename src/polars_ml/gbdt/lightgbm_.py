from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Union

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame, Series

from polars_ml.component import Component

if TYPE_CHECKING:
    import lightgbm as lgb


class LightGBM(Component):
    def __init__(
        self,
        label: str,
        params: dict[str, Any],
        *,
        exclude: str | Iterable[str] | None = None,
        weight: str | None = None,
        group: Callable[[DataFrame], DataFrame] | None = None,
        init_score: str | None = None,
        categorical_feature: list[str] | None = None,
        dataset_params: dict[str, Any] | None = None,
        position: str | None = None,
        num_boost_round: int = 100,
        feval: Callable[..., Any] | None = None,
        init_model: Union[str, Path, "lgb.Booster"] | None = None,
        keep_training_booster: bool = False,
        callbacks: list[Callable[..., Any]] | None = None,
        start_iteration: int = 0,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        prediction_name: str = "lightgbm",
        include_input: bool = True,
    ):
        self.label = label
        self.params = params
        if isinstance(exclude, str):
            self.exclude = [exclude]
        elif isinstance(exclude, Iterable):
            self.exclude = list(exclude)
        else:
            self.exclude = []
        self.weight = weight
        self.group = group
        self.init_score = init_score
        self.categorical_feature = categorical_feature
        self.dataset_params = dataset_params
        self.position = position
        self.num_boost_round = num_boost_round
        self.feval = feval
        self.init_model = init_model
        self.keep_training_booster = keep_training_booster
        self.callbacks = callbacks
        self.start_iteration = start_iteration
        self.num_iteration = num_iteration
        self.raw_score = raw_score
        self.pred_leaf = pred_leaf
        self.pred_contrib = pred_contrib
        self.data_has_header = data_has_header
        self.validate_features = validate_features
        self.prediction_name = prediction_name
        self.include_input = include_input

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Component:
        import lightgbm as lgb

        train_features = data.select(pl.exclude(self.label, *self.exclude))
        train_label = data.select(self.label)
        self.feature_names = train_features.columns

        train_dataset = lgb.Dataset(
            train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            feature_name=self.feature_names,
            weight=(
                data.select(self.weight).to_numpy().squeeze()
                if self.weight is not None
                else None
            ),
            group=(
                self.group(data).to_numpy().squeeze()
                if self.group is not None
                else None
            ),
            init_score=(
                data.select(self.init_score).to_numpy().squeeze()
                if self.init_score is not None
                else None
            ),
            categorical_feature=self.categorical_feature or "auto",
            free_raw_data=True,
            params=self.dataset_params,
            position=(
                data.select(self.position).to_numpy().squeeze()
                if self.position is not None
                else None
            ),
        )

        valid_sets = []
        valid_names = []
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.feature_names)
                valid_label = validation_data.select(self.label)

                valid_dataset = train_dataset.create_valid(
                    valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    weight=(
                        validation_data.select(self.weight).to_numpy().squeeze()
                        if self.weight is not None
                        else None
                    ),
                    group=(
                        self.group(validation_data).to_numpy().squeeze()
                        if self.group is not None
                        else None
                    ),
                    init_score=(
                        validation_data.select(self.init_score).to_numpy().squeeze()
                        if self.init_score is not None
                        else None
                    ),
                    params=self.dataset_params,
                    position=(
                        validation_data.select(self.position).to_numpy().squeeze()
                        if self.position is not None
                        else None
                    ),
                )
                valid_sets.append(valid_dataset)
                valid_names.append("valid")
            else:
                for name, valid_data in validation_data.items():
                    valid_features = valid_data.select(self.feature_names)
                    valid_label = valid_data.select(self.label)

                    valid_dataset = train_dataset.create_valid(
                        valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
                        weight=(
                            valid_data.select(self.weight).to_numpy().squeeze()
                            if self.weight is not None
                            else None
                        ),
                        group=(
                            self.group(valid_data).to_numpy().squeeze()
                            if self.group is not None
                            else None
                        ),
                        init_score=(
                            valid_data.select(self.init_score).to_numpy().squeeze()
                            if self.init_score is not None
                            else None
                        ),
                        params=self.dataset_params,
                        position=(
                            valid_data.select(self.position).to_numpy().squeeze()
                            if self.position is not None
                            else None
                        ),
                    )
                    valid_sets.append(valid_dataset)
                    valid_names.append(name)

        valid_sets.append(train_dataset)
        valid_names.append("train")

        self.model = lgb.train(
            self.params,
            train_dataset,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=self.feval,
            init_model=self.init_model,
            keep_training_booster=self.keep_training_booster,
            callbacks=self.callbacks,
        )

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.feature_names)
        pred: NDArray[Any] = self.model.predict(
            input.to_numpy(),
            start_iteration=self.start_iteration,
            num_iteration=self.num_iteration,
            raw_score=self.raw_score,
            pred_leaf=self.pred_leaf,
            pred_contrib=self.pred_contrib,
            data_has_header=self.data_has_header,
            validate_features=self.validate_features,
        )  # type: ignore

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
