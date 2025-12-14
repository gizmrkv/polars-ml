from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Self,
    TypedDict,
    Union,
)

import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import Transformer

if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import BaseCrossValidator


class LGBDatasetParams(TypedDict, total=False):
    weight: Callable[[DataFrame], NDArray[Any]]
    group: Callable[[DataFrame], NDArray[Any]]
    init_score: Callable[[DataFrame], NDArray[Any]]
    categorical_feature: list[str]
    params: dict[str, Any]
    position: Callable[[DataFrame], NDArray[Any]]


class LGBTrainParams(TypedDict, total=False):
    num_boost_round: int
    feval: Callable[..., Any]
    init_model: Union[str, Path, "lgb.Booster"]
    keep_training_booster: bool
    callbacks: list[Callable[..., Any]]


class LGBPredictParams(TypedDict, total=False):
    start_iteration: int
    num_iteration: int
    raw_score: bool
    pred_leaf: bool
    pred_contrib: bool
    data_has_header: bool
    validate_features: bool


class LGBTunerParams(TypedDict, total=False):
    num_boost_round: int
    feval: Callable[..., Any]
    categorical_feature: str
    keep_training_booster: bool
    callbacks: list[Callable[..., Any]]
    time_budget: int
    sample_size: int
    study: "optuna.study.Study"
    optuna_callbacks: list[
        Callable[["optuna.study.Study", "optuna.trial.FrozenTrial"], None]
    ]
    model_dir: str
    show_progress_bar: bool
    optuna_seed: int


class LGBTunerCVParams(TypedDict, total=False):
    num_boost_round: int
    folds: (
        Generator[tuple[int, int], None, None]
        | Iterator[tuple[int, int]]
        | "BaseCrossValidator"
    )
    nfold: int
    stratified: bool
    shuffle: bool
    feval: Callable[..., Any]
    categorical_feature: str
    fpreproc: Callable[..., Any]
    seed: int
    callbacks: list[Callable[..., Any]]
    time_budget: int
    sample_size: int
    study: "optuna.study.Study"
    optuna_callbacks: list[
        Callable[["optuna.study.Study", "optuna.trial.FrozenTrial"], None]
    ]
    show_progress_bar: bool
    model_dir: str
    optuna_seed: int


class BaseLightGBM(Transformer, ABC):
    def __init__(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        *,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
    ):
        self.label = label
        self.params = params
        self.features_selector = features
        self.dataset_params = dataset_params or {}
        self.predict_params = predict_params or {}
        self.prediction_name = prediction_name

    @abstractmethod
    def get_booster(self) -> Union["lgb.Booster", dict[str, "lgb.Booster"]]: ...

    def make_dataset_params(self, data: DataFrame) -> dict[str, Any]:
        train_label = data.select(self.label)
        if self.features_selector is None:
            self.features_selector = cs.exclude(*train_label.columns)

        train_features = data.select(self.features_selector)

        params: dict[str, Any] = {
            "data": train_features.to_pandas(),
            "label": train_label.to_pandas(),
        }
        if weight := self.dataset_params.get("weight"):
            params["weight"] = weight(data)
        if group := self.dataset_params.get("group"):
            params["group"] = group(data)
        if init_score := self.dataset_params.get("init_score"):
            params["init_score"] = init_score(data)
        if categorical_feature := self.dataset_params.get("categorical_feature"):
            params["categorical_feature"] = categorical_feature
        if position := self.dataset_params.get("position"):
            params["position"] = position(data)

        return params

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple["lgb.Dataset", list["lgb.Dataset"], list[str]]:
        import lightgbm as lgb

        self.feature_names = (
            data.lazy().select(self.features_selector).collect_schema().names()
        )

        dataset_params = self.make_dataset_params(data)
        train_dataset = lgb.Dataset(
            **dataset_params,
            feature_name=self.feature_names,
            free_raw_data=True,
        )
        valid_sets = []
        valid_names = []
        for name, valid_data in more_data.items():
            dataset_params = self.make_dataset_params(valid_data)
            valid_dataset = train_dataset.create_valid(**dataset_params)
            valid_sets.append(valid_dataset)
            valid_names.append(name)

        valid_sets.append(train_dataset)
        valid_names.append("train")

        return train_dataset, valid_sets, valid_names

    def transform(self, data: DataFrame) -> DataFrame:
        import lightgbm as lgb

        input = data.select(self.features_selector)
        boosters = self.get_booster()
        if isinstance(boosters, lgb.Booster):
            boosters = {self.prediction_name: boosters}

        predictions = []
        for name, b in boosters.items():
            pred: NDArray[Any] = b.predict(input.to_numpy(), **self.predict_params)  # type: ignore
            predictions.append(
                pl.from_numpy(
                    pred,
                    schema=[name]
                    if pred.ndim == 1
                    else [f"{name}_{i}" for i in range(pred.shape[1])],
                )
            )

        return pl.concat([data, *predictions], how="horizontal")


class LightGBM(BaseLightGBM):
    def __init__(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        *,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        train_params: LGBTrainParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ):
        super().__init__(
            label=label,
            params=params,
            features=features,
            dataset_params=dataset_params,
            predict_params=predict_params,
            prediction_name=prediction_name,
        )
        self.train_params = train_params or {}
        self.out_dir = Path(out_dir) if out_dir else None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import lightgbm as lgb

        train_dataset, valid_sets, valid_names = self.make_train_valid_sets(
            data, **more_data
        )

        self.booster = lgb.train(
            self.params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **self.train_params,
        )

        if self.out_dir:
            save_lightgbm_booster(self.booster, self.out_dir)

        return self

    def get_booster(self) -> "lgb.Booster":
        return self.booster


class LightGBMTuner(BaseLightGBM):
    def __init__(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        *,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        tuner_params: LGBTunerParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ):
        super().__init__(
            label=label,
            params=params,
            features=features,
            dataset_params=dataset_params,
            predict_params=predict_params,
            prediction_name=prediction_name,
        )
        self.tuner_params = tuner_params or {}
        self.out_dir = Path(out_dir) if out_dir else None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        from optuna_integration.lightgbm import LightGBMTuner

        train_dataset, valid_sets, valid_names = self.make_train_valid_sets(
            data, **more_data
        )

        self.tuner = LightGBMTuner(
            self.params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **self.tuner_params,
        )
        self.tuner.run()
        self.best_booster = self.tuner.get_best_booster()

        if self.out_dir:
            save_lightgbm_booster(self.best_booster, self.out_dir)

        return self

    def get_booster(self) -> "lgb.Booster":
        return self.best_booster


class LightGBMTunerCV(BaseLightGBM):
    def __init__(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        *,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: LGBDatasetParams | None = None,
        tuner_params: LGBTunerCVParams | None = None,
        predict_params: LGBPredictParams | None = None,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ):
        super().__init__(
            label=label,
            params=params,
            features=features,
            dataset_params=dataset_params,
            predict_params=predict_params,
            prediction_name=prediction_name,
        )
        self.tuner_params = tuner_params or {}
        self.prediction_name = prediction_name
        self.out_dir = Path(out_dir) if out_dir else None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import lightgbm as lgb
        from optuna_integration.lightgbm import LightGBMTunerCV

        train_dataset, _, _ = self.make_train_valid_sets(data)
        self.tuner = LightGBMTunerCV(
            self.params,
            train_dataset,
            return_cvbooster=True,
            **self.tuner_params,
        )
        self.tuner.run()
        self.boosters = self.tuner.get_best_booster().boosters

        if self.out_dir:
            for i, booster in enumerate(self.boosters):
                save_lightgbm_booster(booster, self.out_dir / f"cv{i}")

        return self

    def get_booster(self) -> dict[str, "lgb.Booster"]:
        return {f"cv{i}": booster for i, booster in enumerate(self.boosters)}


def save_lightgbm_booster(booster: "lgb.Booster", out_dir: str | Path):
    import json

    import lightgbm as lgb
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.txt")

    params = booster.params
    with open(out_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    DataFrame(
        {
            "feature": booster.feature_name(),
            **{
                importance_type: booster.feature_importance(
                    importance_type=importance_type
                )
                for importance_type in ["gain", "split"]
            },
        }
    ).write_csv(out_dir / "feature_importance.csv")

    lgb.plot_importance(booster, importance_type="gain")
    plt.savefig(out_dir / "importance_gain.png")
    plt.tight_layout()
    plt.close()

    lgb.plot_importance(booster, importance_type="split")
    plt.savefig(out_dir / "importance_split.png")
    plt.tight_layout()
    plt.close()
