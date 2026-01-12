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
    Literal,
    Mapping,
    Self,
    Sequence,
    Union,
)

import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import HasFeatureImportance, Transformer

if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    from optuna import Study
    from optuna.trial import FrozenTrial
    from sklearn.model_selection import BaseCrossValidator


class BaseLightGBM(Transformer, HasFeatureImportance, ABC):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
        prediction_name: str | Sequence[str] = "prediction",
    ):
        self.params = params
        self.label = label
        self.features_selector = features
        self.categorical_feature = categorical_feature
        self.prediction_name = prediction_name

    @abstractmethod
    def get_booster(self) -> lgb.Booster | dict[str, lgb.Booster]: ...

    def create_train(self, data: DataFrame) -> lgb.Dataset:
        import lightgbm as lgb

        if self.features_selector is None:
            label_columns = data.lazy().select(self.label).collect_schema().names()
            self.features_selector = cs.exclude(*label_columns)

        features = data.select(self.features_selector)
        label = data.select(self.label)
        self.feature_names = features.columns

        return lgb.Dataset(
            features.to_pandas(),
            label.to_pandas(),
            feature_name=self.feature_names,
            categorical_feature=self.categorical_feature,
            free_raw_data=True,
        )

    def create_valid(self, data: DataFrame, reference: lgb.Dataset) -> lgb.Dataset:
        features = data.select(self.feature_names)
        label = data.select(self.label)

        return reference.create_valid(
            features.to_pandas(),
            label.to_pandas(),
        )

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple[lgb.Dataset, list[lgb.Dataset], list[str]]:
        train_dataset = self.create_train(data)
        valid_sets = []
        valid_names = []
        for name, valid_data in more_data.items():
            valid_dataset = self.create_valid(valid_data, train_dataset)
            valid_sets.append(valid_dataset)
            valid_names.append(name)

        valid_sets.append(train_dataset)
        valid_names.append("train")

        return train_dataset, valid_sets, valid_names

    def predict(self, data: DataFrame) -> NDArray:
        import lightgbm as lgb
        import numpy as np

        input_data = data.select(self.feature_names).to_numpy()
        boosters = self.get_booster()

        if isinstance(boosters, lgb.Booster):
            return boosters.predict(input_data)

        preds = [b.predict(input_data) for b in boosters.values()]
        return np.mean(preds, axis=0)

    def transform(self, data: DataFrame) -> DataFrame:
        pred = self.predict(data)
        name = self.prediction_name

        if isinstance(name, str):
            schema = (
                [name]
                if pred.ndim == 1
                else [f"{name}_{i}" for i in range(pred.shape[1])]
            )
        else:
            n_cols = 1 if pred.ndim == 1 else pred.shape[1]
            if len(name) != n_cols:
                raise ValueError(
                    f"prediction_name length ({len(name)}) does not match prediction shape ({n_cols})"
                )
            schema = list(name)

        prediction_df = pl.from_numpy(
            pred,
            schema=schema,
        )

        return prediction_df

    def save(self, save_dir: str | Path) -> None:
        import lightgbm as lgb

        boosters = self.get_booster()
        save_dir = Path(save_dir)

        if isinstance(boosters, lgb.Booster):
            save_lightgbm_booster(boosters, save_dir)
        else:
            for name, booster in boosters.items():
                save_lightgbm_booster(booster, save_dir / name)

    def get_feature_importance(self) -> DataFrame:
        import lightgbm as lgb

        boosters = self.get_booster()
        if isinstance(boosters, lgb.Booster):
            return DataFrame(
                {
                    "feature": boosters.feature_name(),
                    **{
                        importance_type: boosters.feature_importance(
                            importance_type=importance_type
                        )
                        for importance_type in ["gain", "split"]
                    },
                }
            )

        # For CV, average the importance
        all_importances = []
        for booster in boosters.values():
            all_importances.append(
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
                )
            )

        return (
            DataFrame.concat(all_importances)
            .group_by("feature", maintain_order=True)
            .mean()
            .select("feature", "gain", "split")
        )


class LightGBM(BaseLightGBM):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        num_boost_round: int = 100,
        feval: Callable[..., Any] | None = None,
        init_model: Union[str, Path, lgb.Booster] | None = None,
        keep_training_booster: bool = False,
        callbacks: list[Callable] | None = None,
        categorical_feature: list[str] | list[int] | Literal["auto"] = "auto",
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ):
        super().__init__(
            params,
            label,
            features,
            categorical_feature=categorical_feature,
            prediction_name=prediction_name,
        )
        self.num_boost_round = num_boost_round
        self.feval = feval
        self.init_model = init_model
        self.keep_training_booster = keep_training_booster
        self.callbacks = callbacks
        self.save_dir = Path(save_dir) if save_dir else None

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
            num_boost_round=self.num_boost_round,
            feval=self.feval,
            init_model=self.init_model,
            keep_training_booster=self.keep_training_booster,
            callbacks=self.callbacks,
        )

        if self.save_dir:
            self.save(self.save_dir)

        return self

    def get_booster(self) -> "lgb.Booster":
        return self.booster


class LightGBMTuner(BaseLightGBM):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
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
    ):
        super().__init__(
            params,
            label,
            features,
            categorical_feature=categorical_feature,
            prediction_name=prediction_name,
        )
        self.num_boost_round = num_boost_round
        self.feval = feval
        self.keep_training_booster = keep_training_booster
        self.callbacks = callbacks
        self.time_budget = time_budget
        self.sample_size = sample_size
        self.study = study
        self.optuna_callbacks = optuna_callbacks
        self.model_dir = model_dir
        self.show_progress_bar = show_progress_bar
        self.optuna_seed = optuna_seed
        self.save_dir = Path(save_dir) if save_dir else None

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
            num_boost_round=self.num_boost_round,
            feval=self.feval,
            keep_training_booster=self.keep_training_booster,
            callbacks=self.callbacks,
            time_budget=self.time_budget,
            sample_size=self.sample_size,
            study=self.study,
            optuna_callbacks=self.optuna_callbacks,
            model_dir=self.model_dir,
            show_progress_bar=self.show_progress_bar,
            optuna_seed=self.optuna_seed,
        )
        self.tuner.run()
        self.best_booster = self.tuner.get_best_booster()

        if self.save_dir:
            self.save(self.save_dir)

        return self

    def get_booster(self) -> "lgb.Booster":
        return self.best_booster


class LightGBMTunerCV(BaseLightGBM):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        num_boost_round: int = 1000,
        folds: (
            Generator[tuple[int, int], None, None]
            | Iterator[tuple[int, int]]
            | BaseCrossValidator
            | None
        ) = None,
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
    ):
        super().__init__(
            params,
            label,
            features,
            categorical_feature=categorical_feature,
            prediction_name=prediction_name,
        )
        self.num_boost_round = num_boost_round
        self.folds = folds
        self.nfold = nfold
        self.stratified = stratified
        self.shuffle = shuffle
        self.feval = feval
        self.fpreproc = fpreproc
        self.seed = seed
        self.callbacks = callbacks
        self.time_budget = time_budget
        self.sample_size = sample_size
        self.study = study
        self.optuna_callbacks = optuna_callbacks
        self.model_dir = model_dir
        self.show_progress_bar = show_progress_bar
        self.optuna_seed = optuna_seed
        self.prediction_name = prediction_name
        self.save_dir = Path(save_dir) if save_dir else None

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import lightgbm as lgb
        from optuna_integration.lightgbm import LightGBMTunerCV

        train_dataset, _, _ = self.make_train_valid_sets(data)
        self.tuner = LightGBMTunerCV(
            self.params,
            train_dataset,
            num_boost_round=self.num_boost_round,
            folds=self.folds,
            nfold=self.nfold,
            stratified=self.stratified,
            shuffle=self.shuffle,
            feval=self.feval,
            fpreproc=self.fpreproc,
            seed=self.seed,
            callbacks=self.callbacks,
            time_budget=self.time_budget,
            sample_size=self.sample_size,
            study=self.study,
            optuna_callbacks=self.optuna_callbacks,
            model_dir=self.model_dir,
            show_progress_bar=self.show_progress_bar,
            optuna_seed=self.optuna_seed,
            return_cvbooster=True,
        )
        self.tuner.run()
        self.boosters = self.tuner.get_best_booster().boosters

        if self.save_dir:
            self.save(self.save_dir)

        return self

    def get_booster(self) -> dict[str, "lgb.Booster"]:
        return {f"cv{i}": booster for i, booster in enumerate(self.boosters)}


def save_lightgbm_booster(booster: "lgb.Booster", save_dir: str | Path):
    import json

    import lightgbm as lgb
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(save_dir / "model.txt")

    params = booster.params
    with open(save_dir / "params.json", "w") as f:
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
    ).write_csv(save_dir / "feature_importance.csv")

    lgb.plot_importance(booster, importance_type="gain")
    plt.tight_layout()
    plt.savefig(save_dir / "importance_gain.png")
    plt.close()

    lgb.plot_importance(booster, importance_type="split")
    plt.tight_layout()
    plt.savefig(save_dir / "importance_split.png")
    plt.close()
