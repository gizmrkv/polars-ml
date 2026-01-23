from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import lightgbm as lgb


class BaseLightGBM(Transformer, HasFeatureImportance, ABC):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        params: Mapping[str, Any],
    ):
        self._target_selector = target
        self._prediction = (
            [prediction] if isinstance(prediction, str) else list(prediction)
        )
        self._features_selector = (
            features if features is not None else cs.exclude(target)
        )
        self._params = dict(params)

        self._target: list[str] | None = None
        self._features: list[str] | None = None
        self._dataset_params: dict[str, Any] | None = None
        self._booster: lgb.Booster | dict[str, lgb.Booster] | None = None

    @property
    def target(self) -> list[str]:
        if self._target is None:
            raise NotFittedError()
        return self._target

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise NotFittedError()
        return self._features

    @property
    def dataset_params(self) -> dict[str, Any]:
        if self._dataset_params is None:
            raise NotFittedError()
        return self._dataset_params

    @property
    def booster(self) -> lgb.Booster | dict[str, lgb.Booster]:
        if self._booster is None:
            raise NotFittedError()
        return self._booster

    def init_target(self, data: DataFrame) -> list[str]:
        return data.lazy().select(self._target_selector).collect_schema().names()

    def init_features(self, data: DataFrame) -> list[str]:
        return data.lazy().select(self._features_selector).collect_schema().names()

    def init_dataset_params(self, data: DataFrame) -> dict[str, Any]:
        return {}

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple[lgb.Dataset, list[lgb.Dataset], list[str]]:
        import lightgbm as lgb

        train_dataset = lgb.Dataset(
            data.select(*self.features).to_pandas(),
            data.select(*self.target).to_pandas(),
            feature_name=self.features,
            **self.dataset_params,
        )
        valid_sets = []
        valid_names = []
        for name, valid_data in more_data.items():
            valid_dataset = train_dataset.create_valid(
                valid_data.select(*self.features).to_pandas(),
                valid_data.select(*self.target).to_pandas(),
            )
            valid_sets.append(valid_dataset)
            valid_names.append(name)

        valid_sets.append(train_dataset)
        valid_names.append("train")

        return train_dataset, valid_sets, valid_names

    def transform(self, data: DataFrame) -> DataFrame:
        import lightgbm as lgb

        input_data = data.select(*self.features).to_numpy()
        if isinstance(self.booster, lgb.Booster):
            pred = self.booster.predict(input_data)
            return pl.from_numpy(pred, schema=self._prediction)  # type: ignore
        else:
            preds = {
                name: booster.predict(input_data)
                for name, booster in self.booster.items()
            }
            return pl.concat(
                [
                    pl.from_numpy(
                        pred,  # type: ignore
                        schema=[f"{p}_{name}" for p in self._prediction],
                    )
                    for name, pred in preds.items()
                ],
                how="horizontal",
            )

    def save(self, fit_dir: str | Path):
        import lightgbm as lgb

        fit_dir = Path(fit_dir)

        if isinstance(self.booster, lgb.Booster):
            save_lightgbm_booster(self.booster, fit_dir)
        else:
            for name, booster in self.booster.items():
                save_lightgbm_booster(booster, fit_dir / name)

    def get_feature_importance(self) -> DataFrame:
        import lightgbm as lgb

        if isinstance(self.booster, lgb.Booster):
            return DataFrame(
                {
                    "feature": self.booster.feature_name(),
                    **{
                        importance_type: self.booster.feature_importance(
                            importance_type=importance_type
                        )
                        for importance_type in ["gain", "split"]
                    },
                }
            )

        all_importances = []
        for booster in self.booster.values():
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
            pl.concat(all_importances)
            .group_by("feature", maintain_order=True)
            .mean()
            .select("feature", "gain", "split")
        )


class LightGBM(BaseLightGBM):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        params: Mapping[str, Any],
        fit_dir: str | Path | None = None,
        **train_params: Any,
    ):
        super().__init__(target, prediction, features, params=params)
        self._fit_dir = Path(fit_dir) if fit_dir else None
        self._train_params = train_params

    def init_train_params(self, data: DataFrame) -> dict[str, Any]:
        return self._train_params

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import lightgbm as lgb

        self._target = self.init_target(data)
        self._features = self.init_features(data)
        self._dataset_params = self.init_dataset_params(data)
        self._train_params = self.init_train_params(data)

        train_dataset, valid_sets, valid_names = self.make_train_valid_sets(
            data, **more_data
        )

        self._booster = lgb.train(
            self._params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **self._train_params,
        )

        if self._fit_dir:
            self.save(self._fit_dir)

        return self


class LightGBMTuner(BaseLightGBM):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        params: Mapping[str, Any],
        fit_dir: str | Path | None = None,
        **tuner_params: Any,
    ):
        super().__init__(target, prediction, features, params=params)
        self._fit_dir = Path(fit_dir) if fit_dir else None
        self._tuner_params = tuner_params

    def init_tuner_params(self, data: DataFrame) -> dict[str, Any]:
        return self._tuner_params

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        from optuna_integration.lightgbm import LightGBMTuner

        self._target = self.init_target(data)
        self._features = self.init_features(data)
        self._dataset_params = self.init_dataset_params(data)
        self._tuner_params = self.init_tuner_params(data)

        train_dataset, valid_sets, valid_names = self.make_train_valid_sets(
            data, **more_data
        )

        self.tuner = LightGBMTuner(
            self._params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **self._tuner_params,
        )
        self.tuner.run()
        self._booster = self.tuner.get_best_booster()

        if self._fit_dir:
            self.save(self._fit_dir)

        return self


class LightGBMTunerCV(BaseLightGBM):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        params: Mapping[str, Any],
        fit_dir: str | Path | None = None,
        **tuner_params: Any,
    ):
        super().__init__(target, prediction, features, params=params)
        self._fit_dir = Path(fit_dir) if fit_dir else None
        self._tuner_params = tuner_params

    def init_tuner_params(self, data: DataFrame) -> dict[str, Any]:
        return self._tuner_params

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        from optuna_integration.lightgbm import LightGBMTunerCV

        self._target = self.init_target(data)
        self._features = self.init_features(data)
        self._dataset_params = self.init_dataset_params(data)
        self._tuner_params = self.init_tuner_params(data)

        train_dataset, _, _ = self.make_train_valid_sets(data)
        self.tuner = LightGBMTunerCV(
            self._params, train_dataset, return_cvbooster=True, **self._tuner_params
        )
        self.tuner.run()
        self._booster = {
            f"cv_{i}": booster
            for i, booster in enumerate(self.tuner.get_best_booster().boosters)
        }

        if self._fit_dir:
            self.save(self._fit_dir)

        return self


def save_lightgbm_booster(booster: lgb.Booster, fit_dir: str | Path):
    import json

    import lightgbm as lgb
    import matplotlib.pyplot as plt

    fit_dir = Path(fit_dir)
    fit_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(fit_dir / "model.txt")

    params = booster.params
    with open(fit_dir / "params.json", "w") as f:
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
    ).write_csv(fit_dir / "feature_importance.csv")

    lgb.plot_importance(booster, importance_type="gain")
    plt.tight_layout()
    plt.savefig(fit_dir / "importance_gain.png")
    plt.close()

    lgb.plot_importance(booster, importance_type="split")
    plt.tight_layout()
    plt.savefig(fit_dir / "importance_split.png")
    plt.close()
