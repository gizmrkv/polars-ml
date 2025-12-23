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

from polars_ml.base import Transformer

if TYPE_CHECKING:
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import BaseCrossValidator


class BaseLightGBM(Transformer, ABC):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str = "prediction",
    ):
        self.label = label
        self.params = params
        self.features_selector = features
        self.prediction_name = prediction_name

    @abstractmethod
    def get_booster(self) -> lgb.Booster | dict[str, lgb.Booster]:
        """Return the trained booster(s)."""
        ...

    def create_train(self, data: DataFrame) -> lgb.Dataset:
        """Create a LightGBM Dataset for training."""
        import lightgbm as lgb

        if self.features_selector is None:
            # Determine features lazily if not specified
            label_cols = data.lazy().select(self.label).collect_schema().names()
            self.features_selector = cs.exclude(*label_cols)

        features = data.select(self.features_selector)
        label = data.select(self.label)
        self.feature_names = features.columns

        params: dict[str, Any] = {
            "data": features.to_pandas(),
            "label": label.to_pandas(),
        }

        return lgb.Dataset(
            **params,
            feature_name=self.feature_names,
            free_raw_data=True,
        )

    def create_valid(self, data: DataFrame, reference: lgb.Dataset) -> lgb.Dataset:
        """Create a LightGBM Dataset for validation based on a reference training dataset."""
        # Use stored feature_names to ensure consistency
        features = data.select(self.feature_names)
        label = data.select(self.label)

        params: dict[str, Any] = {
            "data": features.to_pandas(),
            "label": label.to_pandas(),
        }

        return reference.create_valid(**params)

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple[lgb.Dataset, list[lgb.Dataset], list[str]]:
        """Prepare training and validation datasets."""
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
        """Generate raw predictions using the booster(s). Matches LightGBM.Booster.predict annotation."""
        import lightgbm as lgb
        import numpy as np

        # Use stored feature_names to ensure consistency with training
        input_data = data.select(self.feature_names).to_numpy()
        boosters = self.get_booster()

        if isinstance(boosters, lgb.Booster):
            return boosters.predict(input_data)

        # Ensemble (average) predictions if multiple boosters exist
        preds = [b.predict(input_data) for b in boosters.values()]
        return np.mean(preds, axis=0)

    def transform(self, data: DataFrame) -> DataFrame:
        """Transform the data by adding prediction columns."""
        pred = self.predict(data)
        name = self.prediction_name

        prediction_df = pl.from_numpy(
            pred,
            schema=[name]
            if pred.ndim == 1
            else [f"{name}_{i}" for i in range(pred.shape[1])],
        )

        return pl.concat([data, prediction_df], how="horizontal")

    def save(self, out_dir: str | Path) -> None:
        """Save the booster(s) and relevant metadata."""
        import lightgbm as lgb

        boosters = self.get_booster()
        out_dir = Path(out_dir)

        if isinstance(boosters, lgb.Booster):
            save_lightgbm_booster(boosters, out_dir)
        else:
            for name, booster in boosters.items():
                save_lightgbm_booster(booster, out_dir / name)


class LightGBM(BaseLightGBM):
    def __init__(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ):
        super().__init__(
            params,
            label,
            features,
            prediction_name=prediction_name,
        )
        self.out_dir = Path(out_dir) if out_dir else None

    def get_train_params(self) -> Mapping[str, Any]:
        """Return parameters for LightGBM training. Override this for customization."""
        return {}

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
            **self.get_train_params(),
        )

        if self.out_dir:
            self.save(self.out_dir)

        return self

    def get_booster(self) -> "lgb.Booster":
        return self.booster


class LightGBMTuner(BaseLightGBM):
    def __init__(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ):
        super().__init__(
            params,
            label,
            features,
            prediction_name=prediction_name,
        )
        self.out_dir = Path(out_dir) if out_dir else None

    def get_tuner_params(self) -> Mapping[str, Any]:
        """Return parameters for Optuna LightGBMTuner. Override this for customization."""
        return {}

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
            **self.get_tuner_params(),
        )
        self.tuner.run()
        self.best_booster = self.tuner.get_best_booster()

        if self.out_dir:
            self.save(self.out_dir)

        return self

    def get_booster(self) -> "lgb.Booster":
        return self.best_booster


class LightGBMTunerCV(BaseLightGBM):
    def __init__(
        self,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
        **params: Any,
    ):
        super().__init__(
            params,
            label,
            features,
            prediction_name=prediction_name,
        )
        self.prediction_name = prediction_name
        self.out_dir = Path(out_dir) if out_dir else None

    def get_tuner_params(self) -> Mapping[str, Any]:
        """Return parameters for Optuna LightGBMTunerCV. Override this for customization."""
        return {}

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import lightgbm as lgb
        from optuna_integration.lightgbm import LightGBMTunerCV

        train_dataset, _, _ = self.make_train_valid_sets(data)
        self.tuner = LightGBMTunerCV(
            self.params,
            train_dataset,
            return_cvbooster=True,
            **self.get_tuner_params(),
        )
        self.tuner.run()
        self.boosters = self.tuner.get_best_booster().boosters

        if self.out_dir:
            self.save(self.out_dir)

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
