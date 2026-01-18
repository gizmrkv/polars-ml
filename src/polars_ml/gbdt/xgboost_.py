from __future__ import annotations

import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Self,
    Sequence,
)

import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import xgboost as xgb


class XGBoost(Transformer, HasFeatureImportance):
    def __init__(
        self,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        params: Mapping[str, Any],
        fit_dir: str | Path | None = None,
        **train_params: Any,
    ) -> None:
        self._target_selector = target
        self._prediction = (
            [prediction] if isinstance(prediction, str) else list(prediction)
        )
        self._features_selector = features
        self._params = dict(params)
        self._fit_dir = Path(fit_dir) if fit_dir else None
        self._train_params = train_params

        self._target: list[str] | None = None
        self._features: list[str] | None = None
        self._dmatrix_params: dict[str, Any] | None = None
        self._booster: xgb.Booster | None = None

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
    def dmatrix_params(self) -> dict[str, Any]:
        if self._dmatrix_params is None:
            raise NotFittedError()
        return self._dmatrix_params

    @property
    def booster(self) -> xgb.Booster:
        if self._booster is None:
            raise NotFittedError()
        return self._booster

    def init_target(self, data: DataFrame) -> list[str]:
        return data.lazy().select(self._target_selector).collect_schema().names()

    def init_features(self, data: DataFrame) -> list[str]:
        return data.lazy().select(self._features_selector).collect_schema().names()

    def init_train_params(self, data: DataFrame) -> dict[str, Any]:
        return self._train_params

    def init_dmatrix_params(self, data: DataFrame) -> dict[str, Any]:
        return {}

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple[xgb.DMatrix, list[tuple[xgb.DMatrix, str]]]:
        dtrain = xgb.DMatrix(
            data.select(*self.features).to_pandas(),
            label=data.select(*self.target).to_pandas(),
            feature_names=self.features,
            **self.dmatrix_params,
        )
        evals = []
        for name, valid_data in more_data.items():
            dvalid = xgb.DMatrix(
                valid_data.select(*self.features).to_pandas(),
                label=valid_data.select(*self.target).to_pandas(),
                feature_names=self.features,
                **self.dmatrix_params,
            )
            evals.append((dvalid, name))

        evals.append((dtrain, "train"))
        return dtrain, evals

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self._target = self.init_target(data)
        self._features = self.init_features(data)
        self._dmatrix_params = self.init_dmatrix_params(data)
        self._train_params = self.init_train_params(data)

        dtrain, evals = self.make_train_valid_sets(data, **more_data)
        self._booster = xgb.train(
            self._params, dtrain, evals=evals, **self._train_params
        )

        if self._fit_dir:
            self.save(self._fit_dir)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_data = xgb.DMatrix(
            data.select(self.features).to_pandas(),
            feature_names=self.features,
        )
        pred = self.booster.predict(input_data)
        return pl.from_numpy(pred, schema=self._prediction)

    def save(self, fit_dir: str | Path) -> None:
        booster = self.booster
        fit_dir = Path(fit_dir)
        save_xgboost_booster(booster, fit_dir)

    def get_feature_importance(self) -> DataFrame:
        importance_types = [
            "gain",
            "weight",
            "cover",
            "total_gain",
            "total_cover",
        ]

        importance_data: dict[str, Any] = {"feature": self.features}
        for it in importance_types:
            scores = self.booster.get_score(importance_type=it)
            importance_data[it] = [scores.get(fn, 0.0) for fn in self.features]

        return DataFrame(importance_data)


def save_xgboost_booster(booster: xgb.Booster, fit_dir: str | Path) -> None:
    import json

    import matplotlib.pyplot as plt

    fit_dir = Path(fit_dir)
    fit_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(fit_dir / "model.json")

    config = json.loads(booster.save_config())
    with open(fit_dir / "params.json", "w") as f:
        json.dump(config, f, indent=4)

    if feature_names := booster.feature_names:
        DataFrame(
            {
                "feature": feature_names,
                **{
                    importance_type: [
                        booster.get_score(importance_type=importance_type).get(fn, 0.0)
                        for fn in feature_names
                    ]
                    for importance_type in [
                        "gain",
                        "weight",
                        "cover",
                        "total_gain",
                        "total_cover",
                    ]
                },
            }
        ).write_csv(fit_dir / "feature_importance.csv")

    xgb.plot_importance(booster, importance_type="gain")
    plt.savefig(fit_dir / "importance_gain.png")
    plt.tight_layout()
    plt.close()

    xgb.plot_importance(booster, importance_type="weight")
    plt.savefig(fit_dir / "importance_weight.png")
    plt.tight_layout()
    plt.close()
