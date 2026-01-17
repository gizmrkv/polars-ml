from __future__ import annotations

import os
from ctypes import Union
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
from polars._typing import IntoExpr

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    import xgboost as xgb


class XGBoost(Transformer, HasFeatureImportance):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        num_boost_round: int = 10,
        evals: Sequence[tuple[xgb.DMatrix, str]] | None = None,
        obj: xgb.Objective | None = None,
        maximize: bool | None = None,
        early_stopping_rounds: int | None = None,
        evals_result: xgb.TrainingCallback.EvalsLog | None = None,
        verbose_eval: bool | int | None = True,
        xgb_model: str | os.PathLike | xgb.Booster | bytearray | None = None,
        callbacks: Sequence[xgb.TrainingCallback] | None = None,
        custom_metric: xgb.Metric | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ):
        self.label = label
        self.features_selector = features
        self.params = params
        self.num_boost_round = num_boost_round
        self.evals = evals
        self.obj = obj
        self.maximize = maximize
        self.early_stopping_rounds = early_stopping_rounds
        self.evals_result = evals_result
        self.verbose_eval = verbose_eval
        self.xgb_model = xgb_model
        self.callbacks = callbacks
        self.custom_metric = custom_metric
        self.prediction_name = prediction_name
        self.save_dir = Path(save_dir) if save_dir else None

    def get_booster(self) -> xgb.Booster:
        return self.booster

    def create_dmatrix(self, data: DataFrame) -> xgb.DMatrix:
        import xgboost as xgb

        features = data.select(self.feature_names)
        label = data.select(self.label)

        return xgb.DMatrix(
            features.to_pandas(),
            label=label.to_pandas(),
            feature_names=self.feature_names,
        )

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple[xgb.DMatrix, list[tuple[xgb.DMatrix, str]]]:
        dtrain = self.create_dmatrix(data)
        evals = []
        for name, valid_data in more_data.items():
            dvalid = self.create_dmatrix(valid_data)
            evals.append((dvalid, name))

        evals.append((dtrain, "train"))
        return dtrain, evals

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import xgboost as xgb

        if self.features_selector is None:
            label_columns = data.lazy().select(self.label).collect_schema().names()
            self.features_selector = cs.exclude(*label_columns)

        self.feature_names = data.select(self.features_selector).columns

        dtrain, evals = self.make_train_valid_sets(data, **more_data)

        self.booster = xgb.train(
            self.params,
            dtrain,
            self.num_boost_round,
            evals=evals,
            obj=self.obj,
            maximize=self.maximize,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=self.verbose_eval,
            xgb_model=self.xgb_model,
            callbacks=self.callbacks,
            custom_metric=self.custom_metric,
        )

        if self.save_dir:
            self.save(self.save_dir)

        return self

    def predict(self, data: DataFrame) -> NDArray:
        import xgboost as xgb

        if not hasattr(self, "feature_names"):
            raise NotFittedError()

        input_data = xgb.DMatrix(
            data.select(self.feature_names).to_pandas(),
            feature_names=self.feature_names,
        )
        booster = self.get_booster()

        return booster.predict(input_data)

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
        booster = self.get_booster()
        save_dir = Path(save_dir)
        save_xgboost_booster(booster, save_dir)

    def get_feature_importance(self) -> DataFrame:
        booster = self.get_booster()
        feature_names = booster.feature_names
        if feature_names is None:
            feature_names = self.feature_names

        importance_types = [
            "gain",
            "weight",
            "cover",
            "total_gain",
            "total_cover",
        ]

        importance_data = {"feature": feature_names}
        for it in importance_types:
            scores = booster.get_score(importance_type=it)
            importance_data[it] = [scores.get(fn, 0.0) for fn in feature_names]

        return DataFrame(importance_data)


def save_xgboost_booster(booster: xgb.Booster, save_dir: str | Path):
    import json

    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(save_dir / "model.json")

    config = json.loads(booster.save_config())
    with open(save_dir / "params.json", "w") as f:
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
        ).write_csv(save_dir / "feature_importance.csv")

    import xgboost as xgb

    xgb.plot_importance(booster, importance_type="gain")
    plt.savefig(save_dir / "importance_gain.png")
    plt.tight_layout()
    plt.close()

    xgb.plot_importance(booster, importance_type="weight")
    plt.savefig(save_dir / "importance_weight.png")
    plt.tight_layout()
    plt.close()
