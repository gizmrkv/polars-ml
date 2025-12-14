from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Self,
    Sequence,
    TypedDict,
    Union,
)

import polars as pl
import polars.selectors as cs
from numpy.typing import ArrayLike, NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import Transformer

if TYPE_CHECKING:
    import xgboost as xgb


class XGBDatasetParams(TypedDict, total=False):
    weight: Callable[[DataFrame], ArrayLike]
    base_margin: Callable[[DataFrame], ArrayLike]
    missing: float
    silent: bool
    nthread: int
    group: Callable[[DataFrame], ArrayLike]
    qid: Callable[[DataFrame], ArrayLike]
    label_lower_bound: Callable[[DataFrame], ArrayLike]
    label_upper_bound: Callable[[DataFrame], ArrayLike]
    feature_weights: Callable[[DataFrame], ArrayLike]
    enable_categorical: bool


class XGBTrainParams(TypedDict, total=False):
    num_boost_round: int
    obj: "xgb.core.Objective"
    feval: "xgb.core.Metric"
    maximize: bool
    early_stopping_rounds: int
    evals_result: Any
    verbose_eval: Union[bool, int]
    xgb_model: Union[str, os.PathLike[Any], "xgb.Booster", bytearray]
    callbacks: Sequence[Any]
    custom_metric: "xgb.core.Metric"


class XGBPredictParams(TypedDict, total=False):
    iteration_range: tuple[int, int]
    output_margin: bool
    ntree_limit: int
    pred_leaf: bool
    pred_contribs: bool
    approx_contribs: bool
    pred_interactions: bool
    validate_features: bool
    iteration_range: tuple[int, int]
    strict_shape: bool


class BaseXGBoost(Transformer, ABC):
    def __init__(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        *,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: XGBDatasetParams | None = None,
        predict_params: XGBPredictParams | None = None,
        prediction_name: str = "prediction",
    ):
        self.label = label
        self.params = params
        self.features_selector = features
        self.dataset_params = dataset_params or {}
        self.predict_params = predict_params or {}
        self.prediction_name = prediction_name

    @abstractmethod
    def get_booster(self) -> Union["xgb.Booster", dict[str, "xgb.Booster"]]: ...

    def make_dmatrix_params(self, data: DataFrame) -> dict[str, Any]:
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
        if base_margin := self.dataset_params.get("base_margin"):
            params["base_margin"] = base_margin(data)
        if group := self.dataset_params.get("group"):
            params["group"] = group(data)
        if qid := self.dataset_params.get("qid"):
            params["qid"] = qid(data)
        if label_lower_bound := self.dataset_params.get("label_lower_bound"):
            params["label_lower_bound"] = label_lower_bound(data)
        if label_upper_bound := self.dataset_params.get("label_upper_bound"):
            params["label_upper_bound"] = label_upper_bound(data)
        if feature_weights := self.dataset_params.get("feature_weights"):
            params["feature_weights"] = feature_weights(data)

        return params

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple["xgb.DMatrix", list[tuple["xgb.DMatrix", str]]]:
        import xgboost as xgb

        self.feature_names = (
            data.lazy().select(self.features_selector).collect_schema().names()
        )

        dtrain_params = self.make_dmatrix_params(data)
        dtrain = xgb.DMatrix(**dtrain_params, feature_names=self.feature_names)

        evals = []
        for name, valid_data in more_data.items():
            dvalid_params = self.make_dmatrix_params(valid_data)
            dvalid = xgb.DMatrix(**dvalid_params, feature_names=self.feature_names)
            evals.append((dvalid, name))

        evals.append((dtrain, "train"))
        return dtrain, evals

    def transform(self, data: DataFrame) -> DataFrame:
        import xgboost as xgb

        input = data.select(self.features_selector).to_pandas()
        input = xgb.DMatrix(input, feature_names=self.feature_names)

        boosters = self.get_booster()
        if isinstance(boosters, xgb.Booster):
            boosters = {self.prediction_name: boosters}

        predictions = []
        for name, b in boosters.items():
            pred: NDArray[Any] = b.predict(input, **self.predict_params)  # type: ignore
            predictions.append(
                pl.from_numpy(
                    pred,
                    schema=[name]
                    if pred.ndim == 1
                    else [f"{name}_{i}" for i in range(pred.shape[1])],
                )
            )

        return pl.concat([data, *predictions], how="horizontal")


class XGBoost(BaseXGBoost):
    def __init__(
        self,
        label: IntoExpr,
        params: dict[str, Any],
        *,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        dataset_params: XGBDatasetParams | None = None,
        train_params: XGBTrainParams | None = None,
        predict_params: XGBPredictParams | None = None,
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
        import xgboost as xgb

        dtrain, evals = self.make_train_valid_sets(data, **more_data)

        self.booster = xgb.train(
            self.params,
            dtrain,
            evals=evals,
            **self.train_params,
        )

        if self.out_dir:
            save_xgboost_booster(self.booster, self.out_dir)

        return self

    def get_booster(self) -> "xgb.Booster":
        return self.booster


def save_xgboost_booster(booster: "xgb.Booster", out_dir: str | Path):
    import json

    import matplotlib.pyplot as plt
    import xgboost as xgb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.json")

    config = json.loads(booster.save_config())
    with open(out_dir / "params.json", "w") as f:
        json.dump(config, f, indent=4)

    if feature_names := booster.feature_names:
        DataFrame(
            {
                "feature": feature_names,
                **{
                    importance_type: [
                        booster.get_score(importance_type=importance_type).get(fn, None)
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
        )

    xgb.plot_importance(booster, importance_type="gain")
    plt.savefig(out_dir / "importance_gain.png")
    plt.tight_layout()
    plt.close()

    xgb.plot_importance(booster, importance_type="weight")
    plt.savefig(out_dir / "importance_weight.png")
    plt.tight_layout()
    plt.close()
