from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Self,
    Sequence,
)

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import HasFeatureImportance, Transformer

if TYPE_CHECKING:
    import catboost


class CatBoost(Transformer, HasFeatureImportance):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        fit_params: Mapping[str, Any] | None = None,
        prediction_name: str | Sequence[str] = "prediction",
        save_dir: str | Path | None = None,
    ):
        self.label = label
        self.features_selector = features
        self.params = params
        self.fit_params = fit_params or {}
        self.prediction_name = prediction_name
        self.save_dir = Path(save_dir) if save_dir else None

    def get_booster(self) -> catboost.CatBoost:
        return self.model

    def create_pool(self, data: DataFrame) -> catboost.Pool:
        import catboost

        features = data.select(self.feature_names)
        label = data.select(self.label)

        return catboost.Pool(
            data=features.to_pandas(),
            label=label.to_pandas(),
            feature_names=self.feature_names,
        )

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import catboost

        if self.features_selector is None:
            label_cols = data.lazy().select(self.label).collect_schema().names()
            self.features_selector = cs.exclude(*label_cols)

        self.feature_names = data.select(self.features_selector).columns

        train_pool = self.create_pool(data)
        eval_sets = []
        for valid_data in more_data.values():
            eval_sets.append(self.create_pool(valid_data))

        self.model = catboost.CatBoost(dict(self.params))
        self.model.fit(
            train_pool,
            eval_set=eval_sets if eval_sets else None,
            **self.fit_params,
        )

        if self.save_dir:
            self.save(self.save_dir)

        return self

    def predict(self, data: DataFrame) -> NDArray:
        input_data = data.select(self.feature_names).to_pandas()
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
        save_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(save_dir / "model.cbm"))

    def get_feature_importance(self) -> DataFrame:
        booster = self.get_booster()
        importance = booster.get_feature_importance()
        return DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance,
            }
        )
