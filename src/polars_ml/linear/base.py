from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Self, Sequence

import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import HasFeatureImportance, Transformer


class BaseLinear(Transformer, HasFeatureImportance, ABC):
    def __init__(
        self,
        model: Any,
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str | Sequence[str] = "prediction",
    ):
        self.model = model
        self.label = label
        self.features_selector = features
        self.prediction_name = prediction_name

    @abstractmethod
    def get_model(self) -> Any: ...

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        if self.features_selector is None:
            label_cols = data.lazy().select(self.label).collect_schema().names()
            self.features_selector = cs.exclude(*label_cols)

        features = data.select(self.features_selector)
        label = data.select(self.label)
        self.feature_names = features.columns

        X = features.to_pandas()
        y = label.to_pandas().values.ravel()
        self.get_model().fit(X, y)

        return self

    def predict(self, data: DataFrame) -> NDArray:
        input_data = data.select(self.feature_names).to_pandas()
        return self.get_model().predict(input_data)

    def transform(self, data: DataFrame) -> DataFrame:
        pred = self.predict(data)
        name = self.prediction_name

        if isinstance(name, str):
            if pred.ndim == 1:
                schema = [name]
            elif pred.shape[1] == 1:
                schema = [name]
                pred = pred.reshape(-1)
            else:
                schema = [f"{name}_{i}" for i in range(pred.shape[1])]
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

    def get_feature_importance(self) -> DataFrame:
        model = self.get_model()
        coef = model.coef_

        if coef.ndim == 1:
            return DataFrame(
                {
                    "feature": self.feature_names,
                    "coefficient": coef,
                }
            )
        else:
            # Multi-class
            data = {"feature": self.feature_names}
            for i in range(coef.shape[0]):
                data[f"coefficient_class_{i}"] = coef[i]
            return DataFrame(data)
