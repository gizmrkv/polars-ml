from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Self, Sequence

import numpy as np
import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator, TransformerMixin


class SKLearnLinearModel(Transformer, HasFeatureImportance):
    def __init__(
        self,
        estimator: Any,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **fit_params: Any,
    ):
        self._estimator = estimator
        self._target_selector = target
        self._prediction = (
            [prediction] if isinstance(prediction, str) else list(prediction)
        )
        self._features_selector = (
            features if features is not None else cs.exclude(target)
        )
        self._fit_params = fit_params

        self._target: list[str] | None = None
        self._features: list[str] | None = None

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
    def estimator(self) -> BaseEstimator:
        return self._estimator

    def init_fit_params(self, data: DataFrame) -> dict[str, Any]:
        return self._fit_params

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        X = data.select(*self.features).to_numpy()
        y = data.select(*self.target).to_numpy().squeeze()

        self._fit_params = self.init_fit_params(data)

        self._estimator.fit(X, y, **self._fit_params)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self._target is None:
            raise NotFittedError()

        X = data.select(*self.features).to_numpy()
        pred = self._estimator.predict(X)
        return pl.from_numpy(pred, schema=self._prediction)

    def get_feature_importance(self) -> DataFrame:
        if self._target is None:
            raise NotFittedError()

        if hasattr(self._estimator, "coef_"):
            coef = self._estimator.coef_
            if len(coef.shape) == 1:
                return DataFrame({"feature": self.features, "coefficient": coef})
            else:
                data = {"feature": self.features}
                for i in range(coef.shape[0]):
                    data[f"coefficient_{i}"] = coef[i]
                return DataFrame(data)

        raise AttributeError(
            f"Estimator {type(self._estimator).__name__} has no coef_ attribute."
        )
