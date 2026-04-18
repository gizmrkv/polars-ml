from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class SKLearnLinear(Transformer):
    def __init__(
        self,
        estimator: Any,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
    ):
        self._estimator = estimator
        self._target_selector = target
        self._prediction = (
            [prediction] if isinstance(prediction, str) else list(prediction)
        )
        self._features_selector = (
            features if features is not None else cs.exclude(target)
        )

        self._target: list[str] | None = None
        self._features: list[str] | None = None
        self._fit_params: dict[str, Any] = {}

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

    def set_fit_params(self, **kwargs: Any) -> Self:
        self._fit_params = kwargs
        return self

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        X = data.select(*self.features).to_numpy()
        y = data.select(*self.target).to_numpy().squeeze()
        self._estimator.fit(X, y, **self._fit_params)

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._target is None:
            raise NotFittedError()

        X = data.select(*self.features).to_numpy()
        pred = self._estimator.predict(X)
        return pl.from_numpy(pred, schema=self._prediction)
